"""Build cpdino whole-cell ``_seg_cleaned`` masks from VSCyto3D ``_vs`` stores.

Eval-faithful cell segmentation for the A549 mantis regen campaign: replaces the
legacy cellpose-v3 ``6-segmentation`` chain with the exact Cellpose-DINO (cpdino)
backend the eval pipeline uses (:mod:`dynacell.evaluation.segmentation_cpdino`).

For each ``_vs.zarr`` position and timepoint, whole cells are segmented from the
``membrane_prediction`` channel on the max-Z-projection, the resulting 2-D label
footprint is repeated across Z, and a 3-D min-size filter (matching the legacy
``clean_up_seg.py`` whole-cell fragment cut) drops spurious instances. Output is a
single-channel ``["segmentation"]`` uint16 store ``<stem>_seg_cleaned.zarr``
colocated with the input (``SEC61B_mock_vs.zarr`` -> ``SEC61B_mock_seg_cleaned.zarr``).

The masks are organelle-independent (segmented from the shared nucleus/membrane VS
predictions), so one run per ``(plate, condition)`` serves every organelle. The
v3->cpdino switch intentionally shifts every downstream A549 feature/whole-cell-AP
metric to match the eval instance-seg backend.

Run in the cpdino-eval venv (cubic + cellpose-DINO)::

    UV_PROJECT_ENVIRONMENT=/hpc/mydata/alex.kalinin/VisCy-venvs/cpdino-eval \
        uv run --no-sync python applications/dynacell/tools/build_cpdino_seg_cleaned.py \
        --vs-store /hpc/projects/.../a549/mantis/test/SEC61B_mock_vs.zarr
"""

import argparse
from pathlib import Path

import numpy as np
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.segmentation_cellpose import load_cellpose_model
from dynacell.evaluation.segmentation_cpdino import segment_cpdino_instances

MEMBRANE_CHANNEL = "membrane_prediction"

# Whole-cell fragment cut, in voxels. Identical to the legacy
# ``clean_up_seg.py::MIN_SIZE_3D`` so the cpdino masks keep the same size floor as
# the cellpose-v3 ``_seg_cleaned`` they replace (a 2-D footprint repeated over
# Z=48 => ~2083 px^2 minimum cell area at 0.1494 um/px).
MIN_SIZE_3D = 100000

# Nominal lateral pixel size (um). Only forwarded to segment_cpdino_instances for
# the ``do_3d`` anisotropy term, which is unused on the 2-D max-projection path.
LATERAL_UM = 0.1494


def remove_small_instances_3d(label_vol: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 3-D instances below ``min_size`` voxels and relabel 1..N sequentially.

    Copied from the legacy ``clean_up_seg.py`` so the cleanup is byte-for-byte the
    same rule as the cellpose-v3 chain.

    Parameters
    ----------
    label_vol : numpy.ndarray
        3-D ``(Z, Y, X)`` integer instance-label volume.
    min_size : int
        Minimum instance size in voxels; smaller instances are set to background.

    Returns
    -------
    numpy.ndarray
        Relabeled ``(Z, Y, X)`` int64 volume with sequential ids 1..N.
    """
    if label_vol.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {label_vol.shape}")

    label_vol = label_vol.astype(np.int64, copy=False)

    counts = np.bincount(label_vol.ravel())
    small_labels = np.where(counts < min_size)[0]
    small_labels = small_labels[small_labels != 0]

    cleaned = label_vol.copy()
    if len(small_labels) > 0:
        cleaned[np.isin(cleaned, small_labels)] = 0

    unique_labels = np.unique(cleaned)
    unique_labels = unique_labels[unique_labels != 0]

    relabeled = np.zeros_like(cleaned, dtype=np.int64)
    for new_label, old_label in enumerate(unique_labels, start=1):
        relabeled[cleaned == old_label] = new_label

    return relabeled


def _seg_cleaned_path(vs_store: Path) -> Path:
    """Map ``<stem>_vs.zarr`` -> ``<stem>_seg_cleaned.zarr`` (colocated)."""
    stem = vs_store.stem
    base = stem[:-3] if stem.endswith("_vs") else stem
    return vs_store.parent / f"{base}_seg_cleaned.zarr"


def build_one(vs_store: Path, model) -> Path:
    """Segment one ``_vs.zarr`` into a cpdino ``_seg_cleaned.zarr``.

    Parameters
    ----------
    vs_store : pathlib.Path
        Input VSCyto3D prediction store with a ``membrane_prediction`` channel.
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose-DINO model.

    Returns
    -------
    pathlib.Path
        The written ``_seg_cleaned.zarr`` path.
    """
    out_path = _seg_cleaned_path(vs_store)
    with (
        open_ome_zarr(out_path, mode="w", layout="hcs", version="0.5", channel_names=["segmentation"]) as out,
        open_ome_zarr(vs_store, mode="r") as vs,
    ):
        n_pos = sum(1 for _ in vs.positions())
        for i, (pos_name, pos) in enumerate(vs.positions()):
            memb_idx = pos.get_channel_index(MEMBRANE_CHANNEL)
            t_len, _, d_len, h_len, w_len = pos.data.shape
            out_vol = np.zeros((t_len, 1, d_len, h_len, w_len), dtype=np.uint16)
            for t in range(t_len):
                memb = np.asarray(pos.data[t, memb_idx])  # (Z, Y, X)
                memb_2d = np.max(memb, axis=0)  # whole-cell max projection
                labels_2d = segment_cpdino_instances(memb_2d, (LATERAL_UM, LATERAL_UM), model, do_3d=False)
                vol = np.repeat(labels_2d[None, ...], d_len, axis=0)
                vol = remove_small_instances_3d(vol, MIN_SIZE_3D)
                out_vol[t, 0] = vol.astype(np.uint16)
            row, col, fov = pos_name.split("/")
            seg_pos = out.create_position(row, col, fov)
            seg_pos.create_image("0", out_vol)
            n_cells = int(out_vol.max())
            print(f"[{i + 1}/{n_pos}] {pos_name}: shape={out_vol.shape} max_label={n_cells}", flush=True)
    return out_path


def main() -> None:
    """Segment one ``_vs.zarr`` (``--vs-store``) into a cpdino ``_seg_cleaned.zarr``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vs-store",
        type=Path,
        required=True,
        help="Path to a VSCyto3D _vs.zarr store (membrane_prediction channel).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Load the model on CPU (debug only; cpdino inference is GPU-only).",
    )
    args = parser.parse_args()

    if not args.vs_store.exists():
        raise FileNotFoundError(f"_vs store not found: {args.vs_store}")

    model = load_cellpose_model(use_gpu=not args.no_gpu, model_name="cpdino")
    out_path = build_one(args.vs_store, model)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
