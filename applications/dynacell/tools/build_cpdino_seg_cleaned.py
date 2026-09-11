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

# Channel whose focus_slice zattrs drive --focus-slab-halfwidth.
FOCUS_CHANNEL = "Phase3D"

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

    # Single-pass LUT: drop-and-relabel in one gather instead of a full-volume
    # boolean comparison per surviving instance. The original loop was
    # O(n_instances * n_voxels) — on a 64x960x1184 HEK volume with a few hundred
    # cells that dominated the entire segmentation run. Semantics are unchanged:
    # labels under min_size go to background, survivors are numbered 1..N in
    # ascending original-label order (what np.unique gave).
    counts = np.bincount(label_vol.ravel())
    # `counts > 0` is load-bearing, not redundant: bincount yields a zero count
    # for every label id absent from the volume, and those must stay background.
    # Without it, min_size=0 would resurrect absent ids as instances, whereas the
    # original walked np.unique and only ever saw present labels.
    keep = np.where((counts >= min_size) & (counts > 0))[0]
    keep = keep[keep != 0]
    lut = np.zeros(counts.size, dtype=np.int64)
    lut[keep] = np.arange(1, keep.size + 1, dtype=np.int64)
    return lut[label_vol]


def _membrane_projection(memb: np.ndarray, position, t: int, halfwidth: int | None) -> np.ndarray:
    """Max-project the membrane volume for 2-D whole-cell segmentation.

    ``halfwidth=None`` projects the whole stack — the A549 behaviour, where 48
    planes at 0.174 um is a ~8.4 um slab and the walls stay crisp. On a deeper
    stack that projection superimposes basal and apical membranes of different
    cells and cpdino under-segments badly (HEK: 64 planes at 0.205 um is ~13.1 um,
    and HEK cells grow in 3-D, which is why the release ships >48 planes). Passing
    a halfwidth restricts the projection to ``focus_plane +/- halfwidth``, read
    from the store's ``focus_slice`` zattrs — the same in-focus-slab idea the eval
    itself uses for its 2-D instance path and deep-feature crops.
    """
    if halfwidth is None:
        return np.max(memb, axis=0)
    focus = position.zattrs.get("focus_slice", {}).get(FOCUS_CHANNEL, {}).get("per_timepoint", {})
    if str(t) not in focus:
        raise ValueError(
            f"--focus-slab-halfwidth needs focus_slice.{FOCUS_CHANNEL}.per_timepoint[{t}] zattrs; "
            "write them with dynacell.evaluation.focus.write_focus_slice_metadata first"
        )
    plane = int(focus[str(t)])
    lo = max(0, plane - halfwidth)
    hi = min(memb.shape[0], plane + halfwidth + 1)
    return np.max(memb[lo:hi], axis=0)


def _seg_cleaned_path(vs_store: Path) -> Path:
    """Map ``<stem>_vs.zarr`` -> ``<stem>_seg_cleaned.zarr`` (colocated)."""
    stem = vs_store.stem
    base = stem[:-3] if stem.endswith("_vs") else stem
    return vs_store.parent / f"{base}_seg_cleaned.zarr"


def build_one(
    vs_store: Path,
    model,
    membrane_channel: str = MEMBRANE_CHANNEL,
    min_size_3d: int = MIN_SIZE_3D,
    focus_slab_halfwidth: int | None = None,
    force: bool = False,
) -> Path:
    """Segment one store's membrane channel into a cpdino ``_seg_cleaned.zarr``.

    Parameters
    ----------
    vs_store : pathlib.Path
        Input store carrying a whole-cell membrane channel. Normally a VSCyto3D
        ``_vs.zarr`` (virtually stained ``membrane_prediction``); the HEK
        third-cell-type probe instead points this at an experimental membrane
        label (``Membrane_label``) in its GT store, since it has no VS store.
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose-DINO model.
    membrane_channel : str
        Channel to segment whole cells from.
    min_size_3d : int
        Instance size floor in voxels. Scales with Z and pixel size: the A549
        floor of 100000 is ~2083 px^2 over Z=48 at 0.1494 um/px (~46.5 um^2), so
        a store with a different depth needs that area x its own Z.
    focus_slab_halfwidth : int or None
        ``None`` (default) max-projects the full stack, preserving the A549
        behaviour. An int restricts the projection to ``focus_plane +/- hw``; see
        :func:`_membrane_projection`.
    force : bool
        Overwrite an existing output store. Off by default: ``mode="w"`` destroys
        the destination at ``__enter__``, so a rerun that fails afterwards would
        leave no segmentation at all.

    Returns
    -------
    pathlib.Path
        The written ``_seg_cleaned.zarr`` path.

    Raises
    ------
    FileExistsError
        If the output store exists and ``force`` is not set.
    ValueError
        If ``membrane_channel`` is not a channel of ``vs_store``.
    """
    out_path = _seg_cleaned_path(vs_store)
    # Everything that can raise must run BEFORE the writer opens: iohub's
    # mode="w" destroys the destination at __enter__ and only logs a warning, so
    # a later failure (bad channel, OOM, unreadable input) would leave the
    # canonical io.cell_segmentation_path store truncated and invalid.
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass --force to rebuild")
    with open_ome_zarr(vs_store, mode="r") as vs:
        if membrane_channel not in vs.channel_names:
            raise ValueError(
                f"channel {membrane_channel!r} not in {vs_store}; available channels: {list(vs.channel_names)}"
            )
    with (
        open_ome_zarr(out_path, mode="w", layout="hcs", version="0.5", channel_names=["segmentation"]) as out,
        open_ome_zarr(vs_store, mode="r") as vs,
    ):
        n_pos = sum(1 for _ in vs.positions())
        for i, (pos_name, pos) in enumerate(vs.positions()):
            memb_idx = pos.get_channel_index(membrane_channel)
            t_len, _, d_len, h_len, w_len = pos.data.shape
            out_vol = np.zeros((t_len, 1, d_len, h_len, w_len), dtype=np.uint16)
            for t in range(t_len):
                memb = np.asarray(pos.data[t, memb_idx])  # (Z, Y, X)
                memb_2d = _membrane_projection(memb, pos, t, focus_slab_halfwidth)
                labels_2d = segment_cpdino_instances(memb_2d, (LATERAL_UM, LATERAL_UM), model, do_3d=False)
                vol = np.repeat(labels_2d[None, ...], d_len, axis=0)
                vol = remove_small_instances_3d(vol, min_size_3d)
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
        help="Store holding a whole-cell membrane channel. Normally a VSCyto3D _vs.zarr "
        "(membrane_prediction); any store works via --membrane-channel, e.g. the HEK GT "
        "stores' experimental Membrane_label. Output is <stem>_seg_cleaned.zarr alongside it.",
    )
    parser.add_argument(
        "--membrane-channel",
        default=MEMBRANE_CHANNEL,
        help="Channel to segment whole cells from (default: %(default)s). The HEK probe "
        "uses Membrane_label, its experimental membrane label, having no _vs store.",
    )
    parser.add_argument(
        "--min-size-3d",
        type=int,
        default=MIN_SIZE_3D,
        help="Instance size floor in voxels (default: %(default)s, the A549 floor at Z=48 "
        "and 0.1494 um/px). Scale by the store's own Z to keep the physical area equal.",
    )
    parser.add_argument(
        "--focus-slab-halfwidth",
        type=int,
        default=None,
        help="Max-project only focus_plane +/- N instead of the whole stack (default: whole "
        "stack, the A549 behaviour). Needed on deep stacks where a full projection "
        "superimposes membranes from different cells; requires focus_slice zattrs.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Load the model on CPU (debug only; cpdino inference is GPU-only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing <stem>_seg_cleaned.zarr. Without this a rerun "
        "refuses rather than truncating the store the eval campaign reads.",
    )
    args = parser.parse_args()

    if not args.vs_store.exists():
        raise FileNotFoundError(f"store not found: {args.vs_store}")

    model = load_cellpose_model(use_gpu=not args.no_gpu, model_name="cpdino")
    out_path = build_one(
        args.vs_store,
        model,
        membrane_channel=args.membrane_channel,
        min_size_3d=args.min_size_3d,
        focus_slab_halfwidth=args.focus_slab_halfwidth,
        force=args.force,
    )
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
