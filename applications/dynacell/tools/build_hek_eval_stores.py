#!/usr/bin/env python3
"""Build the HEK293T evaluation stores for the third-cell-type probe.

Derives one OME-Zarr store per QC-passing well of the mantis release's HEK
figure_3 subset, in the ``a549xy`` geometry the ``hek-mantis-*-a549xy`` manifests
declare. Evaluation-only: no HEK training happens, so no train store is built.

Three transformations, and nothing else:

1. **Channel rename** to the dynacell vocabulary, mirroring the A549 test stores
   (``CAAX_mock.zarr`` = ``[Phase3D, Brightfield, Membrane]``)::

       Phase       -> Phase3D          (source)
       Brightfield -> Brightfield      (auxiliary passthrough)
       Organelle   -> Membrane         (KRAS well: the KRAS-mNG target)
                   -> Structure        (TOMM70A well: the TOMM70A-mNG target)
       Membrane    -> Membrane_label   (the release's separate membrane label,
                                        present in BOTH wells; the cell-mask
                                        source, not a prediction target)

2. **XY resample 0.116 -> 0.1494 um**, reusing the A549 assembly's own
   :func:`~dynacell.preprocess.a549_mantis.assemble._resample_yx_to_pixel_size`
   so the interpolation is identical to the one the A549 training stores went
   through. This is the whole point: ``generate_pool_manifests`` records that the
   A549 v2 nucleus/membrane plate is 0.116 at source and its published store is
   resampled to 0.1494, i.e. HEK's native pitch IS the A549 source pitch. Feeding
   0.116 to models trained at 0.1494 would inject a 1.29x lateral scale shift
   that has nothing to do with cell type. Then centre-crop to a multiple of 32
   (the FCMAE stem / ViT tiling constraint).

3. **Metadata the eval and predict paths require**, neither of which the source
   store carries:
   - per-FOV ``normalization`` statistics. Mandatory, not defensive:
     ``NormalizeSampled.__call__`` indexes ``sample["norm_meta"][key][level]``
     with no fallback, so predict would die without them.
   - ``focus_slice`` zattrs, computed with the canonical estimator at the
     ``eval.yaml`` default ``lambda_ill``. The value MUST match what the eval and
     the GT prewarm use: ``focus.estimator_sig`` hashes
     ``{na_det, lambda_ill, pixel_size}`` into every deep-feature
     ``preprocess_version``, so a mismatch silently drops and re-extracts every
     prewarmed DINOv3 / DynaCLR / CellDINO / MorphEm cache.

Z is deliberately left at the native mantis 0.205 um: matching it to A549's 0.174
would mean upsampling 11 planes that were never measured. Lateral sampling is
matched, axial sampling is not, and the acquisition also differs optically from
A549 (NA_det 1.35 / lambda 0.5 vs the A549 preprocess configs' 1.25 / 0.405), so
this is "same instrument, different acquisition configuration" — never "same
microscope".

Usage::

    uv run python applications/dynacell/tools/build_hek_eval_stores.py --dry-run
    uv run python applications/dynacell/tools/build_hek_eval_stores.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.focus import write_focus_slice_metadata

# The A549 assembly's own resample / crop / zattr-filter helpers. Importing these
# rather than reimplementing them is deliberate: the scientific claim is that the
# HEK store went through the SAME lateral resample as the A549 training data.
from dynacell.preprocess.a549_mantis.assemble import (
    _OME_NGFF_VERSION,
    _Z_CHUNK,
    _apply_center_crop_yx,
    _provenance_only_zattrs,
    _resample_yx_to_pixel_size,
    _scale_transform,
)
from viscy_utils.meta_utils import generate_normalization_metadata

_SOURCE = Path(
    "/hpc/projects/comp.micro/mantis/mantis_paper_data_release/dynacell/HEK/figure_3_flipped_crop32_z64.zarr"
)
_OUT_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell/hek/mantis/test")

# Source geometry (verified against the store + the release README).
_SOURCE_YX_UM = 0.116
_SOURCE_Z_UM = 0.205
# A549 assembly target lateral pitch (assemble.TARGET_YX_PIXEL_SIZE_UM).
_TARGET_YX_UM = 0.1494
# Post-resample centre crop. 1248 * 0.116/0.1494 = 968.9, 1536 * 0.116/0.1494 =
# 1192.5, so the resample lands on 969 x 1193; crop to the enclosed multiples of 32.
_CROP_YX: tuple[int, int] = (960, 1184)

# Focus-estimator params. na_det / lambda_ill are the eval.yaml defaults ON
# PURPOSE (see module docstring): they feed the deep-feature cache tag, so the
# store, the GT prewarm and the eval must all agree. The HEK acquisition's own
# 1.35 / 0.5 is recorded in the table caption, not here.
_FOCUS_NA_DET = 1.35
_FOCUS_LAMBDA_ILL = 0.450
# The source was cropped to 64 planes with the focus plane at output index 24.
# Z is untouched here, so the estimator must still land there; a larger deviation
# means the resample corrupted Z and the run must stop.
_EXPECTED_FOCUS_PLANE = 24
_FOCUS_PLANE_TOLERANCE = 3


@dataclass(frozen=True)
class Well:
    """One QC-passing HEK well and its dynacell channel mapping."""

    name: str  # source well name, also the output store stem
    target_channel: str  # dynacell name for the well's `Organelle` channel
    manifest: str  # the manifest slug this store backs

    @property
    def out_path(self) -> Path:
        """Output store path, matching the ``stores.test`` entry in this well's manifest."""
        return _OUT_ROOT / f"{self.name}_a549xy.zarr"


_WELLS: tuple[Well, ...] = (
    Well(name="KRAS", target_channel="Membrane", manifest="hek-mantis-kras-a549xy"),
    Well(name="TOMM70A", target_channel="Structure", manifest="hek-mantis-tomm70a-a549xy"),
)

_SOURCE_CHANNELS = ("Phase", "Organelle", "Membrane", "Brightfield")


def _channel_plan(well: Well) -> tuple[list[int], list[str]]:
    """Return (source channel indices, output names) in A549 channel order.

    A549 test stores lead with ``[Phase3D, Brightfield, <target>]``; the HEK
    stores append ``Membrane_label`` (the cell-mask source) as a 4th channel.
    """
    order = [
        ("Phase", "Phase3D"),
        ("Brightfield", "Brightfield"),
        ("Organelle", well.target_channel),
        ("Membrane", "Membrane_label"),
    ]
    return [_SOURCE_CHANNELS.index(src) for src, _ in order], [out for _, out in order]


def build_store(well: Well, source: Path, dry_run: bool = False) -> Path:
    """Build one well's ``a549xy`` evaluation store and return its path."""
    indices, out_names = _channel_plan(well)
    out_path = well.out_path
    print(f"\n=== {well.name} -> {out_path}")
    print(f"    channels {[_SOURCE_CHANNELS[i] for i in indices]} -> {out_names}")

    with open_ome_zarr(source, mode="r", layout="hcs") as src:
        if list(src.channel_names) != list(_SOURCE_CHANNELS):
            raise ValueError(f"unexpected source channels {src.channel_names!r}; expected {list(_SOURCE_CHANNELS)}")
        positions = [(name, pos) for name, pos in src.positions() if name.split("/")[1] == well.name]
        if len(positions) != 4:
            raise ValueError(f"expected 4 QC-passing FOVs in well {well.name}, found {len(positions)}")

        if dry_run:
            for pos_name, pos in positions:
                print(f"    {pos_name} src shape {pos['0'].shape} scale {pos.scale}")
            return out_path

        if out_path.exists():
            raise FileExistsError(f"{out_path} already exists; remove it explicitly before rebuilding")

        with open_ome_zarr(
            out_path,
            mode="w",
            layout="hcs",
            channel_names=out_names,
            version=_OME_NGFF_VERSION,
        ) as writer:
            # Provenance only. Copying the source `ome` block would make this
            # store report the source's channel names, so channel_names.index
            # ("Phase3D") would raise in the focus pass and the normalization
            # keys would land under the old names.
            writer.zattrs.update(_provenance_only_zattrs(src.zattrs))
            for pos_name, src_pos in positions:
                row, col, fov = pos_name.split("/")
                block = np.asarray(src_pos["0"].oindex[:, indices])
                resampled = _resample_yx_to_pixel_size(
                    block,
                    source_y_um=_SOURCE_YX_UM,
                    source_x_um=_SOURCE_YX_UM,
                    target_yx_um=_TARGET_YX_UM,
                )
                cropped = _apply_center_crop_yx(resampled, _CROP_YX)
                t_dim, c_dim, z_dim, y_dim, x_dim = cropped.shape
                if (y_dim, x_dim) != _CROP_YX:
                    raise ValueError(f"{pos_name}: crop produced {(y_dim, x_dim)}, expected {_CROP_YX}")
                if z_dim != block.shape[2]:
                    raise ValueError(f"{pos_name}: Z changed ({block.shape[2]} -> {z_dim}); Z must be untouched")
                z_chunk = min(_Z_CHUNK, z_dim)
                out_pos = writer.create_position(row, col, fov)
                out_pos.create_image(
                    name="0",
                    data=cropped.astype(np.float32, copy=False),
                    chunks=(1, 1, z_chunk, y_dim, x_dim),
                    shards_ratio=(1, c_dim, (z_dim + z_chunk - 1) // z_chunk, 1, 1),
                    transform=_scale_transform((_SOURCE_Z_UM, _TARGET_YX_UM, _TARGET_YX_UM)),
                )
                # Keeps the release's `z_focus_crop` provenance (not NGFF-managed).
                out_pos.zattrs.update(_provenance_only_zattrs(src_pos.zattrs))
                print(f"    {pos_name}: {block.shape} -> {cropped.shape}")

    # Only Phase3D and the target are ever normalized by the predict/eval
    # transforms, and _grid_sample's strided read decompresses whole shards, so
    # restrict the pass rather than walking all four channels.
    norm_channels = [out_names.index("Phase3D"), out_names.index(well.target_channel)]
    print(f"    normalization metadata for channels {norm_channels} ({[out_names[i] for i in norm_channels]})")
    generate_normalization_metadata(out_path, channel_ids=norm_channels)

    print(f"    focus_slice (na_det={_FOCUS_NA_DET}, lambda_ill={_FOCUS_LAMBDA_ILL}, px={_TARGET_YX_UM})")
    stats = write_focus_slice_metadata(
        str(out_path),
        channel_name="Phase3D",
        na_det=_FOCUS_NA_DET,
        lambda_ill=_FOCUS_LAMBDA_ILL,
        pixel_size=_TARGET_YX_UM,
    )
    print(f"    focus planes: {stats}")
    return out_path


def _verify_channel_identity(well: Well, source: Path) -> list[str]:
    """Check each output channel really holds the SOURCE channel it is named for.

    The read in :func:`build_store` is ``oindex[:, indices]`` with ``indices``
    unsorted (``[0, 3, 1, 2]``), while ``channel_names`` is set independently from
    ``out_names``. zarr does honour the requested order, but nothing in the write
    path enforces it: were a selection ever returned in ascending order instead,
    the mito store's ``Structure`` would silently hold the release's ``Membrane``
    label and ``Brightfield`` would hold the mNG target -- with no error, and
    passing every other check here. So compare identity, not just names.

    Mean and std are compared rather than pixels because the output is laterally
    resampled; interpolation preserves both to well within this tolerance, while
    the source channels differ from each other by orders of magnitude (phase
    ~1e-3, fluorescence ~1e2, brightfield ~2e3), so any mix-up is unmissable.

    Tolerance is scaled by the source channel's own std, NOT by the statistic's
    magnitude. Phase is zero-centred (mean ~1e-6), so a relative test on its mean
    compares two rounding-noise values and always fails; its std is the only
    meaningful scale it has.
    """
    errors: list[str] = []
    indices, out_names = _channel_plan(well)
    with (
        open_ome_zarr(source, mode="r", layout="hcs") as src,
        open_ome_zarr(well.out_path, mode="r", layout="hcs") as out,
    ):
        src_pos = next(p for n, p in src.positions() if n.split("/")[1] == well.name)
        out_pos = next(p for _, p in out.positions())
        z = _EXPECTED_FOCUS_PLANE
        for out_idx, (src_idx, out_name) in enumerate(zip(indices, out_names, strict=True)):
            a = np.asarray(src_pos["0"][0, src_idx, z])
            b = np.asarray(out_pos["0"][0, out_idx, z])
            scale = max(abs(float(np.mean(a))), float(np.std(a)), 1e-12)
            for stat, fn in (("mean", np.mean), ("std", np.std)):
                want, got = float(fn(a)), float(fn(b))
                if abs(got - want) / scale > 0.05:
                    errors.append(
                        f"{well.out_path.name}: channel {out_name!r} (out index {out_idx}) {stat} {got:.6g} "
                        f"does not match source {_SOURCE_CHANNELS[src_idx]!r} {stat} {want:.6g} -- the "
                        f"channel selection did not preserve the requested order"
                    )
    return errors


def verify_store(well: Well) -> list[str]:
    """Re-open a built store and check every contract the manifests rely on."""
    errors: list[str] = []
    out_path = well.out_path
    _, out_names = _channel_plan(well)
    with open_ome_zarr(out_path, mode="r", layout="hcs") as plate:
        if list(plate.channel_names) != out_names:
            errors.append(f"{out_path.name}: channel_names {plate.channel_names!r} != {out_names!r}")
        for pos_name, pos in plate.positions():
            arr = pos["0"]
            if arr.shape[-2:] != _CROP_YX:
                errors.append(f"{pos_name}: YX {arr.shape[-2:]} != {_CROP_YX}")
            if arr.dtype != np.float32:
                errors.append(f"{pos_name}: dtype {arr.dtype} != float32")
            spacing = tuple(round(pos.scale[pos.get_axis_index(a)], 6) for a in "zyx")
            expected = (_SOURCE_Z_UM, _TARGET_YX_UM, _TARGET_YX_UM)
            if spacing != expected:
                errors.append(f"{pos_name}: spacing {spacing} != {expected}")
            norm = pos.zattrs.get("normalization", {})
            for channel in ("Phase3D", well.target_channel):
                stats = norm.get(channel, {}).get("fov_statistics", {})
                if not {"mean", "std"} <= set(stats):
                    errors.append(f"{pos_name}: normalization.{channel}.fov_statistics missing mean/std")
            planes = pos.zattrs.get("focus_slice", {}).get("Phase3D", {}).get("per_timepoint", {})
            if not planes:
                errors.append(f"{pos_name}: focus_slice.Phase3D.per_timepoint absent")
            else:
                plane = int(planes["0"])
                if abs(plane - _EXPECTED_FOCUS_PLANE) > _FOCUS_PLANE_TOLERANCE:
                    errors.append(
                        f"{pos_name}: focus plane {plane} is more than {_FOCUS_PLANE_TOLERANCE} from the "
                        f"expected {_EXPECTED_FOCUS_PLANE} (Z was not resampled, so this suggests the "
                        f"XY resample corrupted the volume)"
                    )
                else:
                    print(f"    {pos_name}: focus plane {plane} (expected ~{_EXPECTED_FOCUS_PLANE})")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Build (or dry-run) the HEK evaluation stores and verify them."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=_SOURCE, help=f"HEK source store (default: {_SOURCE})")
    ap.add_argument(
        "--well",
        action="append",
        choices=[w.name for w in _WELLS],
        help="build only this well (repeatable); default builds both",
    )
    ap.add_argument("--dry-run", action="store_true", help="report the plan and source shapes, write nothing")
    args = ap.parse_args(argv)

    wells = [w for w in _WELLS if args.well is None or w.name in args.well]
    # --dry-run "writes nothing"; creating the output root is still a write.
    if not args.dry_run:
        _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for well in wells:
        build_store(well, args.source, dry_run=args.dry_run)
        if not args.dry_run:
            errors.extend(verify_store(well))
            errors.extend(_verify_channel_identity(well, args.source))

    if errors:
        print("\n[FAIL] store verification found problems:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\n[ok] " + ("dry run complete" if args.dry_run else f"{len(wells)} store(s) built and verified"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
