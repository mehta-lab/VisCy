"""Generate a549-mantis condition-pool manifests + splits.

Walks every authoring plate, partitions positions by (target, condition,
split), and writes the bundled ``dynacell._manifests`` registry:

- ``_manifests/a549-mantis-<target>-<condition>/manifest.yaml``
- ``_manifests/a549-mantis-<target>-<condition>/splits/<target>_train_test.yaml``

This is the single home for the registry (there is no longer a dual
``configs/`` + ``_configs/`` mirror); the resolver + name-lookup API read
these files via the ``dynacell.manifest_roots`` entry point.

The pool's FOV list is computed deterministically by replicating
``assemble_pool``'s plate-iteration order: plates are sorted by
``_list_authored_plates`` (filename order under ``authoring/platemaps/``),
and within each plate positions are read in the order the authoring splits
YAML lists them. Sequential indexing ``0/0/fov<NNNN>`` then maps onto the
assembled HCS store.

The manifest's top-level Z spacing is set per-pool from the dominant
contributing plate's source spacing (read live from the source plate
NGFF); XY is the assembly resample target ``TARGET_YX_PIXEL_SIZE_UM``, not
the source pitch, because the assembled stores are resampled to it. For
mixed-microscope pools the Z pick is an approximation; the per-position
OME-NGFF transform inside the assembled store is the source of truth at
metric-eval time.

Usage::

    uv run python -m dynacell.preprocess.a549_mantis.generate_pool_manifests
    # or write to a scratch registry (e.g. to diff before committing)
    uv run python -m dynacell.preprocess.a549_mantis.generate_pool_manifests \
        --manifests-root /tmp/scratch_manifests
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.resources import files
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from dynacell.preprocess.a549_mantis import GENE_TO_FILENAME, VALID_CONDITIONS
from dynacell.preprocess.a549_mantis.assemble import (
    _gather_plate_contributions,
    _list_authored_plates,
)
from dynacell.preprocess.a549_mantis.channels import CANONICAL_TARGET_CHANNELS

DEFAULT_MANIFESTS_ROOT = Path(str(files("dynacell") / "_manifests"))
DEFAULT_AUTHORING_ROOT = Path(str(files("dynacell") / "_configs" / "datasets" / "a549-mantis" / "authoring"))
DEFAULT_OUTPUT_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell/a549/mantis")
DEFAULT_PLATE_ZARR_ROOT = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")

GRID_STRIDE_H = 2.0
GRID_WINDOW = (5.0, 23.0)
GRID_TAIL_TOL_H = 1.5
T_CAP = 10
# XY pixel size of the assembled stores. All A549 mantis pools are resampled to
# this target at assembly time (_configs/preprocess/dynacell/a549_assemble_pool.yaml
# {train,test}_target_yx_pixel_size_um), so the manifest XY spacing must be the
# TARGET, not the source-plate pitch — the v2 nucleus/membrane plate is 0.116 at
# source but the store is resampled to 0.1494. The eval reads pixel_metrics.spacing
# from this manifest field (evaluation/_ref_hook.py), so it must match the store.
TARGET_YX_PIXEL_SIZE_UM = 0.1494

GENE_TO_ORGANELLE: dict[str, str] = {
    "sec61b": "er",
    "tomm20": "mitochondria",
    "h2b": "nuclei",
    "caax": "membrane",
}
GENE_TO_DISPLAY: dict[str, str] = {
    "sec61b": "ER (Sec61b)",
    "tomm20": "Mitochondria (TOMM20)",
    "h2b": "Nuclei (H2B)",
    "caax": "Membrane (CAAX)",
}
GENE_TO_TARGET_CHANNEL = CANONICAL_TARGET_CHANNELS
# Gene → on-disk store stem in mantis/. Nucleus (h2b) and membrane (caax)
# were merged into one combined store ``dual_nucl_memb`` (channels
# [Phase3D, Brightfield, Nuclei, Membrane]); ER/mito keep separate stores.
GENE_TO_STORE_STEM: dict[str, str] = {
    "sec61b": "SEC61B",
    "tomm20": "TOMM20",
    "h2b": "dual_nucl_memb",
    "caax": "dual_nucl_memb",
}


def _pool_fov_list(
    *,
    target: str,
    condition: str,
    split: str,
    plates: list[str],
    authoring_root: Path,
    plate_zarr_root: Path,
    pool_condition: str | None,
    start_index: int = 0,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Compute the deterministic ``0/0/fov<NNNN>`` list for one pool + plate spacings.

    Mirrors the iteration order in ``_assemble_one_pool``.

    ``pool_condition`` is the condition the on-disk store was assembled with --
    ``None`` for the all-condition ``<TARGET>_all`` train store -- and therefore
    fixes the fov index space. ``condition`` then selects which of that store's
    positions belong to this pool. The two differ on the train side, where every
    pool reads the same pooled store: filtering the walk by ``condition`` there
    renumbered the survivors from zero and produced indices into a store that
    does not exist.

    Returns
    -------
    fov_names : list[str]
        Pool position names ``"0/0/fov<NNNN>"`` in assembly order.
    plate_spacings : list[tuple[float, float, float]]
        One spacing per contributing plate, used by callers to derive the
        manifest-level spacing.
    """
    contribs = _gather_plate_contributions(
        target=target,
        condition=pool_condition,
        split=split,
        plates=plates,
        plate_zarr_root=plate_zarr_root,
        authoring_root=authoring_root,
        stride_h=GRID_STRIDE_H,
        window=GRID_WINDOW,
        tail_tol_h=GRID_TAIL_TOL_H,
        t_cap=T_CAP,
    )
    fov_names: list[str] = []
    plate_spacings: list[tuple[float, float, float]] = []
    idx = start_index
    for c in contribs:
        kept_any = False
        for well_id, _fov in c.positions:
            if c.platemap.wells[well_id].condition == condition:
                fov_names.append(f"0/0/fov{idx:04d}")
                kept_any = True
            idx += 1
        if kept_any:
            plate_spacings.append(c.spacing)
    return fov_names, plate_spacings


def _dominant_spacing(
    plate_spacings: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Pick the modal spacing across plates (with first-plate tiebreak)."""
    if not plate_spacings:
        return (0.174, 0.1494, 0.1494)
    counts = Counter(plate_spacings)
    most_common, _ = counts.most_common(1)[0]
    return most_common


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _splits_yaml(
    *,
    target: str,
    condition: str,
    train_fovs: list[str],
    test_fovs: list[str],
) -> dict:
    return {
        "split_version": "1.0",
        "random_seed": 0,
        "selection_criteria": {
            "source": "a549-mantis condition-pooled assembly",
            "target": target,
            "condition": condition,
            "pool_naming": "0/0/fov<NNNN> sequential across contributing plates",
        },
        "train": {"count": len(train_fovs), "fovs": train_fovs},
        "test": {"count": len(test_fovs), "fovs": test_fovs},
    }


def _manifest_yaml(
    *,
    target: str,
    condition: str,
    spacing: tuple[float, float, float],
    output_root: Path,
    splits_relpath: str,
) -> dict:
    name = f"a549-mantis-{target}-{condition.lower()}"
    store_stem = GENE_TO_STORE_STEM[target]
    # Train pools all conditions into ``<stem>_all.zarr`` (no per-condition
    # train store exists in mantis/); test is per-condition, casing verbatim.
    train_path = output_root / "train" / f"{store_stem}_all.zarr"
    test_path = output_root / "test" / f"{store_stem}_{condition}.zarr"
    # Eval reads whole-cell instances from a fixed cell_segmentation store
    # (Cellpose on VS nucleus+membrane, organelle-independent) and caches GT
    # features under gt_cache_dir. Both are additive stores fields the eval
    # chain depends on, so they must be emitted here (test store + _seg_cleaned;
    # eval_cache/<target>_<condition> under the a549 root).
    cell_segmentation_path = output_root / "test" / f"{store_stem}_{condition}_seg_cleaned.zarr"
    gt_cache_dir = output_root.parent / "eval_cache" / f"{target}_{condition.lower()}"
    return {
        "name": name,
        "version": "1",
        "description": (
            f"A549 mantis condition-pooled — {target} on {condition} "
            f"(pool-internal 0/0/fov<NNNN> naming, plate provenance in "
            f"per-position zattrs and the colocated provenance.json sidecar)."
        ),
        "cell_type": "A549",
        "imaging_modality": "mantis-lightsheet",
        "spacing": {"z": spacing[0], "y": spacing[1], "x": spacing[2]},
        "channels": {
            "source": "Phase3D",
            "auxiliary": ["Brightfield"],
        },
        "targets": {
            target: {
                "gene": GENE_TO_FILENAME[target],
                "organelle": GENE_TO_ORGANELLE[target],
                "display_name": GENE_TO_DISPLAY[target],
                "target_channel": GENE_TO_TARGET_CHANNEL[target],
                "stores": {
                    "train": str(train_path),
                    "test": str(test_path),
                    "cell_segmentation": str(cell_segmentation_path),
                    "gt_cache_dir": str(gt_cache_dir),
                },
                "splits": splits_relpath,
            }
        },
    }


def _generate_pair(
    *,
    target: str,
    condition: str,
    plates: list[str],
    authoring_root: Path,
    plate_zarr_root: Path,
    output_root: Path,
    dataset_dir: Path,
) -> bool:
    """Write manifest + splits for one (target, condition). Skip if pool empty."""
    train_fovs, train_spacings = _pool_fov_list(
        target=target,
        condition=condition,
        split="train",
        plates=plates,
        authoring_root=authoring_root,
        plate_zarr_root=plate_zarr_root,
        # Train reads the pooled <TARGET>_all store, assembled with condition=None.
        pool_condition=None,
    )
    test_fovs, test_spacings = _pool_fov_list(
        target=target,
        condition=condition,
        split="test",
        plates=plates,
        authoring_root=authoring_root,
        plate_zarr_root=plate_zarr_root,
        # Test stores are per-condition, so the pool condition IS the condition.
        pool_condition=condition,
    )
    if not train_fovs and not test_fovs:
        print(f"  skip {target}/{condition}: no contributing positions")
        return False

    # Z spacing = modal across both splits' contributing plates (Z is not
    # resampled); XY = the assembly resample target (see TARGET_YX_PIXEL_SIZE_UM).
    z_spacing = _dominant_spacing(train_spacings + test_spacings)[0]
    spacing = (z_spacing, TARGET_YX_PIXEL_SIZE_UM, TARGET_YX_PIXEL_SIZE_UM)

    splits_relpath = f"splits/{target}_train_test.yaml"
    splits_yaml = _splits_yaml(
        target=target,
        condition=condition,
        train_fovs=train_fovs,
        test_fovs=test_fovs,
    )
    manifest_yaml = _manifest_yaml(
        target=target,
        condition=condition,
        spacing=spacing,
        output_root=output_root,
        splits_relpath=splits_relpath,
    )

    _write_yaml(dataset_dir / "manifest.yaml", manifest_yaml)
    _write_yaml(dataset_dir / splits_relpath, splits_yaml)
    print(f"  wrote {dataset_dir.name}/manifest.yaml (train={len(train_fovs)} test={len(test_fovs)} spacing={spacing})")
    return True


def main() -> None:
    """Parse CLI args and generate one manifest + splits per (target, condition)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=DEFAULT_MANIFESTS_ROOT,
        help="Registry root; writes <root>/a549-mantis-<target>-<condition>/.",
    )
    parser.add_argument(
        "--authoring-root",
        type=Path,
        default=DEFAULT_AUTHORING_ROOT,
        help="Where authoring/platemaps and authoring/splits live.",
    )
    parser.add_argument(
        "--plate-zarr-root",
        type=Path,
        default=DEFAULT_PLATE_ZARR_ROOT,
        help="Source plate zarrs root (read-only at manifest gen time).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Pooled-store output root referenced in stores.{train,test}.",
    )
    args = parser.parse_args()

    authoring_root = args.authoring_root
    plates = _list_authored_plates(authoring_root)
    print(f"Authoring plates ({len(plates)}):")
    for p in plates:
        print(f"  {p}")

    print()
    print(f"Output root:    {args.output_root}")
    print(f"Manifests root: {args.manifests_root}")

    written = 0
    for target in sorted(GENE_TO_FILENAME):
        # The combined store is referenced by the h2b/caax manifests via
        # GENE_TO_STORE_STEM; no standalone dual manifest is emitted.
        if target == "dual_nucl_memb":
            continue
        for condition in VALID_CONDITIONS:
            name = f"a549-mantis-{target}-{condition.lower()}"
            dataset_dir = args.manifests_root / name
            if _generate_pair(
                target=target,
                condition=condition,
                plates=plates,
                authoring_root=authoring_root,
                plate_zarr_root=args.plate_zarr_root,
                output_root=args.output_root,
                dataset_dir=dataset_dir,
            ):
                written += 1
    print()
    print(f"Done. Wrote {written} pool manifest(s).")


if __name__ == "__main__":
    main()
