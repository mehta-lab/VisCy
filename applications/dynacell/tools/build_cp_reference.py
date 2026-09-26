r"""Build a target's CP (GLCM+) reference from existing GT CP feature caches.

The reference fixes the CP feature space every eval of a target is scored in: a
feature keep-mask selected on GT cells only, plus one shared per-feature mean/std
over the same pooled GT (see :mod:`dynacell.evaluation.cp_reference`). It is fit
on the target's benchmark test sets -- iPSC + A549 mock/denv/zikv; HEK and the
movie sets are excluded.

This tool only READS. Each dataset ref is resolved exactly as an eval resolves it
(``apply_dataset_ref`` on the packaged ``eval.yaml`` defaults), the GT cache is
opened through the pipeline's own ``init_cache_context`` with
``require_complete_cache=true`` (so a cache of another dataset or CP recipe is
refused), and every ``(position, timepoint)`` of the GT store must already be in
the CP cache. A miss raises; features are never computed. Non-finite cells are
dropped the way the pipeline drops them (``drop_paired_nonfinite_rows``).

Run::

    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py \
        --target er --dry-run
    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py \
        --target er                  # writes the registry path under DATA_ROOT
    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py \
        --target er --out /tmp/er__3d.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf

import dynacell.evaluation
from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.cache import StaleCacheError, open_features_group, read_features_from_group
from dynacell.evaluation.cp_reference import (
    CP_REFERENCE_DIMENSION,
    fit_cp_reference,
    write_cp_reference,
)
from dynacell.evaluation.metrics import CP_FEATURE_VERSION, active_cp_feature_names, drop_paired_nonfinite_rows
from dynacell.evaluation.paths import cp_reference_path
from dynacell.evaluation.pipeline_cache import cp_recipe_identity, init_cache_context

#: The fit set per eval ``target_name``: iPSC + A549 mock/denv/zikv, as
#: ``(dataset, manifest target)`` refs -- the same refs the grouped leaves carry in
#: ``benchmark.dataset_ref``. HEK and the movie sets are deliberately absent.
DEFAULT_DATASETS: dict[str, tuple[tuple[str, str], ...]] = {
    "nucleus": (
        ("aics-hipsc", "nucleus"),
        ("a549-mantis-h2b-mock", "h2b"),
        ("a549-mantis-h2b-denv", "h2b"),
        ("a549-mantis-h2b-zikv", "h2b"),
    ),
    "membrane": (
        ("aics-hipsc", "membrane"),
        ("a549-mantis-caax-mock", "caax"),
        ("a549-mantis-caax-denv", "caax"),
        ("a549-mantis-caax-zikv", "caax"),
    ),
    "er": (
        ("aics-hipsc", "sec61b"),
        ("a549-mantis-sec61b-mock", "sec61b"),
        ("a549-mantis-sec61b-denv", "sec61b"),
        ("a549-mantis-sec61b-zikv", "sec61b"),
    ),
    "mitochondria": (
        ("aics-hipsc", "tomm20"),
        ("a549-mantis-tomm20-mock", "tomm20"),
        ("a549-mantis-tomm20-denv", "tomm20"),
        ("a549-mantis-tomm20-zikv", "tomm20"),
    ),
}

_EVAL_YAML = Path(dynacell.evaluation.__file__).parent / "_configs" / "eval.yaml"


@dataclass
class DatasetCells:
    """Finite GT CP cells read from one dataset's cache, plus what was read."""

    record: dict[str, Any]
    cells: np.ndarray


def eval_config_for(target_name: str, dataset: str, manifest_target: str) -> DictConfig:
    """Return the packaged eval defaults resolved for one dataset ref, cache reads mandatory."""
    config = OmegaConf.load(_EVAL_YAML)
    config.target_name = target_name
    config.benchmark = {"dataset_ref": {"dataset": dataset, "target": manifest_target}}
    config.io.require_complete_cache = True
    apply_dataset_ref(config)
    return config


def cp_identity_for(config: DictConfig) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the CP recipe identity and column names an eval of ``config`` would use."""
    norm = OmegaConf.to_container(config.feature_metrics.cp.norm, resolve=True)
    glcm = OmegaConf.to_container(config.feature_metrics.cp.glcm, resolve=True)
    return (
        cp_recipe_identity(CP_FEATURE_VERSION, norm, glcm),
        tuple(active_cp_feature_names(bool(glcm["enabled"]))),
    )


def read_dataset_cells(config: DictConfig, n_features: int) -> DatasetCells:
    """Read every GT CP cell of one dataset from its cache, read-only.

    Parameters
    ----------
    config : DictConfig
        Output of :func:`eval_config_for`.
    n_features : int
        Expected CP column count.

    Returns
    -------
    DatasetCells
        Finite cells stacked in position/timepoint order, and the provenance record.

    Raises
    ------
    StaleCacheError
        If the cache belongs to another store or CP recipe, a ``(position,
        timepoint)`` of the GT store is missing from it, or an entry has the wrong
        column count.
    ValueError
        If a GT position is not a 3-D volume.
    """
    ctx = init_cache_context(config, side="gt")
    blocks: list[np.ndarray] = []
    n_positions = n_timepoints = n_dropped = 0
    with (
        open_ome_zarr(Path(config.io.gt_path), mode="r") as plate,
        open_features_group(ctx.paths, "cp", mode="r") as group,
    ):
        if group is None:
            raise StaleCacheError(f"no CP feature cache at {ctx.paths.cp_features()}")
        for pos_name, pos in plate.positions():
            t_count, _, z_depth = pos.data.shape[:3]
            if z_depth < 2:
                raise ValueError(f"{config.io.gt_path}/{pos_name} has Z={z_depth}; the CP reference is 3-D")
            n_positions += 1
            for t in range(t_count):
                feats = read_features_from_group(group, pos_name, t)
                if feats is None:
                    raise StaleCacheError(f"CP cache miss at {pos_name}/t{t} in {ctx.paths.cp_features()}")
                n_timepoints += 1
                if feats.shape[0] == 0:
                    continue
                if feats.shape[1] != n_features:
                    raise StaleCacheError(f"{pos_name}/t{t} has {feats.shape[1]} CP columns, expected {n_features}")
                finite, _ = drop_paired_nonfinite_rows(feats, feats)
                n_dropped += feats.shape[0] - finite.shape[0]
                blocks.append(np.asarray(finite, dtype=np.float64))
    cells = np.concatenate(blocks, axis=0) if blocks else np.empty((0, n_features))
    record = {
        "dataset": config.benchmark.dataset_ref.dataset,
        "target": config.benchmark.dataset_ref.target,
        "gt_path": str(config.io.gt_path),
        "gt_cache_dir": str(ctx.paths.root),
        "cp_cache_path": str(ctx.paths.cp_features()),
        "n_positions": n_positions,
        "n_timepoints": n_timepoints,
        "n_cells": int(cells.shape[0]),
        "n_cells_dropped_nonfinite": int(n_dropped),
    }
    return DatasetCells(record=record, cells=cells)


def build(
    target_name: str, datasets: list[tuple[str, str]], dimension: str
) -> tuple[list[DatasetCells], dict[str, Any], tuple[str, ...]]:
    """Read every fit dataset and check they share one CP recipe.

    Returns
    -------
    tuple
        ``(per-dataset cells, cp_identity, feature_names)``.

    Raises
    ------
    ValueError
        If ``dimension`` is not :data:`CP_REFERENCE_DIMENSION`, or the datasets
        disagree on the CP recipe.
    """
    if dimension != CP_REFERENCE_DIMENSION:
        raise ValueError(f"only the {CP_REFERENCE_DIMENSION!r} CP reference exists; got {dimension!r}")
    identity: dict[str, Any] | None = None
    names: tuple[str, ...] = ()
    read: list[DatasetCells] = []
    for dataset, manifest_target in datasets:
        config = eval_config_for(target_name, dataset, manifest_target)
        ds_identity, ds_names = cp_identity_for(config)
        if identity is None:
            identity, names = ds_identity, ds_names
        elif (ds_identity, ds_names) != (identity, names):
            raise ValueError(f"{dataset} resolves a different CP recipe: {ds_identity} vs {identity}")
        read.append(read_dataset_cells(config, len(names)))
    if identity is None:
        raise ValueError("no datasets given")
    return read, identity, names


def _parse_dataset(token: str) -> tuple[str, str]:
    """Parse a ``dataset:target`` CLI token."""
    dataset, sep, target = token.partition(":")
    if not sep or not dataset or not target:
        raise argparse.ArgumentTypeError(f"expected DATASET:TARGET, got {token!r}")
    return dataset, target


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--dimension", default=CP_REFERENCE_DIMENSION, choices=[CP_REFERENCE_DIMENSION])
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=_parse_dataset,
        default=None,
        metavar="DATASET:TARGET",
        help="fit datasets as manifest refs (default: the target's iPSC + A549 mock/denv/zikv)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output JSON (default: the registry path)")
    parser.add_argument("--dry-run", action="store_true", help="read and count cells; write nothing")
    args = parser.parse_args(argv)

    datasets = args.datasets if args.datasets is not None else list(DEFAULT_DATASETS[args.target])
    out = args.out if args.out is not None else cp_reference_path(args.target, args.dimension)
    start = time.perf_counter()
    read, identity, names = build(args.target, datasets, args.dimension)
    for ds in read:
        r = ds.record
        print(
            f"{r['dataset']}:{r['target']}  {r['n_positions']} positions, {r['n_timepoints']} timepoints, "
            f"{r['n_cells']} cells ({r['n_cells_dropped_nonfinite']} non-finite dropped)\n    {r['cp_cache_path']}"
        )
    print(f"total {sum(ds.record['n_cells'] for ds in read)} cells, {len(names)} CP features, identity {identity}")
    if args.dry_run:
        print(f"dry run: would write {out} ({time.perf_counter() - start:.1f} s)")
        return
    payload = fit_cp_reference(
        np.concatenate([ds.cells for ds in read], axis=0),
        target_name=args.target,
        dimension=args.dimension,
        feature_names=names,
        cp_identity=identity,
        datasets=[ds.record for ds in read],
    )
    write_cp_reference(payload, out)
    print(
        f"kept {len(payload['kept_feature_names'])}/{len(names)}: {payload['kept_feature_names']}\n"
        f"wrote {out} sha256={payload['sha256']} ({time.perf_counter() - start:.1f} s)"
    )


if __name__ == "__main__":
    main()
