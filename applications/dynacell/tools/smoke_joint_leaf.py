r"""CPU-only smoke for BatchedConcatDataModule joint train leaves.

Composes the leaf, instantiates the datamodule end-to-end via
``jsonargparse`` (matches LightningCLI's recursive type-driven
instantiation), opens the real zarrs, and iterates a handful of train
and val batches to confirm the loader/transform wiring holds.

The GPU augmentation pipeline runs in ``on_after_batch_transfer`` and
needs an actual GPU — this smoke explicitly does not cover it. For a
full end-to-end smoke including GPU transforms + forward + backward,
use ``uv run dynacell fit -c <leaf> --trainer.fast_dev_run=5`` on a
GPU node (disables checkpointing and loggers; 5 train + 5 val batches).

Usage
-----
    # default: celldiff ER joint leaf with small ipsc override
    uv run python applications/dynacell/tools/smoke_joint_leaf.py

    # different leaf, same ipsc override:
    uv run python applications/dynacell/tools/smoke_joint_leaf.py \\
        --leaf applications/dynacell/configs/benchmarks/virtual_staining/\\
mito/celldiff/joint_ipsc_confocal_a549_mantis/train.yml

    # full-size ipsc (slower startup but exercises the real path):
    uv run python applications/dynacell/tools/smoke_joint_leaf.py --no-ipsc-override
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

from jsonargparse import ArgumentParser

from viscy_data.combined import BatchedConcatDataModule
from viscy_utils.compose import load_composed_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LEAF = (
    _REPO_ROOT / "applications/dynacell/configs/benchmarks/virtual_staining/"
    "er/celldiff/joint_ipsc_confocal_a549_mantis/train.yml"
)
_SMALL_IPSC_SEC61B = "/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/SEC61B_test12.zarr"
_MAX_TRAIN_BATCHES = 3
_MAX_VAL_BATCHES = 1


def _instantiate(data_cfg: dict) -> BatchedConcatDataModule:
    """Instantiate via jsonargparse — recurses into data_modules + transforms."""
    parser = ArgumentParser()
    parser.add_subclass_arguments(BatchedConcatDataModule, "data")
    ns = parser.parse_object({"data": data_cfg})
    ns = parser.instantiate_classes(ns)
    return ns.data


def _apply_overrides(data_cfg: dict, ipsc_override: str | None) -> None:
    """Replace ipsc data_path (in-place) and drop preload/persistent-worker settings."""
    children = data_cfg["init_args"]["data_modules"]
    if ipsc_override:
        children[0]["init_args"]["data_path"] = ipsc_override
    for ch in children:
        ch["init_args"]["mmap_preload"] = False
        ch["init_args"].pop("scratch_dir", None)
        ch["init_args"]["persistent_workers"] = False
        ch["init_args"]["num_workers"] = 0


def _summarize_transforms(dm: BatchedConcatDataModule) -> None:
    for i, ch in enumerate(dm.data_modules):
        print(f"  child[{i}]: {type(ch).__name__}")
        print(f"    data_path: {ch.data_path}")
        print(f"    normalizations: {[type(t).__name__ for t in ch.normalizations]}")
        print(f"    augmentations:  {[type(t).__name__ for t in ch.augmentations]}")


def _check_micro_batch(mb: dict, ds_idx_expected: int | None = None) -> int:
    """Assert the micro-batch contract and return its sample count."""
    assert "_dataset_idx" in mb, "micro-batch missing _dataset_idx"
    assert "source" in mb, "micro-batch missing source"
    assert "target" in mb, "micro-batch missing target"
    assert mb["source"].ndim == 5, f"source ndim={mb['source'].ndim}"
    assert mb["target"].ndim == 5, f"target ndim={mb['target'].ndim}"
    if ds_idx_expected is not None:
        assert mb["_dataset_idx"] == ds_idx_expected
    if "norm_meta" in mb and mb["norm_meta"]:
        for ch_name, levels in mb["norm_meta"].items():
            assert "fov_statistics" in levels, (
                f"fov_statistics missing on ds_idx={mb['_dataset_idx']} channel={ch_name}"
            )
    return mb["source"].shape[0]


def main() -> int:
    """Smoke the joint train leaf: compose, instantiate, iterate a few batches."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--leaf", type=Path, default=_DEFAULT_LEAF, help="Joint train leaf YAML")
    ap.add_argument(
        "--ipsc-override",
        default=_SMALL_IPSC_SEC61B,
        help="Override child[0].data_path to this zarr. Default: 12-FOV subsampled SEC61B",
    )
    ap.add_argument(
        "--no-ipsc-override",
        dest="ipsc_override",
        action="store_const",
        const=None,
        help="Use the leaf's original child[0].data_path (full-size zarr)",
    )
    args = ap.parse_args()

    print(f"=== Real-zarr smoke for {args.leaf.name} ===\n")
    cfg = load_composed_config(args.leaf)
    data_cfg = copy.deepcopy(cfg["data"])
    _apply_overrides(data_cfg, args.ipsc_override)

    t0 = time.time()
    dm = _instantiate(data_cfg)
    print(f"jsonargparse instantiate: {time.time() - t0:.2f}s")
    print(f"  type: {type(dm).__name__}, n children: {len(dm.data_modules)}, batch_size: {dm.batch_size}")
    _summarize_transforms(dm)
    print()

    t0 = time.time()
    dm.setup(stage="fit")
    print(f"setup(fit): {time.time() - t0:.2f}s")
    print(f"  train_dataset (joint): {len(dm.train_dataset)} windows")
    print(f"  val_dataset   (joint): {len(dm.val_dataset)} windows")
    for i, ch in enumerate(dm.data_modules):
        print(f"  child[{i}].train: {len(ch.train_dataset)}, val: {len(ch.val_dataset)}")
    print()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"train sampler: {type(train_loader.sampler).__name__}")
    print(f"val   sampler: {type(val_loader.sampler).__name__}\n")

    # RandWeightedCropd(num_samples=N) on each child makes each dataset
    # index yield N samples. BatchedConcatDataModule.train_dataloader
    # requests batch_size indices, so total per batch = batch_size * N.
    patches_per_stack = dm.train_patches_per_stack
    expected_per_batch = dm.batch_size * patches_per_stack
    print("=== iterating train batches ===")
    print(
        f"  batch_size={dm.batch_size}, patches_per_stack={patches_per_stack}, "
        f"expected samples/batch={expected_per_batch}"
    )
    t0 = time.time()
    per_dataset_counts: dict[int, int] = {}
    for batch_idx, batch in enumerate(train_loader):
        assert isinstance(batch, list), f"expected list, got {type(batch).__name__}"
        total_samples = 0
        shapes = []
        for mb in batch:
            n = _check_micro_batch(mb)
            per_dataset_counts[mb["_dataset_idx"]] = per_dataset_counts.get(mb["_dataset_idx"], 0) + n
            total_samples += n
            shapes.append((mb["_dataset_idx"], tuple(mb["source"].shape)))
        assert total_samples == expected_per_batch, (
            f"batch {batch_idx}: total {total_samples} != expected {expected_per_batch}"
        )
        print(f"  batch[{batch_idx}]: {len(batch)} micro-batch(es), total {total_samples} samples, shapes={shapes}")
        if batch_idx + 1 >= _MAX_TRAIN_BATCHES:
            break
    print(f"iterated {_MAX_TRAIN_BATCHES} train batches in {time.time() - t0:.2f}s")
    print(f"  per-dataset sample counts: {dict(per_dataset_counts)}\n")

    print("=== iterating val batches ===")
    t0 = time.time()
    for batch_idx, batch in enumerate(val_loader):
        assert isinstance(batch, list)
        for mb in batch:
            _check_micro_batch(mb)
        shapes = [(mb["_dataset_idx"], tuple(mb["source"].shape)) for mb in batch]
        print(f"  val[{batch_idx}]: {len(batch)} micro-batch(es), shapes={shapes}")
        if batch_idx + 1 >= _MAX_VAL_BATCHES:
            break
    print(f"iterated {_MAX_VAL_BATCHES} val batches in {time.time() - t0:.2f}s\n")

    print("=== SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
