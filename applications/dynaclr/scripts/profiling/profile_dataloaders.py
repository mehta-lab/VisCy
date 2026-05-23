"""Profile BatchedConcatDataModule + TripletDatasets vs MultiExperimentDataModule.

Benchmarks setup time, raw __getitems__ latency, and full dataloader
throughput for:
- Old: BatchedConcatDataModule wrapping 2 TripletDataModules (one per experiment)
- New: Single MultiExperimentDataModule with flat parquet index

Uses two real datasets:
- 2025_07_24 G3BP1 (stress granules)
- 2025_04_15 H2B (chromatin)

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/profile_dataloaders.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

COLLECTION_YAML = "applications/dynaclr/configs/collections/benchmark_2exp.yml"
CELL_INDEX_PARQUET = "applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"

DATASETS = {
    "G3BP1": {
        "data_path": (
            "/hpc/projects/organelle_phenotyping/datasets/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
            "/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
        ),
        "tracks_path": (
            "/hpc/projects/organelle_phenotyping/datasets/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/tracking.zarr"
        ),
        "source_channel": ["raw GFP EX488 EM525-45"],
        "include_wells": ["C/1", "C/2"],
    },
    "H2B": {
        "data_path": (
            "/hpc/projects/organelle_phenotyping/datasets/2025_04_15_A549_H2B_CAAX_ZIKV_DENV"
            "/2025_04_15_A549_H2B_CAAX_ZIKV_DENV.zarr"
        ),
        "tracks_path": (
            "/hpc/projects/organelle_phenotyping/datasets/2025_04_15_A549_H2B_CAAX_ZIKV_DENV/tracking.zarr"
        ),
        "source_channel": ["raw Cy5 EX639 EM698-70"],
        "include_wells": ["B/1", "B/2"],
    },
}

# Shared benchmark parameters
BATCH_SIZES = [8, 32, 64, 128]
N_BATCHES = 20
WARMUP_BATCHES = 3
CACHE_POOL_BYTES = 500_000_000  # 500 MB
Z_RANGE = (30, 46)  # 16 z-slices, 3D benchmark


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


# ======================================================================
# Old: BatchedConcatDataModule wrapping 2 TripletDataModules
# ======================================================================


def setup_old():
    """Set up legacy BatchedConcatDataModule with 2 TripletDataModules."""
    from viscy_data.combined import BatchedConcatDataModule
    from viscy_data.triplet import TripletDataModule

    dms = []
    for name, cfg in DATASETS.items():
        dm = TripletDataModule(
            data_path=cfg["data_path"],
            tracks_path=cfg["tracks_path"],
            source_channel=cfg["source_channel"],
            z_range=Z_RANGE,
            initial_yx_patch_size=(192, 192),
            final_yx_patch_size=(160, 160),
            split_ratio=0.8,
            batch_size=BATCH_SIZES[-1],
            num_workers=1,
            time_interval=3,
            return_negative=False,
            cache_pool_bytes=CACHE_POOL_BYTES,
            fit_include_wells=cfg["include_wells"],
        )
        dms.append(dm)
        print(f"    Created TripletDataModule for {name}")

    concat_dm = BatchedConcatDataModule(data_modules=dms)
    concat_dm.setup("fit")
    return concat_dm


# ======================================================================
# New: MultiExperimentDataModule
# ======================================================================


def setup_new():
    """Set up MultiExperimentDataModule with pre-built parquet."""
    from dynaclr.data.datamodule import MultiExperimentDataModule

    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        z_window=Z_RANGE[1] - Z_RANGE[0],  # 16
        yx_patch_size=(192, 192),
        final_yx_patch_size=(160, 160),
        channels_per_sample=None,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=["perturbation"],
        split_ratio=0.8,
        batch_size=BATCH_SIZES[-1],
        num_workers=1,
        seed=42,
        cache_pool_bytes=CACHE_POOL_BYTES,
        normalizations=[],
        augmentations=[],
    )
    dm.setup("fit")
    return dm


# ======================================================================
# Benchmark helpers
# ======================================================================


def benchmark_getitems(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    n_batches: int = N_BATCHES,
    warmup: int = WARMUP_BATCHES,
) -> dict:
    """Time raw __getitems__ calls.

    Parameters
    ----------
    dataset : Dataset
        Must implement __getitems__(indices).
    batch_size : int
        Number of indices per call.
    n_batches : int
        Total batches to time (excluding warmup).
    warmup : int
        Batches to discard for cache warmup.

    Returns
    -------
    dict
        Timing statistics.
    """
    n_samples = len(dataset)
    rng = np.random.default_rng(42)
    total = warmup + n_batches

    times = []
    for i in range(total):
        indices = rng.integers(0, n_samples, size=batch_size).tolist()
        t0 = time.perf_counter()
        _ = dataset.__getitems__(indices)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)

    times_arr = np.array(times)
    return {
        "batch_size": batch_size,
        "mean_ms": times_arr.mean() * 1000,
        "std_ms": times_arr.std() * 1000,
        "median_ms": np.median(times_arr) * 1000,
        "p95_ms": np.percentile(times_arr, 95) * 1000,
        "throughput_samples_per_sec": batch_size / times_arr.mean(),
    }


def benchmark_dataloader(
    dataloader,
    n_batches: int = N_BATCHES,
    warmup: int = WARMUP_BATCHES,
) -> dict:
    """Time full dataloader iteration.

    Parameters
    ----------
    dataloader : DataLoader
        Configured dataloader.
    n_batches : int
        Batches to time after warmup.
    warmup : int
        Batches to discard.

    Returns
    -------
    dict
        Timing statistics.
    """
    timestamps = []
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= warmup + n_batches:
            break
        now = time.perf_counter()
        if i >= warmup:
            timestamps.append(now)
            # Count samples in batch
            if isinstance(batch, list):
                # BatchedConcatDataModule returns list of micro-batches
                for mb in batch:
                    if isinstance(mb, dict) and "anchor" in mb:
                        total_samples += mb["anchor"].shape[0]
            elif isinstance(batch, dict) and "anchor" in batch:
                total_samples += batch["anchor"].shape[0]

    if len(timestamps) < 2:
        return {"note": "not enough batches"}

    inter_batch = np.diff(timestamps)
    return {
        "n_batches": len(inter_batch),
        "total_samples": total_samples,
        "mean_inter_batch_ms": inter_batch.mean() * 1000,
        "std_inter_batch_ms": inter_batch.std() * 1000,
        "median_inter_batch_ms": np.median(inter_batch) * 1000,
        "throughput_samples_per_sec": total_samples / inter_batch.sum() if inter_batch.sum() > 0 else 0,
    }


# ======================================================================
# Main
# ======================================================================


def main():
    """Profile and compare dataloader implementations."""
    results = []

    print("=" * 70)
    print("DATALOADER PROFILING")
    print("BatchedConcatDataModule + TripletDatasets  vs  MultiExperimentDataModule")
    print("=" * 70)
    print("\nDatasets: G3BP1 (2025_07_24) + H2B (2025_04_15)")
    print(f"Z range: {Z_RANGE} ({Z_RANGE[1] - Z_RANGE[0]} slices)")
    print("Patch: 192x192 -> 160x160")
    print(f"Cache: {CACHE_POOL_BYTES / 1e6:.0f} MB")

    # ------------------------------------------------------------------
    # Setup timing
    # ------------------------------------------------------------------
    print("\n## Setup: Old (BatchedConcatDataModule + 2x TripletDataModule)")
    t0 = time.perf_counter()
    old_dm = setup_old()
    old_setup_time = time.perf_counter() - t0
    n_old_train = len(old_dm.train_dataset)
    n_old_val = len(old_dm.val_dataset)
    print(f"  Setup time: {_fmt(old_setup_time)}")
    print(f"  Train samples: {n_old_train}  |  Val samples: {n_old_val}")

    print("\n## Setup: New (MultiExperimentDataModule)")
    t0 = time.perf_counter()
    new_dm = setup_new()
    new_setup_time = time.perf_counter() - t0
    n_new_train = len(new_dm.train_dataset)
    n_new_val = len(new_dm.val_dataset) if new_dm.val_dataset else 0
    print(f"  Setup time: {_fmt(new_setup_time)}")
    print(f"  Train samples: {n_new_train}  |  Val samples: {n_new_val}")

    # ------------------------------------------------------------------
    # Benchmark 1: Raw __getitems__
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Raw __getitems__ (no dataloader, no transforms)")
    print("=" * 70)

    for bs in BATCH_SIZES:
        print(f"\n### batch_size={bs}")

        stats_old = benchmark_getitems(old_dm.train_dataset, bs)
        stats_old["dataset"] = "Old (BatchedConcatDataset)"
        results.append(stats_old)
        print(
            f"  Old: {stats_old['mean_ms']:.1f} ± {stats_old['std_ms']:.1f} ms/batch "
            f"| p95={stats_old['p95_ms']:.1f} ms "
            f"| {stats_old['throughput_samples_per_sec']:.0f} samples/s"
        )

        stats_new = benchmark_getitems(new_dm.train_dataset, bs)
        stats_new["dataset"] = "New (MultiExperimentTripletDataset)"
        results.append(stats_new)
        print(
            f"  New: {stats_new['mean_ms']:.1f} ± {stats_new['std_ms']:.1f} ms/batch "
            f"| p95={stats_new['p95_ms']:.1f} ms "
            f"| {stats_new['throughput_samples_per_sec']:.0f} samples/s"
        )

        speedup = stats_old["mean_ms"] / stats_new["mean_ms"] if stats_new["mean_ms"] > 0 else float("inf")
        direction = "faster" if speedup > 1 else "slower"
        print(f"  New is {abs(speedup):.2f}x {direction}")

    # ------------------------------------------------------------------
    # Benchmark 2: Full dataloader
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Full ThreadDataLoader iteration")
    print("=" * 70)

    for bs in [32, 64]:
        print(f"\n### batch_size={bs}")

        # Old
        old_dm.batch_size = bs
        for sub_dm in old_dm.data_modules:
            sub_dm.batch_size = bs
        old_dl = old_dm.train_dataloader()
        dl_old = benchmark_dataloader(old_dl)
        print(
            f"  Old: {dl_old.get('mean_inter_batch_ms', 0):.1f} ± "
            f"{dl_old.get('std_inter_batch_ms', 0):.1f} ms/batch "
            f"| {dl_old.get('throughput_samples_per_sec', 0):.0f} samples/s"
        )

        # New
        new_dm.batch_size = bs
        new_dl = new_dm.train_dataloader()
        dl_new = benchmark_dataloader(new_dl)
        print(
            f"  New: {dl_new.get('mean_inter_batch_ms', 0):.1f} ± "
            f"{dl_new.get('std_inter_batch_ms', 0):.1f} ms/batch "
            f"| {dl_new.get('throughput_samples_per_sec', 0):.0f} samples/s"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n### __getitems__ throughput (samples/sec)")
    summary = pd.DataFrame(results)
    pivot = summary.pivot_table(
        index="batch_size",
        columns="dataset",
        values="throughput_samples_per_sec",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.0f}"))

    print("\n### Setup times")
    print("| Pipeline | Setup Time |")
    print("|----------|-----------|")
    print(f"| Old (BatchedConcatDataModule) | {_fmt(old_setup_time)} |")
    print(f"| New (MultiExperimentDataModule) | {_fmt(new_setup_time)} |")


if __name__ == "__main__":
    main()
