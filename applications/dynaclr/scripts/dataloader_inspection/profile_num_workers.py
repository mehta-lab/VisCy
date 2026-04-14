"""Sweep num_workers to find optimal dataloader parallelism.

Holds all other parameters constant and measures end-to-end ThreadDataLoader
throughput (samples/sec and inter-batch latency) for num_workers in [1, 2, 4, 8].

Unlike profile_stages.py (which isolates individual pipeline stages) or
profile_dataloaders.py (which compares two dataloader implementations), this
script answers: does adding more CPU workers reduce GPU starvation?

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/profile_num_workers.py
"""

from __future__ import annotations

import time

import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CELL_INDEX_PARQUET = "applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"

BATCH_SIZE = 128
N_BATCHES = 30
WARMUP = 5
CACHE_POOL_BYTES = 500_000_000  # 500 MB

Z_WINDOW = 16
Z_EXTRACTION_WINDOW = 45
YX_PATCH = (192, 192)
FINAL_YX_PATCH = (160, 160)

NUM_WORKERS_SWEEP = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_dm(num_workers: int) -> MultiExperimentDataModule:
    """Build a MultiExperimentDataModule with the given num_workers."""
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        z_window=Z_WINDOW,
        z_extraction_window=Z_EXTRACTION_WINDOW,
        z_focus_offset=0.3,
        yx_patch_size=YX_PATCH,
        final_yx_patch_size=FINAL_YX_PATCH,
        channels_per_sample=1,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=["perturbation"],
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        seed=42,
        cache_pool_bytes=CACHE_POOL_BYTES,
        normalizations=[],
        augmentations=[],
    )
    dm.setup("fit")
    return dm


def benchmark_dataloader(dataloader, n_batches: int = N_BATCHES, warmup: int = WARMUP) -> dict:
    """Measure inter-batch latency and throughput over the dataloader.

    Parameters
    ----------
    dataloader : ThreadDataLoader
        Configured training dataloader.
    n_batches : int
        Number of batches to time after warmup.
    warmup : int
        Batches to discard for cache/thread warmup.

    Returns
    -------
    dict
        Inter-batch timing stats, throughput in samples/sec, and VAST bandwidth in MB/s.
    """
    timestamps = []
    total_samples = 0
    read_mb_per_batch = None

    for i, batch in enumerate(dataloader):
        if i >= warmup + n_batches:
            break
        now = time.perf_counter()
        if i >= warmup:
            timestamps.append(now)
            if isinstance(batch, dict) and "anchor" in batch:
                total_samples += batch["anchor"].shape[0]
                if read_mb_per_batch is None:
                    # anchor + positive (fit mode). Lower bound — ignores chunk alignment overhead.
                    n_tensors = 2 if "positive" in batch else 1
                    read_mb_per_batch = batch["anchor"].nelement() * batch["anchor"].element_size() * n_tensors / 1e6

    if len(timestamps) < 2:
        return {"note": "not enough batches"}

    inter_batch = np.diff(timestamps)
    mean_s = inter_batch.mean()
    bandwidth_mb_s = read_mb_per_batch / mean_s if read_mb_per_batch else 0.0
    return {
        "mean_ms": mean_s * 1000,
        "std_ms": inter_batch.std() * 1000,
        "median_ms": float(np.median(inter_batch) * 1000),
        "p95_ms": float(np.percentile(inter_batch, 95) * 1000),
        "throughput_samples_per_sec": total_samples / inter_batch.sum(),
        "read_mb_per_batch": read_mb_per_batch or 0.0,
        "bandwidth_mb_s": bandwidth_mb_s,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Sweep num_workers and report throughput."""
    print("=" * 60)
    print("num_workers SWEEP — ThreadDataLoader throughput")
    print("=" * 60)
    print(f"batch_size={BATCH_SIZE}, z={Z_EXTRACTION_WINDOW}→{Z_WINDOW}")
    print(f"patch={YX_PATCH}→{FINAL_YX_PATCH}, channels_per_sample=1")
    print(f"warmup={WARMUP} batches, measured over {N_BATCHES} batches")
    print()

    # Setup is shared across runs — only the dataloader changes.
    # Re-setup for each num_workers since ThreadDataLoader is created in train_dataloader().
    results = []
    for nw in NUM_WORKERS_SWEEP:
        print(f"## num_workers={nw}")
        dm = setup_dm(nw)
        dl = dm.train_dataloader()
        stats = benchmark_dataloader(dl)
        stats["num_workers"] = nw
        results.append(stats)
        print(
            f"   {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms/batch"
            f"  |  p95={stats['p95_ms']:.1f} ms"
            f"  |  {stats['throughput_samples_per_sec']:.0f} samples/sec"
            f"  |  {stats['bandwidth_mb_s']:.0f} MB/s"
        )
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    read_mb = results[0]["read_mb_per_batch"] if results else 0.0
    print(f"Read volume per batch (lower bound): {read_mb:.0f} MB")
    print()
    print("| num_workers | mean ms/batch | p95 ms | samples/sec | MB/s (VAST) |")
    print("|-------------|---------------|--------|-------------|-------------|")
    for r in results:
        print(
            f"| {r['num_workers']:11d} | {r['mean_ms']:13.1f} | {r['p95_ms']:6.1f}"
            f" | {r['throughput_samples_per_sec']:11.0f} | {r['bandwidth_mb_s']:11.0f} |"
        )


if __name__ == "__main__":
    main()
