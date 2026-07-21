"""Sweep num_workers to find optimal dataloader parallelism.

Holds all other parameters constant and measures ThreadDataLoader-only
throughput (samples/sec and inter-batch latency). Defaults match the Reef
Stage-1 self-positive recipe and sweep num_workers in [4, 8, 12, 16].

Unlike profile_stages.py (which isolates individual pipeline stages) or
profile_dataloaders.py (which compares two dataloader implementations), this
script answers whether adding thread workers improves loader service rate. It
does not measure GPU utilization; validate the winner in a real ``fit`` run.

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/profile_num_workers.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule

# ---------------------------------------------------------------------------
# Config — overridable via env; geometry/loader defaults match Reef Stage 1.
# ---------------------------------------------------------------------------

CELL_INDEX_PARQUET = os.environ.get(
    "PROFILE_PARQUET", "applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"
)

BATCH_SIZE = int(os.environ.get("PROFILE_BATCH_SIZE", "256"))
N_BATCHES = int(os.environ.get("PROFILE_N_BATCHES", "30"))
WARMUP = int(os.environ.get("PROFILE_WARMUP", "5"))
CACHE_POOL_BYTES = int(os.environ.get("PROFILE_CACHE_POOL_BYTES", "0"))

FILE_IO_CONCURRENCY = int(os.environ.get("PROFILE_FILE_IO_CONCURRENCY", "32"))
Z_WINDOW = int(os.environ.get("PROFILE_Z_WINDOW", "1"))
Z_EXTRACTION_WINDOW = int(os.environ.get("PROFILE_Z_EXTRACTION_WINDOW", "16"))
_YX = int(os.environ.get("PROFILE_YX_PATCH", "256"))
_FINAL_YX = int(os.environ.get("PROFILE_FINAL_YX_PATCH", "160"))
YX_PATCH = (_YX, _YX)
FINAL_YX_PATCH = (_FINAL_YX, _FINAL_YX)
POSITIVE_CELL_SOURCE = os.environ.get("PROFILE_POSITIVE_CELL_SOURCE", "self")
PREFETCH_FACTOR = int(os.environ.get("PROFILE_PREFETCH_FACTOR", "1"))
BUFFER_SIZE = int(os.environ.get("PROFILE_BUFFER_SIZE", "1"))

FOCUS_CHANNEL = os.environ.get("PROFILE_FOCUS_CHANNEL", "Phase3D")
REFERENCE_PIXEL_SIZE_XY_UM = float(os.environ.get("PROFILE_REFERENCE_PIXEL_SIZE_XY_UM", "0.1494"))
NUM_WORKERS_SWEEP = [int(x) for x in os.environ.get("PROFILE_NUM_WORKERS_SWEEP", "4,8,12,16").split(",")]


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
        focus_channel=FOCUS_CHANNEL,
        reference_pixel_size_xy_um=REFERENCE_PIXEL_SIZE_XY_UM,
        yx_patch_size=YX_PATCH,
        final_yx_patch_size=FINAL_YX_PATCH,
        channels_per_sample=1,
        positive_cell_source=POSITIVE_CELL_SOURCE,
        positive_match_columns=None if POSITIVE_CELL_SOURCE == "self" else ["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        batch_group_by="marker",
        stratify_by="experiment",
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        buffer_size=BUFFER_SIZE,
        seed=42,
        cache_pool_bytes=CACHE_POOL_BYTES,
        normalizations=[],
        file_io_concurrency=FILE_IO_CONCURRENCY,
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
        Inter-batch timing stats, throughput, logical tensor rate, and queued batch size.
    """
    timestamps = []
    measured_batch_size = None
    logical_mb_per_batch = None
    output_mb_per_batch = None

    for i, batch in enumerate(dataloader):
        if i >= warmup + n_batches:
            break
        now = time.perf_counter()
        if i >= warmup:
            timestamps.append(now)
            if isinstance(batch, dict) and "anchor" in batch:
                measured_batch_size = batch["anchor"].shape[0]
                if logical_mb_per_batch is None:
                    anchor_bytes = batch["anchor"].nelement() * batch["anchor"].element_size()
                    storage_sourced_tensors = 1 if POSITIVE_CELL_SOURCE == "self" else 2
                    logical_mb_per_batch = anchor_bytes * storage_sourced_tensors / 1e6
                    output_mb_per_batch = sum(
                        batch[key].nelement() * batch[key].element_size()
                        for key in ("anchor", "positive")
                        if key in batch
                    ) / 1e6

    if len(timestamps) < 2:
        return {"note": "not enough batches"}

    inter_batch = np.diff(timestamps)
    mean_s = inter_batch.mean()
    logical_mb_s = logical_mb_per_batch / mean_s if logical_mb_per_batch else 0.0
    return {
        "mean_ms": mean_s * 1000,
        "std_ms": inter_batch.std() * 1000,
        "median_ms": float(np.median(inter_batch) * 1000),
        "p95_ms": float(np.percentile(inter_batch, 95) * 1000),
        "throughput_samples_per_sec": (measured_batch_size or 0) / mean_s,
        "logical_mb_per_batch": logical_mb_per_batch or 0.0,
        "logical_mb_s": logical_mb_s,
        "output_mb_per_batch": output_mb_per_batch or 0.0,
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
    print(f"positive_source={POSITIVE_CELL_SOURCE}, prefetch_factor={PREFETCH_FACTOR}, buffer_size={BUFFER_SIZE}")
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
        queued_batches = nw * PREFETCH_FACTOR + BUFFER_SIZE
        stats["queued_host_gib"] = stats["output_mb_per_batch"] * queued_batches / 1024
        results.append(stats)
        print(
            f"   {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f} ms/batch"
            f"  |  p95={stats['p95_ms']:.1f} ms"
            f"  |  {stats['throughput_samples_per_sec']:.0f} samples/sec"
            f"  |  {stats['logical_mb_s']:.0f} logical MB/s"
            f"  |  ≤{stats['queued_host_gib']:.1f} GiB queued tensors"
        )
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    logical_mb = results[0]["logical_mb_per_batch"] if results else 0.0
    print(f"Storage-sourced logical tensor bytes per batch: {logical_mb:.0f} MB")
    print("Queued-host-memory values are upper-bound tensor payloads; pinning and metadata add overhead.")
    print()
    print("| num_workers | mean ms/batch | p95 ms | samples/sec | logical MB/s | queued GiB upper bound |")
    print("|-------------|---------------|--------|-------------|--------------|------------------------|")
    for r in results:
        print(
            f"| {r['num_workers']:11d} | {r['mean_ms']:13.1f} | {r['p95_ms']:6.1f}"
            f" | {r['throughput_samples_per_sec']:11.0f} | {r['logical_mb_s']:12.0f}"
            f" | {r['queued_host_gib']:22.1f} |"
        )


if __name__ == "__main__":
    main()
