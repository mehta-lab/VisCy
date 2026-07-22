"""Profile per-stage breakdown: I/O vs normalization vs augmentation vs crop.

Isolates each stage of the training batch pipeline to find the bottleneck:
1. Data fetch:     __getitems__ (storage read + decompression + interpolation,
                    tensor construction, and positive sampling)
2. CPU→GPU:        .to(device) transfer
3. Normalization:  NormalizeSampled (fov/timepoint stats)
4. Augmentation:   affine + flip + contrast + scale + smooth + noise
5. Final crop:     BatchedRandSpatialCropd (z_extraction → z_window)

TensorStore counters report compressed file bytes and inner-chunk reads.

Uses MultiExperimentDataModule with environment-overridable Reef defaults.
Requires a GPU unless ``PROFILE_DEVICE=cpu`` is set.

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/profile_stages.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import tensorstore as ts
import torch
from monai.transforms import Compose

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_transforms import (
    BatchedChannelWiseZReductiond,
    BatchedRandAdjustContrastd,
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedRandGaussianNoised,
    BatchedRandGaussianSmoothd,
    BatchedRandScaleIntensityd,
    BatchedRandSpatialCropd,
    NormalizeSampled,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# All knobs are environment-overridable. Geometry and data-path defaults match
# the Reef Stage-1 recipe; PROFILE_PARQUET selects the collection.
COLLECTION_YAML = "applications/dynaclr/configs/collections/benchmark_2exp.yml"
CELL_INDEX_PARQUET = os.environ.get(
    "PROFILE_PARQUET", "applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"
)

BATCH_SIZE = int(os.environ.get("PROFILE_BATCH_SIZE", "256"))
N_BATCHES = int(os.environ.get("PROFILE_N_BATCHES", "15"))
WARMUP = int(os.environ.get("PROFILE_WARMUP", "3"))
FILE_IO_CONCURRENCY = int(os.environ.get("PROFILE_FILE_IO_CONCURRENCY", "32"))
CACHE_POOL_BYTES = int(os.environ.get("PROFILE_CACHE_POOL_BYTES", "0"))

Z_WINDOW = int(os.environ.get("PROFILE_Z_WINDOW", "1"))
Z_EXTRACTION_WINDOW = int(os.environ.get("PROFILE_Z_EXTRACTION_WINDOW", "16"))
_YX = int(os.environ.get("PROFILE_YX_PATCH", "256"))
_FINAL_YX = int(os.environ.get("PROFILE_FINAL_YX_PATCH", "160"))
YX_PATCH = (_YX, _YX)
FINAL_YX_PATCH = (_FINAL_YX, _FINAL_YX)
# None = all-channels mode; 1 = bag-of-channels (single "channel_0"). The
# transforms below key on "channel_0", so 1 matches production BoC configs.
_CPS = os.environ.get("PROFILE_CHANNELS_PER_SAMPLE", "1")
FOCUS_CHANNEL = os.environ.get("PROFILE_FOCUS_CHANNEL", "Phase3D")
REFERENCE_PIXEL_SIZE_XY_UM = float(os.environ.get("PROFILE_REFERENCE_PIXEL_SIZE_XY_UM", "0.1494"))
CHANNELS_PER_SAMPLE = None if _CPS.lower() in ("none", "null", "") else int(_CPS)
POSITIVE_CELL_SOURCE = os.environ.get("PROFILE_POSITIVE_CELL_SOURCE", "self")

CHANNEL_KEY = "channel_0"
DEVICE = os.environ.get("PROFILE_DEVICE", "cuda")


_TS_METRICS = {
    "bytes_read": "/tensorstore/kvstore/file/bytes_read",
    "file_reads": "/tensorstore/kvstore/file/read",
    "chunk_reads": "/tensorstore/cache/chunk_cache/reads",
}


def _tensorstore_metrics() -> dict[str, int]:
    """Return cumulative TensorStore counters used by this benchmark."""
    collected = {metric["name"]: metric for metric in ts.experimental_collect_matching_metrics()}
    snapshot = {}
    for label, name in _TS_METRICS.items():
        values = collected.get(name, {}).get("values", [])
        snapshot[label] = int(sum(value.get("value", 0) for value in values))
    return snapshot


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def setup():
    """Set up MultiExperimentDataModule with production-like config."""
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        z_window=Z_WINDOW,
        z_extraction_window=Z_EXTRACTION_WINDOW,
        z_focus_offset=0.3,
        focus_channel=FOCUS_CHANNEL,
        reference_pixel_size_xy_um=REFERENCE_PIXEL_SIZE_XY_UM,
        yx_patch_size=YX_PATCH,
        final_yx_patch_size=FINAL_YX_PATCH,
        channels_per_sample=CHANNELS_PER_SAMPLE,
        positive_cell_source=POSITIVE_CELL_SOURCE,
        positive_match_columns=None if POSITIVE_CELL_SOURCE == "self" else ["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=["perturbation"],
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        file_io_concurrency=FILE_IO_CONCURRENCY,
        num_workers=1,
        seed=42,
        cache_pool_bytes=CACHE_POOL_BYTES,
        normalizations=[],
        augmentations=[],
    )
    dm.setup("fit")
    return dm


def build_transforms():
    """Build transform stages matching the Reef Stage-1 2D-MIP recipe."""
    normalization = NormalizeSampled(
        keys=[CHANNEL_KEY],
        level="timepoint_statistics",
        subtrahend="mean",
        divisor="std",
    )

    augmentations = [
        BatchedRandAffined(
            keys=[CHANNEL_KEY],
            prob=0.8,
            scale_range=[[0.8, 1.3], [0.8, 1.3], [0.8, 1.3]],
            rotate_range=[3.14, 0.0, 0.0],
            shear_range=[0.05, 0.05, 0.0, 0.05, 0.0, 0.05],
        ),
        BatchedRandFlipd(
            keys=[CHANNEL_KEY],
            spatial_axes=[1, 2],
            prob=0.5,
        ),
        BatchedRandAdjustContrastd(
            keys=[CHANNEL_KEY],
            prob=0.5,
            gamma=(0.6, 1.6),
        ),
        BatchedRandScaleIntensityd(
            keys=[CHANNEL_KEY],
            prob=0.5,
            factors=0.5,
        ),
        BatchedRandGaussianSmoothd(
            keys=[CHANNEL_KEY],
            prob=0.5,
            sigma_x=[0.25, 0.50],
            sigma_y=[0.25, 0.50],
            sigma_z=[0.0, 0.0],
        ),
        BatchedRandGaussianNoised(
            keys=[CHANNEL_KEY],
            prob=0.5,
            mean=0.0,
            std=0.1,
        ),
        BatchedRandSpatialCropd(keys=[CHANNEL_KEY], roi_size=(10, 192, 192)),
        BatchedChannelWiseZReductiond(keys=[CHANNEL_KEY], allow_missing_keys=True),
    ]

    final_crop = BatchedRandSpatialCropd(
        keys=[CHANNEL_KEY],
        roi_size=(Z_WINDOW, FINAL_YX_PATCH[0], FINAL_YX_PATCH[1]),
    )

    return normalization, augmentations, final_crop


def time_stage(fn, n_batches=N_BATCHES, warmup=WARMUP, on_measure_start=None):
    """Time a callable over multiple iterations, return stats.

    Parameters
    ----------
    fn : callable
        Function to time. Called with no arguments.
    n_batches : int
        Iterations to time after warmup.
    warmup : int
        Iterations to discard.
    on_measure_start : callable or None
        Called after warmup and immediately before the first measured iteration.

    Returns
    -------
    dict
        mean_ms, std_ms, median_ms.
    """
    times = []
    for i in range(warmup + n_batches):
        if i == warmup and on_measure_start is not None:
            on_measure_start()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    arr = np.array(times)
    return {
        "mean_ms": arr.mean() * 1000,
        "std_ms": arr.std() * 1000,
        "median_ms": np.median(arr) * 1000,
    }, result


def main():
    """Profile individual dataloader pipeline stages."""
    print("=" * 70)
    print("STAGE BREAKDOWN: Data fetch → Transfer → Normalize → Augment → Crop")
    print("=" * 70)
    print(f"batch_size={BATCH_SIZE}, z_extraction={Z_EXTRACTION_WINDOW}→z_window={Z_WINDOW}")
    print(f"patch={YX_PATCH}→{FINAL_YX_PATCH}, device={DEVICE}")
    print()

    # Setup
    dm = setup()
    dataset = dm.train_dataset
    normalization, augmentations, final_crop = build_transforms()
    rng = np.random.default_rng(42)
    n_samples = len(dataset)

    def random_indices():
        return rng.integers(0, n_samples, size=BATCH_SIZE).tolist()

    # Pre-generate index lists so index generation doesn't pollute timing
    all_indices = [random_indices() for _ in range(WARMUP + N_BATCHES + 5)]
    idx_iter = iter(all_indices)

    # ── Stage 1: Data fetch (__getitems__) ──
    print("## Stage 1: Data fetch (__getitems__)")
    sample_batch = None

    def io_step():
        nonlocal sample_batch
        indices = next(idx_iter)
        sample_batch = dataset.__getitems__(indices)
        return sample_batch

    metrics_before = {}
    io_stats, _ = time_stage(io_step, on_measure_start=lambda: metrics_before.update(_tensorstore_metrics()))
    metrics_after = _tensorstore_metrics()
    metric_delta = {name: metrics_after[name] - metrics_before[name] for name in metrics_before}

    # Use the last batch for subsequent stages
    assert sample_batch is not None
    anchor = sample_batch["anchor"]
    positive = sample_batch.get("positive")

    # Self positives clone the anchor and therefore perform one storage read
    # even though the returned batch contains two tensors.
    storage_sourced_tensors = 1 if POSITIVE_CELL_SOURCE == "self" else 2
    logical_bytes = anchor.nelement() * anchor.element_size() * storage_sourced_tensors
    read_bytes = metric_delta["bytes_read"] / N_BATCHES
    read_mb = read_bytes / 1e6
    bandwidth_mb_s = read_mb / (io_stats["mean_ms"] / 1000)
    io_stats["read_mb"] = read_mb
    io_stats["bandwidth_mb_s"] = bandwidth_mb_s
    io_stats["logical_mb"] = logical_bytes / 1e6
    io_stats["storage_amplification"] = read_bytes / logical_bytes

    print(f"   {io_stats['mean_ms']:.1f} ± {io_stats['std_ms']:.1f} ms")
    print(
        f"   TensorStore file bytes: {read_mb:.0f} MB/batch | bandwidth: {bandwidth_mb_s:.0f} MB/s | "
        f"storage/logical: {io_stats['storage_amplification']:.2f}x"
    )
    print(
        f"   measured reads: {metric_delta['chunk_reads'] / N_BATCHES:.1f} chunks/batch, "
        f"{metric_delta['file_reads'] / N_BATCHES:.1f} file ranges/batch; "
        f"storage-sourced tensor bytes: {io_stats['logical_mb']:.0f} MB/batch"
    )
    print(f"   anchor shape: {anchor.shape}, dtype: {anchor.dtype}")

    # ── Stage 2: CPU→GPU transfer ──
    print("\n## Stage 2: CPU → GPU transfer")

    def transfer_step():
        gpu_positive = positive.to(DEVICE, non_blocking=True) if positive is not None else None
        return anchor.to(DEVICE, non_blocking=True), gpu_positive

    transfer_stats, (gpu_anchor, gpu_positive) = time_stage(transfer_step)
    print(f"   {transfer_stats['mean_ms']:.1f} ± {transfer_stats['std_ms']:.1f} ms")
    output_bytes = anchor.nelement() * anchor.element_size()
    output_bytes += positive.nelement() * positive.element_size() if positive is not None else 0
    print(f"   output tensor size: {output_bytes / 1e6:.1f} MB")

    # ── Stage 3: Normalization ──
    print("\n## Stage 3: Normalization (subtract mean, divide std — manual)")
    # NormalizeSampled via _transform_channel_wise requires channel-name
    # alignment that depends on the full DataModule context. Time the raw
    # arithmetic instead: this is what NormalizeSampled does per channel.

    def normalize_view(view):
        x = view.clone()
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        std = x.std(dim=(-3, -2, -1), keepdim=True)
        return (x - mean) / (std + 1e-8)

    def norm_step():
        normed_positive = normalize_view(gpu_positive) if gpu_positive is not None else None
        return normalize_view(gpu_anchor), normed_positive

    norm_stats, (normed, normed_positive) = time_stage(norm_step)
    print(f"   {norm_stats['mean_ms']:.1f} ± {norm_stats['std_ms']:.1f} ms")

    # ── Stage 4: Augmentations (individually) ──
    print("\n## Stage 4: Augmentations (individual)")
    aug_names = [
        "RandAffined",
        "RandFlipd",
        "RandAdjustContrastd",
        "RandScaleIntensityd",
        "RandGaussianSmoothd",
        "RandGaussianNoised",
        "RandSpatialCropd",
        "ChannelWiseZReductiond",
    ]
    aug_total = 0.0
    current_input = normed
    current_positive = normed_positive

    for aug_name, aug_transform in zip(aug_names, augmentations):
        t = Compose([aug_transform])
        inp = current_input
        inp_positive = current_positive

        def aug_step(transform=t, data=inp, positive_data=inp_positive):
            d = {CHANNEL_KEY: data.clone()}
            transformed = transform(d)[CHANNEL_KEY]
            if positive_data is not None:
                positive = transform({CHANNEL_KEY: positive_data.clone()})[CHANNEL_KEY]
                return transformed, positive
            return transformed, None

        stats, (current_input, current_positive) = time_stage(aug_step)
        aug_total += stats["mean_ms"]
        print(f"   {aug_name:30s} {stats['mean_ms']:8.1f} ± {stats['std_ms']:.1f} ms")

    print(f"   {'TOTAL':30s} {aug_total:8.1f} ms")

    # ── Stage 5: Final crop ──
    print("\n## Stage 5: Final crop (BatchedRandSpatialCropd)")
    crop_input = current_input
    crop_positive = current_positive

    def crop_step():
        d = {CHANNEL_KEY: crop_input.clone()}
        cropped = final_crop(d)[CHANNEL_KEY]
        if crop_positive is not None:
            positive = final_crop({CHANNEL_KEY: crop_positive.clone()})[CHANNEL_KEY]
            return cropped, positive
        return cropped, None

    crop_stats, _ = time_stage(crop_step)
    print(f"   {crop_stats['mean_ms']:.1f} ± {crop_stats['std_ms']:.1f} ms")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY (mean ms per batch)")
    print("=" * 70)

    stages = {
        "Data fetch (__getitems__)": io_stats["mean_ms"],
        "CPU→GPU transfer": transfer_stats["mean_ms"],
        "Normalization": norm_stats["mean_ms"],
        "Augmentations (total)": aug_total,
        "Final crop": crop_stats["mean_ms"],
    }
    total = sum(stages.values())

    print("\n| Stage | Time (ms) | % of total | Bandwidth |")
    print("|-------|-----------|------------|-----------|")
    for name, ms in stages.items():
        if name == "Data fetch (__getitems__)":
            bw = f"{io_stats['bandwidth_mb_s']:.0f} MB/s ({io_stats['read_mb']:.0f} MB read)"
        else:
            bw = "—"
        print(f"| {name} | {ms:.1f} | {ms / total * 100:.1f}% | {bw} |")
    print(f"| **Total** | **{total:.1f}** | **100%** | |")


if __name__ == "__main__":
    main()
