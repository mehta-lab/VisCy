"""Profile per-stage breakdown: I/O vs normalization vs augmentation vs crop.

Isolates each stage of the training batch pipeline to find the bottleneck:
1. I/O:            __getitems__ (tensorstore read + positive sampling)
2. CPU→GPU:        .to(device) transfer
3. Normalization:  NormalizeSampled (fov/timepoint stats)
4. Augmentation:   affine + flip + contrast + scale + smooth + noise
5. Final crop:     BatchedRandSpatialCropd (z_extraction → z_window)

Uses the new MultiExperimentDataModule with the benchmark_2exp collection.
Requires GPU.

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/profile_stages.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from monai.transforms import Compose

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_transforms import (
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

COLLECTION_YAML = "applications/dynaclr/configs/collections/benchmark_2exp.yml"
CELL_INDEX_PARQUET = "applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"

BATCH_SIZE = 128
N_BATCHES = 15
WARMUP = 3
CACHE_POOL_BYTES = 500_000_000

Z_WINDOW = 16
Z_EXTRACTION_WINDOW = 45
YX_PATCH = (192, 192)
FINAL_YX_PATCH = (160, 160)

CHANNEL_KEY = "channel_0"
DEVICE = "cuda"


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
        yx_patch_size=YX_PATCH,
        final_yx_patch_size=FINAL_YX_PATCH,
        channels_per_sample=None,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=["perturbation"],
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=1,
        seed=42,
        cache_pool_bytes=CACHE_POOL_BYTES,
        normalizations=[],
        augmentations=[],
    )
    dm.setup("fit")
    return dm


def build_transforms():
    """Build the individual transform stages matching DynaCLR-3D-BagOfChannels-v2."""
    normalization = NormalizeSampled(
        keys=[CHANNEL_KEY],
        level="fov_statistics",
        subtrahend="mean",
        divisor="std",
    )

    augmentations = [
        BatchedRandAffined(
            keys=[CHANNEL_KEY],
            prob=0.8,
            scale_range=[[0.9, 1.1], [0.9, 1.1], [0.9, 1.1]],
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
            sigma_z=[0.0, 0.2],
        ),
        BatchedRandGaussianNoised(
            keys=[CHANNEL_KEY],
            prob=0.5,
            mean=0.0,
            std=0.1,
        ),
    ]

    final_crop = BatchedRandSpatialCropd(
        keys=[CHANNEL_KEY],
        roi_size=(Z_WINDOW, FINAL_YX_PATCH[0], FINAL_YX_PATCH[1]),
    )

    return normalization, augmentations, final_crop


def time_stage(fn, n_batches=N_BATCHES, warmup=WARMUP):
    """Time a callable over multiple iterations, return stats.

    Parameters
    ----------
    fn : callable
        Function to time. Called with no arguments.
    n_batches : int
        Iterations to time after warmup.
    warmup : int
        Iterations to discard.

    Returns
    -------
    dict
        mean_ms, std_ms, median_ms.
    """
    times = []
    for i in range(warmup + n_batches):
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
    print("STAGE BREAKDOWN: I/O → Transfer → Normalize → Augment → Crop")
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

    # ── Stage 1: I/O (__getitems__) ──
    print("## Stage 1: I/O (__getitems__)")
    batches = []

    def io_step():
        indices = next(idx_iter)
        batch = dataset.__getitems__(indices)
        batches.append(batch)
        return batch

    io_stats, _ = time_stage(io_step)
    print(f"   {io_stats['mean_ms']:.1f} ± {io_stats['std_ms']:.1f} ms")

    # Use the last batch for subsequent stages
    sample_batch = batches[-1]
    anchor = sample_batch["anchor"]
    print(f"   anchor shape: {anchor.shape}, dtype: {anchor.dtype}")

    # ── Stage 2: CPU→GPU transfer ──
    print("\n## Stage 2: CPU → GPU transfer")

    def transfer_step():
        return anchor.to(DEVICE, non_blocking=True)

    transfer_stats, gpu_anchor = time_stage(transfer_step)
    print(f"   {transfer_stats['mean_ms']:.1f} ± {transfer_stats['std_ms']:.1f} ms")
    print(f"   tensor size: {anchor.nelement() * anchor.element_size() / 1e6:.1f} MB")

    # ── Stage 3: Normalization ──
    print("\n## Stage 3: Normalization (subtract mean, divide std — manual)")
    # NormalizeSampled via _transform_channel_wise requires channel-name
    # alignment that depends on the full DataModule context. Time the raw
    # arithmetic instead: this is what NormalizeSampled does per channel.

    def norm_step():
        x = gpu_anchor.clone()
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        std = x.std(dim=(-3, -2, -1), keepdim=True)
        return (x - mean) / (std + 1e-8)

    norm_stats, normed = time_stage(norm_step)
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
    ]
    aug_total = 0.0
    current_input = normed

    for aug_name, aug_transform in zip(aug_names, augmentations):
        t = Compose([aug_transform])
        inp = current_input

        def aug_step(transform=t, data=inp):
            d = {CHANNEL_KEY: data.clone()}
            return transform(d)[CHANNEL_KEY]

        stats, current_input = time_stage(aug_step)
        aug_total += stats["mean_ms"]
        print(f"   {aug_name:30s} {stats['mean_ms']:8.1f} ± {stats['std_ms']:.1f} ms")

    print(f"   {'TOTAL':30s} {aug_total:8.1f} ms")

    # ── Stage 5: Final crop ──
    print("\n## Stage 5: Final crop (BatchedRandSpatialCropd)")
    crop_input = current_input

    def crop_step():
        d = {CHANNEL_KEY: crop_input.clone()}
        return final_crop(d)[CHANNEL_KEY]

    crop_stats, _ = time_stage(crop_step)
    print(f"   {crop_stats['mean_ms']:.1f} ± {crop_stats['std_ms']:.1f} ms")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY (mean ms per batch)")
    print("=" * 70)

    stages = {
        "I/O (__getitems__)": io_stats["mean_ms"],
        "CPU→GPU transfer": transfer_stats["mean_ms"],
        "Normalization": norm_stats["mean_ms"],
        "Augmentations (total)": aug_total,
        "Final crop": crop_stats["mean_ms"],
    }
    total = sum(stages.values())

    print("\n| Stage | Time (ms) | % of total |")
    print("|-------|-----------|-----------|")
    for name, ms in stages.items():
        print(f"| {name} | {ms:.1f} | {ms / total * 100:.1f}% |")
    print(f"| **Total** | **{total:.1f}** | **100%** |")


if __name__ == "__main__":
    main()
