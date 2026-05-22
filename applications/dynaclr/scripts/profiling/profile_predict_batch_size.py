"""Sweep batch_size for prediction to find GPU utilization sweet spot.

Times the full predict pipeline (dataloader I/O + GPU forward) at increasing
batch sizes to find where GPU utilization saturates on the local A40.

Uses the microglia-eval parquet and the 2D MIP checkpoint.

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/profile_predict_batch_size.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_data._utils import _transform_channel_wise
from viscy_models.contrastive import ContrastiveEncoder
from viscy_transforms import BatchedChannelWiseZReductiond, NormalizeSampled

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CELL_INDEX_PARQUET = "/hpc/projects/organelle_phenotyping/models/collections/microglia-eval.parquet"
CKPT_PATH = (
    "/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels"
    "/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11"
    "/DynaCLR-2D-MIP-BagOfChannels/20260403-150013/checkpoints/last.ckpt"
)

BATCH_SIZES = [256, 512, 1024, 2048, 4096]
N_BATCHES = 20
WARMUP = 3
NUM_WORKERS = 4
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_dm(batch_size: int) -> MultiExperimentDataModule:
    """Build a predict-mode MultiExperimentDataModule for the given batch size."""
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        focus_channel="Phase3D",
        reference_pixel_size_xy_um=0.1494,
        z_window=1,
        z_extraction_window=11,
        z_focus_offset=0.5,
        yx_patch_size=(192, 192),
        final_yx_patch_size=(160, 160),
        channels_per_sample=1,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        split_ratio=1.0,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=42,
        normalizations=[
            NormalizeSampled(
                keys=["channel_0"],
                level="timepoint_statistics",
                subtrahend="mean",
                divisor="std",
            ),
            BatchedChannelWiseZReductiond(keys=["channel_0"], allow_missing_keys=True),
        ],
        augmentations=[],
    )
    dm.setup("predict")
    return dm


def load_model() -> torch.nn.Module:
    """Load ConvNeXt-Tiny encoder from the benchmark checkpoint."""
    encoder = ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=1,
        in_stack_depth=1,
        stem_kernel_size=[1, 4, 4],
        stem_stride=[1, 4, 4],
        embedding_dim=768,
        projection_dim=32,
        drop_path_rate=0.0,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    # checkpoint keys are prefixed with "model." since ContrastiveModule stores encoder as self.model
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    encoder.load_state_dict(state)
    encoder.eval()
    encoder.to(DEVICE)
    return encoder


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(batch_size: int, model: torch.nn.Module) -> dict:
    """Time the predict pipeline (I/O + forward) over N_BATCHES after warmup."""
    dm = setup_dm(batch_size)
    dl = dm.predict_dataloader()

    forward_times = []
    samples_processed = 0
    t_start = None

    with torch.inference_mode():
        for i, batch in enumerate(dl):
            if i >= WARMUP + N_BATCHES:
                break

            # Mirror the predict path: apply _predict_transform then forward
            norm_meta = batch.get("anchor_norm_meta")
            if isinstance(norm_meta, list) and all(m is None for m in norm_meta):
                norm_meta = None
            anchor = _transform_channel_wise(
                transform=dm._predict_transform,
                channel_names=dm._channel_names,
                patch=batch["anchor"].to(DEVICE),
                norm_meta=norm_meta,
            )

            if i == WARMUP:
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(anchor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= WARMUP:
                forward_times.append(t1 - t0)
                samples_processed += anchor.shape[0]

        torch.cuda.synchronize()
        t_end = time.perf_counter()

    wall_s = t_end - t_start if t_start else 1.0
    fwd = np.array(forward_times) * 1000

    return {
        "batch_size": batch_size,
        "forward_mean_ms": fwd.mean(),
        "forward_std_ms": fwd.std(),
        "e2e_samples_per_sec": samples_processed / wall_s,
        "gpu_mem_mib": torch.cuda.max_memory_allocated() // (1024**2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Sweep batch sizes and print a throughput summary table."""
    if not torch.cuda.is_available():
        print("No GPU available.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_mib = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print("=" * 65)
    print(f"Predict batch_size sweep — {gpu_name} ({total_mib} MiB)")
    print("=" * 65)
    print(f"num_workers={NUM_WORKERS}, warmup={WARMUP}, measured={N_BATCHES} batches")
    print("model: ConvNeXt-Tiny 2D MIP, input 1×1×160×160")
    print()

    print("Loading model...")
    model = load_model()
    torch.cuda.reset_peak_memory_stats()

    results = []
    for bs in BATCH_SIZES:
        print(f"batch_size={bs} ...", end=" ", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            r = benchmark(bs, model)
            results.append(r)
            print(
                f"{r['forward_mean_ms']:.1f} ms fwd | "
                f"{r['e2e_samples_per_sec']:.0f} samples/sec | "
                f"{r['gpu_mem_mib']} MiB"
            )
        except torch.cuda.OutOfMemoryError:
            print("OOM")
            break

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print()
    print("| batch_size | fwd ms | samples/sec | GPU MiB |")
    print("|------------|--------|-------------|---------|")
    for r in results:
        print(
            f"| {r['batch_size']:10d} | {r['forward_mean_ms']:6.1f} | "
            f"{r['e2e_samples_per_sec']:11.0f} | {r['gpu_mem_mib']:7d} |"
        )


if __name__ == "__main__":
    main()
