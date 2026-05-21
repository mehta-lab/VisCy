"""Production-config DataLoader benchmark + batch-composition sanity check.

Exercises the real
``DynaCLR-2D-MIP-BagOfChannels.yml`` training settings against the
committed v2 parquet to measure end-to-end DataLoader throughput and
verify that batch-grouping/stratification actually do what the config
says.

Two parts
---------

**1. Composition check** — forces ``batch_group_by="marker"`` and checks
the first 20 batches:

- every batch contains exactly one marker (single-marker batches),
- different batches surface different markers (proves the grouping is
  shuffled across the epoch, not stuck on one value).

**2. Throughput A/B** — runs the production config
(``batch_size=256``, ``channels_per_sample=1``, ``stratify_by=[perturbation, marker]``,
``num_workers=2``) under two ``recheck_cached_data`` settings:

- ``None`` — TensorStore driver default.
- ``"open"`` — validate at open only (our merge's default).

Reports median/p95 per-iter latency, iter/s, samples/s for each leg.
Because this runs on the real VAST-resident parquet with 7k+ FOVs, the
FOV-open amortisation is representative of real training.

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/benchmark_boc2d_real.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_transforms import (
    BatchedChannelWiseZReductiond,
    BatchedRandSpatialCropd,
    NormalizeSampled,
)

CELL_INDEX_PARQUET = "/hpc/projects/organelle_phenotyping/models/collections/DynaCLR-2D-MIP-BagOfChannels-v2.parquet"

BATCH_SIZE = 256
NUM_WORKERS = 2
WARMUP_BATCHES = 10
N_BATCHES = 60
SEED = 42

Z_WINDOW = 1
Z_EXTRACTION_WINDOW = 20
Z_FOCUS_OFFSET = 0.3
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)

COMPOSITION_BATCHES = 20

RECHECK_LEGS: list[tuple[str, str | bool | None]] = [
    ("None (driver default)", None),
    ("open (our default)", "open"),
]


@dataclass
class LegResult:
    """Timing outcome for one recheck_cached_data leg on the real parquet."""

    label: str
    iter_latencies_s: list[float]
    total_s: float

    @property
    def median_ms(self) -> float:
        """Return median per-iter latency in milliseconds."""
        return statistics.median(self.iter_latencies_s) * 1000.0

    @property
    def p95_ms(self) -> float:
        """Return p95 per-iter latency in milliseconds."""
        return float(np.percentile(self.iter_latencies_s, 95)) * 1000.0

    @property
    def iter_per_s(self) -> float:
        """Return sustained iterations per second."""
        return len(self.iter_latencies_s) / self.total_s

    @property
    def samples_per_s(self) -> float:
        """Return sustained samples per second."""
        return self.iter_per_s * BATCH_SIZE


def _build_production_dm(
    recheck_cached_data: str | bool | None,
    batch_group_by: str | list[str] | None = None,
    stratify_by: list[str] | None = None,
    num_workers: int = NUM_WORKERS,
) -> MultiExperimentDataModule:
    """Build a DataModule matching the production 2D-MIP-BoC training recipe."""
    normalizations = [
        NormalizeSampled(
            keys=["channel_0"],
            level="timepoint_statistics",
            subtrahend="mean",
            divisor="std",
        ),
    ]
    augmentations = [
        BatchedRandSpatialCropd(keys=["channel_0"], roi_size=(10, 192, 192)),
        BatchedChannelWiseZReductiond(keys=["channel_0"], allow_missing_keys=True),
    ]
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        focus_channel="Phase3D",
        reference_pixel_size_xy_um=0.1494,
        z_window=Z_WINDOW,
        z_extraction_window=Z_EXTRACTION_WINDOW,
        z_focus_offset=Z_FOCUS_OFFSET,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        channels_per_sample=1,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        positive_channel_source="same",
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        batch_group_by=batch_group_by,
        stratify_by=stratify_by if stratify_by is not None else ["perturbation", "marker"],
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        seed=SEED,
        normalizations=normalizations,
        augmentations=augmentations,
    )
    dm.tensorstore_config = dm.tensorstore_config.model_copy(update={"recheck_cached_data": recheck_cached_data})
    return dm


def _composition_check() -> None:
    """Verify batch_group_by='marker' yields single-marker, shuffled batches."""
    print("=" * 72)
    print("Composition check: batch_group_by='marker'")
    print("=" * 72)

    dm = _build_production_dm(
        recheck_cached_data="open",
        batch_group_by="marker",
        stratify_by=None,
        num_workers=0,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    it = iter(loader)

    markers_by_batch: list[set[str]] = []
    for i in range(COMPOSITION_BATCHES):
        batch = next(it)
        metas = batch["anchor_meta"]
        batch_markers = {m["marker"] for m in metas}
        markers_by_batch.append(batch_markers)
        print(f"  batch {i:>2}: {len(batch_markers)} unique markers → {sorted(batch_markers)[:4]}")

    non_singleton = [i for i, ms in enumerate(markers_by_batch) if len(ms) != 1]
    if non_singleton:
        print(f"\n  FAIL: {len(non_singleton)} of {COMPOSITION_BATCHES} batches had >1 marker")
        print(f"  offending batches: {non_singleton}")
        raise AssertionError("batch_group_by='marker' did not produce single-marker batches")

    unique_markers_seen = set().union(*markers_by_batch)
    print(f"\n  PASS: all {COMPOSITION_BATCHES} batches are single-marker")
    print(f"  markers touched across the {COMPOSITION_BATCHES} batches: {len(unique_markers_seen)}")
    print(f"  → {sorted(unique_markers_seen)}")

    if len(unique_markers_seen) < 2:
        print("\n  WARNING: only 1 marker touched across all batches — epoch may be stuck on one group")
    else:
        print("  → grouping is shuffled across markers (good)")

    del it
    del loader


def _run_throughput_leg(label: str, recheck_cached_data: str | bool | None) -> LegResult:
    """Run one throughput leg on the production config."""
    print(f"\n-- Throughput leg: recheck_cached_data = {label} --")
    dm = _build_production_dm(
        recheck_cached_data=recheck_cached_data,
        batch_group_by=None,
        stratify_by=["perturbation", "marker"],
        num_workers=NUM_WORKERS,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    it = iter(loader)

    for _ in range(WARMUP_BATCHES):
        _ = next(it)

    latencies_s: list[float] = []
    t_total = time.perf_counter()
    t_prev = time.perf_counter()
    for _ in range(N_BATCHES):
        _ = next(it)
        t_now = time.perf_counter()
        latencies_s.append(t_now - t_prev)
        t_prev = t_now
    total_s = time.perf_counter() - t_total

    del it
    del loader

    result = LegResult(label=label, iter_latencies_s=latencies_s, total_s=total_s)
    print(
        f"  median {result.median_ms:.1f} ms | p95 {result.p95_ms:.1f} ms | "
        f"{result.iter_per_s:.2f} iter/s | {result.samples_per_s:.1f} samples/s"
    )
    return result


def _print_markdown(results: list[LegResult]) -> None:
    """Emit a markdown-formatted throughput table."""
    print()
    print("## Throughput (real 2D-MIP-BoC v2 parquet)")
    print()
    print(f"- Parquet: `{CELL_INDEX_PARQUET.split('/')[-1]}`")
    print(f"- Batch size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}")
    print(f"- Warmup: {WARMUP_BATCHES} batches; timed: {N_BATCHES} batches")
    print(f"- Z_extraction={Z_EXTRACTION_WINDOW}, YX={YX_PATCH_SIZE}, final_YX={FINAL_YX_PATCH_SIZE}")
    print("- channels_per_sample=1, stratify_by=[perturbation, marker]")
    print()
    print("| recheck_cached_data | median ms | p95 ms | iter/s | samples/s |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r.label} | {r.median_ms:.1f} | {r.p95_ms:.1f} | {r.iter_per_s:.2f} | {r.samples_per_s:.1f} |")
    print()


def main() -> None:
    """Run composition check, then the throughput A/B, and print a summary."""
    _composition_check()

    print()
    print("=" * 72)
    print("Throughput A/B: production config, real parquet")
    print("=" * 72)

    results: list[LegResult] = []
    for label, value in RECHECK_LEGS:
        results.append(_run_throughput_leg(label, value))

    _print_markdown(results)


if __name__ == "__main__":
    main()
