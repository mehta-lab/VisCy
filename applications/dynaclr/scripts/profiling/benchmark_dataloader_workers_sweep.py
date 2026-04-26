"""Sweep num_workers × recheck_cached_data for the DynaCLR dataloader.

Purpose
-------

The first pass A/B (``benchmark_dataloader_recheck.py``) showed a counter-
intuitive result on ``MultiExperimentDataModule.train_dataloader()`` with
``num_workers=4``: ``recheck_cached_data="open"`` was slower than the
driver default. The raw ``ts.stack`` benchmark showed the opposite. Most
likely the p95 tails were dominated by first-touch FOV opens while the
ThreadDataLoader prefetch buffer masked them differently per leg.

This sweep pins down the cause by running every ``recheck_cached_data``
value across several ``num_workers`` settings with generous warmup, so we
can tell:

- Does the ordering flip between ``num_workers=0`` (no fork, no thread
  buffer) and ``num_workers>0`` (forked workers)?
- Is the ``"open"`` penalty paid only on cold FOV opens? If yes, longer
  warmup should close the gap.
- Does the ``p95`` converge once steady-state is reached?

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/benchmark_dataloader_workers_sweep.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule

CELL_INDEX_PARQUET = "/hpc/projects/organelle_phenotyping/models/collections/DynaCLR-2D-MIP-BagOfChannels-v2.parquet"

BATCH_SIZE = 256
WARMUP_BATCHES = 10
N_BATCHES = 40
SEED = 42

Z_WINDOW = 1
Z_EXTRACTION_WINDOW = 20
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)

WORKER_COUNTS: list[int] = [0, 2, 4, 8]
RECHECK_VALUES: list[tuple[str, str | bool | None]] = [
    ("None", None),
    ("open", "open"),
    ("False", False),
]


@dataclass
class SweepResult:
    """One cell of the ``num_workers`` × ``recheck_cached_data`` grid."""

    num_workers: int
    recheck_label: str
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
        """Return sustained iterations per second across timed batches."""
        return len(self.iter_latencies_s) / self.total_s

    @property
    def samples_per_s(self) -> float:
        """Return sustained samples per second (iter/s × batch)."""
        return self.iter_per_s * BATCH_SIZE


def _build(num_workers: int, recheck_cached_data: str | bool | None) -> MultiExperimentDataModule:
    """Build one datamodule with forced num_workers and recheck_cached_data."""
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        z_window=Z_WINDOW,
        z_extraction_window=Z_EXTRACTION_WINDOW,
        z_focus_offset=0.3,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        channels_per_sample=1,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=["perturbation", "marker"],
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        seed=SEED,
        focus_channel="Phase3D",
        reference_pixel_size_xy_um=0.1494,
        normalizations=[],
        augmentations=[],
    )
    dm.tensorstore_config = dm.tensorstore_config.model_copy(update={"recheck_cached_data": recheck_cached_data})
    return dm


def _run_cell(num_workers: int, label: str, recheck_cached_data: str | bool | None) -> SweepResult:
    """Run one cell of the sweep."""
    print(f"\n-- num_workers={num_workers}, recheck_cached_data={label} --")
    dm = _build(num_workers, recheck_cached_data)
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

    result = SweepResult(
        num_workers=num_workers,
        recheck_label=label,
        iter_latencies_s=latencies_s,
        total_s=total_s,
    )
    print(
        f"  median {result.median_ms:.1f} ms | p95 {result.p95_ms:.1f} ms | "
        f"{result.iter_per_s:.2f} iter/s | {result.samples_per_s:.1f} samples/s"
    )
    return result


def _print_markdown(results: list[SweepResult]) -> None:
    """Emit a markdown-formatted sweep table for the PR / Confluence."""
    print()
    print("## Sweep results")
    print()
    print(f"- Parquet: `{CELL_INDEX_PARQUET.split('/')[-1]}`")
    print(f"- Batch size: {BATCH_SIZE}, warmup: {WARMUP_BATCHES}, timed: {N_BATCHES}")
    print(f"- Z={Z_WINDOW}, YX={YX_PATCH_SIZE}, final_YX={FINAL_YX_PATCH_SIZE}")
    print()
    print("| num_workers | recheck | median ms | p95 ms | iter/s | samples/s |")
    print("|---:|---|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.num_workers} | {r.recheck_label} | "
            f"{r.median_ms:.1f} | {r.p95_ms:.1f} | "
            f"{r.iter_per_s:.2f} | {r.samples_per_s:.1f} |"
        )
    print()


def main() -> None:
    """Run the full sweep and print a combined markdown summary."""
    print("=" * 72)
    print("num_workers × recheck_cached_data sweep — MultiExperimentDataModule")
    print("=" * 72)

    results: list[SweepResult] = []
    for nw in WORKER_COUNTS:
        for label, value in RECHECK_VALUES:
            results.append(_run_cell(nw, label, value))

    _print_markdown(results)


if __name__ == "__main__":
    main()
