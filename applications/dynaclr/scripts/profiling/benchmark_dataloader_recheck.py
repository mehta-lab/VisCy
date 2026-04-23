"""Full-pipeline A/B benchmark for TensorStoreConfig.recheck_cached_data.

Drives :class:`dynaclr.data.datamodule.MultiExperimentDataModule`
end-to-end — ``__getitems__`` + ``collate_fn=lambda x:x`` +
PyTorch DataLoader with ``num_workers`` forked workers — to measure the
effect of ``recheck_cached_data`` on sustained training-loader
throughput, the only number that actually matters for GPU utilization.

Three legs are compared against the same parquet, in the same process,
with the same FOVs and the same seed so sampling is deterministic:

- ``"open"`` — validate at open only, trust cache thereafter (our
  expected production setting).
- ``None``   — driver default, revalidate cached chunk metadata every
  read (one stat/GETATTR per chunk per read on NFS).
- ``False``  — never revalidate (included for completeness).

Per leg the script:

1. Constructs a fresh ``MultiExperimentDataModule``, forcibly overriding
   ``self.tensorstore_config.recheck_cached_data`` after ``__init__`` so
   every Plate opens with the configured setting.
2. Runs ``setup("fit")`` once.
3. Warms the DataLoader with ``WARMUP_BATCHES`` batches (discarded).
4. Times ``N_BATCHES`` steady-state batches by wall-clocking the
   iterator yield interval — this is what the training loop sees.
5. Reports median/p95 iteration time and steady-state iter/s.

Because we use forked DataLoader workers, each config opens its own
Plates inside the worker after fork — matching real DDP training.

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/benchmark_dataloader_recheck.py

Requires:

- iohub with ``recheck_cached_data`` on ``TensorStoreConfig``
  (czbiohub-sf/iohub#406 or later).
- A parquet whose ``store_path`` entries are readable on this node.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule

CELL_INDEX_PARQUET = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"

BATCH_SIZE = 32
NUM_WORKERS = 4
WARMUP_BATCHES = 10
N_BATCHES = 100
SEED = 42

Z_WINDOW = 8
YX_PATCH_SIZE = (192, 192)
FINAL_YX_PATCH_SIZE = (160, 160)

LEGS: list[tuple[str, str | bool | None]] = [
    ("open (recommended)", "open"),
    ("None (driver default)", None),
    ("False (never revalidate)", False),
]


@dataclass
class LegResult:
    """Timing outcome for one recheck_cached_data leg."""

    label: str
    iter_latencies_s: list[float]
    total_s: float

    @property
    def median_ms(self) -> float:
        """Return the median inter-batch iteration time in milliseconds."""
        return statistics.median(self.iter_latencies_s) * 1000.0

    @property
    def p95_ms(self) -> float:
        """Return the p95 inter-batch iteration time in milliseconds."""
        return float(np.percentile(self.iter_latencies_s, 95)) * 1000.0

    @property
    def iter_per_s(self) -> float:
        """Return steady-state iterations per second."""
        return len(self.iter_latencies_s) / self.total_s

    @property
    def samples_per_s(self) -> float:
        """Return steady-state samples per second."""
        return self.iter_per_s * BATCH_SIZE


def _build_datamodule(recheck_cached_data: str | bool | None) -> MultiExperimentDataModule:
    """Construct a DataModule and force the recheck_cached_data leg onto its config."""
    dm = MultiExperimentDataModule(
        cell_index_path=CELL_INDEX_PARQUET,
        z_window=Z_WINDOW,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        channels_per_sample=None,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        stratify_by=None,
        split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        normalizations=[],
        augmentations=[],
    )
    # The datamodule sets recheck_cached_data="open" by default; override
    # it here so every leg can dial the knob independently without editing
    # the production code path.
    dm.tensorstore_config = dm.tensorstore_config.model_copy(update={"recheck_cached_data": recheck_cached_data})
    return dm


def _run_leg(label: str, recheck_cached_data: str | bool | None) -> LegResult:
    """Run one A/B leg and return a populated LegResult."""
    print(f"\n-- Leg: recheck_cached_data = {label} --")
    dm = _build_datamodule(recheck_cached_data)
    dm.setup("fit")
    loader = dm.train_dataloader()

    it = iter(loader)

    # Warmup — discard.  Forks workers, populates each worker's plate/ts
    # caches, amortises Python import cost in the forked child.
    for _ in range(WARMUP_BATCHES):
        _ = next(it)

    # Steady-state timing.  We measure the inter-batch yield interval,
    # which is exactly what the training loop observes.
    latencies_s: list[float] = []
    t_total = time.perf_counter()
    t_prev = time.perf_counter()
    for _ in range(N_BATCHES):
        _ = next(it)
        t_now = time.perf_counter()
        latencies_s.append(t_now - t_prev)
        t_prev = t_now
    total_s = time.perf_counter() - t_total

    # Release workers before the next leg so forked processes do not
    # pile up and compete for file descriptors.
    del it
    del loader

    result = LegResult(label=label, iter_latencies_s=latencies_s, total_s=total_s)
    print(
        f"  median {result.median_ms:.1f} ms | p95 {result.p95_ms:.1f} ms | "
        f"{result.iter_per_s:.2f} iter/s | {result.samples_per_s:.1f} samples/s"
    )
    return result


def _print_markdown(results: list[LegResult]) -> None:
    """Emit a markdown-formatted summary for the PR / Confluence."""
    print()
    print("## Results (dataloader-level A/B)")
    print()
    print(f"- Parquet: `{CELL_INDEX_PARQUET.split('/')[-1]}`")
    print(f"- Batch size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}")
    print(f"- Warmup: {WARMUP_BATCHES} batches; timed: {N_BATCHES} batches")
    print(f"- Z={Z_WINDOW}, YX={YX_PATCH_SIZE}, final_YX={FINAL_YX_PATCH_SIZE}")
    print()
    print("| recheck_cached_data | median ms | p95 ms | iter/s | samples/s |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r.label} | {r.median_ms:.1f} | {r.p95_ms:.1f} | {r.iter_per_s:.2f} | {r.samples_per_s:.1f} |")
    print()


def main() -> None:
    """Run all three legs and print a combined markdown summary."""
    print("=" * 72)
    print("Dataloader-level recheck_cached_data benchmark — MultiExperimentDataModule")
    print("=" * 72)

    results: list[LegResult] = []
    for label, value in LEGS:
        results.append(_run_leg(label, value))

    _print_markdown(results)


if __name__ == "__main__":
    main()
