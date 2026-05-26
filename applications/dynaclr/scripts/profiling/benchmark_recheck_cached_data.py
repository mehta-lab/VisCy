"""Measure the impact of ``TensorStoreConfig.recheck_cached_data`` on NFS reads.

Single-process raw ``ts.stack(...).read().result()`` loop against a
2-experiment parquet for three TensorStoreConfig settings:

- ``none`` — driver default, revalidate on every read (one stat/GETATTR
  per chunk per read).
- ``open`` — validate only at open time, trust the cache thereafter.
- ``false`` — never revalidate.

The loop issues ``N_BATCHES`` batches of stacked 3D crops sampled from
random FOVs, reports median/p95 read latency and sustained patches/s.
For the DataLoader-driven end-to-end view see
``benchmark_dataloader_workers_sweep.py``.

Usage
-----
    uv run python applications/dynaclr/scripts/profiling/benchmark_recheck_cached_data.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tensorstore as ts
from iohub import open_ome_zarr
from iohub.core.config import TensorStoreConfig

CELL_INDEX_PARQUET = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/benchmark_2exp.parquet"

BATCH_SIZE = 32
N_BATCHES = 50
PATCH_Z = 8
PATCH_YX = (192, 192)
SEED = 0

DATA_COPY_CONCURRENCY = 16
FILE_IO_CONCURRENCY = 64
CACHE_POOL_BYTES: int | None = None

CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("none (driver default)", {}),
    ("open", {"recheck_cached_data": "open"}),
    ("false", {"recheck_cached_data": False}),
]


@dataclass
class Result:
    """Timing results for one ``recheck_cached_data`` configuration."""

    label: str
    batch_latencies_ms: list[float]
    total_bytes: int
    total_s: float

    @property
    def median_ms(self) -> float:
        """Return the median per-batch read latency in milliseconds."""
        return statistics.median(self.batch_latencies_ms)

    @property
    def p95_ms(self) -> float:
        """Return the p95 per-batch read latency in milliseconds."""
        return float(np.percentile(self.batch_latencies_ms, 95))

    @property
    def patches_per_s(self) -> float:
        """Return the sustained patch-read throughput."""
        return BATCH_SIZE * len(self.batch_latencies_ms) / self.total_s

    @property
    def mib_per_s(self) -> float:
        """Return the sustained read throughput in MiB/s."""
        return (self.total_bytes / (1024 * 1024)) / self.total_s


def _load_fov_index() -> pd.DataFrame:
    """Return unique (store_path, well, fov, shape) rows from the benchmark parquet."""
    df = pd.read_parquet(CELL_INDEX_PARQUET)
    unique = df[["store_path", "well", "fov", "C_shape", "Z_shape", "Y_shape", "X_shape"]].drop_duplicates(
        subset=["store_path", "well", "fov"]
    )
    return unique.reset_index(drop=True)


def _open_stores(fov_df: pd.DataFrame, ts_config: TensorStoreConfig) -> dict[str, Any]:
    """Open each unique zarr store once with the given TensorStoreConfig."""
    store_paths = fov_df["store_path"].drop_duplicates().tolist()
    plates: dict[str, Any] = {}
    for sp in store_paths:
        plates[sp] = open_ome_zarr(
            sp,
            mode="r",
            implementation="tensorstore",
            implementation_config=ts_config,
        )
    return plates


def _sample_patches(
    fov_df: pd.DataFrame,
    plates: dict[str, Any],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[ts.TensorStore], int]:
    """Pick ``batch_size`` random (fov, z, y, x) crops and return lazy slices + byte count.

    Returns a list of tensorstore lazy slices (one per crop) plus the
    total number of bytes the resulting stacked read will pull.
    """
    lazies: list[ts.TensorStore] = []
    total_bytes = 0
    rows = fov_df.sample(n=batch_size, replace=True, random_state=rng.integers(0, 2**31 - 1))
    for _, row in rows.iterrows():
        plate = plates[row["store_path"]]
        position_path = f"{row['well']}/{row['fov']}"
        arr = plate[position_path]["0"].native
        z_start = int(rng.integers(0, max(1, row["Z_shape"] - PATCH_Z + 1)))
        y_start = int(rng.integers(0, max(1, row["Y_shape"] - PATCH_YX[0] + 1)))
        x_start = int(rng.integers(0, max(1, row["X_shape"] - PATCH_YX[1] + 1)))
        lazy = arr[
            0,  # t=0 — keep indexing simple; timepoint is not what we're benchmarking
            :,
            z_start : z_start + PATCH_Z,
            y_start : y_start + PATCH_YX[0],
            x_start : x_start + PATCH_YX[1],
        ]
        lazies.append(lazy)
        total_bytes += PATCH_Z * PATCH_YX[0] * PATCH_YX[1] * row["C_shape"] * 4  # assume float32
    return lazies, total_bytes


def _run_one_config(label: str, extra_cfg: dict[str, Any], fov_df: pd.DataFrame) -> Result:
    """Run the read-loop benchmark for one recheck_cached_data setting."""
    ts_config = TensorStoreConfig(
        data_copy_concurrency=DATA_COPY_CONCURRENCY,
        file_io_concurrency=FILE_IO_CONCURRENCY,
        cache_pool_bytes=CACHE_POOL_BYTES,
        **extra_cfg,
    )
    plates = _open_stores(fov_df, ts_config)

    def _translate_all(lazies: list[ts.TensorStore]) -> list[ts.TensorStore]:
        """Translate each lazy slice to origin so ts.stack can combine them."""
        return [p.translate_to[0] for p in lazies]  # noqa: PD013

    rng_warm = np.random.default_rng(SEED)
    warm_lazies, _ = _sample_patches(fov_df, plates, BATCH_SIZE, rng_warm)
    _ = ts.stack(_translate_all(warm_lazies)).read().result()

    rng = np.random.default_rng(SEED + 1)
    latencies_ms: list[float] = []
    total_bytes = 0
    t_total = time.perf_counter()
    for _ in range(N_BATCHES):
        lazies, batch_bytes = _sample_patches(fov_df, plates, BATCH_SIZE, rng)
        t0 = time.perf_counter()
        _ = ts.stack(_translate_all(lazies)).read().result()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        total_bytes += batch_bytes
    total_s = time.perf_counter() - t_total

    for plate in plates.values():
        plate.close()

    return Result(label=label, batch_latencies_ms=latencies_ms, total_bytes=total_bytes, total_s=total_s)


def _print_markdown_table(results: list[Result]) -> None:
    """Print a markdown-formatted results table suitable for Confluence/PR pasting."""
    print()
    print("## Results")
    print()
    print(f"- Parquet: `{CELL_INDEX_PARQUET.split('/')[-1]}`")
    print(f"- Batch size: {BATCH_SIZE}, N batches: {N_BATCHES}")
    print(f"- Patch shape: (C, Z={PATCH_Z}, Y={PATCH_YX[0]}, X={PATCH_YX[1]})")
    print(f"- data_copy_concurrency={DATA_COPY_CONCURRENCY}, file_io_concurrency={FILE_IO_CONCURRENCY}")
    print()
    print("| recheck_cached_data | median ms | p95 ms | patches/s | MiB/s | total s |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.label} | {r.median_ms:.1f} | {r.p95_ms:.1f} | "
            f"{r.patches_per_s:.1f} | {r.mib_per_s:.1f} | {r.total_s:.2f} |"
        )
    print()


def main() -> None:
    """Run the three configurations back-to-back and print a markdown summary."""
    print("=" * 72)
    print("recheck_cached_data benchmark — DynaCLR contrastive read pattern on VAST")
    print("=" * 72)

    fov_df = _load_fov_index()
    print(f"Loaded {len(fov_df)} unique FOVs across {fov_df['store_path'].nunique()} stores")

    results: list[Result] = []
    for label, extra_cfg in CONFIGS:
        print(f"\n-- Running: recheck_cached_data = {label} --")
        r = _run_one_config(label, extra_cfg, fov_df)
        print(f"  median {r.median_ms:.1f} ms | p95 {r.p95_ms:.1f} ms | {r.patches_per_s:.1f} patches/s")
        results.append(r)

    _print_markdown_table(results)


if __name__ == "__main__":
    main()
