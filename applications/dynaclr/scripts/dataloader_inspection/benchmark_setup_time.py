"""Benchmark MultiExperimentDataModule setup time.

Measures the time for _compute_valid_anchors and _build_match_lookup
on the DynaCLR-2D-MIP-BagOfChannels parquet (3.3M rows) to quantify
the speedup from the vectorized implementations.

Usage
-----
    uv run python applications/dynaclr/scripts/dataloader_inspection/benchmark_setup_time.py
"""

from __future__ import annotations

import time

CELL_INDEX_PARQUET = "applications/dynaclr/configs/cell_index/DynaCLR-2D-MIP-BagOfChannels.parquet"
TAU_RANGE = (0.5, 2.0)
YX_PATCH_SIZE = (256, 256)


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def main() -> None:
    """Run the MultiExperimentDataModule setup benchmark and print a timing summary."""
    from dynaclr.data.experiment import ExperimentRegistry
    from dynaclr.data.index import MultiExperimentIndex
    from viscy_data.cell_index import read_cell_index

    print("=" * 60)
    print("MultiExperimentDataModule setup benchmark")
    print(f"Parquet: {CELL_INDEX_PARQUET}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Parquet read (shared cost)
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    df = read_cell_index(CELL_INDEX_PARQUET)
    parquet_time = time.perf_counter() - t0
    print(f"\nParquet read:     {_fmt(parquet_time)}  ({len(df):,} rows)")

    # ----------------------------------------------------------------
    # Registry build (shared cost)
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    registry, _ = ExperimentRegistry.from_cell_index(
        CELL_INDEX_PARQUET,
        z_window=1,
        z_extraction_window=20,
        z_focus_offset=0.3,
        focus_channel="Phase3D",
        reference_pixel_size_xy_um=0.1494,
    )
    registry_time = time.perf_counter() - t0
    print(f"Registry build:   {_fmt(registry_time)}  ({len(registry.experiments)} experiments)")

    # ----------------------------------------------------------------
    # MultiExperimentIndex (includes _compute_valid_anchors)
    # ----------------------------------------------------------------
    print("\n--- MultiExperimentIndex (cell_index_df path) ---")
    t0 = time.perf_counter()
    index = MultiExperimentIndex(
        registry=registry,
        yx_patch_size=YX_PATCH_SIZE,
        tau_range_hours=TAU_RANGE,
        cell_index_df=df,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
    )
    index_time = time.perf_counter() - t0
    print(f"  Total:          {_fmt(index_time)}")
    print(f"  Tracks:         {len(index.tracks):,}  Valid anchors: {len(index.valid_anchors):,}")

    # ----------------------------------------------------------------
    # _build_match_lookup (MultiExperimentTripletDataset init)
    # ----------------------------------------------------------------
    print("\n--- _build_match_lookup (dataset init) ---")
    from dynaclr.data.dataset import MultiExperimentTripletDataset

    t0 = time.perf_counter()
    MultiExperimentTripletDataset(
        index=index,
        fit=True,
        tau_range_hours=TAU_RANGE,
        cache_pool_bytes=0,
        channels_per_sample=1,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
    )
    dataset_time = time.perf_counter() - t0
    print(f"  _build_match_lookup: {_fmt(dataset_time)}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    total = parquet_time + registry_time + index_time + dataset_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("| Step                    | Time           |")
    print("|-------------------------|----------------|")
    print(f"| Parquet read            | {_fmt(parquet_time):>14} |")
    print(f"| Registry build          | {_fmt(registry_time):>14} |")
    print(f"| Index (_valid_anchors)  | {_fmt(index_time):>14} |")
    print(f"| Dataset (_match_lookup) | {_fmt(dataset_time):>14} |")
    print("|-------------------------|----------------|")
    print(f"| **Total**               | {_fmt(total):>14} |")


if __name__ == "__main__":
    main()
