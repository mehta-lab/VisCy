"""Step 0: verify Zuben's cell-index parquet is consistent with its zarr stores.

Checks, for every store referenced in the parquet:
- the store opens and exposes the expected channel names,
- each (well, fov) referenced resolves to a real position,
- cell centroids fit inside the FOV with room for the requested patch half-width.

Run::

    uv run --no-sync python applications/dynaclr/configs/collections/zuben_gut/verify_parquet_zarr.py
"""

import sys

import pandas as pd
from iohub import open_ome_zarr

PARQUET = "/hpc/projects/jacobo_group/zuben/proj/gutCellClassifier/data/dynaclr_cell_index_bbox_center.parquet"
EXPECTED_CHANNELS = ["nuclear", "septate", "brush_border", "SuH"]
YX_PATCH_SIZE = (256, 256)  # extraction patch used by the bag-of-channels config


def main() -> int:
    """Check every store opens, channels match, positions resolve, and report border-OOB cells."""
    df = pd.read_parquet(PARQUET)
    y_half = YX_PATCH_SIZE[0] // 2
    x_half = YX_PATCH_SIZE[1] // 2

    problems: list[str] = []
    n_cells_out_of_bounds = 0

    for store_path, store_group in df.groupby("store_path", observed=True):
        store_path = str(store_path)
        try:
            with open_ome_zarr(store_path, mode="r") as plate:
                channels = list(plate.channel_names)
                if channels != EXPECTED_CHANNELS:
                    problems.append(f"{store_path}: channels {channels} != {EXPECTED_CHANNELS}")
                positions = {name for name, _ in plate.positions()}
        except Exception as exc:  # noqa: BLE001 - surface any open failure
            problems.append(f"{store_path}: failed to open ({exc})")
            continue

        for (well, fov), fov_group in store_group.groupby(["well", "fov"], observed=True):
            pos_key = f"{well}/{fov}"
            if pos_key not in positions:
                problems.append(f"{store_path}: position {pos_key} not found")
                continue
            y_shape = int(fov_group["Y_shape"].iloc[0])
            x_shape = int(fov_group["X_shape"].iloc[0])
            y = fov_group["y"].to_numpy()
            x = fov_group["x"].to_numpy()
            oob = (y < y_half) | (y > y_shape - y_half) | (x < x_half) | (x > x_shape - x_half)
            n_cells_out_of_bounds += int(oob.sum())

    n_stores = df["store_path"].nunique()
    n_cells = df["cell_id"].nunique()

    print("# Step 0 — parquet↔zarr consistency\n")
    print(f"- stores checked: **{n_stores}**")
    print(f"- unique cells: **{n_cells}**")
    print(f"- expected channels: `{EXPECTED_CHANNELS}`")
    print(
        f"- cells whose {YX_PATCH_SIZE} patch would fall out of bounds: "
        f"**{n_cells_out_of_bounds}** "
        f"({100 * n_cells_out_of_bounds / len(df):.1f}% of rows)"
    )

    if problems:
        print(f"\n**{len(problems)} problem(s):**")
        for p in problems[:50]:
            print(f"  - {p}")
        return 1

    print("\n**OK** — all stores open, channels match, positions resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
