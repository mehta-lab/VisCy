"""Step 2: conform Zuben's cell-index parquet to the canonical schema.

- Repoints ``store_path`` from the v2 originals to the v3 copies under
  ``/hpc/projects/organelle_phenotyping/datasets/zuben_gut_development`` (same basename).
- Fills the few required/derived columns not present in the source
  (``tracks_path``, ``microscope``, ``T_shape``, ``C_shape``).
- ``write_cell_index`` adds every other missing schema column as null and casts dtypes.

Norm columns are populated afterwards by ``dynaclr preprocess-cell-index`` reading the v3
``.zattrs`` written by ``viscy preprocess``. Run this only after Step 1 completes.

Run::

    uv run --no-sync python applications/dynaclr/configs/collections/zuben_gut/conform_parquet.py
"""

import os

import pandas as pd

from viscy_data.cell_index import read_cell_index, write_cell_index

SRC = "/hpc/projects/jacobo_group/zuben/proj/gutCellClassifier/data/dynaclr_cell_index_bbox_center.parquet"
V3_ROOT = "/hpc/projects/organelle_phenotyping/datasets/zuben_gut_development"
OUT = "/hpc/projects/jacobo_group/collab/ed/dynaclr/dynaclr_cell_index_gut_v1.parquet"
N_CHANNELS = 4


def main() -> None:
    """Repoint store paths to the v3 copies and conform the parquet to CELL_INDEX_SCHEMA."""
    df = pd.read_parquet(SRC)
    n_rows_in = len(df)

    # Repoint store_path to the v3 copies (same basename).
    df["store_path"] = df["store_path"].astype(str).map(lambda p: f"{V3_ROOT}/{os.path.basename(p)}")

    # Required columns absent from the source parquet.
    df["tracks_path"] = ""  # ignored by ExperimentRegistry.from_cell_index
    df["microscope"] = ""
    df["T_shape"] = 1  # static: single timepoint
    df["C_shape"] = N_CHANNELS

    # write_cell_index adds remaining nullable schema columns as None and casts
    # to CELL_INDEX_SCHEMA; validate_cell_index runs inside.
    write_cell_index(df, OUT)

    check = read_cell_index(OUT)
    print("# Step 2 — conform parquet\n")
    print(f"- rows in: **{n_rows_in}**, rows out: **{len(check)}**")
    print(f"- output: `{OUT}`")
    print(f"- example store_path: `{check['store_path'].iloc[0]}`")
    print("- next: `dynaclr preprocess-cell-index` with `--focus-channel nuclear` to fill norm_*")


if __name__ == "__main__":
    main()
