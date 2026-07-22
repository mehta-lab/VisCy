"""Step 2: conform Zuben's cell-index parquet to the canonical schema (end-to-end).

Single authoritative step, run after the v3 stores are preprocessed (Step 1):

1. Repoint ``store_path`` from the v2 originals to the v3 copies (same basename).
2. Fill required/derived columns absent from the source
   (``tracks_path``, ``microscope``, ``T_shape``, ``C_shape``).
3. ``preprocess_cell_index`` fills the ``norm_*`` columns from the v3 ``.zattrs``.
4. Seed ``z_focus = z`` LAST — gut has no ``focus_slice`` zattrs, so the bbox-center
   ``z`` IS the focus. This must come after step 3 because ``preprocess_cell_index``
   also writes ``z_focus`` (as NaN here, since there is no focus_slice) and would
   otherwise clobber it. The datamodule centers the Z window on ``z_focus``.

Run::

    uv run --no-sync python applications/dynaclr/configs/collections/zuben_gut/conform_parquet.py
"""

import os

import pandas as pd

from viscy_data.cell_index import preprocess_cell_index, read_cell_index, write_cell_index

SRC = "/hpc/projects/jacobo_group/zuben/proj/gutCellClassifier/data/dynaclr_cell_index_bbox_center.parquet"
V3_ROOT = "/hpc/projects/organelle_phenotyping/datasets/zuben_gut_development"
OUT = "/hpc/projects/jacobo_group/collab/ed/dynaclr/dynaclr_cell_index_gut_v1.parquet"
N_CHANNELS = 4


def main() -> None:
    """Conform the parquet, fill norm stats, and seed z_focus from the bbox-center z."""
    df = pd.read_parquet(SRC)
    n_rows_in = len(df)

    # Repoint store_path to the v3 copies (same basename).
    df["store_path"] = df["store_path"].astype(str).map(lambda p: f"{V3_ROOT}/{os.path.basename(p)}")

    # Required columns absent from the source parquet.
    df["tracks_path"] = ""  # ignored by ExperimentRegistry.from_cell_index
    df["microscope"] = ""
    df["T_shape"] = 1  # static: single timepoint
    df["C_shape"] = N_CHANNELS

    write_cell_index(df, OUT)

    # Fill norm_* from the v3 .zattrs (writes z_focus as NaN — no focus_slice).
    preprocess_cell_index(OUT, focus_channel="nuclear")

    # Seed z_focus = z LAST so it survives preprocess_cell_index.
    out_df = read_cell_index(OUT)
    out_df["z_focus"] = out_df["z"].astype("float32")
    write_cell_index(out_df, OUT)

    check = read_cell_index(OUT)
    print("# Step 2 — conform parquet\n")
    print(f"- rows in: **{n_rows_in}**, rows out: **{len(check)}**")
    print(f"- output: `{OUT}`")
    print(f"- norm_mean NaN: **{int(check['norm_mean'].isna().sum())}**")
    print(f"- z_focus NaN: **{int(check['z_focus'].isna().sum())}** (should be 0)")


if __name__ == "__main__":
    main()
