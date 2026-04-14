"""Stage 3b: Export DTW results as annotated AnnData zarr copies.

Merges alignment + classification results back into copies of the
original embedding zarr stores, adding obs columns:
  dtw_pseudotime, dtw_cost, warping_speed, response_group, template_id

Usage::

    uv run python export_anndata.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def main() -> None:
    """Export DTW-annotated AnnData copies."""
    parser = argparse.ArgumentParser(description="Export DTW results as AnnData zarr (Stage 3b)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--alignments", type=str, default=None, help="Path to alignments parquet file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    script_dir = Path(__file__).resolve().parent
    pseudotime_dir = script_dir.parent
    anndata_dir = script_dir / "anndata"
    anndata_dir.mkdir(parents=True, exist_ok=True)

    alignments_path = (
        Path(args.alignments)
        if args.alignments
        else pseudotime_dir / "1-align_cells" / "alignments" / "alignments.parquet"
    )
    merged = pd.read_parquet(alignments_path)

    alignment_cfg = config["alignment"]
    for ds in alignment_cfg["datasets"]:
        dataset_id = ds["dataset_id"]
        _logger.info(f"Exporting {dataset_id}")

        adata = ad.read_zarr(ds["embeddings_path"])
        adata.obs_names_make_unique()

        # Add integer position column for safe merging
        adata.obs["_iloc"] = np.arange(len(adata.obs))

        # Get this dataset's alignment results
        ds_merged = merged[merged["dataset_id"] == dataset_id].copy()
        if len(ds_merged) == 0:
            _logger.warning(f"  No alignment results for {dataset_id}, skipping")
            continue

        # Build lookup: (fov_name, track_id, t) → dtw columns
        dtw_cols = ["pseudotime", "dtw_cost", "warping_speed", "template_id", "cell_uid"]
        ds_lookup = ds_merged.set_index(["fov_name", "track_id", "t"])[dtw_cols]

        # Build matching index from adata.obs
        obs_key = list(zip(adata.obs["fov_name"], adata.obs["track_id"], adata.obs["t"]))
        obs_multi = pd.MultiIndex.from_tuples(obs_key, names=["fov_name", "track_id", "t"])

        # Reindex dtw columns to match adata obs order
        dtw_aligned = ds_lookup.reindex(obs_multi)

        # Only keep cells that were aligned (have pseudotime)
        aligned_mask = dtw_aligned["pseudotime"].notna().to_numpy()
        adata = adata[aligned_mask].copy()
        dtw_aligned = dtw_aligned[aligned_mask]

        # Write new columns
        adata.obs["dtw_pseudotime"] = dtw_aligned["pseudotime"].to_numpy()
        adata.obs["dtw_cost"] = dtw_aligned["dtw_cost"].to_numpy()
        adata.obs["warping_speed"] = dtw_aligned["warping_speed"].to_numpy()
        adata.obs["template_id"] = dtw_aligned["template_id"].to_numpy()
        adata.obs["cell_uid"] = dtw_aligned["cell_uid"].to_numpy()

        # Drop helper column
        adata.obs = adata.obs.drop(columns=["_iloc"])

        _logger.info(f"  {len(adata)} aligned cells (from {aligned_mask.sum()} matches)")

        # Rebuild obs/var as plain numpy-backed DataFrames (anndata zarr writer
        # cannot serialize Arrow-backed string arrays)
        with pd.option_context("mode.copy_on_write", False, "future.infer_string", False):
            new_obs = pd.DataFrame(index=pd.RangeIndex(len(adata.obs)).astype(str))
            for col in adata.obs.columns:
                vals = adata.obs[col].to_numpy()
                new_obs[col] = vals
            adata.obs = new_obs

            if len(adata.var) > 0:
                new_var = pd.DataFrame(index=pd.Index(np.arange(adata.n_vars).astype(str)))
                for col in adata.var.columns:
                    new_var[col] = adata.var[col].to_numpy()
                adata.var = new_var

        out_path = anndata_dir / f"{dataset_id}_dtw.zarr"
        adata.write_zarr(str(out_path), convert_strings_to_categoricals=False)
        _logger.info(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
