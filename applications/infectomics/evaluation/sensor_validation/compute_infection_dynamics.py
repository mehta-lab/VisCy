# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import anndata

from utils.embedding_utils import convert_to_dataframe

# %% paths
sensor_zarr_path = Path("sensor_embeddings.zarr")
model_path = Path("models/sensor_model.joblib")
well_map_path = Path("well_map.csv")
output_csv = Path("percent_infected.csv")

# Time-axis metadata
time_interval_hours = 0.5   # hours between consecutive timepoints
start_hpi = 0.0             # hours post-infection at t=0

# Class definition (must match the training labels)
INFECTED_CLASS = 2


# %% load embeddings
print(f"Loading sensor embeddings from {sensor_zarr_path} ...")
sensor_adata = anndata.read_zarr(sensor_zarr_path)
print(f"  shape: {sensor_adata.shape}")

# %% convert to DataFrame
sensor_df = convert_to_dataframe(sensor_adata)

# %% load pre-trained classifier
print(f"Loading classifier from {model_path} ...")
clf_sensor = joblib.load(model_path)

# %% identify embedding feature columns
obs_cols = list(sensor_adata.obs.columns)
feature_cols = [c for c in sensor_df.columns if c not in obs_cols]

# %% run inference
print("Running inference ...")
sensor_df["predicted_state"] = clf_sensor.predict(sensor_df[feature_cols].values)

# %% map timepoint indices to hours post-infection
if "t" in sensor_df.columns:
    sensor_df["hpi"] = start_hpi + sensor_df["t"] * time_interval_hours
else:
    sensor_df["hpi"] = np.nan

# %% load well map and merge condition labels (optional – gracefully skipped if absent)
if well_map_path.exists():
    well_map = pd.read_csv(well_map_path)
    # Expect columns: well_id, condition (e.g. DENV MOI 1, uninfected …)
    if "well_id" in well_map.columns and "condition" in well_map.columns:
        sensor_df = sensor_df.merge(
            well_map[["well_id", "condition"]],
            left_on="fov_name",
            right_on="well_id",
            how="left",
        )
        print("  Well map merged.")
    else:
        print(f"  [WARN] well_map.csv lacks 'well_id' or 'condition' columns — skipping merge.")
else:
    print(f"  [INFO] {well_map_path} not found — no condition labels added.")

# %% compute percent infected per well / FOV / timepoint
groupby_cols = ["fov_name"]
if "t" in sensor_df.columns:
    groupby_cols.append("t")
if "hpi" in sensor_df.columns and sensor_df["hpi"].notna().any():
    groupby_cols.append("hpi")
if "condition" in sensor_df.columns:
    groupby_cols = ["condition"] + groupby_cols


def pct_infected(series: pd.Series) -> float:
    """Return percentage of cells predicted as infected in *series*."""
    return (series == INFECTED_CLASS).mean() * 100


pct_df = (
    sensor_df
    .groupby(groupby_cols)["predicted_state"]
    .agg(
        n_cells="count",
        percent_infected=pct_infected,
    )
    .reset_index()
)

# %% save
pct_df.to_csv(output_csv, index=False)
print(f"\nSaved {len(pct_df)} rows to {output_csv}")
print(pct_df.head(10))
