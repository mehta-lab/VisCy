# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import anndata
from sklearn.metrics import confusion_matrix

from utils.embedding_utils import convert_to_dataframe

# %% paths
sensor_zarr_path = Path("sensor_embeddings.zarr")
estain_zarr_path = Path("estain_embeddings.zarr")

sensor_model_path = Path("models/sensor_model.joblib")
estain_model_path = Path("models/estain_model.joblib")

output_metrics_csv = Path("infection_metrics.csv")
output_infection_pct_csv = Path("infection_percentage_by_MOI.csv")

# Well lists (one entry per MOI level, ordered low → high)
# GFP sensor dataset uses numeric-style well names (e.g. "0/4", "0/5")
# mCherry / estain dataset uses letter-numeric well names (e.g. "A/1", "B/1")
# Adjust these lists to match the actual plate layout.
ZIKV_wells_sensor = ["0/4", "0/5", "0/6", "0/7"]
DENV_wells_sensor = ["1/4", "1/5", "1/6", "1/7"]

ZIKV_wells_estain = ["A/1", "A/2", "A/3", "A/4"]
DENV_wells_estain = ["B/1", "B/2", "B/3", "B/4"]

# Human-readable MOI labels (same length as the well lists above)
MOI_names = ["0.01", "0.1", "1", "10"]

# Annotation class values
UNINFECTED_CLASS = 1
INFECTED_CLASS = 2


# %% helper – per-well classification metrics
def compute_well_metrics(
    df: pd.DataFrame,
    well_col: str,
    pred_col: str,
    true_col: str,
    well_id: str,
) -> dict:
    """Compute accuracy, precision, recall, and F1 for a single well.

    Rows where ``true_col`` == 0 are excluded (class 0 = unlabelled /
    ambiguous cells).

    Parameters
    ----------
    df:
        DataFrame containing predictions and ground-truth labels.
    well_col:
        Column name that identifies the well.
    pred_col:
        Column name for predicted class labels.
    true_col:
        Column name for ground-truth class labels.
    well_id:
        Well identifier to filter on.

    Returns
    -------
    dict
        Metric dict for this well.
    """
    subset = df[(df[well_col] == well_id) & (df[true_col] != 0)].copy()
    if subset.empty:
        return {"well_id": well_id, "n_cells": 0, "accuracy": np.nan,
                "precision": np.nan, "recall": np.nan, "F1": np.nan,
                "pct_infected_true": np.nan, "pct_infected_pred": np.nan}

    y_true = subset[true_col].astype(int).values
    y_pred = subset[pred_col].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[UNINFECTED_CLASS, INFECTED_CLASS])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision is not np.nan and recall is not np.nan
              and (precision + recall) > 0) else np.nan)

    pct_infected_true = (y_true == INFECTED_CLASS).mean() * 100
    pct_infected_pred = (y_pred == INFECTED_CLASS).mean() * 100

    return {
        "well_id": well_id,
        "n_cells": len(subset),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "pct_infected_true": pct_infected_true,
        "pct_infected_pred": pct_infected_pred,
    }


# %% load embeddings
print("Loading sensor embeddings ...")
sensor_adata = anndata.read_zarr(sensor_zarr_path)
sensor_df = convert_to_dataframe(sensor_adata)

print("Loading estain embeddings ...")
estain_adata = anndata.read_zarr(estain_zarr_path)
estain_df = convert_to_dataframe(estain_adata)

# %% load models
clf_sensor = joblib.load(sensor_model_path)
clf_estain = joblib.load(estain_model_path)
print("Models loaded.")

# %% identify feature columns
obs_cols_sensor = list(sensor_adata.obs.columns)
obs_cols_estain = list(estain_adata.obs.columns)
feature_cols_sensor = [c for c in sensor_df.columns if c not in obs_cols_sensor]
feature_cols_estain = [c for c in estain_df.columns if c not in obs_cols_estain]

# %% run inference
sensor_df["predicted_state"] = clf_sensor.predict(sensor_df[feature_cols_sensor].values)
estain_df["predicted_state"] = clf_estain.predict(estain_df[feature_cols_estain].values)

# Determine which column holds the ground-truth label (fall back gracefully)
sensor_true_col = "infection_state" if "infection_state" in sensor_df.columns else "predicted_state"
estain_true_col = "infection_state" if "infection_state" in estain_df.columns else "predicted_state"

# %% compute per-well metrics for sensor and estain
metrics_records = []

for virus_label, sensor_wells, estain_wells in [
    ("ZIKV", ZIKV_wells_sensor, ZIKV_wells_estain),
    ("DENV", DENV_wells_sensor, DENV_wells_estain),
]:
    for moi_label, s_well, e_well in zip(MOI_names, sensor_wells, estain_wells):
        s_metrics = compute_well_metrics(
            sensor_df, "fov_name", "predicted_state", sensor_true_col, s_well
        )
        e_metrics = compute_well_metrics(
            estain_df, "fov_name", "predicted_state", estain_true_col, e_well
        )

        metrics_records.append(
            {"virus": virus_label, "MOI": moi_label, "modality": "sensor", **s_metrics}
        )
        metrics_records.append(
            {"virus": virus_label, "MOI": moi_label, "modality": "estain", **e_metrics}
        )

metrics_df = pd.DataFrame(metrics_records)

# %% compute % infected over MOI
pct_records = []

for virus_label, sensor_wells, estain_wells in [
    ("ZIKV", ZIKV_wells_sensor, ZIKV_wells_estain),
    ("DENV", DENV_wells_sensor, DENV_wells_estain),
]:
    for moi_label, s_well, e_well in zip(MOI_names, sensor_wells, estain_wells):
        # Sensor
        s_sub = sensor_df[sensor_df["fov_name"] == s_well]
        s_pct = (s_sub["predicted_state"] == INFECTED_CLASS).mean() * 100 if len(s_sub) > 0 else np.nan

        # Estain
        e_sub = estain_df[estain_df["fov_name"] == e_well]
        e_pct = (e_sub["predicted_state"] == INFECTED_CLASS).mean() * 100 if len(e_sub) > 0 else np.nan

        pct_records.append(
            {
                "virus": virus_label,
                "MOI": moi_label,
                "sensor_pct_infected": s_pct,
                "estain_pct_infected": e_pct,
            }
        )

pct_df = pd.DataFrame(pct_records)

# %% save
metrics_df.to_csv(output_metrics_csv, index=False)
pct_df.to_csv(output_infection_pct_csv, index=False)

print(f"\nSaved metrics to {output_metrics_csv}")
print(metrics_df)
print(f"\nSaved % infected to {output_infection_pct_csv}")
print(pct_df)
