# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import anndata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import sem  # noqa: F401 – available for downstream use

from utils.embedding_utils import match_embeddings_to_annotations, convert_to_dataframe

# %% paths
train_embeddings_path = Path("phase_embeddings_train.zarr")
test_embeddings_path = Path("phase_embeddings_test.zarr")
train_annotations_path = Path("train_infection_annotations.csv")
test_annotations_path = Path("test_infection_annotations.csv")

output_model_path = Path("lc_phase_infection.joblib")
output_predictions_csv = Path("phase_infection_predictions.csv")
output_metrics_csv = Path("phase_infection_metrics_over_time.csv")

# Wells used for training (remaining wells are held out for testing)
train_wells = ["A/2"]

# Time axis configuration
time_interval_hours = 0.5   # hours between consecutive t-indices
start_hpi = 3.0             # hours post-infection at t=0

# Class labels (must match annotation CSV values)
UNINFECTED_CLASS = 1
INFECTED_CLASS = 2


# %% helper – TP/TN/FP/FN classification result label
def classification_result(y_true: int, y_pred: int) -> str:
    """Return the classification result string for a single prediction.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted class labels.

    Returns
    -------
    str
        One of "TP", "TN", "FP", "FN".
    """
    if y_true == INFECTED_CLASS and y_pred == INFECTED_CLASS:
        return "TP"
    elif y_true == UNINFECTED_CLASS and y_pred == UNINFECTED_CLASS:
        return "TN"
    elif y_true == UNINFECTED_CLASS and y_pred == INFECTED_CLASS:
        return "FP"
    else:
        return "FN"


# %% section 1 – load training embeddings and annotations
print(f"Loading training embeddings from {train_embeddings_path} ...")
train_adata = anndata.read_zarr(train_embeddings_path)
train_df = convert_to_dataframe(train_adata)

obs_cols = list(train_adata.obs.columns)
feature_cols = [c for c in train_df.columns if c not in obs_cols]

# Match embeddings to annotations
# match_embeddings_to_annotations returns a DataFrame with an added
# 'infection_state' column aligned to the embedding obs.
train_df = match_embeddings_to_annotations(
    train_df,
    train_annotations_path,
    obs_id_col="fov_name",
    ann_id_col="fov_name",
    ann_state_col="infection_state",
)

# Filter to specified training wells
train_subset = train_df[train_df["fov_name"].isin(train_wells)].dropna(
    subset=["infection_state"]
)
print(f"Training cells: {len(train_subset)}")
print(train_subset["infection_state"].value_counts())

# %% section 2 – train classifier
print("\nTraining logistic regression classifier ...")
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
clf.fit(
    train_subset[feature_cols].values,
    train_subset["infection_state"].astype(int).values,
)

joblib.dump(clf, output_model_path)
print(f"Model saved to {output_model_path}")

# %% section 3 – load test embeddings and run inference
print(f"\nLoading test embeddings from {test_embeddings_path} ...")
test_adata = anndata.read_zarr(test_embeddings_path)
test_df = convert_to_dataframe(test_adata)

# Predict class and probability
test_obs_cols = list(test_adata.obs.columns)
test_feature_cols = [c for c in test_df.columns if c not in test_obs_cols]

test_preds = clf.predict(test_df[test_feature_cols].values)
test_proba = clf.predict_proba(test_df[test_feature_cols].values)

# Find the column index for INFECTED_CLASS in clf.classes_
infected_class_idx = list(clf.classes_).index(INFECTED_CLASS)

test_df["predicted_state"] = test_preds
test_df["prob_infected"] = test_proba[:, infected_class_idx]

# Add HPI column from t index
if "t" in test_df.columns:
    test_df["hpi"] = start_hpi + test_df["t"] * time_interval_hours

# %% section 4 – match test predictions to annotations
test_df = match_embeddings_to_annotations(
    test_df,
    test_annotations_path,
    obs_id_col="fov_name",
    ann_id_col="fov_name",
    ann_state_col="infection_state",
)

# Add classification result label where ground truth is available
has_gt = test_df["infection_state"].notna()
test_df.loc[has_gt, "classification_result"] = test_df.loc[has_gt].apply(
    lambda r: classification_result(
        int(r["infection_state"]), int(r["predicted_state"])
    ),
    axis=1,
)

# %% section 5 – overall metrics
labeled = test_df.dropna(subset=["infection_state"])

y_true_all = labeled["infection_state"].astype(int).values
y_pred_all = labeled["predicted_state"].astype(int).values
y_prob_all = labeled["prob_infected"].values

overall_accuracy = accuracy_score(y_true_all, y_pred_all)
overall_f1 = f1_score(y_true_all, y_pred_all, pos_label=INFECTED_CLASS, zero_division=0)
try:
    overall_roc_auc = roc_auc_score(y_true_all, y_prob_all)
except ValueError:
    overall_roc_auc = np.nan

print("\nOverall metrics on test set:")
print(f"  Accuracy : {overall_accuracy:.4f}")
print(f"  F1       : {overall_f1:.4f}")
print(f"  ROC-AUC  : {overall_roc_auc:.4f}")

# %% section 6 – per-timepoint metrics
timepoint_records = []

if "t" in labeled.columns:
    for t_val, t_group in labeled.groupby("t"):
        hpi = start_hpi + t_val * time_interval_hours
        y_t = t_group["infection_state"].astype(int).values
        p_t = t_group["predicted_state"].astype(int).values
        pr_t = t_group["prob_infected"].values

        acc_t = accuracy_score(y_t, p_t)
        f1_t = f1_score(y_t, p_t, pos_label=INFECTED_CLASS, zero_division=0)
        try:
            auc_t = roc_auc_score(y_t, pr_t)
        except ValueError:
            auc_t = np.nan

        pct_infected_true = (y_t == INFECTED_CLASS).mean() * 100
        pct_infected_pred = (p_t == INFECTED_CLASS).mean() * 100

        timepoint_records.append(
            {
                "t": t_val,
                "hpi": hpi,
                "n_cells": len(t_group),
                "accuracy": acc_t,
                "F1": f1_t,
                "ROC_AUC": auc_t,
                "pct_infected_true": pct_infected_true,
                "pct_infected_pred": pct_infected_pred,
            }
        )

metrics_df = pd.DataFrame(timepoint_records)

# %% save outputs
test_df.to_csv(output_predictions_csv, index=False)
metrics_df.to_csv(output_metrics_csv, index=False)

print(f"\nSaved predictions ({len(test_df)} rows) to {output_predictions_csv}")
print(f"Saved per-timepoint metrics ({len(metrics_df)} rows) to {output_metrics_csv}")
print(metrics_df.head(10))
