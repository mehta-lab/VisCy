# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import joblib
import anndata
from sklearn.linear_model import LogisticRegression

from utils.embedding_utils import convert_to_dataframe

# %% paths
sensor_zarr_path = Path("sensor_embeddings.zarr")
estain_zarr_path = Path("estain_embeddings.zarr")

sensor_annotations_path = Path("extracted_sensor_annotations.csv")
estain_annotations_path = Path("extracted_estain_annotations.csv")

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

sensor_model_path = model_dir / "sensor_model.joblib"
estain_model_path = model_dir / "estain_model.joblib"

# FOVs used as the training set (remaining FOVs are used for evaluation only)
# Edit this list to adjust the train/test split.
train_fovs = [
    "0/4",
    "0/5",
    "0/6",
    "1/4",
    "1/5",
]

# Annotation encoding
# infection_state: 1 = uninfected, 2 = infected
UNINFECTED_LABEL = 1
INFECTED_LABEL = 2


# %% helper – load and align annotation CSV to an AnnData obs index
def load_annotation(
    adata: anndata.AnnData,
    annotation_path: Path,
    state_col: str = "infection_state",
    categories: dict = None,
) -> pd.Series:
    """Align per-cell infection-state annotations to *adata*.

    The annotation CSV must contain columns ``fov_name`` and ``id``.
    Rows are matched to *adata* obs by a (fov_name, id) multi-index.

    Parameters
    ----------
    adata:
        AnnData object whose obs DataFrame has ``fov_name`` and ``id`` columns.
    annotation_path:
        Path to the annotation CSV.
    state_col:
        Name of the infection-state column in the CSV.
    categories:
        Optional dict to rename category values, e.g. {1: "uninfected", 2: "infected"}.

    Returns
    -------
    pd.Series
        Infection-state labels aligned to the adata obs index.
        Cells with no matching annotation receive NaN.
    """
    ann_df = pd.read_csv(annotation_path)
    ann_df = ann_df.set_index(["fov_name", "id"])

    obs = adata.obs.copy()
    obs_index = pd.MultiIndex.from_arrays([obs["fov_name"], obs["id"]])

    labels = ann_df[state_col].reindex(obs_index)
    labels.index = adata.obs.index

    if categories is not None:
        labels = labels.map(categories)

    return labels


# %% load embeddings
print("Loading sensor embeddings ...")
sensor_adata = anndata.read_zarr(sensor_zarr_path)
print(f"  sensor: {sensor_adata.shape}")

print("Loading estain embeddings ...")
estain_adata = anndata.read_zarr(estain_zarr_path)
print(f"  estain: {estain_adata.shape}")

# %% convert to DataFrames (features + obs metadata)
sensor_df = convert_to_dataframe(sensor_adata)
estain_df = convert_to_dataframe(estain_adata)

# %% load and align annotations
sensor_labels = load_annotation(sensor_adata, sensor_annotations_path)
estain_labels = load_annotation(estain_adata, estain_annotations_path)

sensor_df["infection_state"] = sensor_labels.values
estain_df["infection_state"] = estain_labels.values

# Drop cells with missing annotations
sensor_df = sensor_df.dropna(subset=["infection_state"])
estain_df = estain_df.dropna(subset=["infection_state"])

# %% identify embedding feature columns (all columns that are not obs metadata)
obs_cols = list(sensor_adata.obs.columns) + ["infection_state"]
feature_cols = [c for c in sensor_df.columns if c not in obs_cols]

# %% split into train / test by FOV
sensor_train_mask = sensor_df["fov_name"].isin(train_fovs)
estain_train_mask = estain_df["fov_name"].isin(train_fovs)

sensor_train = sensor_df[sensor_train_mask]
estain_train = estain_df[estain_train_mask]

print(f"Sensor  train: {len(sensor_train)} cells  |  test: {len(sensor_df) - len(sensor_train)} cells")
print(f"Estain  train: {len(estain_train)} cells  |  test: {len(estain_df) - len(estain_train)} cells")

# %% train sensor classifier
print("\nTraining sensor logistic regression classifier ...")
clf_sensor = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf_sensor.fit(
    sensor_train[feature_cols].values,
    sensor_train["infection_state"].astype(int).values,
)
joblib.dump(clf_sensor, sensor_model_path)
print(f"  Saved sensor model → {sensor_model_path}")

# %% train estain classifier
print("Training estain logistic regression classifier ...")
clf_estain = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf_estain.fit(
    estain_train[feature_cols].values,
    estain_train["infection_state"].astype(int).values,
)
joblib.dump(clf_estain, estain_model_path)
print(f"  Saved estain model → {estain_model_path}")

# %% quick sanity check – predict on the full dataset
sensor_preds = clf_sensor.predict(sensor_df[feature_cols].values)
estain_preds = clf_estain.predict(estain_df[feature_cols].values)

sensor_df["predicted_state"] = sensor_preds
estain_df["predicted_state"] = estain_preds

print("\nSensor predictions (value counts):")
print(sensor_df["predicted_state"].value_counts())
print("\nEstain predictions (value counts):")
print(estain_df["predicted_state"].value_counts())
