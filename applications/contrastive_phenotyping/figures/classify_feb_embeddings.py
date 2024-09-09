# %% Importing Necessary Libraries
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation

# %% Defining Paths for February Dataset
feb_features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/"
)


# %% Load and Process February Dataset (Embedding Features)
feb_embedding_dataset = read_embedding_dataset(
    feb_features_path / "febtest_predict.zarr"
)
print(feb_embedding_dataset)

# Extract the embedding feature values as the input matrix (X)
X = feb_embedding_dataset["features"].values

# Prepare a DataFrame for the embeddings with id and fov_name
embedding_df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
embedding_df["id"] = feb_embedding_dataset["id"].values
embedding_df["fov_name"] = feb_embedding_dataset["fov_name"].values
print(embedding_df.head())

# %% Load the ground truth infection labels
feb_ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)
feb_infection = load_annotation(
    feb_embedding_dataset,
    feb_ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %% Merge embedding features with infection labels on 'fov_name' and 'id'
merged_df = pd.merge(embedding_df, feb_infection.reset_index(), on=["fov_name", "id"])
print(merged_df.head())
# %% Prepare the full dataset for training
X = merged_df.drop(
    columns=["id", "fov_name", "infection_state"]
).values  # Use embeddings as features
y = merged_df["infection_state"]  # Use infection state as labels
print(X.shape)
print(y.shape)
#  %% Print class distribution before applying SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts())

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution after applying SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Train Logistic Regression Classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_resampled, y_resampled)

# Predict Labels for the Entire Dataset
y_pred = model.predict(X)

# Compute metrics based on the entire original dataset
print("Classification Report for Entire Dataset:")
print(classification_report(y, y_pred))

print("Confusion Matrix for Entire Dataset:")
print(confusion_matrix(y, y_pred))

# %%
# Save the predicted labels to a CSV
save_path_csv = feb_features_path / "feb_test_regression_predicted_labels_embedding.csv"
predicted_labels_df = pd.DataFrame(
    {
        "id": merged_df["id"].values,
        "fov_name": merged_df["fov_name"].values,
        "Predicted_Label": y_pred,
    }
)

predicted_labels_df.to_csv(save_path_csv, index=False)
print(f"Predicted labels saved to {save_path_csv}")

# %%
