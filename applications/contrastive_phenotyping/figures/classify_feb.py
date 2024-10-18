# %% Importing Necessary Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation
from viscy.representation.evaluation.dimensionality_reduction import compute_pca

# %% Defining Paths for February Dataset
feb_features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/output/June_140Patch_2chan/phaseRFP_140patch_99ckpt_Feb.zarr"
)


# %% Load and Process February Dataset
feb_embedding_dataset = read_embedding_dataset(feb_features_path)
print(feb_embedding_dataset)
pca_df = compute_pca(feb_embedding_dataset, n_components=6)

# Print shape before merge
print("Shape of pca_df before merge:", pca_df.shape)

# Load the ground truth infection labels
feb_ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track"
)
feb_infection = load_annotation(
    feb_embedding_dataset,
    feb_ann_root / "tracking_v1_infection.csv",
    "infection class",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# Print shape of feb_infection
print("Shape of feb_infection:", feb_infection.shape)

# Merge PCA results with ground truth labels on both 'fov_name' and 'id'
pca_df = pd.merge(pca_df, feb_infection.reset_index(), on=["fov_name", "id"])

# Print shape after merge
print("Shape of pca_df after merge:", pca_df.shape)

# Prepare the full dataset
X = pca_df[["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6"]]
y = pca_df["infection class"]

# Apply SMOTE to balance the classes in the full dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print shape after SMOTE
print(
    f"Shape after SMOTE - X_resampled: {X_resampled.shape}, y_resampled: {y_resampled.shape}"
)

# %% Train Logistic Regression Classifier with Progress Bar
model = LogisticRegression(max_iter=1000, random_state=42)

# Wrap the training with tqdm to show a progress bar
for _ in tqdm(range(1)):
    model.fit(X_resampled, y_resampled)

# %% Predict Labels for the Entire Dataset
pca_df["Predicted_Label"] = model.predict(X)

# Compute metrics based on the entire original dataset
print("Classification Report for Entire Dataset:")
print(classification_report(pca_df["infection class"], pca_df["Predicted_Label"]))

print("Confusion Matrix for Entire Dataset:")
print(confusion_matrix(pca_df["infection class"], pca_df["Predicted_Label"]))

# %% Plotting the Results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=pca_df["PCA1"], y=pca_df["PCA2"], hue=pca_df["infection class"], s=7, alpha=0.8
)
plt.title("PCA with Ground Truth Labels")
plt.savefig("up_pca_ground_truth_labels.png", format="png", dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=pca_df["PCA1"], y=pca_df["PCA2"], hue=pca_df["Predicted_Label"], s=7, alpha=0.8
)
plt.title("PCA with Logistic Regression Predicted Labels")
plt.savefig("up_pca_predicted_labels.png", format="png", dpi=300)
plt.show()

# %% Save Predicted Labels to CSV
save_path_csv = "up_logistic_regression_predicted_labels_feb_pca.csv"
pca_df[["id", "fov_name", "Predicted_Label"]].to_csv(save_path_csv, index=False)
print(f"Predicted labels saved to {save_path_csv}")
