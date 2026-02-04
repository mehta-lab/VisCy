# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from viscy.representation.embedding_writer import read_embedding_dataset

test_data_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_rpe_fucci_leger_weigert/0-phenotyping/bf_only_timeaware_ntxent_lr2e-5_temp_7e-2_tau1_w_augmentations_2_ckpt306.zarr"
)
cell_cycle_labels_path = "/hpc/projects/organelle_phenotyping/models/rpe_fucci/dynaclr/pseudolabels/cell_cycle_labels_w_mitosis.csv"

# %%
# Load the data
cell_cycle_labels_df = pd.read_csv(cell_cycle_labels_path, dtype={"dataset_name": str})
test_embeddings = read_embedding_dataset(test_data_features_path)

# Extract features (768-dimensional embeddings)
features = test_embeddings.features.values

# %%
sample_coords = test_embeddings.coords["sample"].values
fov_names = [coord[0] for coord in sample_coords]
ids = [coord[1] for coord in sample_coords]

# Create DataFrame with embeddings and identifiers
embedding_df = pd.DataFrame(
    {
        "dataset_name": fov_names,
        "timepoint": ids,
    }
)

# Merge with cell cycle labels
merged_data = embedding_df.merge(
    cell_cycle_labels_df, on=["dataset_name", "timepoint"], how="inner"
)

print(f"Original embeddings: {len(embedding_df)}")
print(f"Cell cycle labels: {len(cell_cycle_labels_df)}")
print(f"Merged data: {len(merged_data)}")
print(f"Cell cycle distribution:\n{merged_data['cell_cycle_state'].value_counts()}")

# Get corresponding features for merged samples
merged_indices = merged_data.index.values
X = features[merged_indices]
y = merged_data["cell_cycle_state"].values

# %%
# First split: 80% train+val, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# %%
# Train logistic regression model
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nTest set classification report:")
print(classification_report(y_test, y_test_pred))

# %%
# Enhanced evaluation and visualization
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. Confusion Matrix - shows which classes are confused with each other
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=["G1", "G2", "S", "M"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 2. Per-class errors breakdown
print("\nDetailed per-class analysis:")
for class_name in ["G1", "G2", "S", "M"]:
    mask = y_test == class_name
    correct = (y_test_pred[mask] == class_name).sum()
    total = mask.sum()
    print(f"{class_name}: {correct}/{total} correct ({correct / total:.3f})")

    # Show what this class was misclassified as
    if total > correct:
        wrong_preds = y_test_pred[mask & (y_test_pred != class_name)]
        unique, counts = np.unique(wrong_preds, return_counts=True)
        print(f"  Misclassified as: {dict(zip(unique, counts))}")

# 3. Prediction confidence (probabilities)
y_test_proba = clf.predict_proba(X_test)
class_names = clf.classes_

plt.figure(figsize=(12, 4))
for i, class_name in enumerate(class_names):
    plt.subplot(1, 4, i + 1)
    plt.hist(
        y_test_proba[:, i],
        bins=20,
        alpha=0.7,
        color=["blue", "orange", "green", "red"][i],
    )
    plt.title(f"Confidence for {class_name}")
    plt.xlabel("Probability")
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 4. Most confident correct and incorrect predictions
print("\nMost confident predictions:")
max_proba = np.max(y_test_proba, axis=1)
pred_correct = y_test == y_test_pred

# Most confident correct predictions
correct_idx = np.where(pred_correct)[0]
most_confident_correct = correct_idx[np.argsort(max_proba[correct_idx])[-5:]]
print("Top 5 most confident CORRECT predictions:")
for idx in most_confident_correct:
    print(
        f"  True: {y_test[idx]}, Pred: {y_test_pred[idx]}, Confidence: {max_proba[idx]:.3f}"
    )

# Most confident incorrect predictions
incorrect_idx = np.where(~pred_correct)[0]
if len(incorrect_idx) > 0:
    most_confident_wrong = incorrect_idx[np.argsort(max_proba[incorrect_idx])[-5:]]
    print("\nTop 5 most confident WRONG predictions:")
    for idx in most_confident_wrong:
        print(
            f"  True: {y_test[idx]}, Pred: {y_test_pred[idx]}, Confidence: {max_proba[idx]:.3f}"
        )

# %%
