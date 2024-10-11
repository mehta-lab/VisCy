# %% import statements 
from pathlib import Path
import pandas as pd
from viscy.representation.embedding_writer import read_embedding_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from viscy.representation.evaluation import load_annotation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from viscy.representation.evaluation import GMMClustering
from viscy.representation.evaluation import compute_pca
from viscy.representation.evaluation import compute_umap
# %% Paths and parameters.

features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)


feature_path_no_track = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
)


features_path_any_time = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr"
)

features_path_june = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr")


# %% visualize distribution of embeddings
embedding_dataset = read_embedding_dataset(features_path_30_min)
features_data = embedding_dataset['features']
n_samples, n_features = features_data.shape

random_dimensions = np.random.choice(n_features, 5, replace=False)

plt.figure(figsize=(15, 10))
for i, dim in enumerate(random_dimensions, 1):
    plt.subplot(2, 3, i)
    sns.histplot(features_data[:, dim], kde=True)
    plt.title(f"Dimension {dim} Distribution")

plt.tight_layout()
plt.show()

# %% initialize GMM clustering and ground truth labels

embedding_dataset = read_embedding_dataset(features_path_june)
features_data = embedding_dataset['features']

cluster_evaluator = GMMClustering(features_data)

# %% Find best n_clusters 

aic_scores, bic_scores = cluster_evaluator.find_best_n_clusters()

plt.figure(figsize=(8, 6))
plt.plot(cluster_evaluator.n_clusters_range, aic_scores, label='AIC', marker='o')
plt.plot(cluster_evaluator.n_clusters_range, bic_scores, label='BIC', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('AIC / BIC Score')
plt.title('AIC and BIC Scores for Different Numbers of Clusters')
plt.legend()
plt.show()

# %%
# Choose the best model (with the lowest BIC score)
best_gmm = cluster_evaluator.fit_best_model(criterion='bic')
cluster_labels = cluster_evaluator.predict_clusters()

# %% ground truth labels (if available!)
ann_root = Path(
   "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)

infection = load_annotation(
   features_data,
   ann_root / "extracted_inf_state.csv",
   "infection_state",
   {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %%  the confusion matrix with ground truth states

ground_truth_labels_numeric = infection.cat.codes

cm = confusion_matrix(ground_truth_labels_numeric, cluster_labels)

cm_df = pd.DataFrame(cm, index=["Background", "Uninfected", "Infected"], 
                     columns=["Cluster 0", "Cluster 1", "Cluster 2"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')

plt.title('Confusion Matrix: Clusters vs Ground Truth')
plt.ylabel('Ground Truth Labels')
plt.xlabel('Cluster Labels')
plt.show()

# %%
# Reduce dimensions to 2 for vis
_, _, pca_df = compute_pca(embedding_dataset, n_components=2)

pca1 = pca_df["PCA1"]
pca2 = pca_df["PCA2"]

color_map = {'background': 'gray', 'uninfected': 'blue', 'infected': 'red'}
colors = infection.map(color_map)

plt.figure(figsize=(10, 8))

# Plot Cluster 0 with circle markers ('o')
plt.scatter(pca1[cluster_labels == 0], pca2[cluster_labels == 0],
            c=colors[cluster_labels == 0], edgecolor='black', s=50, alpha=0.7, label='Cluster 0 (circle)', marker='o')

# Plot Cluster 1 with X markers ('x')
plt.scatter(pca1[cluster_labels == 1], pca2[cluster_labels == 1],
            c=colors[cluster_labels == 1], edgecolor='black', s=50, alpha=0.7, label='Cluster 1 (X)', marker='x')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f"Ground Truth Colors with GMM Cluster Marker Types")

handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=color_map[label], markersize=10, markeredgecolor='black')
           for label in color_map.keys()]
plt.legend(handles=handles, title="Ground Truth")

plt.show()

# %% Visualize GMM Clusters in PCA space (without ground truth)
_, _, pca_df = compute_pca(embedding_dataset, n_components=2)

pca1 = pca_df["PCA1"]
pca2 = pca_df["PCA2"]

plt.figure(figsize=(10, 8))

# Plot Cluster 0 with circle markers ('o')
plt.scatter(pca1[cluster_labels == 0], pca2[cluster_labels == 0],
            c='green', edgecolor='black', s=50, alpha=0.7, label='Cluster 0 (GMM)', marker='o')

# Plot Cluster 1 with X markers ('x')
plt.scatter(pca1[cluster_labels == 1], pca2[cluster_labels == 1],
            c='orange', edgecolor='black', s=50, alpha=0.7, label='Cluster 1 (GMM)', marker='x')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f"GMM Clusters")

plt.legend()
plt.show()


# %% Visualize UMAP embeddings colored by GMM cluster weights 
umap_features, umap_projection, umap_df = compute_umap(embedding_dataset)

gmm_weights = best_gmm.weights_

plt.figure(figsize=(10, 8))
plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=gmm_weights[cluster_labels], cmap='viridis', s=50, alpha=0.8, edgecolor='k')
plt.colorbar(label='GMM Cluster Weights')
plt.title('UMAP Embeddings Colored by GMM Cluster Weights')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()


# %% Visualize UMAP embeddings colored by cluster labels
umap_features, umap_projection, umap_df = compute_umap(embedding_dataset)

plt.figure(figsize=(10, 8))

plt.scatter(umap_df["UMAP1"][cluster_labels == 0], umap_df["UMAP2"][cluster_labels == 0],
            c='green', edgecolor='black', s=50, alpha=0.7, label='Cluster 0 (GMM)', marker='o')

plt.scatter(umap_df["UMAP1"][cluster_labels == 1], umap_df["UMAP2"][cluster_labels == 1],
            c='orange', edgecolor='black', s=50, alpha=0.7, label='Cluster 1 (GMM)', marker='o')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title(f"GMM Clusters in UMAP Space")

plt.legend()
plt.show()

# %% UMAP vis (w/ ground truth colors and GMM cluster markers)
umap_features, umap_projection, umap_df = compute_umap(embedding_dataset, normalize_features=True)

umap1 = umap_df["UMAP1"]
umap2 = umap_df["UMAP2"]

color_map = {'background': 'gray', 'uninfected': 'blue', 'infected': 'red'}
colors = infection.map(color_map)

plt.figure(figsize=(10, 8))

# Plot Cluster 0 with circle markers ('o')
plt.scatter(umap1[cluster_labels == 0], umap2[cluster_labels == 0],
            c=colors[cluster_labels == 0], edgecolor='black', s=50, alpha=0.7, label='Cluster 0 (circle)', marker='o')

# Plot Cluster 1 with X markers ('x')
plt.scatter(umap1[cluster_labels == 1], umap2[cluster_labels == 1],
            c=colors[cluster_labels == 1], edgecolor='black', s=50, alpha=0.7, label='Cluster 1 (X)', marker='x')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title(f"Ground Truth Colors with GMM Cluster Marker Types in UMAP Space")

handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=color_map[label], markersize=10, markeredgecolor='black')
           for label in color_map.keys()]
plt.legend(handles=handles, title="Ground Truth")

plt.show()

# %%
