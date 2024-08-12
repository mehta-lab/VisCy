# %%
from seaborn import scatterplot
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from viscy.light.embedding_writer import read_embedding_dataset

# %%
dataset = read_embedding_dataset(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/contrastive_tune_augmentations/predict/2024_02_04-tokenized-drop_path_0_1.zarrr"
)
dataset

# %%
features = dataset["features"].sel(fov_name="/A/4/1")
features

# %%
scaled_features = StandardScaler().fit_transform(features.values)

# %%
umap = UMAP()

# %%
embedding = umap.fit_transform(scaled_features)
embedding.shape

# %%
scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=features["t"], s=10)
# %%
