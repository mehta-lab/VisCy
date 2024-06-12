# %%
"""
Script to visualize the encoder feature maps of a trained model.
Using PCA to visualize feature maps is inspired by
https://doi.org/10.48550/arXiv.2304.07193 (Oquab et al., 2023).
"""

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from iohub import open_ome_zarr
from matplotlib.patches import Rectangle
from skimage.exposure import rescale_intensity
from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from viscy.light.engine import VSUNet

# %%
# prepare sample images
dataset = open_ome_zarr("data.hcs.ome.zarr")
center_index = 48
depth = 5
crop = 1024
# normalize phase
norm_meta = dataset.zattrs["normalization"]["Phase3D"]["dataset_statistics"]
img = dataset["row/col/0/0"][
    :, :, center_index - depth // 2 : center_index + depth // 2 + 1, :crop, :crop
]
phase = img[:, 0:1]
phase = (phase - norm_meta["median"]) / norm_meta["iqr"]
plt.imshow(phase[0, 0, 2], cmap="gray")

# %%
# load model
model = VSUNet.load_from_checkpoint(
    "model.ckpt",
    architecture="UNeXt2",
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 5,
        "backbone": "convnextv2_tiny",
        "stem_kernel_size": (5, 4, 4),
        "decoder_mode": "pixelshuffle",
        "head_expansion_ratio": 4,
        "head_pool": True,
    },
)

# %%
# extract features
with torch.inference_mode():
    features = model.model.stem(torch.from_numpy(phase).to(model.device))
    features = [f.detach().cpu().numpy() for f in model.model.encoder_stages(features)]

for f in features:
    print(f.shape)


# %%
def feature_map_pca(feature_map: np.array, n_components: int = 8) -> PCA:
    """
    Compute PCA on a feature map.
    :param np.array feature_map: (C, H, W) feature map
    :param int n_components: number of components to keep
    :return: PCA: fit sklearn PCA object
    """
    # (C, H, W) -> (C, H*W)
    feat = feature_map.reshape(feature_map.shape[0], -1)
    pca = PCA(n_components=n_components)
    pca.fit(feat)
    return pca


# %%
f, ax = plt.subplots(2, 5, figsize=(20, 8))

n_components = 8

ax[0, 0].imshow(img[0, 0, 2], cmap="gray")
ax[0, 0].set_title(f"Phase {phase.shape[1:]}")
fluo = img[0, 1:3, 2]
fluo = [
    rescale_intensity(np.clip(ch, np.percentile(ch, 1), np.percentile(ch, 99)))
    for ch in fluo
]
fluo = np.stack([*fluo, np.zeros_like(fluo[0])], axis=-1)
ax[1, 0].imshow(fluo)
ax[1, 0].set_title("Fluorescence")

for level, feat in enumerate(features):
    pca = feature_map_pca(feat[0], n_components=n_components)
    pc_first_3 = pca.components_[:3].reshape(3, *features[level].shape[-2:])
    rgb = np.stack([rescale_intensity(pc) for pc in pc_first_3], axis=-1)
    ax[0, level + 1].imshow(rgb)
    ax[0, level + 1].set_title(f"Level {level+1} {features[level].shape[1:]}")
    ax[1, level + 1].plot(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax[1, level + 1].set_xlabel("Principal component")
    ax[1, level + 1].set_ylabel("Explained variance ratio")


legend_table = {"PC1": (1, 0, 0), "PC2": (0, 1, 0), "PC3": (0, 0, 1)}
handles = [Rectangle((0, 0), 1, 1, color=v) for v in legend_table.values()]
f.legend(
    handles,
    legend_table.keys(),
    title="feature map",
    bbox_to_anchor=(1.01, 0.8),
    loc="upper left",
)

plt.tight_layout()

# %%
level3_feat = features[3][0]
level3_pca = feature_map_pca(level3_feat, n_components=n_components)
level3_pcs = level3_pca.components_[:8].reshape(8, *level3_feat.shape[-2:])
nuc_pcs = (
    2 * rescale_intensity(level3_pcs[0])
    + rescale_intensity(level3_pcs[1])
    + rescale_intensity(level3_pcs[3])
)
# mem_pcs = rescale_intensity(level3_pc_first_3[0]) + rescale_intensity(
#     level3_pc_first_3[2]
# )

g, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(level3_pcs[i], cmap="magma")
    ax.set_title(f"PC{i+1}")
    # axes[3].imshow(nuc_pcs, cmap="magma")
    # axes[3].set_title("normalized PC1 + PC2")
    # axes[4].imshow(mem_pcs, cmap="magma")
    # axes[4].set_title("normalized PC1 + PC3")
    # for i, ax in enumerate(axes):
    ax.axis("off")
plt.tight_layout()

# %%
down_fluo = downscale_local_mean(fluo, (32, 32, 1))

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    level3_pcs[0].ravel(),
    level3_pcs[1].ravel(),
    level3_pcs[2].ravel(),
    s=1,
    c=down_fluo.reshape(-1, 3),
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
g_cmap = matplotlib.colors.ListedColormap(
    np.linspace((0, 0, 0), (0, 1, 0), 256), name="green"
)
r_cmap = matplotlib.colors.ListedColormap(
    np.linspace((0, 0, 0), (1, 0, 0), 256), name="red"
)
plt.subplots_adjust(wspace=1000)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=r_cmap),
    label="Membrane fluorescence",
    shrink=0.5,
    ax=ax,
    # location="left",
)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=g_cmap),
    label="Nuclei fluorescence",
    shrink=0.5,
    ax=ax,
    # location="left",
)


# plt.tight_layout()
# %%
def feature_map_tsne(feature_map: np.array, n_components: int = 8) -> TSNE:
    """
    Compute t-SNE manifold on a feature map.
    :param np.array feature_map: (C, H, W) feature map
    :param int n_components: number of components in the manifold
    :return: TSNE: fit sklearn TSNE object
    """
    # (C, H, W) -> (H*W, C)
    feat = feature_map.reshape(feature_map.shape[0], -1).T
    tsne = TSNE(n_components=n_components)
    tsne.fit(feat)
    return tsne


# %%
level_3_tsne = feature_map_tsne(level3_feat, n_components=2)
# %%
plt.scatter(
    level_3_tsne.embedding_[:, 0],
    level_3_tsne.embedding_[:, 1],
    s=1,
    c=down_fluo.reshape(-1, 3),
)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=r_cmap),
    label="Membrane fluorescence",
    shrink=0.5,
)
plt.colorbar(
    matplotlib.cm.ScalarMappable(cmap=g_cmap),
    label="Nuclei fluorescence",
    shrink=0.5,
)
# %%
