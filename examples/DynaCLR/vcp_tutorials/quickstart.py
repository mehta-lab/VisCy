# %% [markdown]
"""
# Quick Start: DynaCLR (Cell Dynamics Contrastive Learning of Representations)

**Estimated time to complete:** 15 minutes
"""

# %% [markdown]
"""
# Learning Goals

* Download the DynaCLR model and run it on an example dataset
* Visualize the learned embeddings
"""

# %% [markdown]
"""
# Prerequisites
Python>=3.11

"""

# %% [markdown]
"""
# Introduction

## Model
The DynaCLR model architecture consists of three main components designed to map 3D multi-channel patches of single cells to a temporally regularized embedding space.

## Example Dataset

The A549 example dataset used in this quick-start guide contains
quantitative phase and paired fluorescence images of viral sensor reporter.
It is stored in OME-Zarr format and can be downloaded from
[here](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/registered_test.zarr/).

It has pre-computed statistics for normalization, generated using the `viscy preprocess` CLI.

Refer to our [preprint](https://arxiv.org/abs/2410.11281) for more details
about how the dataset and model were generated.

## User Data

The DynaCLR-DENV-VS+Ph model only requires label-free (quantitative phase) and fluorescence images for inference.

To run inference on your own data (Experimental): 
- Convert the label-free images into the OME-Zarr data format using iohub or other
[tools](https://ngff.openmicroscopy.org/tools/index.html#file-conversion),
- Run [pre-processing](https://github.com/mehta-lab/VisCy/blob/main/docs/usage.md#preprocessing)
with the `viscy preprocess` CLI
- Generate pseudo-tracks or tracking data from [Ultrack](https://github.com/royerlab/ultrack)
"""

# %% [markdown]
"""
# Setup

The commands below will install the required packages and download the example dataset and model checkpoint.
It may take a few minutes to download all the files.

## Setup Google Colab

To run this quick-start guide using Google Colab,
choose the 'T4' GPU runtime from the "Connect" dropdown menu
in the upper-right corner of this notebook for faster execution.
Using a GPU significantly speeds up running model inference, but CPU compute can also be used.

## Setup Local Environment

The commands below assume a Unix-like shell with `wget` installed.
On Windows, the files can be downloaded manually from the URLs.
"""

# %%
# Install VisCy with the optional dependencies for this example
# See the [repository](https://github.com/mehta-lab/VisCy) for more details
# !pip install "viscy[metrics,visual]==0.4.0a3"

# %%
# restart kernel if running in Google Colab
if "get_ipython" in globals():
    session = get_ipython()  # noqa: F821
    if "google.colab" in str(session):
        print("Shutting down colab session.")
        session.kernel.do_shutdown(restart=True)

# %%
# Validate installation
# !viscy --help

# %%
# Download the example tracks data
# !wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/track_test.zarr/"
# Download the example registered timelapse data
# !wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/registered_test.zarr/"
# Download the model checkpoint
# !wget -m -np -nH --cut-dirs=5 "index.html*" "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_models/DynaCLR-DENV/VS_n_Ph/epoch=94-step=2375.ckpt"

# %% [markdown]
"""
# Run Model Inference

The following code will run inference on a single field of view (FOV) of the example dataset.
This can also be achieved by using the VisCy CLI.
"""

# %%
# %%
from pathlib import Path  # noqa: E402

from iohub import open_ome_zarr  # noqa: E402
from torchview import draw_graph  # noqa: E402

from viscy.data.triplet import TripletDataModule  # noqa: E402
from viscy.trainer import VisCyTrainer  # noqa: E402
from viscy.transforms import NormalizeSampled  # noqa: E402
from viscy.representation.embedding_writer import EmbeddingWriter  # noqa: E402
from viscy.representation.engine import ContrastiveModule  # noqa: E402
from anndata import read_zarr # noqa: E402
import matplotlib.pyplot as plt # noqa: E402
import seaborn as sns # noqa: E402
import pandas as pd # noqa: E402

# %%
# NOTE: Nothing needs to be changed in this code block for the example to work.
# If using your own data, please modify the paths below.

# TODO: Set download paths, by default the working directory is used
root_dir = Path()
# TODO: modify the path to the input dataset
input_data_path = root_dir / "registered_test.zarr"
# TODO: modify the path to the track dataset
tracks_path= root_dir/ "track_test.zarr"
# TODO: modify the path to the model checkpoint
model_ckpt_path = root_dir / "epoch=94-step=2375.ckpt"
#TODO" modify the path to load the extracted infected cell annotation
annotations_path = root_dir / "extracted_inf_state.csv"

# TODO: modify the path to save the predictions
output_path = root_dir / "dynaclr_prediction.zarr"

#%%
# Default parameters for the test dataset
z_range = (15, 45)
yx_patch_size = (160, 160)
channels_to_display = ["Phase3D", "RFP"]

# %%
# Configure the data module for loading example images in prediction mode.
# See API documentation for how to use it with a different dataset.
# For example, View the documentation for the HCSDataModule class by running:
# ?HCSDataModule

# %%
# Setup the data module to use the example dataset
datamodule = TripletDataModule(
    data_path=input_data_path,
    tracks_path=tracks_path,
    source_channel=channels_to_display,
    z_range=z_range,
    initial_yx_patch_size=yx_patch_size,
    final_yx_patch_size=yx_patch_size,
    predict_cells=True,
    batch_size=8,
)
datamodule.setup("predict")

# %%
# Load the DynaCLR checkpoint from the downloaded checkpoint
# See this module for options to configure the model:

# ?contrastive.ContrastiveEncoder

# %%
dynaclr_model = ContrastiveEncoder.load_from_checkpoint(
    model_ckpt_path, # checkpoint path
    model_config={
        backbone: 'convnext_tiny',
        in_channels: len(channels_to_display), 
        in_stack_depth: z_range,
        stem_kernel_size: (5,4,4),
        stem_stride:(5,4,4),
        embedding_dim: 768,
        projection_dim: 32,
        drop_path_rate: 0.0,
    },
)

# %%
# Visualize the model graph
model_graph = draw_graph(
    dynaclr_model,
    torch.ones((1,2,30,256,256),
    graph_name="DynaCLR",
    roll=True,
    depth=3,
    expand_nested=True,
)

model_graph.visual_graph

# %%
# Setup the trainer for prediction
# The trainer can be further configured to better utilize the available hardware,
# For example using GPUs and half precision.
# Callbacks can also be used to customize logging and prediction writing.
# See the API documentation for more details:
# ?VisCyTrainer

# %%
# Initialize the trainer
# The prediction writer callback will save the predictions to an OME-Zarr store
trainer = VisCyTrainer(callbacks=[EmbeddingWriter(output_path, pca_kwargs={"n_components":8}, phate_kwargs={"knn":5, "decay":40,"n_jobs":-1})])

# Run prediction
trainer.predict(model=dynaclr_model, datamodule=datamodule, return_predictions=False)

# %% [markdown]
"""
# Model Outputs

The model outputs are also stored in an ANNData. The embeddings can then be visualized with a dimensionality reduction method (i.e UMAP, PHATE, PCA)
"""

# NOTE: We have chosen these tracks to be representative of the data. Feel free to open the dataset and select other tracks
features_anndata = read_zarr(output_path)
annotation = pd.read_csv(annotations_path)
ANNOTATION_COLUMN = 'infection-state'

# Combine embeddings and annotations
annotation["fov_name"] = annotation["fov_name"].str.strip("/")
annotation["fov_name"] = annotation["fov_name"].str.strip("/")

annotation = annotation.set_index(["fov_name", "id"])

mi = pd.MultiIndex.from_arrays(
    [features_anndata.obs["fov_name"], features_anndata.obs["id"]], names=["fov_name", "id"]
)
features_anndata.obs['annotations_infections_state'] = annotation.reindex(mi)[ANNOTATION_COLUMN] 

# Plot the PCA and PHATE embeddings colored by infection state
# Prepare data for plotting
plot_df = pd.DataFrame({
    'PC1': features_anndata.obsm['X_pca'][:, 0],
    'PC2': features_anndata.obsm['X_pca'][:, 1],
    'PHATE1': features_anndata.obsm['X_phate'][:, 0],
    'PHATE2': features_anndata.obsm['X_phate'][:, 1],
    'infection_state': features_anndata.obs['annotations_infections_state'].fillna('unknown')
})

# Define color palette
color_palette = {
    'infected': 'orange',
    'uninfected': 'blue',
    'unknown': 'gray'
}

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot PCA
sns.scatterplot(
    data=plot_df,
    x='PC1',
    y='PC2',
    hue='infection_state',
    palette=color_palette,
    ax=axes[0],
    alpha=0.6,
    s=20
)
axes[0].set_title('PCA Embedding')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Plot PHATE
sns.scatterplot(
    data=plot_df,
    x='PHATE1',
    y='PHATE2',
    hue='infection_state',
    palette=color_palette,
    ax=axes[1],
    alpha=0.6,
    s=20
)
axes[1].set_title('PHATE Embedding')
axes[1].set_xlabel('PHATE 1')
axes[1].set_ylabel('PHATE 2')

plt.tight_layout()
plt.show()


# %%
# NOTE: We have chosen these tracks to be representative of the data. Feel free to open the dataset and select other tracks
fov_name_mock = "A/3/9"
track_id_mock = [19]
fov_name_inf = "B/4/9"
track_id_inf = [42]

## Show the images over time
def get_patch(data, cell_centroid, patch_size):
    """Extract patch centered on cell centroid across all channels.

    Parameters
    ----------
    data : ndarray
        Image data with shape (C, Y, X) or (Y, X)
    cell_centroid : tuple
        (y, x) coordinates of cell centroid
    patch_size : int
        Size of the square patch to extract

    Returns
    -------
    ndarray
        Extracted patch with shape (C, patch_size, patch_size) or (patch_size, patch_size)
    """
    y_centroid, x_centroid = cell_centroid
    x_start = max(0, x_centroid - patch_size // 2)
    x_end = min(data.shape[-1], x_centroid + patch_size // 2)
    y_start = max(0, y_centroid - patch_size // 2)
    y_end = min(data.shape[-2], y_centroid + patch_size // 2)

    if data.ndim == 3:  # CYX format
        patch = data[:, int(y_start) : int(y_end), int(x_start) : int(x_end)]
    else:  # YX format
        patch = data[int(y_start) : int(y_end), int(x_start) : int(x_end)]
    return patch

# Open the dataset
plate = open_ome_zarr(input_data_path)
uninfected_position = plate[fov_name_mock][0]
infected_position = plate[fov_name_inf][0]

# Filter the centroids of these two tracks
filtered_centroid_mock = features_anndata[(features_anndata["fov_name"] == fov_name_mock) &(features_anndata['track_id']==track_id_mock)]
filtered_centroid_inf = features_anndata[(features_anndata["fov_name"] == fov_name_inf) &(features_anndata['track_id']==track_id_inf)]

uinfected_stack= []
for idx, row in filtered_centroid_mock.iterrows():
    uinfected_stack.append(get_patch(cyx,(row['y'],row['x']),patch_size))
uinfected_stack = np.array(uinfected_stack)

infected_stack = []
for idx, row in filtered_centroid_mock.iterrows():
    infected_stack.append(get_patch(cyx,(row['y'],row['x']),patch_size))
infected_stack = np.array(infected_stack)

# Plot 10 timepoints 

# %% [markdown]
"""
## Responsible Use

We are committed to advancing the responsible development and use of artificial intelligence.
Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with our services.

Should you have any security or privacy issues or questions related to the services,
please reach out to our team at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com) or [privacy@chanzuckerberg.com](mailto:privacy@chanzuckerberg.com) respectively.
"""
