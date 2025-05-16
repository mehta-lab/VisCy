# %% [markdown]
"""
# Cytoland Tutorial: Virtual Staining of Zebrafish Neuromasts with VSNeuromast

**Estimated time to complete:** 15 minutes
"""

# %% [markdown]
"""
# Learning Goals

* Download the VSNeuromast model and an example dataset containing time-lapse images of zebrafish neuromasts.
* Pre-compute normalization statistics for the images using the `viscy preprocess` command-line interface (CLI).
* Run inference for joint virtual staining of cell nuclei and plasma membrane via the `viscy predict` CLI.
* Visualize the effect of photobleaching in fluorescence imaging and how virtual staining can mitigate this issue.
"""

# %% [markdown]
"""
# Prerequisites

Python>=3.11
"""

# %% [markdown]
"""
# Introduction

The zebrafish neuromasts are sensory organs on the lateral lines.
Given their relatively simple structure and high accessibility to live imaging,
they are used as a model system to study organogenesis _in vivo_.
However, multiplexed long-term fluorescence imaging at high spatial-temporal resolution
is often limited by photobleaching and phototoxicity.
Also, engineering fish lines with a combination of landmark fluorescent labels
(e.g. nuclei and plasma membrane) and functional reporters increases experimental complexity.
\
VSNeuromast is a 3D UNeXt2 model that has been trained zebrafish neuromasts using the Cytoland approach.
(See the [model card](https://virtualcellmodels.cziscience.com/paper/cytoland2025)
for more details about the Cytoland models.)
This model enables users to jointly stain cell nuclei and plasma membranes from 3D label-free images
for downstream analysis such as cell segmentation and tracking.
"""

# %% [markdown]
"""
# Setup

The commands below will install the required packages and download the example dataset and model checkpoint.
It may take a **few minutes** to download all the files.

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
# Here stackview and ipycanvas are installed for visualization
# !pip install -U "viscy[metrics,visual]==0.3"

# %%
# Restart kernel if running in Google Colab
# This is required to use the packages installed above
# The 'kernel crashed' message is expected here
if "get_ipython" in globals():
    session = get_ipython()
    if "google.colab" in str(session):
        print("Shutting down colab session.")
        session.kernel.do_shutdown(restart=True)

# %%
# Download the example dataset
# !wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSNeuromast/test/isim-bleaching-example.zarr/"

# %%
# Rename the downloaded dataset to what the example prediction config expects (`input.ome.zarr`)
# And validate the OME-Zarr metadata with iohub
# !mv isim-bleaching-example.zarr input.ome.zarr
# !iohub info -v input.ome.zarr

# %%
# Download the VSNeuromast model checkpoint and prediction config
# !wget "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSNeuromast/epoch=64-step=24960.ckpt"
# !wget "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSNeuromast/predict.yml"

# %% [markdown]
"""
# Use Case

## Example Dataset

The neuromast example dataset used in this tutorial contains
quantitative phase and paired fluorescence images of cell nuclei and plasma membrane.
It is a subsampled time-lapse from a test set used to evaluate the VSNeuromast model.
The full dataset can be downloaded from the
[BioImage Archive](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1702).

Refer to our [preprint](https://doi.org/10.1101/2024.05.31.596901) for more details
about how the dataset and model were generated.

## Using Custom Data

The model only requires label-free images for inference.
To run inference on your own data,
convert them into the OME-Zarr data format using iohub or other
[tools](https://ngff.openmicroscopy.org/tools/index.html#file-conversion),
and edit the `predict.yml` file to specify the input data path.
Specifically, the `data.init_args.data_path` field should be updated:

```diff
-     data_path: input.ome.zarr
+     data_path: /path/to/your.ome.zarr
```

The image may need to be resampled to roughly match the voxel size of the example dataset
(0.2x0.1x0.1 µm, ZYX).
"""

# %% [markdown]
"""
# Run Model Inference

On Google Colab, the preprocessing step takes about **1 minute**,
and the inference step takes about **2 minutes** (T4 GPU).
"""

# %%
# Run the CLI command to pre-compute normalization statistics
# This includes the median and interquartile range (IQR)
# Used to shift and scale the intensity distribution of the input images
# !viscy preprocess --data_path=input.ome.zarr

# %%
# Run the CLI command to run inference
# !viscy predict -c predict.yml

# %% [markdown]
"""
# Analysis of Model Outputs

Measure photobleaching in the fluorescence images
and how virtual staining can mitigate this issue.
"""
