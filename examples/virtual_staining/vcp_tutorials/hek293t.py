# %% [markdown]
"""
# Cytoland Tutorial: Virtual Staining of HEK293T Cells with VSCyto3D

**Estimated time to complete:** 15 minutes
"""

# %% [markdown]
"""
# Learning Goals

* Download the VSCyto3D model and an example dataset containing HEK293T cell images.
* Pre-compute normalization statistics for the images using the `viscy preprocess` command line interface (CLI).
* Run inference for joint virtual staining of cell nuclei and plasma membrane via the `viscy predict` CLI.
* Compare virtually and experimentally stained cells and see how virtual staining can rescue missing labels.
"""

# %% [markdown]
"""
# Prerequisites

Python>=3.11
"""

# %% [markdown]
"""
# Introduction

See the [model card](https://virtualcellmodels.cziscience.com/paper/cytoland2025)
for more details about the Cytoland models. 

VSCyto3D is a 3D UNeXt2 model that has been trained on A549, HEK293T, and hiPSC cells using the Cytoland approach.
This model enables users to jointly stain cell nuclei and plasma membranes from 3D label-free images
for downstream analysis such as cell segmentation and tracking without the need for human annotation of volumetric data.
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
# !pip install -U "viscy[metrics,visual]==0.3" stackview ipycanvas==0.11

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
# !wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto3D/test/HEK293T-Phase3D-H2B-CAAX-example.zarr/"

# %%
# Rename the downloaded dataset to what the example prediction config expects (`input.ome.zarr`)
# And validate the OME-Zarr metadata with iohub
# !mv HEK293T-Phase3D-H2B-CAAX-example.zarr input.ome.zarr
# !iohub info -v input.ome.zarr

# %%
# Download the VSCyto3D model checkpoint and prediction config
# !wget "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto3D/epoch=83-step=14532-loss=0.492.ckpt"
# !wget "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto3D/predict.yml"

# %% [markdown]
"""
# Use Case

## Example Dataset

The HEK293T example dataset used in this quick-start guide contains
quantitative phase and paired fluorescence images of cell nuclei and plasma membrane.
It is a subset (one cropped region of interest) from a test set used to evaluate the VSCyto3D model.
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

Visualize the experimental and virtually stained images using the `stackview` package.
"""

# %% [markdown]
"""
Visualizing large 3D multichannel images in a Jupyter notebook
**is prone to performance issues and may crash the notebook** if the images are too large
(the free Colab instances have limited CPU cores and memory).
The visualization code below is only intended for demonstration.
We strongly recommend downloading the images (from the 'files' bar in Colab)
and using a standalone viewer such as [napari](https://napari.org/).
"""

# %%
from pathlib import Path

import numpy as np
import stackview
from iohub import open_ome_zarr
from skimage.exposure import rescale_intensity

try:
    from google.colab import output

    output.enable_custom_widget_manager()
except ImportError:
    pass


# %%
# open the images
def split_and_rescale_channels(timepoint: np.ndarray) -> tuple[np.ndarray, ...]:
    return (rescale_intensity(channel, out_range=(0, 1)) for channel in timepoint)


fov_name = "plate/0/11"
input_image = open_ome_zarr("input.ome.zarr")[fov_name]["0"]
prediction_image = open_ome_zarr("prediction.ome.zarr")[fov_name]["0"]

phase, fluor_nucleus, fluor_membrane = split_and_rescale_channels(input_image[0])
vs_nucleus, vs_membrane = split_and_rescale_channels(prediction_image[0])

# %%
# Drag the slider to start rendering
# Click on the numbered buttons to toggle the channels
stackview.switch(
    # the 0, 1, 2, 3, 4 buttons will correspond to these 5 channels
    # We apply a gamma adjustment to the phase channel to improve visibility in the overlay
    images=[phase**2.5, fluor_nucleus, fluor_membrane, vs_nucleus, vs_membrane],
    colormap=["gray", "pure_green", "pure_magenta", "pure_blue", "pure_yellow"],
    toggleable=True,
    zoom_factor=0.5,
    display_min=0.0,
    display_max=0.9,
)

# %% [markdown]
"""
Note how the experimental fluorescence is missing for a subset of cells.
This is due to loss of genetic labeling.
The virtually stained images is not affected by this issue and can robustly label all cells.
"""

# %% [markdown]
"""
# Summary

In the above example, we demonstrated how to use the VSCyto3D model
for virtual staining of cell nuclei and plasma membranes, which can rescue missing labels.
"""

# %% [markdown]
"""
## Contact & Feedback

For issues or feedback about this tutorial please contact Ziwen Liu at [ziwen.liu@czbiohub.org](mailto:ziwen.liu@czbiohub.org).

## Responsible Use

We are committed to advancing the responsible development and use of artificial intelligence.
Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with our services.

Should you have any security or privacy issues or questions related to the services,
please reach out to our team at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com) or [privacy@chanzuckerberg.com](mailto:privacy@chanzuckerberg.com) respectively.
"""
