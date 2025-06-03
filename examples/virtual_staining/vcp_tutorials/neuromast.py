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
VSNeuromast is a 3D UNeXt2 model that has been trained on images of
zebrafish neuromasts using the Cytoland approach.
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
quantitative phase and paired fluorescence images of the cell nuclei and the plasma membrane.
\
**It is a subsampled time-lapse from a test set used to evaluate the VSNeuromast model.**
\
The full dataset can be downloaded from the
[BioImage Archive](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1702).

Refer to our [preprint](https://doi.org/10.1101/2024.05.31.596901) for more details
about how the dataset and model were generated.

## Using Custom Data

The model only requires label-free images for inference.
To run inference on your own data,
convert them into the [OME-Zarr](https://ngff.openmicroscopy.org/)
data format using iohub or other
[tools](https://ngff.openmicroscopy.org/tools/index.html#file-conversion),
and edit the `predict.yml` file to specify the input data path.
Specifically, the `data.init_args.data_path` field should be updated:

```diff
-     data_path: input.ome.zarr
+     data_path: /path/to/your.ome.zarr
```

The image may need to be resampled to roughly match the voxel size of the example dataset
(0.25x0.108x0.108 µm, ZYX).
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

1. Visualize predicted images over time and compare with the fluorescence images.
2. Measure photobleaching in the fluorescence images
and how virtual staining can mitigate this issue.
Since most pixels in the images are background,
we will use the 99th percentile (brightest 1%)
of the intensity distribution as a proxy for foreground signal.
"""

# %%
# imports
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from iohub import open_ome_zarr
from numpy.typing import NDArray
from skimage.exposure import rescale_intensity


def render_rgb(
    image: np.ndarray, colormap: Colormap
) -> tuple[NDArray, plt.cm.ScalarMappable]:
    """Render a 2D grayscale image as RGB using a colormap.

    Parameters
    ----------
    image : np.ndarray
        intensity image
    colormap : Colormap
        colormap

    Returns
    -------
    tuple[NDArray, plt.cm.ScalarMappable]
        rendered RGB image and the color mapping
    """
    image = rescale_intensity(image, out_range=(0, 1))
    image = colormap(image)
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(0, 1), cmap=colormap.to_matplotlib()
    )
    return image, mappable


# %%
# read a single Z-slice for visualization
z_slice = 30

with open_ome_zarr("input.ome.zarr/0/3/0") as fluor_store:
    fluor_nucleus = fluor_store[0][:, 1, z_slice]
    fluor_membrane = fluor_store[0][:, 0, z_slice]

with open_ome_zarr("prediction.ome.zarr/0/3/0") as vs_store:
    vs_nucleus = vs_store[0][:, 0, z_slice]
    vs_membrane = vs_store[0][:, 1, z_slice]


# Render the images as RGB in false colors
vs_nucleus_rgb, vs_nucleus_mappable = render_rgb(vs_nucleus, Colormap("bop_blue"))
vs_membrane_rgb, vs_membrane_mappable = render_rgb(vs_membrane, Colormap("bop_orange"))
merged_vs = (vs_nucleus_rgb + vs_membrane_rgb).clip(0, 1)

fluor_nucleus_rgb, fluor_nucleus_mappable = render_rgb(fluor_nucleus, Colormap("green"))
fluor_membrane_rgb, fluor_membrane_mappable = render_rgb(
    fluor_membrane, Colormap("magenta")
)
merged_fluor = (fluor_nucleus_rgb + fluor_membrane_rgb).clip(0, 1)

# Plot
fig = plt.figure(figsize=(12, 7), layout="constrained")

images = {"fluorescence": merged_fluor, "virtual staining": merged_vs}

for row, (subfig, (name, img)) in enumerate(
    zip(fig.subfigures(nrows=2, ncols=1), images.items())
):
    subfig.suptitle(name)
    cax_nuc = subfig.add_axes([1, 0.55, 0.02, 0.3])
    cax_mem = subfig.add_axes([1, 0.15, 0.02, 0.3])
    axes = subfig.subplots(ncols=len(merged_vs))
    for t, ax in enumerate(axes):
        if row == 1:
            ax.set_title(f"{t * 30} min", y=-0.1)
        ax.imshow(img[t])
        ax.axis("off")
    if row == 0:
        subfig.colorbar(fluor_nucleus_mappable, cax=cax_nuc, label="Nuclei (GFP)")
        subfig.colorbar(
            fluor_membrane_mappable, cax=cax_mem, label="Membrane (mScarlett)"
        )
    elif row == 1:
        subfig.colorbar(vs_nucleus_mappable, cax=cax_nuc, label="Nuclei (VS)")
        subfig.colorbar(vs_membrane_mappable, cax=cax_mem, label="Membrane (VS)")

plt.show()

# %% [markdown]
"""
The plasma membrane fluorescence decreases over time,
while the virtual staining remains stable.
How significant is this effect? Is it consistent with photobleaching?
Analysis below will answer these questions.
"""


# %%
def highlight_intensity_normalized(fov_path: str, channel_name: str) -> list[float]:
    """
    Compute highlight (99th percentile) intensity of each timepoint,
    normalized to the first timepoint.

    Parameters
    ----------
    fov_path : str
        Path to the field of view (FOV).
    channel_name : str
        Name of the channel to compute highlight intensity for.

    Returns
    -------
    NDArray
        List of intensity values.
    """
    with open_ome_zarr(fov_path) as fov:
        channel_index = fov.get_channel_index(channel_name)
        channel = fov["0"].dask_array()[:, channel_index]
        highlights = []
        for t, volume in enumerate(channel):
            highlights.append(np.percentile(volume.compute(), 99))
        return [h / highlights[0] for h in highlights]


# %%
# Plot intensity over time
mean_fl = highlight_intensity_normalized("input.ome.zarr/0/3/0", "mScarlett")
mean_vs = highlight_intensity_normalized(
    "prediction.ome.zarr/0/3/0", "membrane_prediction"
)
time = np.arange(0, 100, 30)

plt.plot(time, mean_fl, label="membrane fluorescence")
plt.plot(time, mean_vs, label="membrane virtual staining")
plt.xlabel("time / min")
plt.ylabel("normalized highlight intensity")
plt.legend()

# %% [markdown]
"""
Here the highlight intensity of the fluorescence images decreases over time,
following a exponential decay pattern, indicating photobleaching.
The virtual staining is not affected by this issue.
(The object drifts slightly over time, so some inherent noise is expected.)
"""

# %% [markdown]
"""
# Summary

In the above example, we demonstrated how to use the VSNeuromast model
for virtual staining of cell nuclei and plasma membranes of the zebrafish neuromast _in vivo_,
which can avoid photobleaching in long-term live imaging.
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
