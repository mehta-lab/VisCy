# %%
"""
Example script for converting xarray embeddings to AnnData format and loading annotations.

This script demonstrates:
1. Converting xarray embeddings to AnnData format
2. Loading annotations into AnnData objects
3. Simple Plotting example
"""

from pathlib import Path

import pandas as pd
import seaborn as sns

# Optional for plotting directly with AnnData objects w/o manual accessing patterns
# import scanpy as sc 
import xarray as xr

from viscy.representation.evaluation import (
    convert_xarray_annotation_to_anndata,
    load_annotation_anndata,
)

# %%
# Define paths
data_path = Path("/hpc/mydata/sricharan.varra/repos/VisCy/data/2024_11_21_A549_TOMM20_DENV/")
annotations_path = data_path / "annotations" / "track_infection_annotation.csv"
embeddings_path = data_path / "embeddings" / "phase_160patch_104ckpt_ver3max.zarr"
output_path = data_path / "track_data_anndata.zarr"

# %%
# Load embeddings
embeddings_dataset = xr.open_zarr(embeddings_path)

# %%
# Convert xarray to AnnData
adata = convert_xarray_annotation_to_anndata(
    embeddings_dataset,
    output_path,
    overwrite=True,
    return_anndata=True,
)

# %%
# Load annotations
adata_annotated = load_annotation_anndata(
    adata=adata,
    path=annotations_path,
    name="infection_status",
)

# %%
# Show results
print(adata_annotated.obs)

# %%
# Simple Accessing and Plotting (matplotlib)
# Plot the first two PCs colored by fov_name

sns.scatterplot(
    data=pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names).join(adata.obs),
    x=0,
    y=1,
    hue="fov_name",
)


# %% 
# Simple Plotting with scanpy if you have it installed.
# Plot the first two PCs colored by fov_name

# sc.pl.pca(adata, color="fov_name")
