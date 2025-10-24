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

from viscy.representation.evaluation import load_annotation_anndata
from viscy.representation.evaluation.annotation import convert

# %%
# Define paths
embeddings_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr"
)
annotations_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_infection_annotation.csv"
)
output_path = Path(
    "./hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/anndata/tmp/track_data_anndata.zarr"
)

# %%
# Load embeddings
embeddings_dataset = xr.open_zarr(embeddings_path)

# %%
# Convert xarray to AnnData
adata = convert(
    embeddings_dataset,
    output_path,
    overwrite=True,
    return_anndata=True,
)
print(adata)

# %%
# Load annotations
adata_annotated = load_annotation_anndata(
    adata=adata,
    path=annotations_path,
    name="infection_status",
)

# %%
# Show results
print(adata_annotated)

# %%
# Simple Accessing and Plotting (matplotlib)
# Plot the first two Phate embeddings colored by fov_name

sns.scatterplot(
    data=pd.DataFrame(adata.obsm["X_phate"], index=adata.obs_names).join(adata.obs),
    x=0,
    y=1,
    hue="fov_name",
)


# %%
# Simple Plotting with scanpy if you have it installed.
# Plot the first two Phate embeddings colored by fov_name

# sc.pl.embedding(basis="phate", adata=adata, color="fov_name")
