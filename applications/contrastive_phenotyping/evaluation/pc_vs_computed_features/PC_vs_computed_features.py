"""Script to compute the correlation between PCA and UMAP features and computed features
* finds the computed features best representing the PCA and UMAP components
* outputs a heatmap of the correlation between PCA and UMAP features and computed features
"""

# %%
from anndata import read_zarr
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from compute_pca_features import compute_correlation_and_save_png

# %% for organelle features

def convert_to_dataframe(embeddings_dataset):
    features_df = embeddings_dataset.obs
    embedding_features = embeddings_dataset.X
    embedding_features_df = pd.DataFrame(embedding_features, columns=[f"feature_{i+1}" for i in range(embedding_features.shape[1])], index=features_df.index)
    features_df = pd.concat([features_df, embedding_features_df], axis=1)
    return features_df

def match_computed_features_to_embeddings(
    embeddings_path: Path,
    computed_features_path: Path,
    wells: list,
):
    embeddings_dataset = read_zarr(embeddings_path)
    computed_features = pd.read_csv(computed_features_path)
    
    # Extract well from fov_name with better handling
    computed_features_split = computed_features["fov_name"].str.split("/")
    computed_features["well"] = computed_features_split.str[1].fillna("") + "/" + computed_features_split.str[2].fillna("")
    
    features_df = convert_to_dataframe(embeddings_dataset)
    
    # Extract well from fov_name with better handling
    features_split = features_df["fov_name"].str.split("/")
    features_df["well"] = features_split.str[0].fillna("") + "/" + features_split.str[1].fillna("")
    
    features_df_filtered = features_df[features_df["well"].isin(wells)].copy()
    computed_features_df = computed_features[computed_features["well"].isin(wells)].copy()
    
    # Prepare columns for merge - strip '/' from fov_name in both dataframes
    computed_features_df["fov_name_clean"] = computed_features_df["fov_name"].str.strip('/')
    features_df_filtered["fov_name_clean"] = features_df_filtered["fov_name"].astype(str).str.strip('/')
    
    # Rename columns to avoid conflicts during merge
    # Rename 't' in features_df_filtered to 'time_point' to match computed_features
    features_df_filtered = features_df_filtered.rename(columns={"t": "time_point"})
    
    # Ensure merge keys have compatible types - use float for track_id
    computed_features_df["track_id"] = pd.to_numeric(computed_features_df["track_id"], errors='coerce')
    features_df_filtered["track_id"] = pd.to_numeric(features_df_filtered["track_id"], errors='coerce')
    computed_features_df["time_point"] = pd.to_numeric(computed_features_df["time_point"], errors='coerce')
    features_df_filtered["time_point"] = pd.to_numeric(features_df_filtered["time_point"], errors='coerce')
    
    # This performs a vectorized join operation
    merged_df = pd.merge(
        computed_features_df,
        features_df_filtered,
        on=["fov_name_clean", "track_id", "time_point"],
        how="inner",
        suffixes=("_computed", "_embedding")
    )
    
    # Drop the temporary fov_name_clean column
    merged_df = merged_df.drop(columns=["fov_name_computed"])
    
    return merged_df

embeddings_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/organelle_160patch_104ckpt_ver3max.zarr"
)
computed_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/feature_list_G3BP1_2025_07_22_160patch.csv"
)

wells = ["C/2"]
computed_features_df = match_computed_features_to_embeddings(
embeddings_path,
computed_features_path,
wells,
)

# Save the features dataframe to a CSV file
computed_features_df.to_csv(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/features_organelle_wellC2_160patch.csv",
    index=False,
)


# %% compute the correlation between PCA and computed features and plot

computed_features_df = pd.read_csv("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/features_organelle_wellC2_160patch.csv")
correlation_organelle = compute_correlation_and_save_png(
    computed_features_df,
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/PC_vs_CF_organelle_wellC2_160patch.svg",
)

# features_organelle = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/features_twoChan_organelle_multiwell_refinedPCA.csv")


# %%
