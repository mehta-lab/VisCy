#!/usr/bin/env python3
#%%
import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from viscy.data.triplet import TripletDataModule
from viscy.representation.evaluation.pseudotime_plotting import (
    align_image_stacks,
    plot_pc_trajectories,
)

# Use the new integrated DTW API
from viscy.representation.pseudotime import CytoDtw, identify_lineages

logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration
NAPARI = False
if NAPARI:
    import os

    import napari
    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()

# File paths
input_data_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)
infection_annotations_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_infection_annotation.csv"
)

pretrain_features_root = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/prediction_pretrained_models"
)

# Embedding paths
dynaclr_features_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions_infection/2chan_192patch_100ckpt_timeAware_ntxent_GT.zarr"
imagenet_features_path = pretrain_features_root / "ImageNet/20241107_sensor_n_phase_imagenet.zarr"
openphenom_features_path = pretrain_features_root / "OpenPhenom/20241107_sensor_n_phase_openphenom.zarr"

#%% Check that the directories exist
print(f"Input data path exists: {input_data_path.exists()}")
print(f"Tracks path exists: {tracks_path.exists()}")
print(f"Infection annotations path exists: {infection_annotations_path.exists()}")
print(f"Pretrain features root exists: {pretrain_features_root.exists()}")
print(f"Dynaclr features path exists: {dynaclr_features_path.exists()}")
print(f"Imagenet features path exists: {imagenet_features_path.exists()}")
print(f"Openphenom features path exists: {openphenom_features_path.exists()}")
#%%

output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/output"
)
output_root.mkdir(parents=True, exist_ok=True)
#%%
def main():
    """Main analysis pipeline using the new DTW API."""
    
    # Initialize DTW analyzers for each embedding method
    analyzers = {
        "dynaclr": CytoDtw(dynaclr_features_path),
        "imagenet": CytoDtw(imagenet_features_path), 
        "openphenom": CytoDtw(openphenom_features_path),
    }
    
    # Load infection annotations
    infection_annotations_df = pd.read_csv(infection_annotations_path)
    infection_annotations_df["fov_name"] = "/C/2/000001"
    
    # Identify lineages from the first dataset
    feature_df = analyzers["dynaclr"].embeddings["sample"].to_dataframe().reset_index(drop=True)
    all_lineages = identify_lineages(feature_df)
    logger.info(f"Found {len(all_lineages)} distinct lineages")
    
    # Filter lineages by minimum timepoints
    min_timepoints = 20
    filtered_lineages = []
    for fov_id, track_ids in all_lineages:
        lineage_rows = feature_df[
            (feature_df["fov_name"] == fov_id) & (feature_df["track_id"].isin(track_ids))
        ]
        total_timepoints = len(lineage_rows)
        if total_timepoints >= min_timepoints:
            filtered_lineages.append((fov_id, track_ids))
    
    logger.info(f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints")
    
    # Reference pattern configuration
    reference_lineage_fov = "/C/2/000001"
    reference_lineage_track_id = [129]
    reference_timepoints = (8, 70)  # sensor relocalization and partial remodelling
    
    # Find a valid reference lineage from filtered lineages
    for fov_id, track_ids in filtered_lineages:
        if reference_lineage_fov == fov_id:
            reference_lineage_track_id = track_ids
            break
    
    # Perform DTW analysis for each embedding method
    alignment_results = {}
    
    for name, analyzer in analyzers.items():
        logger.info(f"Processing {name} embeddings")
        
        try:
            # Extract reference pattern
            reference_pattern = analyzer.get_reference_pattern(
                fov_name=reference_lineage_fov,
                track_id=reference_lineage_track_id,
                timepoints=reference_timepoints
            )
            
            logger.info(f"Found reference pattern for {name} with shape {reference_pattern.shape}")
            
            # Find pattern matches
            matches = analyzer.find_pattern_matches(
                reference_pattern=reference_pattern,
                filtered_lineages=filtered_lineages,
                window_step_fraction=0.1,
                num_candidates=4,
                method="bernd_clifford",
                metric="cosine",
                save_path=output_root / f"{name}_matching_lineages_cosine.csv"
            )
            
            alignment_results[name] = matches
            logger.info(f"Found {len(matches)} matches for {name}")
            
            # Generate PC trajectory visualization
            if not matches.empty:
                plot_pc_trajectories(
                    reference_lineage_fov=reference_lineage_fov,
                    reference_lineage_track_id=reference_lineage_track_id,
                    reference_timepoints=list(reference_timepoints),
                    match_positions=matches,
                    embeddings_dataset=analyzer.embeddings,
                    filtered_lineages=filtered_lineages,
                    name=name,
                    save_path=output_root / f"{name}_pc_lineage_alignment.png",
                )
                
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            continue
    
    # Compare DTW performance between embedding methods
    create_dtw_comparison_plots(alignment_results, output_root)
    
    # Demonstrate image alignment for the best model
    if alignment_results:
        best_model = min(alignment_results.keys(), 
                        key=lambda k: alignment_results[k]["distance"].min() 
                        if not alignment_results[k].empty else float('inf'))
        
        logger.info(f"Best performing model: {best_model}")
        demonstrate_image_alignment(
            analyzers[best_model], 
            alignment_results[best_model], 
            reference_pattern,
            output_root
        )

def create_dtw_comparison_plots(alignment_results, output_root):
    """Create comparison plots for DTW performance across models."""
    
    # Collect alignment statistics
    match_data = []
    for name, match_positions in alignment_results.items():
        if match_positions is not None and not match_positions.empty:
            for i, row in match_positions.head(10).iterrows():
                warp_path = (
                    ast.literal_eval(row["warp_path"])
                    if isinstance(row["warp_path"], str)
                    else row["warp_path"]
                )
                match_data.append({
                    "model": name,
                    "match_position": row["start_timepoint"],
                    "dtw_distance": row["distance"],
                    "path_skewness": row["skewness"],
                    "path_length": len(warp_path),
                })

    if not match_data:
        logger.warning("No match data available for comparison plots")
        return
        
    comparison_df = pd.DataFrame(match_data)

    # Create comparison visualizations
    plt.figure(figsize=(12, 10))

    # DTW distances comparison
    plt.subplot(2, 2, 1)
    sns.boxplot(x="model", y="dtw_distance", data=comparison_df)
    plt.title("DTW Distance by Model")
    plt.ylabel("DTW Distance (lower is better)")

    # Path skewness comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(x="model", y="path_skewness", data=comparison_df)
    plt.title("Path Skewness by Model")
    plt.ylabel("Skewness (lower is better)")

    # Path lengths comparison
    plt.subplot(2, 2, 3)
    sns.boxplot(x="model", y="path_length", data=comparison_df)
    plt.title("Warping Path Length by Model")
    plt.ylabel("Path Length")

    # Distance vs skewness scatterplot
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        x="dtw_distance", y="path_skewness", hue="model", data=comparison_df
    )
    plt.title("DTW Distance vs Path Skewness")
    plt.xlabel("DTW Distance")
    plt.ylabel("Path Skewness")
    plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig(output_root / "dtw_alignment_comparison.png", dpi=300)
    plt.close()
    
    logger.info("Saved DTW comparison plots")

def demonstrate_image_alignment(analyzer, matches, reference_pattern, output_root):
    """Demonstrate image alignment using DTW results."""
    
    if matches.empty:
        logger.warning("No matches available for image alignment")
        return
        
    # Configuration for image alignment
    source_channels = [
        "Phase3D",
        "raw GFP EX488 EM525-45", 
        "raw mCherry EX561 EM600-37",
    ]
    yx_patch_size = (192, 192)
    z_range = (10, 30)
    
    # Get top aligned cells
    top_aligned_cells = matches.head(5)
    napari_viewer = viewer if NAPARI else None
    
    try:
        # Align image stacks
        all_lineage_images, all_aligned_stacks = align_image_stacks(
            reference_pattern=reference_pattern,
            top_aligned_cells=top_aligned_cells,
            input_data_path=input_data_path,
            tracks_path=tracks_path,
            source_channels=source_channels,
            yx_patch_size=yx_patch_size,
            z_range=z_range,
            view_ref_sector_only=True,
            napari_viewer=napari_viewer,
        )
        
        logger.info(f"Aligned {len(all_aligned_stacks)} image stacks")
        
        # Display aligned stacks in napari if available
        if NAPARI and napari_viewer:
            for idx, stack in enumerate(all_aligned_stacks):
                # Display different channels
                gfp_mip = np.max(stack[:, 1, :, :], axis=1)
                mcherry_mip = np.max(stack[:, 2, :, :], axis=1)
                phase_slice = stack[:, 0, 15, :]  # middle z-slice
                
                napari_viewer.add_image(
                    gfp_mip,
                    name=f"Aligned_GFP_{idx}",
                    colormap="green",
                    contrast_limits=(106, 215),
                )
                napari_viewer.add_image(
                    mcherry_mip,
                    name=f"Aligned_mCherry_{idx}",
                    colormap="magenta", 
                    contrast_limits=(106, 190),
                )
                napari_viewer.add_image(
                    phase_slice,
                    name=f"Aligned_Phase_{idx}",
                    colormap="gray",
                    contrast_limits=(-0.74, 0.4),
                )
            
            napari_viewer.grid.enabled = True
            napari_viewer.grid.shape = (-1, 3)
            
    except Exception as e:
        logger.error(f"Failed to align images: {e}")

if __name__ == "__main__":
    main()
# %%
