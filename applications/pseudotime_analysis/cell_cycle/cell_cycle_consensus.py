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

"""
TODO

- We need to find a way to save the annotations, features and track information into one file. 
- We need to standardize the naming convention. i.e The annotations fov_name is missing a / at the beginning. 
- It would be nice to also select which will be the reference lineages and add that as a column. 
- Figure out what is the best format to save the consensus lineage
- Does the consensus track generalize?
- There is a lot of fragmentation. Which tracking was used for the annotations? There is a script that unifies this but no record of which one was it. We can append these as extra columns

"""

# Configuration
NAPARI = True
if NAPARI:
    import os

    import napari
    os.environ["DISPLAY"] = ":1"
    viewer = napari.Viewer()

# File paths

# ANNOTATIONS
cell_cycle_annotations_denv_dict= {
    # "tomm20_cc_1": 
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'fov_name': "/C/2/001000",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr",
    #     # 'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/tracking.zarr",
    #     },
    # "tomm20_cc_2": 
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'fov_name': "/B/3/000001",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr",
    #     # 'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_21_A549_TOMM20_DENV/2-assemble/tracking.zarr",
    #     },
    # "sec61b_cc_1": 
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr/B/3/001000",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_cell_state_annotation.csv",
    #     'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/tracking.zarr",
    #     },
    # "sec61b_cc_2": 
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr/C/2/000001",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/0-annotation/track_cell_state_annotation.csv",
    #     },
    "g3bp1_cc_1": 
        {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr",
        'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/cytospeak_annotations/2025_07_24_annotations.csv",
        'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/phase_160patch_104ckpt_ver3max.zarr",
        'fov_name': "/C/1/001000",
        },
}
output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/output/cell_cycle_consensus"
)
output_root.mkdir(parents=True, exist_ok=True)

#%%
color_dict = {
    "interphase": "blue",
    "mitosis": "orange",
}
ANNOTATION_CELL_CYCLE = "predicted_cellstate"

# Load each dataframe and find the lineages
key , cell_cycle_annotations_denv=next(iter(cell_cycle_annotations_denv_dict.items()))
cell_cycle_annotations_df = pd.read_csv(cell_cycle_annotations_denv["annotations_path"])
data_path = cell_cycle_annotations_denv["data_path"]
fov_name = cell_cycle_annotations_denv["fov_name"]

cytodtw=CytoDtw(cell_cycle_annotations_denv["features_path"])

feature_df = cytodtw.embeddings["sample"].to_dataframe().reset_index(drop=True)
all_lineages = identify_lineages(feature_df)
logger.info(f"Found {len(all_lineages)} distinct lineages in the whole plate")

# Filter lineages by minimum timepoints
min_timepoints = 8

filtered_lineages = []
for fov_id, track_ids in all_lineages:
    lineage_rows = feature_df[
        (feature_df["fov_name"] == fov_id) & (feature_df["track_id"].isin(track_ids))
    ]
    total_timepoints = len(lineage_rows)
    if total_timepoints >= min_timepoints:
        filtered_lineages.append((fov_id, track_ids))
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])

# Filter to lineages only to the fov_name 
filtered_lineages = filtered_lineages[filtered_lineages["fov_name"] == fov_name]
logger.info(f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints")

#%%
# Filter all lineages that meet the minimum timepoints
min_timepoints = 15
all_filtered_lineages = []
for fov_id, track_ids in all_lineages:
    lineage_rows = feature_df[
        (feature_df["fov_name"] == fov_id) & (feature_df["track_id"].isin(track_ids))
    ]
    total_timepoints = len(lineage_rows)
    if total_timepoints >= min_timepoints:
        all_filtered_lineages.append((fov_id, track_ids))
all_filtered_lineages = pd.DataFrame(all_filtered_lineages, columns=["fov_name", "track_id"])
logger.info(f"Found {len(all_filtered_lineages)} lineages with at least {min_timepoints} timepoints")

#%%
#FIXME the dataframe has a missing / at the beginning of the fov_name
# Ensure all fov_name entries start with a leading slash
cell_cycle_annotations_df["fov_name"] = cell_cycle_annotations_df["fov_name"].apply(
    lambda x: x if str(x).startswith("/") else "/" + str(x)
)
filtered_cell_cycle_annotations_df = cell_cycle_annotations_df[cell_cycle_annotations_df["fov_name"] == fov_name]
filtered_cell_cycle_annotations_df[filtered_cell_cycle_annotations_df[ANNOTATION_CELL_CYCLE] == "mitosis"]

#%%
# Find the lineage that have the 
mitosis_track_ids = set(cell_cycle_annotations_df[cell_cycle_annotations_df[ANNOTATION_CELL_CYCLE] == "mitosis"]["track_id"])
filtered_lineages_mitosis = filtered_lineages[filtered_lineages["track_id"].apply(lambda track_list: any(track_id in mitosis_track_ids for track_id in track_list))]
logger.info(f"Found {len(filtered_lineages_mitosis)} lineages with mitosis annotation")
#%%
# TODO: find random 5 lineages that have the mitosis annotation in the filtered_lineages or top 5 mitosis lineages
annotated_examples = []

n_timepoints_before = min_timepoints//2
n_timepoints_after = min_timepoints//2
selected_lineages = filtered_lineages_mitosis.sample(min(5, len(filtered_lineages_mitosis)))
#%%
for _, lineage in selected_lineages.iterrows():
    track_ids_list = lineage["track_id"]  # This is a list of track IDs (includes pre and post mitosis)
    fov_name_lineage = lineage["fov_name"]
    
    # Debug: Check what data exists for this lineage
    lineage_feature_data = feature_df[
        (feature_df["fov_name"] == fov_name_lineage) & 
        (feature_df["track_id"].isin(track_ids_list))
    ]
    logger.info(f"Lineage {fov_name_lineage} with tracks {track_ids_list}: "
               f"feature data timepoints {lineage_feature_data['t'].min()}-{lineage_feature_data['t'].max()}")
    
    # Find mitosis annotations for any track in this lineage
    mitosis_annotations_for_lineage = cell_cycle_annotations_df[
        (cell_cycle_annotations_df["track_id"].isin(track_ids_list)) &
        (cell_cycle_annotations_df[ANNOTATION_CELL_CYCLE] == "mitosis")
    ]
    
    if not mitosis_annotations_for_lineage.empty:
        # Use the first mitosis timepoint as reference
        mitosis_timepoint = mitosis_annotations_for_lineage["t"].iloc[0]
        
        # Calculate min_t and max_t around the mitosis event
        requested_min_t = mitosis_timepoint - n_timepoints_before
        requested_max_t = mitosis_timepoint + n_timepoints_after
        
        # Constrain to available data range
        available_min_t = lineage_feature_data['t'].min()
        available_max_t = lineage_feature_data['t'].max()
        
        min_t = max(requested_min_t, available_min_t)
        max_t = min(requested_max_t, available_max_t)
        
        logger.info(f"  Mitosis at t={mitosis_timepoint}, requested range ({requested_min_t}, {requested_max_t}), "
                   f"constrained to ({min_t}, {max_t})")
        
        # Create annotations list for each timepoint in the range
        timepoint_annotations = []
        for t in range(int(min_t), int(max_t) + 1):
            # Find annotation for this specific timepoint and any track in the lineage
            annotation_for_t = cell_cycle_annotations_df[
                (cell_cycle_annotations_df["track_id"].isin(track_ids_list)) &
                (cell_cycle_annotations_df["t"] == t)
            ]
            
            if not annotation_for_t.empty:
                timepoint_annotations.append(annotation_for_t[ANNOTATION_CELL_CYCLE].iloc[0])
            else:
                # Default to interphase if no annotation found
                logger.warning(f"No annotation found for timepoint {t} for lineage {fov_name_lineage}")
                timepoint_annotations.append("interphase")
        
        annotated_examples.append({
            'fov_name': fov_name_lineage,
            'track_id': track_ids_list,  # Use the full lineage track IDs (pre + post mitosis)
            'timepoints': (int(min_t), int(max_t)),
            'annotations': timepoint_annotations,
            'weight': 1.0
        })
logger.info(f"Created {len(annotated_examples)} annotated examples from mitosis lineages")

# Filter out examples with empty patterns
valid_annotated_examples = []
for i, example in enumerate(annotated_examples):
    logger.info(f"Example {i}: fov={example['fov_name']}, track_ids={len(example['track_id'])}, "
               f"timepoints={example['timepoints']}, annotations={len(example['annotations'])}")
    
    # Check if we can extract a valid pattern
    try:
        test_pattern = cytodtw.get_reference_pattern(
            fov_name=example['fov_name'],
            track_id=example['track_id'], 
            timepoints=example['timepoints']
        )
        logger.info(f"  Pattern shape: {test_pattern.shape}")
        
        # Only keep examples with non-empty patterns AND sufficient post-mitosis data
        if test_pattern.shape[0] > 0:
            # Check if we have sufficient timepoints after mitosis
            # Find the mitosis timepoint for this example
            lineage_mitosis_annotations = cell_cycle_annotations_df[
                (cell_cycle_annotations_df["track_id"].isin(example['track_id'])) &
                (cell_cycle_annotations_df[ANNOTATION_CELL_CYCLE] == "mitosis")
            ]
            
            if not lineage_mitosis_annotations.empty:
                mitosis_t = lineage_mitosis_annotations["t"].iloc[0]
                logger.info(f"    Mitosis timepoint: {mitosis_t}, example timepoints: {example['timepoints']}")
                timepoints_after_mitosis = example['timepoints'][1] - mitosis_t
                
                if timepoints_after_mitosis >= n_timepoints_after:
                    valid_annotated_examples.append(example)
                    logger.info(f"  ✓ Valid pattern with {timepoints_after_mitosis} timepoints after mitosis - keeping example {i}")
                else:
                    logger.info(f"  ✗ Only {timepoints_after_mitosis} timepoints after mitosis (need ≥{n_timepoints_after}) - skipping example {i}")
            else:
                logger.info(f"  ✗ No mitosis annotation found - skipping example {i}")
        else:
            logger.info(f"  ✗ Empty pattern - skipping example {i}")
            
    except Exception as e:
        logger.error(f"  ✗ Failed to extract pattern: {e}")

logger.info(f"Filtered to {len(valid_annotated_examples)} valid examples from {len(annotated_examples)} total")
#%%
n_timepoints_before = 10
n_timepoints_after = 10
valid_annotated_examples=[{
    'fov_name': "/C/1/001000",
    'track_id': [47,48],
    'timepoints': (45-n_timepoints_before, 45+n_timepoints_after),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
{
    'fov_name': "/C/1/001000",
    'track_id': [59,60],
    'timepoints': (52-n_timepoints_before, 52+n_timepoints_after),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
{
    'fov_name': "/C/1/001001",
    'track_id': [93,94],
    'timepoints': (29-n_timepoints_before, 29+n_timepoints_after),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
# {
#     'fov_name': "/C/1/001001",
#     'track_id': [138,139],
#     'timepoints': (11-n_timepoints_before, 11+n_timepoints_after),
#     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
#     'weight': 1.0
# },
]
#%% 
# Plot the annotated examples features to PC1,PC2, and PC3 over time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Extract all reference patterns
patterns = []
pattern_info = []

for i, example in enumerate(valid_annotated_examples):
    pattern = cytodtw.get_reference_pattern(
        fov_name=example['fov_name'],
        track_id=example['track_id'],
        timepoints=example['timepoints']
    )
    patterns.append(pattern)
    pattern_info.append({
        'index': i,
        'fov_name': example['fov_name'], 
        'track_id': example['track_id'],
        'timepoints': example['timepoints'],
        'annotations': example['annotations']
    })

# Concatenate all patterns to fit PCA on full dataset
all_patterns_concat = np.vstack(patterns)

# Fit PCA on all data
scaler = StandardScaler()
scaled_patterns = scaler.fit_transform(all_patterns_concat)
pca = PCA(n_components=3)
pca.fit(scaled_patterns)

# Create subplots for PC1, PC2, PC3 over time
n_patterns = len(patterns)
fig, axes = plt.subplots(n_patterns, 3, figsize=(12, 3*n_patterns))
if n_patterns == 1:
    axes = axes.reshape(1, -1)

# Plot each pattern
for i, (pattern, info) in enumerate(zip(patterns, pattern_info)):
    # Transform this pattern to PC space
    scaled_pattern = scaler.transform(pattern)
    pc_pattern = pca.transform(scaled_pattern)
    
    # Create time axis
    time_axis = np.arange(len(pattern))
    
    # Plot PC1, PC2, PC3
    for pc_idx in range(3):
        ax = axes[i, pc_idx]
        
        # Plot PC trajectory with colorblind-friendly colors
        ax.plot(time_axis, pc_pattern[:, pc_idx], 'o-', color='blue', linewidth=2, markersize=4)
        
        # Color timepoints by annotation
        annotations = info['annotations']
        for t, annotation in enumerate(annotations):
            if annotation == 'mitosis':
                ax.axvline(t, color='orange', alpha=0.7, linestyle='--', linewidth=2)
                ax.scatter(t, pc_pattern[t, pc_idx], c='orange', s=50, zorder=5)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel(f'PC{pc_idx+1}')
        ax.set_title(f'Pattern {i+1}: FOV {info["fov_name"]}, Tracks {info["track_id"]}\nPC{pc_idx+1} over time')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Create consensus pattern if we have valid examples
if len(valid_annotated_examples) >= 2:
    consensus_result = cytodtw.create_consensus_reference_pattern(
        annotated_examples=valid_annotated_examples,
        reference_selection="median_length", 
        aggregation_method="mean"
    )
    consensus_lineage = consensus_result['consensus_pattern']
    consensus_annotations = consensus_result.get('consensus_annotations', None)
    consensus_metadata = consensus_result['metadata']
    
    logger.info(f"Created consensus pattern with shape: {consensus_lineage.shape}")
    logger.info(f"Consensus method: {consensus_metadata['aggregation_method']}")
    logger.info(f"Reference pattern: {consensus_metadata['reference_pattern']}")
    if consensus_annotations:
        logger.info(f"Consensus annotations length: {len(consensus_annotations)}")
else:
    logger.warning("Not enough valid lineages found to create consensus pattern")

#%%
#Plot the consensus pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaled_consensus_lineage = scaler.fit_transform(consensus_lineage)
pca = PCA(n_components=2)
pca.fit(scaled_consensus_lineage)
pca_consensus_lineage = pca.transform(scaled_consensus_lineage)

# Plot the consensus pattern PC1 over time and PC2 over time
plt.subplot(1, 2, 1)
plt.plot(pca_consensus_lineage[:, 0])
plt.xlabel("Time")
plt.ylabel("PC1")
plt.title("Consensus Pattern PC1 over Time")
plt.subplot(1, 2, 2)
plt.plot(pca_consensus_lineage[:, 1])
plt.xlabel("Time")
plt.ylabel("PC2")
plt.title("Consensus Pattern PC2 over Time")
plt.tight_layout()
plt.show()

#%% 
# Perform DTW analysis for each embedding method
alignment_results = {}

name = "consensus_lineage"
# Find pattern matches
matches = cytodtw.find_pattern_matches(
    reference_pattern=consensus_lineage,
    filtered_lineages=all_filtered_lineages.to_numpy(),
    window_step=2,
    num_candidates=20,
    method="bernd_clifford",
    metric="cosine",
    save_path=output_root / f"{name}_matching_lineages_cosine.csv"
)

alignment_results[name] = matches
logger.info(f"Found {len(matches)} matches for {name}")

#%%
# Plot aligned PC trajectories for top N cells aligned to consensus
import ast
top_n = 20
top_matches = matches.head(top_n)

# Get consensus pattern for PCA reference
consensus_pattern = consensus_lineage

# Prepare all data for consistent PCA fitting
all_patterns_for_pca = [consensus_pattern]
all_aligned_patterns = []

# Extract and align each matched pattern
for idx, row in top_matches.iterrows():
    fov_name = row['fov_name']
    track_ids = row['track_ids']
    if isinstance(track_ids, str):
        track_ids = ast.literal_eval(track_ids)
    warp_path = row['warp_path'] 
    if isinstance(warp_path, str):
        warp_path = ast.literal_eval(warp_path)
    start_time = row['start_timepoint']
    distance = row['distance']
    
    # Get lineage embeddings
    lineage_embeddings = []
    for track_id in track_ids:
        try:
            track_emb = cytodtw.embeddings.sel(sample=(fov_name, track_id)).features.values
            lineage_embeddings.append(track_emb)
        except KeyError:
            continue
    
    if not lineage_embeddings:
        continue
        
    lineage_embeddings = np.concatenate(lineage_embeddings, axis=0)
    
    # Create aligned pattern using warping path
    aligned_pattern = np.zeros_like(consensus_pattern)
    
    # Map each consensus timepoint to lineage timepoint via warping path
    ref_to_lineage = {}
    for ref_idx, query_idx in warp_path:
        lineage_idx = int(start_time + query_idx) if not pd.isna(start_time) else query_idx
        if 0 <= lineage_idx < len(lineage_embeddings):
            ref_to_lineage[ref_idx] = lineage_idx
            aligned_pattern[ref_idx] = lineage_embeddings[lineage_idx]
    
    # Fill missing values with nearest neighbor
    for ref_idx in range(len(consensus_pattern)):
        if ref_idx not in ref_to_lineage and ref_to_lineage:
            closest_ref_idx = min(ref_to_lineage.keys(), key=lambda x: abs(x - ref_idx))
            aligned_pattern[ref_idx] = lineage_embeddings[ref_to_lineage[closest_ref_idx]]
    
    all_aligned_patterns.append({
        'pattern': aligned_pattern,
        'info': f'FOV {fov_name}, Track {track_ids[0]}, Dist={distance:.3f}'
    })
    all_patterns_for_pca.append(aligned_pattern)

# Fit PCA on all patterns together for consistency
all_data = np.vstack(all_patterns_for_pca)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data)
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# Split back to individual patterns
consensus_pca = pca_data[:len(consensus_pattern)]
aligned_patterns_pca = []
start_idx = len(consensus_pattern)
for pattern_data in all_aligned_patterns:
    end_idx = start_idx + len(pattern_data['pattern'])
    aligned_patterns_pca.append({
        'pca': pca_data[start_idx:end_idx],
        'info': pattern_data['info']
    })
    start_idx = end_idx

# Plot PC1, PC2, PC3 trajectories
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['orange', 'purple', 'brown', 'pink', 'gray']
time_axis = np.arange(len(consensus_pattern))

for pc_idx in range(3):
    ax = axes[pc_idx]
    
    # Plot consensus pattern (reference)
    ax.plot(time_axis, consensus_pca[:, pc_idx], 'o-', color='black', 
           linewidth=3, markersize=6, label='Consensus', zorder=5)
    
    # Plot aligned patterns
    for i, pattern_data in enumerate(aligned_patterns_pca):
        ax.plot(time_axis, pattern_data['pca'][:, pc_idx], 'o-', 
               color=colors[i % len(colors)], alpha=0.7, linewidth=2,
               markersize=4, label=pattern_data['info'])
    
    # Add consensus annotations if available
    if consensus_annotations:
        for t, annotation in enumerate(consensus_annotations):
            if annotation == 'mitosis':
                ax.axvline(t, color='orange', alpha=0.5, linestyle='--', linewidth=1)
                ax.scatter(t, consensus_pca[t, pc_idx], c='orange', s=100, 
                          marker='s', edgecolor='black', zorder=6)
    
    ax.set_xlabel('Aligned Time')
    ax.set_ylabel(f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.2%})')
    ax.set_title(f'PC{pc_idx+1} Aligned Trajectories')
    ax.grid(True, alpha=0.3)
    
    # if pc_idx == 0:  # Only show legend on first subplot
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle(f'Top {top_n} Cells Aligned to Consensus Pattern\nPC Trajectories Over Time')
plt.tight_layout()
plt.show()

#%%
# Prototype video alignment based on DTW matches
from iohub import open_ome_zarr
from viscy.data.triplet import TripletDataset

z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)
channel_names = ["Phase3D"]
top_n = 20

positions = []
tracks_tables = []
images_plate = open_ome_zarr(data_path)

# Find the positions and tracks tables for the top n matches
top_n_matches_df = top_matches.head(top_n)
for _,pos in images_plate.positions():
    if pos.zgroup.name in top_n_matches_df["fov_name"].values:
        positions.append(pos)
        # filter the tracks per positition append to a list of dataframes
        tracks_df = cell_cycle_annotations_df[cell_cycle_annotations_df["fov_name"] == pos.zgroup.name]
        tracks_tables.append(tracks_df)
        
dataset = TripletDataset(
    positions=positions,
    tracks_tables=tracks_tables,
    channel_names=channel_names,
    initial_yx_patch_size=initial_yx_patch_size,
    z_range=z_range,
    anchor_transform=None,
    positive_transform=None,
    negative_transform=None,
    fit=False,
    predict_cells=False,
    include_fov_names=None,
    include_track_ids=None,
    time_interval=1,
    return_negative=False,
)

# %%
def get_candidate_sequences(dataset, candidates_df):
    """Get image sequences for candidate lineages."""
    import ast
    
    sequences = {}
    for idx, row in candidates_df.iterrows():
        fov_name = row['fov_name'] 
        track_ids = ast.literal_eval(row['track_ids']) if isinstance(row['track_ids'], str) else row['track_ids']
        
        # Find matching indices
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (anchor_row['fov_name'] == fov_name and anchor_row['track_id'] in track_ids):
                matching_indices.append(dataset_idx)
        
        if matching_indices:
            images = dataset.__getitems__(matching_indices)
            images.sort(key=lambda x: x['index']['t'])  # Sort by time
            sequences[idx] = {'images': images, 'row': row}
    
    return sequences

def align_sequences_by_warp_path(sequences, consensus_length):
    """Align sequences to consensus timeline using warp_path."""
    import ast
    
    aligned_sequences = {}
    for idx, seq_data in sequences.items():
        row = seq_data['row']
        images = seq_data['images']
        
        warp_path = ast.literal_eval(row['warp_path']) if isinstance(row['warp_path'], str) else row['warp_path']
        start_time = int(row['start_timepoint']) if not pd.isna(row['start_timepoint']) else 0
        
        # Create mapping from actual time to image
        time_to_image = {img['index']['t']: img for img in images}
        
        # Create warp_path mapping: reference_idx -> query_idx
        ref_to_query = {ref_idx: query_idx for ref_idx, query_idx in warp_path}
        
        # Create aligned sequence with consensus length
        aligned_images = [None] * consensus_length
        
        for ref_idx in range(consensus_length):
            if ref_idx in ref_to_query:
                query_idx = ref_to_query[ref_idx]
                query_time = start_time + query_idx
                
                if query_time in time_to_image:
                    aligned_images[ref_idx] = time_to_image[query_time]
                else:
                    # Find closest available time
                    available_times = list(time_to_image.keys())
                    closest_time = min(available_times, key=lambda x: abs(x - query_time))
                    aligned_images[ref_idx] = time_to_image[closest_time]
        
        # Fill any None values with nearest neighbor
        for i in range(consensus_length):
            if aligned_images[i] is None:
                # Find nearest non-None image
                for offset in range(1, consensus_length):
                    for direction in [-1, 1]:
                        neighbor_idx = i + direction * offset
                        if 0 <= neighbor_idx < consensus_length and aligned_images[neighbor_idx] is not None:
                            aligned_images[i] = aligned_images[neighbor_idx]
                            break
                    if aligned_images[i] is not None:
                        break
        
        aligned_sequences[idx] = {
            'aligned_images': aligned_images,
            'metadata': {
                'fov_name': row['fov_name'],
                'track_ids': ast.literal_eval(row['track_ids']) if isinstance(row['track_ids'], str) else row['track_ids'],
                'distance': row['distance'],
                'consensus_length': consensus_length
            }
        }
    
    return aligned_sequences

# Get sequences for top candidates
sequences = get_candidate_sequences(dataset, top_n_matches_df)
aligned_sequences = align_sequences_by_warp_path(sequences, len(consensus_lineage))

logger.info(f"Retrieved {len(aligned_sequences)} aligned sequences")
for idx, seq in aligned_sequences.items():
    meta = seq['metadata']
    logger.info(f"Sequence {idx}: FOV {meta['fov_name']}, {meta['consensus_length']} aligned images, distance={meta['distance']:.3f}")

# %%
# Load aligned sequences into napari
if NAPARI and len(aligned_sequences) > 0:
    import numpy as np
    
    for idx, seq_data in aligned_sequences.items():
        aligned_images = seq_data['aligned_images']
        meta = seq_data['metadata']
        
        if len(aligned_images) == 0:
            continue
            
        # Stack images into time series (T, C, Z, Y, X)
        image_stack = []
        for img_sample in aligned_images:
            img_tensor = img_sample['anchor']  # Shape should be (Z, C, Y, X)
            img_np = img_tensor.cpu().numpy()
            image_stack.append(img_np)
        
        if len(image_stack) > 0:
            # Stack into (T, Z, C, Y, X) or (T, C, Z, Y, X)
            time_series = np.stack(image_stack, axis=0)
            
            # Add to napari viewer  
            layer_name = f"Seq_{idx}_FOV_{meta['fov_name']}_dist_{meta['distance']:.3f}"
            viewer.add_image(
                time_series,
                name=layer_name,
                contrast_limits=(time_series.min(), time_series.max()),
            )
            logger.info(f"Added {layer_name} with shape {time_series.shape}")

# %%
