#%%
import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.data.triplet import TripletDataset
from viscy.representation.embedding_writer import (
    read_embedding_dataset,
)
from viscy.representation.pseudotime import CytoDtw

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
    #     {'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/2-assemble/2024_11_21_A549_TOMM20_DENV.zarr",
    #     'fov_name': "/C/2/001000",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/phase_160patch_104ckpt_ver3max.zarr",
    #     'tracks_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/2-assemble/tracking.zarr",
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
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/cell_cycle/output"
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
features_path = cell_cycle_annotations_denv["features_path"]

# Unified dataframe that adds the annotations to the features
embeddings = read_embedding_dataset(features_path)
annotations_df = pd.read_csv(cell_cycle_annotations_denv["annotations_path"])

# Instantiate the CytoDtw object
cytodtw=CytoDtw(embeddings,annotations_df)
feature_df=cytodtw.annotations_df

min_timepoints = 7
filtered_lineages = cytodtw.get_lineages(min_timepoints)
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])
logger.info(f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints")

#%%
n_timepoints_before = min_timepoints//2
n_timepoints_after = min_timepoints//2
valid_annotated_examples=[
    {
    'fov_name': "/A/2/001001",
    'track_id': [136,137],
    'timepoints': (43-n_timepoints_before, 43+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
# {
#     'fov_name': "/C/1/001000",
#     'track_id': [47,48],
#     'timepoints': (45-n_timepoints_before, 45+n_timepoints_after+1),
#     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
#     'weight': 1.0
# },
{
    'fov_name': "/C/1/000000",
    'track_id': [118,119],
    'timepoints': (27-n_timepoints_before, 27+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
# {
#     'fov_name': "/C/1/001000",
#     'track_id': [59,60],
#     'timepoints': (52-n_timepoints_before, 52+n_timepoints_after+1),
#     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
#     'weight': 1.0
# },
{
    'fov_name': "/C/1/001001",
    'track_id': [93,94],
    'timepoints': (29-n_timepoints_before, 29+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},

]
#%% 
# Extract all reference patterns
patterns = []
pattern_info = []
REFERENCE_TYPE = "features"
DTW_CONSTRAINT_TYPE="sakoe_chiba"
DTW_BAND_WIDTH_RATIO=0.3

for i, example in enumerate(valid_annotated_examples):
    pattern = cytodtw.get_reference_pattern(
        fov_name=example['fov_name'],
        track_id=example['track_id'],
        timepoints=example['timepoints'],
        reference_type=REFERENCE_TYPE,
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

#%%
# Plot the sample patterns

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
        annotated_samples=valid_annotated_examples,
        reference_selection="median_length", 
        aggregation_method="median",
        reference_type=REFERENCE_TYPE,
    )
    consensus_lineage = consensus_result['pattern']
    consensus_annotations = consensus_result.get('annotations', None)
    consensus_metadata = consensus_result['metadata']
    
    logger.info(f"Created consensus pattern with shape: {consensus_lineage.shape}")
    logger.info(f"Consensus method: {consensus_metadata['aggregation_method']}")
    logger.info(f"Reference pattern: {consensus_metadata['reference_pattern']}")
    if consensus_annotations:
        logger.info(f"Consensus annotations length: {len(consensus_annotations)}")
else:
    logger.warning("Not enough valid lineages found to create consensus pattern")

#%% 
# Perform DTW analysis for each embedding method
alignment_results = {}
top_n = 30

name = "consensus_lineage"
# Find pattern matches
matches = cytodtw.get_matches(
    reference_pattern=consensus_lineage,
    lineages=filtered_lineages.to_numpy(),
    window_step=1,
    num_candidates=top_n,
    method="bernd_clifford",
    metric="cosine",
    save_path=output_root / f"{name}_matching_lineages_cosine.csv",
    reference_type=REFERENCE_TYPE,
    constraint_type=DTW_CONSTRAINT_TYPE,
    band_width_ratio=DTW_BAND_WIDTH_RATIO
)

alignment_results[name] = matches
logger.info(f"Found {len(matches)} matches for {name}")
#%%
# Save matches
print(f'Saving matches to {output_root / f"{name}_matching_lineages_cosine.csv"}')
# cytodtw.save_consensus(output_root / f"{name}_consensus_lineage.pkl")
# Add consensus path to the df all rows
# Add a new column 'consensus_path' to the matches DataFrame, with the same value for all rows.
# This is useful for downstream analysis to keep track of the consensus pattern used for matching.
# Reference: pandas.DataFrame.assign
matches['consensus_path'] = str(output_root / f"{name}_consensus_lineage.pkl")
# Save the pkl
cytodtw.save_consensus(output_root / f"{name}_consensus_lineage.pkl")

matches.to_csv(output_root / f"{name}_matching_lineages_cosine.csv", index=False)
#%%
top_matches = matches.head(top_n)

all_patterns_for_pca = [consensus_lineage]
all_aligned_patterns = []
for idx, row in top_matches.iterrows():
    fov_name = row['fov_name']
    track_ids = ast.literal_eval(row['track_ids']) if isinstance(row['track_ids'], str) else row['track_ids']
    warp_path = ast.literal_eval(row['warp_path']) if isinstance(row['warp_path'], str) else row['warp_path']
    start_time = row['start_track_timepoint']
    distance = row['distance']
    
    # Get lineage embeddings
    lineage_embeddings = []
    for track_id in track_ids:
        try:
            track_emb = cytodtw.embeddings.sel(sample=(fov_name, track_id))[REFERENCE_TYPE].values
            lineage_embeddings.append(track_emb)
        except KeyError:
            continue
    
    if not lineage_embeddings:
        continue
        
    lineage_embeddings = np.concatenate(lineage_embeddings, axis=0)
    
    # Create aligned pattern using warping path
    aligned_pattern = np.zeros_like(consensus_lineage)
    
    # Map each consensus timepoint to lineage timepoint via warping path
    ref_to_lineage = {}
    for ref_idx, query_t in warp_path:
        cumulative_idx = 0
        found_idx = None
        for track_id in track_ids:
            try:
                track_data = cytodtw.embeddings.sel(sample=(fov_name, track_id))
                track_t_values = track_data['t'].values
                if query_t in track_t_values:
                    track_relative_idx = np.where(track_t_values == query_t)[0][0]
                    found_idx = cumulative_idx + track_relative_idx
                    break
                cumulative_idx += len(track_t_values)
            except KeyError:
                continue
        
        if found_idx is not None and 0 <= found_idx < len(lineage_embeddings):
            ref_to_lineage[ref_idx] = found_idx
            aligned_pattern[ref_idx] = lineage_embeddings[found_idx]
    
    # Fill missing values with nearest neighbor
    for ref_idx in range(len(consensus_lineage)):
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
consensus_pca = pca_data[:len(consensus_lineage)]
aligned_patterns_pca = []
start_idx = len(consensus_lineage)
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
time_axis = np.arange(len(consensus_lineage))

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

plt.suptitle(f'Top {top_n} Cells Aligned to Consensus Pattern\nPC Trajectories Over Time')
plt.tight_layout()
plt.show()

#%%
# Prototype video alignment based on DTW matches

z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)
channel_names = ["Phase3D"]

positions = []
tracks_tables = []
images_plate = open_ome_zarr(data_path)

for _,pos in images_plate.positions():
    if pos.zgroup.name in top_matches["fov_name"].values:
        positions.append(pos)
        tracks_df = cytodtw.annotations_df[cytodtw.annotations_df["fov_name"] == pos.zgroup.name]
        tracks_tables.append(tracks_df)
        
dataset = TripletDataset(
    positions=positions,
    tracks_tables=tracks_tables,
    channel_names=channel_names,
    initial_yx_patch_size=initial_yx_patch_size,
    z_range=z_range,
    fit=False,
    predict_cells=False,
    include_fov_names=None,
    include_track_ids=None,
    time_interval=1,
    return_negative=False,
)

# %%
# Simplified sequence alignment using existing DTW results
def get_aligned_image_sequences(dataset, candidates_df):
    """Get image sequences aligned to consensus timeline using DTW warp paths."""
    import ast
    
    aligned_sequences = {}
    for idx, row in candidates_df.iterrows():
        fov_name = row['fov_name']
        track_ids = ast.literal_eval(row['track_ids']) if isinstance(row['track_ids'], str) else row['track_ids']
        warp_path = ast.literal_eval(row['warp_path']) if isinstance(row['warp_path'], str) else row['warp_path']
        start_time = int(row['start_track_timepoint']) if not pd.isna(row['start_track_timepoint']) else 0
        
        # Determine alignment length from warp path
        alignment_length = max(ref_idx for ref_idx, _ in warp_path) + 1
        
        # Find matching dataset indices
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (anchor_row['fov_name'] == fov_name and anchor_row['track_id'] in track_ids):
                matching_indices.append(dataset_idx)
        
        if not matching_indices:
            logger.warning(f"No matching indices found for FOV {fov_name}, tracks {track_ids}")
            continue
            
        # Get images and sort by time
        batch_data = dataset.__getitems__(matching_indices)
        
        # Extract individual images from batch
        images = []
        for i in range(len(matching_indices)):
            img_data = {
                'anchor': batch_data['anchor'][i],
                'index': batch_data['index'][i]
            }
            images.append(img_data)
        
        images.sort(key=lambda x: x['index']['t'])
        time_to_image = {img['index']['t']: img for img in images}
        
        # Create warp_path mapping and align images
        # Note: query_idx is now actual t value, not relative index
        ref_to_query = {ref_idx: query_t for ref_idx, query_t in warp_path}
        aligned_images = [None] * alignment_length
        
        for ref_idx in range(alignment_length):
            if ref_idx in ref_to_query:
                query_time = ref_to_query[ref_idx]  # query_time is already actual t value
                if query_time in time_to_image:
                    aligned_images[ref_idx] = time_to_image[query_time]
                else:
                    # Find closest available time
                    available_times = list(time_to_image.keys())
                    if available_times:
                        closest_time = min(available_times, key=lambda x: abs(x - query_time))
                        aligned_images[ref_idx] = time_to_image[closest_time]
        
        # Fill None values with nearest neighbor
        for i in range(alignment_length):
            if aligned_images[i] is None:
                for offset in range(1, alignment_length):
                    for direction in [-1, 1]:
                        neighbor_idx = i + direction * offset
                        if 0 <= neighbor_idx < alignment_length and aligned_images[neighbor_idx] is not None:
                            aligned_images[i] = aligned_images[neighbor_idx]
                            break
                    if aligned_images[i] is not None:
                        break
        
        aligned_sequences[idx] = {
            'aligned_images': aligned_images,
            'metadata': {
                'fov_name': fov_name,
                'track_ids': track_ids,
                'distance': row['distance'],
                'alignment_length': alignment_length
            }
        }
    
    return aligned_sequences

# Get aligned sequences using consolidated function
aligned_sequences = get_aligned_image_sequences(dataset, top_matches)

logger.info(f"Retrieved {len(aligned_sequences)} aligned sequences")
for idx, seq in aligned_sequences.items():
    meta = seq['metadata']
    index=seq['aligned_images'][0]['index']
    logger.info(f"Track id {index['track_id']}: FOV {meta['fov_name']} aligned images, distance={meta['distance']:.3f}")

# %%
# Load aligned sequences into napari
if NAPARI and len(aligned_sequences) > 0:
    import numpy as np
    
    for idx, seq_data in aligned_sequences.items():
        aligned_images = seq_data['aligned_images']
        meta = seq_data['metadata']
        index=seq_data['aligned_images'][0]['index']
        
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
            layer_name = f"track_id_{index['track_id']}_FOV_{meta['fov_name']}_dist_{meta['distance']:.3f}"
            viewer.add_image(
                time_series,
                name=layer_name,
                contrast_limits=(time_series.min(), time_series.max()),
            )
            logger.info(f"Added {layer_name} with shape {time_series.shape}")


#%%
# Create concatenated aligned + unaligned sequences
def create_concatenated_aligned_sequences(cytodtw, top_matches, consensus_lineage, reference_type="features"):
    """
    Create sequences that concatenate:
    1. DTW-aligned portion (extracted from full lineage and aligned to consensus)
    2. Remaining unaligned timepoints from full lineage (in original temporal order)
    
    This properly aligns the DTW-matched subsequences and appends the rest.
    """
    import ast
    
    concatenated_sequences = {}
    
    for idx, match_row in top_matches.iterrows():
        fov_name = match_row['fov_name']
        track_ids = ast.literal_eval(match_row['track_ids']) if isinstance(match_row['track_ids'], str) else match_row['track_ids']
        warp_path = ast.literal_eval(match_row['warp_path']) if isinstance(match_row['warp_path'], str) else match_row['warp_path']
        
        # Get full lineage embeddings and timepoints
        full_lineage_embeddings = []
        full_timepoints = []
        timepoint_to_global_idx = {}
        
        global_idx = 0
        for track_id in track_ids:
            try:
                track_data = cytodtw.embeddings.sel(sample=(fov_name, track_id))
                track_embeddings = track_data[reference_type].values
                track_timepoints = track_data['t'].values
                
                full_lineage_embeddings.append(track_embeddings)
                full_timepoints.extend(track_timepoints)
                
                # Map timepoints to global indices
                for local_idx, t in enumerate(track_timepoints):
                    timepoint_to_global_idx[t] = global_idx + local_idx
                global_idx += len(track_embeddings)
                
            except KeyError:
                continue
        
        if not full_lineage_embeddings:
            continue
            
        # Concatenate all track embeddings
        full_lineage = np.vstack(full_lineage_embeddings)
        full_timepoints = np.array(sorted(set(full_timepoints)))
        
        # Step 1: Extract aligned portion using DTW warp path
        aligned_portion = np.zeros_like(consensus_lineage)
        aligned_timepoints = set()
        
        for consensus_idx, query_timepoint in warp_path:
            aligned_timepoints.add(query_timepoint)
            if query_timepoint in timepoint_to_global_idx:
                lineage_idx = timepoint_to_global_idx[query_timepoint]
                if 0 <= lineage_idx < len(full_lineage):
                    aligned_portion[consensus_idx] = full_lineage[lineage_idx]
        
        # Fill gaps in aligned portion with interpolation
        filled_indices = []
        for i in range(len(aligned_portion)):
            if any(np.array_equal(aligned_portion[i], full_lineage[timepoint_to_global_idx[t]]) 
                  for t in aligned_timepoints if t in timepoint_to_global_idx):
                filled_indices.append(i)
        
        if len(filled_indices) > 0:
            for i in range(len(aligned_portion)):
                if i not in filled_indices:
                    closest_idx = filled_indices[np.argmin(np.abs(np.array(filled_indices) - i))]
                    aligned_portion[i] = aligned_portion[closest_idx]
        
        # Step 2: Extract unaligned portion (timepoints NOT in DTW alignment)
        unaligned_indices = []
        unaligned_timepoints = []
        for t in full_timepoints:
            if t not in aligned_timepoints and t in timepoint_to_global_idx:
                unaligned_indices.append(timepoint_to_global_idx[t])
                unaligned_timepoints.append(t)
        
        # Sort to maintain temporal order
        sorted_pairs = sorted(zip(unaligned_indices, unaligned_timepoints), key=lambda x: x[1])
        unaligned_indices = [pair[0] for pair in sorted_pairs]
        unaligned_timepoints = [pair[1] for pair in sorted_pairs]
        
        # Extract unaligned embeddings
        if unaligned_indices:
            unaligned_portion = full_lineage[unaligned_indices]
        else:
            unaligned_portion = np.array([]).reshape(0, full_lineage.shape[1])
        
        # Step 3: Concatenate aligned + unaligned portions
        if unaligned_portion.size > 0:
            concatenated = np.vstack([aligned_portion, unaligned_portion])
            segment_boundaries = [len(aligned_portion)]
        else:
            concatenated = aligned_portion
            segment_boundaries = []
        
        concatenated_sequences[idx] = {
            'concatenated': concatenated,
            'aligned_portion': aligned_portion,
            'unaligned_portion': unaligned_portion,
            'aligned_length': len(aligned_portion),
            'unaligned_length': len(unaligned_indices),
            'segment_boundaries': segment_boundaries,
            'aligned_timepoints': sorted(aligned_timepoints),
            'unaligned_timepoints': unaligned_timepoints,
            'info': {
                'fov_name': fov_name,
                'track_ids': track_ids,
                'distance': match_row.get('distance', np.nan),
                'consensus_length': len(consensus_lineage),
                'full_lineage_length': len(full_lineage)
            }
        }
    
    return concatenated_sequences

# Create concatenated sequences
concat_sequences = create_concatenated_aligned_sequences(cytodtw, top_matches, consensus_lineage, REFERENCE_TYPE)

# Log concatenation details
for idx, seq_data in concat_sequences.items():
    info = seq_data['info']
    n_aligned = seq_data['aligned_length']
    n_unaligned = len(seq_data['concatenated']) - n_aligned
    logger.info(f"Track {info['track_ids'][0]}: Aligned={n_aligned} + Unaligned={n_unaligned} = Total={len(seq_data['concatenated'])}")
    logger.info(f"  Aligned timepoints: {seq_data['aligned_timepoints'][:5]}..." if len(seq_data['aligned_timepoints']) > 5 else f"  Aligned timepoints: {seq_data['aligned_timepoints']}")
    logger.info(f"  Unaligned timepoints: {seq_data['unaligned_timepoints'][:5]}..." if len(seq_data['unaligned_timepoints']) > 5 else f"  Unaligned timepoints: {seq_data['unaligned_timepoints']}")

# Prepare all data for PCA
all_data_for_pca = [consensus_lineage]
for seq_data in concat_sequences.values():
    all_data_for_pca.append(seq_data['concatenated'])

# Fit PCA on all concatenated data  
all_concat = np.vstack(all_data_for_pca)
scaler = StandardScaler()
scaled_all = scaler.fit_transform(all_concat) 
pca = PCA(n_components=3)
pca_all = pca.fit_transform(scaled_all)

# Split back to individual sequences
consensus_pca = pca_all[:len(consensus_lineage)]
start_idx = len(consensus_lineage)

sequences_pca = {}
for idx, seq_data in concat_sequences.items():
    seq_len = len(seq_data['concatenated'])
    seq_pca = pca_all[start_idx:start_idx + seq_len]
    
    sequences_pca[idx] = {
        'pca': seq_pca,
        'aligned_length': seq_data['aligned_length'],
        'segment_boundaries': seq_data['segment_boundaries'], 
        'info': seq_data['info']
    }
    start_idx += seq_len

# Visualization with offsets - rows for PC components
n_sequences = min(5, len(sequences_pca))
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
colors = ['purple', 'brown', 'pink', 'gray', 'cyan']

# Calculate offsets for each sequence
y_offset_step = 2.0  # Vertical separation between tracks

for pc_idx in range(3):
    ax = axes[pc_idx]
    
    # Plot consensus (no offset)
    consensus_time = np.arange(len(consensus_pca))
    ax.plot(consensus_time, consensus_pca[:, pc_idx], 'o-', 
           color='black', linewidth=4, markersize=8, 
           label='Consensus', alpha=0.9, zorder=5)
    
    # Add consensus annotations
    if consensus_annotations:
        for t, annotation in enumerate(consensus_annotations):
            if annotation == 'mitosis':
                ax.axvline(t, color='orange', alpha=0.7, 
                         linestyle='--', linewidth=2, zorder=1)
    
    # Plot each concatenated sequence with offset
    for plot_idx, (seq_idx, seq_data) in enumerate(list(sequences_pca.items())[:n_sequences]):
        if plot_idx >= len(colors):
            break
            
        y_offset = -(plot_idx + 1) * y_offset_step  # Negative offset downward
        seq_pca = seq_data['pca'][:, pc_idx] + y_offset
        aligned_length = seq_data['aligned_length']
        boundaries = seq_data['segment_boundaries']
        info = seq_data['info']
        
        time_axis = np.arange(len(seq_pca))
        
        # Plot full concatenated sequence
        ax.plot(time_axis, seq_pca, '.-', 
               color=colors[plot_idx], linewidth=2, markersize=4,
               alpha=0.8, label=f'Track {info["track_ids"][0]} (d={info["distance"]:.3f})')
        
        # Highlight aligned portion (anchor point) - first part of concatenated sequence
        aligned_start = 0
        aligned_end = aligned_length
        aligned_time = time_axis[aligned_start:aligned_end]
        aligned_values = seq_pca[aligned_start:aligned_end]
        
        ax.plot(aligned_time, aligned_values, 's-',
               color=colors[plot_idx], linewidth=5, markersize=8, 
               alpha=0.9, zorder=4, label='DTW Aligned' if plot_idx == 0 else "")
        
        # Highlight unaligned portion - second part of concatenated sequence
        if len(boundaries) > 0:
            unaligned_start = boundaries[0]
            unaligned_end = len(seq_pca)
            unaligned_time = time_axis[unaligned_start:unaligned_end]
            unaligned_values = seq_pca[unaligned_start:unaligned_end]
            
            ax.plot(unaligned_time, unaligned_values, 'o-',
                   color=colors[plot_idx], linewidth=2, markersize=3,
                   alpha=0.6, zorder=3, label='Unaligned' if plot_idx == 0 else "")
        
        # Mark segment boundaries
        for boundary in boundaries:
            ax.axvline(boundary, color=colors[plot_idx], 
                     alpha=0.5, linestyle=':', linewidth=1)
        
        # Add track label with alignment info
        mid_point = len(time_axis) // 2
        track_id = info["track_ids"][0]
        n_aligned = seq_data['aligned_length'] 
        n_total = len(seq_data['pca'])
        n_unaligned = n_total - n_aligned
        ax.text(mid_point, seq_pca[mid_point] - 0.5, 
               f'Track {track_id}\nA:{n_aligned} U:{n_unaligned}', 
               ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[plot_idx], alpha=0.3))
    
    ax.set_xlabel('Concatenated Time: [DTW Aligned] + [Remaining Unaligned]')
    ax.set_ylabel(f'PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.2%}) + Offset')
    ax.set_title(f'PC{pc_idx+1}: DTW Aligned Anchors → Unaligned Continuations')
    ax.grid(True, alpha=0.3)
    
    if pc_idx == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('DTW Partial Alignment Strategy: Anchor Points + Unperturbed Continuations\n'
            'Orange squares = DTW-aligned cell division patterns, Circles = Remaining temporal trajectory',
            fontsize=14)
plt.tight_layout()
plt.show()

#%%
