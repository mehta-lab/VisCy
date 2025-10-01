#%%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_zarr
from iohub import open_ome_zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from viscy.data.triplet import TripletDataset
from viscy.representation.pseudotime import (
    CytoDtw,
    align_embedding_patterns,
    get_aligned_image_sequences,
)

#%%
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

# File paths for infection state analysis
sensor_annotations_dict={
    'denv': {
        'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/train-test/2024_11_21_A549_TOMM20_DENV.zarr",
        'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
        'features_path': "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/phase_160patch_104ckpt_ver3max.zarr",
    },
    # 'zikv': {
    #     'data_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr",
    #     'annotations_path': "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/0-annotations/track_cell_state_annotation.csv",
    #     'features_path': "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/cell_cycle/output/phase_160patch_104ckpt_ver3max.anndata",
    # },
}

# consensus_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output/consensus_lineage_consensus_lineage.pkl"
consensus_path = None
ALIGN_TYPE = "cell_division"  # Options: "cell_division" or "infection_state"

output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/infection_state/output"
)
output_root.mkdir(parents=True, exist_ok=True)

#%%
color_dict = {
    "uninfected": "blue",
    "infected": "orange",
}
ANNOTATION_TYPE = "predicted_infection_state"

for key in sensor_annotations_dict.keys():
    data_path = sensor_annotations_dict[key]['data_path']
    annotations_path = sensor_annotations_dict[key]['annotations_path']
    features_path = sensor_annotations_dict[key]['features_path']
    logger.info(f"Processing dataset: {key}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Annotations path: {annotations_path}")
    logger.info(f"Features path: {features_path}")
    break

# Load AnnData directly
adata = read_zarr(features_path)
print("Loaded AnnData with shape:", adata.shape)
print("Available columns:", adata.obs.columns.tolist())

# Instantiate the CytoDtw object with AnnData
cytodtw = CytoDtw(adata)
feature_df = cytodtw.adata.obs

min_timepoints = 50
filtered_lineages = cytodtw.get_lineages(min_timepoints)
filtered_lineages = pd.DataFrame(filtered_lineages, columns=["fov_name", "track_id"])
logger.info(f"Found {len(filtered_lineages)} lineages with at least {min_timepoints} timepoints")

#%%
n_timepoints_before = min_timepoints//2
n_timepoints_after = min_timepoints//2

cell_div_annotations = [
#     {
#     'fov_name': "C/2/001000",
#     'track_id': [39,40],
#     'timepoints': (5-n_timepoints_before, 5+n_timepoints_after+1),
#     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
#     'weight': 1.0
# },
{
    'fov_name': "C/2/001001",
    'track_id': [20,21],
    'timepoints': (83-n_timepoints_before, 83+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
{
    'fov_name': "B/3/001000",
    'track_id': [71,72],
    'timepoints': (44-n_timepoints_before, 44+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
{
    'fov_name': "B/2/000001",
    'track_id': [147,148],
    'timepoints': (59-n_timepoints_before, 59+n_timepoints_after+1),
    'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
    'weight': 1.0
},
# {
#     'fov_name': "B/1/000001",
#     'track_id': [49,50],
#     'timepoints': (44-n_timepoints_before, 44+n_timepoints_after+1),
#     'annotations': ["interphase"] * (n_timepoints_before) + ["mitosis"] + ["interphase"] * (n_timepoints_after-1),
#     'weight': 1.0
# },



]
infection_annotations=[
    {
        'fov_name': "C/2/001000",
        'track_id': [45],
        'timepoints': (31-n_timepoints_before, 31+n_timepoints_after+1),
        'annotations': ["uinfected"] * (n_timepoints_before) + ["infected"] + ["uninfected"] * (n_timepoints_after-1),
        'weight': 1.0
    },
    {
        'fov_name': "C/2/001000",
        'track_id': [66],
        'timepoints': (19-n_timepoints_before, 19+n_timepoints_after+1),
        'annotations': ["uninfected"] * (n_timepoints_before) + ["infected"] + ["uninfected"] * (n_timepoints_after-1),
        'weight': 1.0
    },
    {
        'fov_name': "C/2/001000",
        'track_id': [54],
        'timepoints': (27-n_timepoints_before, 27+n_timepoints_after+1),
        'annotations': ["uninfected"] * (n_timepoints_before) + ["infected"] + ["uninfected"] * (n_timepoints_after-1),
        'weight': 1.0
    },
    {
        'fov_name': "C/2/001000",
        'track_id': [53],
        'timepoints': (21-n_timepoints_before, 21+n_timepoints_after+1),
        'annotations': ["uninfected"] * (n_timepoints_before) + ["infected"] + ["uninfected"] * (n_timepoints_after-1),
        'weight': 1.0
    },
]

if ALIGN_TYPE == "infection_state":
    aligning_annotations = infection_annotations
else:
    aligning_annotations = cell_div_annotations


#%% 
# Extract all reference patterns
patterns = []
pattern_info = []
REFERENCE_TYPE = "features"
DTW_CONSTRAINT_TYPE="sakoe_chiba"
DTW_BAND_WIDTH_RATIO=0.3

for i, example in enumerate(aligning_annotations):
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
scaler = StandardScaler()
scaled_patterns = scaler.fit_transform(all_patterns_concat)
pca = PCA(n_components=3)
pca.fit(scaled_patterns)

n_patterns = len(patterns)
fig, axes = plt.subplots(n_patterns, 3, figsize=(12, 3*n_patterns))
if n_patterns == 1:
    axes = axes.reshape(1, -1)

for i, (pattern, info) in enumerate(zip(patterns, pattern_info)):
    scaled_pattern = scaler.transform(pattern)
    pc_pattern = pca.transform(scaled_pattern)
    time_axis = np.arange(len(pattern))
    
    for pc_idx in range(3):
        ax = axes[i, pc_idx]
        
        ax.plot(time_axis, pc_pattern[:, pc_idx], 'o-', color='blue', linewidth=2, markersize=4)
        
        annotations = info['annotations']
        for t, annotation in enumerate(annotations):
            if annotation == 'mitosis':
                ax.axvline(t, color='orange', alpha=0.7, linestyle='--', linewidth=2)
                ax.scatter(t, pc_pattern[t, pc_idx], c='orange', s=50, zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'PC{pc_idx+1}')
        ax.set_title(f'Pattern {i+1}: FOV {info["fov_name"]}, Tracks {info["track_id"]}\nPC{pc_idx+1} over time')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Create consensus pattern
if consensus_path is not None and Path(consensus_path).exists():
    consensus_result = np.load(consensus_path, allow_pickle=True)
    cytodtw.consensus_data = consensus_result
else:
    consensus_result = cytodtw.create_consensus_reference_pattern(
        annotated_samples=aligning_annotations,
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



#%%
# Plot all aligned consensus patterns together to validate alignment
# The patterns need to be DTW-aligned to the consensus for proper visualization
# We'll align each pattern to the consensus using the same method used internally

aligned_patterns_list = []
for i, example in enumerate(aligning_annotations):
    # Extract pattern
    pattern = cytodtw.get_reference_pattern(
        fov_name=example['fov_name'],
        track_id=example['track_id'],
        timepoints=example['timepoints'],
        reference_type=REFERENCE_TYPE,
    )

    # Align to consensus
    if len(pattern) == len(consensus_lineage):
        # Already same length, likely the reference pattern
        aligned_patterns_list.append(pattern)
    else:
        # Align to consensus
        alignment_result = align_embedding_patterns(
            query_pattern=pattern,
            reference_pattern=consensus_lineage,
            metric="cosine",
            constraint_type=DTW_CONSTRAINT_TYPE,
            band_width_ratio=DTW_BAND_WIDTH_RATIO,
        )
        aligned_patterns_list.append(alignment_result['pattern'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for pc_idx in range(3):
    ax = axes[pc_idx]

    # Transform each aligned pattern to PC space and plot
    for i, pattern in enumerate(aligned_patterns_list):
        scaled_ref = scaler.transform(pattern)
        pc_ref = pca.transform(scaled_ref)

        time_axis = np.arange(len(pc_ref))
        ax.plot(time_axis, pc_ref[:, pc_idx], 'o-',
               label=f'Ref {i+1}', alpha=0.7, linewidth=2, markersize=4)

    # Overlay consensus pattern
    scaled_consensus = scaler.transform(consensus_lineage)
    pc_consensus = pca.transform(scaled_consensus)
    time_axis = np.arange(len(pc_consensus))
    ax.plot(time_axis, pc_consensus[:, pc_idx], 's-',
           color='black', linewidth=3, markersize=6,
           label='Consensus', zorder=10)

    # Mark infection timepoint if available
    if consensus_annotations:
        for t, annotation in enumerate(consensus_annotations):
            if annotation == 'infected':
                ax.axvline(t, color='orange', alpha=0.7,
                         linestyle='--', linewidth=2)

    ax.set_xlabel('Aligned Time')
    ax.set_ylabel(f'PC{pc_idx+1}')
    ax.set_title(f'PC{pc_idx+1}: All DTW-Aligned References + Consensus')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Consensus Validation: DTW-Aligned References + Computed Consensus',
            fontsize=14)
plt.tight_layout()
plt.show()
logger.info("Plotted DTW-aligned consensus patterns for validation")

#%%
# Perform DTW analysis for each embedding method
alignment_results = {}
top_n = 30

name = "consensus_lineage"
consensus_lineage = cytodtw.consensus_data['pattern']
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
matches['consensus_path'] = str(output_root / f"{name}_consensus_lineage.pkl")
cytodtw.save_consensus(output_root / f"{name}_consensus_lineage.pkl")
matches.to_csv(output_root / f"{name}_matching_lineages_cosine.csv", index=False)
#%%
top_matches = matches.head(top_n)

# Use the new alignment dataframe method instead of manual alignment
alignment_df = cytodtw.create_alignment_dataframe(top_matches, consensus_lineage, alignment_name="cell_division", reference_type=REFERENCE_TYPE)

logger.info(f"Enhanced DataFrame created with {len(alignment_df)} rows")
logger.info(f"Lineages: {alignment_df['lineage_id'].nunique()} (including consensus)")
logger.info(f"Cell division aligned timepoints: {alignment_df['dtw_cell_division_aligned'].sum()}/{len(alignment_df)} ({100*alignment_df['dtw_cell_division_aligned'].mean():.1f}%)")

# Check for duplicate (fov_name, track_id, t) combinations
duplicates = alignment_df[alignment_df.duplicated(subset=['fov_name', 'track_id', 't'], keep=False)]
if len(duplicates) > 0:
    logger.warning(f"Found {len(duplicates)} duplicate (fov_name, track_id, t) combinations!")
    logger.warning("Sample duplicates:")
    print(duplicates[['fov_name', 'track_id', 't', 'lineage_id', 'x', 'y']].head(10))
else:
    logger.info("✓ All (fov_name, track_id, t) combinations are unique")

# PCA plotting and alignment visualization is now handled by the enhanced alignment dataframe method
logger.info("Cell division consensus analysis completed successfully!")
print(f"Enhanced DataFrame columns: {alignment_df.columns.tolist()}")

#%%
# Prototype video alignment based on DTW matches

z_range = slice(0, 1)
initial_yx_patch_size = (192, 192)

positions = []
tracks_tables = []
images_plate = open_ome_zarr(data_path)

# Load matching positions
print(f"Loading positions for {len(top_matches)} FOV matches...")
matches_found = 0
for _, pos in images_plate.positions():
    pos_name = pos.zgroup.name
    pos_normalized = pos_name.lstrip('/')
    
    if pos_normalized in top_matches['fov_name'].values:
        positions.append(pos)
        matches_found += 1
        
        # Get ALL tracks for this FOV to ensure TripletDataset has complete access
        tracks_df = cytodtw.adata.obs[cytodtw.adata.obs["fov_name"] == pos_normalized].copy()
        
        if len(tracks_df) > 0:
            tracks_df = tracks_df.dropna(subset=['x', 'y'])
            tracks_df['x'] = tracks_df['x'].astype(int)
            tracks_df['y'] = tracks_df['y'].astype(int)
            tracks_tables.append(tracks_df)
            
            if matches_found == 1:
                processing_channels = pos.channel_names

print(f"Loaded {matches_found} positions with {sum(len(df) for df in tracks_tables)} total tracks")

# Create TripletDataset if we have valid positions
if len(positions) > 0 and len(tracks_tables) > 0:
    if 'processing_channels' not in locals():
        processing_channels = positions[0].channel_names
    
    # Use all three channels for overlay visualization
    selected_channels = processing_channels  # Use all available channels
    print(f"Creating TripletDataset with {len(selected_channels)} channels: {selected_channels}")
    
    dataset = TripletDataset(
        positions=positions,
        tracks_tables=tracks_tables,
        channel_names=selected_channels,
        initial_yx_patch_size=initial_yx_patch_size,
        z_range=z_range,
        fit=False,
        predict_cells=False,
        include_fov_names=None,
        include_track_ids=None,
        time_interval=1,
        return_negative=False,
    )
    print(f"TripletDataset created with {len(dataset.valid_anchors)} valid anchors")
else:
    print("Cannot create TripletDataset - no valid positions or tracks")
    dataset = None
        
# %%

# Get aligned sequences using consolidated function
if dataset is not None:
    def load_images_from_triplet_dataset(fov_name, track_ids):
        """Load images from TripletDataset for given FOV and track IDs."""
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (anchor_row['fov_name'] == fov_name and anchor_row['track_id'] in track_ids):
                matching_indices.append(dataset_idx)

        if not matching_indices:
            logger.warning(f"No matching indices found for FOV {fov_name}, tracks {track_ids}")
            return {}

        # Get images and create time mapping
        batch_data = dataset.__getitems__(matching_indices)
        images = []
        for i in range(len(matching_indices)):
            img_data = {
                'anchor': batch_data['anchor'][i],
                'index': batch_data['index'][i]
            }
            images.append(img_data)

        images.sort(key=lambda x: x['index']['t'])
        return {img['index']['t']: img for img in images}

    aligned_sequences = get_aligned_image_sequences(
        cytodtw_instance=cytodtw,
        df=top_matches,
        alignment_name="cell_division",
        image_loader_fn=load_images_from_triplet_dataset,
        max_lineages=None
    )
else:
    aligned_sequences = {}

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
# Enhanced DataFrame was already created above with PCA plotting - skip duplicate
logger.info(f"Cell division aligned timepoints: {alignment_df['dtw_cell_division_aligned'].sum()}/{len(alignment_df)} ({100*alignment_df['dtw_cell_division_aligned'].mean():.1f}%)")
logger.info(f"Columns: {list(alignment_df.columns)}")

# Show sample of the enhanced DataFrame
print("\nSample of enhanced DataFrame:")
sample_df = alignment_df[alignment_df['lineage_id'] != -1].head(10)
display_cols = ['lineage_id', 'track_id', 't', 'dtw_cell_division_aligned', 'dtw_cell_division_consensus_mapping', 'PC1']
print(sample_df[display_cols].to_string())

#%%

# Clean function that works directly with enhanced DataFrame
def plot_concatenated_from_dataframe(df, alignment_name="cell_division",
                                    feature_columns=['PC1', 'PC2', 'PC3'],
                                    max_lineages=5, y_offset_step=2.0,
                                    aligned_scale=1.0, unaligned_scale=1.0,
                                    remove_outliers=False, outlier_percentile=(1, 99)):
    """
    Plot concatenated [DTW-aligned portion] + [unaligned portion] sequences
    using ONLY the enhanced DataFrame and alignment information stored in it.

    This function reconstructs the aligned portions using the consensus mapping
    information already stored in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced DataFrame with alignment information
    alignment_name : str
        Name of alignment to plot (e.g., "cell_division")
    feature_columns : list
        Feature columns to plot
    max_lineages : int
        Maximum number of lineages to display
    y_offset_step : float
        Vertical separation between lineages
    aligned_scale : float
        Scale factor for DTW-aligned portions (line width & marker size)
    unaligned_scale : float
        Scale factor for unaligned portions (line width & marker size)
    remove_outliers : bool
        Whether to clip outlier values for better visualization
    outlier_percentile : tuple
        (lower, upper) percentile bounds for clipping (default: 1st-99th percentile)
    """
    import matplotlib.pyplot as plt
    
    # Calculate line widths and marker sizes based on separate scale factors
    aligned_linewidth = 5 * aligned_scale
    unaligned_linewidth = 2 * unaligned_scale
    aligned_markersize = 8 * aligned_scale
    unaligned_markersize = 4 * unaligned_scale
    
    # Dynamic column names based on alignment_name
    aligned_col = f'dtw_{alignment_name}_aligned'
    mapping_col = f'dtw_{alignment_name}_consensus_mapping'
    distance_col = f'dtw_{alignment_name}_distance'
    
    if aligned_col not in df.columns:
        logger.error(f"Alignment '{alignment_name}' not found in DataFrame")
        return
    
    consensus_df = df[df['lineage_id'] == -1].sort_values('t').copy()
    
    if consensus_df.empty:
        logger.error("No consensus found in DataFrame")
        return

    concatenated_seqs = cytodtw.get_concatenated_sequences(
        df=df,
        alignment_name=alignment_name,
        feature_columns=feature_columns,
        max_lineages=max_lineages
    )

    if not concatenated_seqs:
        logger.error(f"No concatenated sequences found")
        return

    # Prepare data for plotting
    concatenated_lineages = {}
    for lineage_id, seq_data in concatenated_seqs.items():
        aligned_features = seq_data['aligned_data']['features']
        unaligned_features = seq_data['unaligned_data'].get('features', {})

        # Concatenate arrays
        concatenated_arrays = {}
        for col in feature_columns:
            if col in unaligned_features and len(unaligned_features[col]) > 0:
                concatenated_arrays[col] = np.concatenate([aligned_features[col], unaligned_features[col]])
            else:
                concatenated_arrays[col] = aligned_features[col]

        concatenated_lineages[lineage_id] = {
            'concatenated': concatenated_arrays,
            'aligned_length': seq_data['aligned_data']['length'],
            'unaligned_length': seq_data['unaligned_data']['length'],
            'dtw_distance': seq_data['metadata']['dtw_distance'],
            'fov_name': seq_data['metadata']['fov_name'],
            'track_ids': seq_data['metadata']['track_ids']
        }

    # Compute outlier bounds per feature if requested
    outlier_bounds = {}
    if remove_outliers:
        for feat_col in feature_columns:
            all_values = []
            # Collect all values for this feature (consensus + all lineages)
            all_values.extend(consensus_df[feat_col].values)
            for data in concatenated_lineages.values():
                all_values.extend(data['concatenated'][feat_col])

            all_values = np.array(all_values)
            all_values = all_values[~np.isnan(all_values)]  # Remove NaNs

            if len(all_values) > 0:
                lower_bound = np.percentile(all_values, outlier_percentile[0])
                upper_bound = np.percentile(all_values, outlier_percentile[1])
                outlier_bounds[feat_col] = (lower_bound, upper_bound)
                logger.info(f"{feat_col}: clipping to [{lower_bound:.3f}, {upper_bound:.3f}]")

    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    cmap = plt.cm.get_cmap('tab10' if len(concatenated_lineages) <= 10 else 'tab20' if len(concatenated_lineages) <= 20 else 'hsv')
    colors = [cmap(i / max(len(concatenated_lineages), 1)) for i in range(len(concatenated_lineages))]
    
    for feat_idx, feat_col in enumerate(feature_columns):
        ax = axes[feat_idx]
        
        # Plot consensus (no offset)
        consensus_values = consensus_df[feat_col].values.copy()
        if remove_outliers and feat_col in outlier_bounds:
            lower, upper = outlier_bounds[feat_col]
            consensus_values = np.clip(consensus_values, lower, upper)
        consensus_time = np.arange(len(consensus_values))
        ax.plot(consensus_time, consensus_values, 'o-',
               color='black', linewidth=4, markersize=8,
               label=f'Consensus ({alignment_name})', alpha=0.9, zorder=5)
        
        # Add consensus annotations if available
        if alignment_name == "cell_division" and 'consensus_annotations' in globals():
            for t, annotation in enumerate(consensus_annotations):
                if annotation == 'mitosis':
                    ax.axvline(t, color='orange', alpha=0.7, 
                             linestyle='--', linewidth=2, zorder=1)
        
        # Plot each concatenated lineage
        for lineage_idx, (lineage_id, data) in enumerate(concatenated_lineages.items()):
            # Remove the color limit - now we have enough colors
                
            y_offset = -(lineage_idx + 1) * y_offset_step
            color = colors[lineage_idx]
            
            # Get concatenated sequence values
            concat_values = data['concatenated'][feat_col].copy()
            if remove_outliers and feat_col in outlier_bounds:
                lower, upper = outlier_bounds[feat_col]
                concat_values = np.clip(concat_values, lower, upper)
            concat_values = concat_values + y_offset
            time_axis = np.arange(len(concat_values))
            
            # Plot full concatenated sequence
            # Format track_ids for display
            track_id_str = ','.join(map(str, data['track_ids']))
            ax.plot(time_axis, concat_values, '.-',
                   color=color, linewidth=unaligned_linewidth, markersize=unaligned_markersize, alpha=0.8,
                   label=f'{data["fov_name"]}, track={track_id_str} (d={data["dtw_distance"]:.3f})')
            
            # Highlight aligned portion with thicker line
            aligned_length = data['aligned_length']
            if aligned_length > 0:
                aligned_time = time_axis[:aligned_length]
                aligned_values = concat_values[:aligned_length]
                
                ax.plot(aligned_time, aligned_values, 's-',
                       color=color, linewidth=aligned_linewidth, markersize=aligned_markersize, 
                       alpha=0.9, zorder=4)
            
            # Mark boundary between aligned and unaligned
            if aligned_length > 0 and aligned_length < len(concat_values):
                ax.axvline(aligned_length, color=color, alpha=0.5, 
                         linestyle=':', linewidth=1)
        
        # Formatting
        ax.set_xlabel('Concatenated Time: [DTW Aligned] + [Unaligned Continuation]')
        ax.set_ylabel(f'{feat_col} (vertically separated)')
        ax.set_title(f'{feat_col}: Concatenated {alignment_name.replace("_", " ").title()} Trajectories')
        ax.grid(True, alpha=0.3)
        
        if feat_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    logger.debug(f"\nConcatenated alignment summary for '{alignment_name}':")
    logger.debug(f"Processed {len(concatenated_lineages)} lineages")
    for lineage_id, data in concatenated_lineages.items():
        logger.debug(f"  Lineage {lineage_id}: A={data['aligned_length']} + U={data['unaligned_length']} = {data['aligned_length'] + data['unaligned_length']}, d={data['dtw_distance']:.3f}")

# Plot using the CytoDtw method
cytodtw.plot_individual_lineages(
    alignment_df,
    alignment_name="cell_division",
    feature_columns=['PC1','PC2','PC3'],
    max_lineages=15,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0
)



#%%
# Create concatenated image sequences using the DataFrame alignment information
# Filter for infection wells only
fov_name_patterns = ['B/3', 'consensus','B/1']
filtered_alignment_df = alignment_df[alignment_df['fov_name'].str.contains('|'.join(fov_name_patterns))]

if dataset is not None:
    # Define TripletDataset-specific image loader
    def load_images_from_triplet_dataset(fov_name, track_ids):
        """Load images from TripletDataset for given FOV and track IDs."""
        matching_indices = []
        for dataset_idx in range(len(dataset.valid_anchors)):
            anchor_row = dataset.valid_anchors.iloc[dataset_idx]
            if (anchor_row['fov_name'] == fov_name and anchor_row['track_id'] in track_ids):
                matching_indices.append(dataset_idx)

        if not matching_indices:
            logger.warning(f"No matching indices found for FOV {fov_name}, tracks {track_ids}")
            return {}

        # Get images and create time mapping
        batch_data = dataset.__getitems__(matching_indices)
        images = []
        for i in range(len(matching_indices)):
            img_data = {
                'anchor': batch_data['anchor'][i],
                'index': batch_data['index'][i]
            }
            images.append(img_data)

        images.sort(key=lambda x: x['index']['t'])
        return {img['index']['t']: img for img in images}

    concatenated_image_sequences = get_aligned_image_sequences(
        cytodtw_instance=cytodtw,
        df=filtered_alignment_df,
        alignment_name="cell_division",
        image_loader_fn=load_images_from_triplet_dataset,
        max_lineages=30
    )
else:
    print("Skipping image sequence creation - no valid dataset available")
    concatenated_image_sequences = {}

# Load concatenated sequences into napari
if NAPARI and dataset is not None and len(concatenated_image_sequences) > 0:
    import numpy as np
    
    for lineage_id, seq_data in concatenated_image_sequences.items():
        concatenated_images = seq_data['concatenated_images']
        meta = seq_data['metadata']
        aligned_length = seq_data['aligned_length']
        unaligned_length = seq_data['unaligned_length']
        
        if len(concatenated_images) == 0:
            continue
            
        # Stack images into time series (T, C, Z, Y, X)
        image_stack = []
        for img_sample in concatenated_images:
            if img_sample is not None:
                img_tensor = img_sample['anchor']  # Shape should be (C, Z, Y, X)
                img_np = img_tensor.cpu().numpy()
                image_stack.append(img_np)
        
        if len(image_stack) > 0:
            # Stack into (T, C, Z, Y, X)
            time_series = np.stack(image_stack, axis=0)
            n_channels = time_series.shape[1]
            
            logger.info(f"Processing lineage {lineage_id} with {n_channels} channels, shape {time_series.shape}")
            
            # Set up colormap based on number of channels
            if n_channels == 2:
                colormap = ['green', 'magenta']
            elif n_channels == 3:
                colormap = ['gray', 'green', 'magenta']
            else:
                colormap = ['gray'] * n_channels  # Default fallback
            
            # Add each channel as a separate layer in napari
            for channel_idx in range(n_channels):
                # Extract single channel: (T, Z, Y, X)
                channel_data = time_series[:, channel_idx, :, :, :]
                
                # Get channel name if available
                channel_name = processing_channels[channel_idx] if channel_idx < len(processing_channels) else f"ch{channel_idx}"
                
                layer_name = f"track_id_{meta['track_ids'][0]}_FOV_{meta['fov_name']}_dist_{meta['dtw_distance']:.3f}_{channel_name}"
                
                viewer.add_image(
                    channel_data,
                    name=layer_name,
                    contrast_limits=(channel_data.min(), channel_data.max()),
                    colormap=colormap[channel_idx],
                    blending='additive'
                )
                logger.debug(f"Added {channel_name} channel for lineage {lineage_id} with shape {channel_data.shape}")
# %%

#Plot the features on the aligned dataframe
computed_features_path='/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/quantify_remodeling/feature_list_TOMM20_DENV.csv'
computed_features_df=pd.read_csv(computed_features_path)


computed_features_df['t']=computed_features_df['time_point']
# Remove the first forward slash from the fov_name
computed_features_df['fov_name']=computed_features_df['fov_name'].str.lstrip('/')

#merge with the enhanced dataframe
align_n_comp_feat_df= filtered_alignment_df.merge(
    computed_features_df,
    on=['fov_name','track_id','t','x','y'],
    how='left',
)
#%%
cytodtw.plot_individual_lineages(
    align_n_comp_feat_df,
    alignment_name=ALIGN_TYPE,
    feature_columns=['PC1','edge_density','contrast','organelle_volume'],
    max_lineages=15,
    aligned_linewidth=2.5,
    unaligned_linewidth=1.4,
    y_offset_step=0.0
)
# %%
# Heatmap showing all tracks
cytodtw.plot_global_trends(align_n_comp_feat_df, alignment_name=ALIGN_TYPE, plot_type="heatmap",feature_columns=['PC1','edge_density', 'organelle_volume', 'contrast',], cmap='magma', remove_outliers=True)

# %%
