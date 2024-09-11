# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.decomposition import PCA

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks, load_annotation
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from viscy.representation.evaluation import calculate_cosine_similarity_cell
from viscy.representation.evaluation import compute_displacement_mean_std
from viscy.representation.evaluation import compute_displacement

# %% Paths and parameters.

features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")

features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_1chan_128patch_32projDim/1chan_128patch_63ckpt_FebTest.zarr")

data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)

tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

# %% Load embedding datasets for all three sampling
fov_name = '/B/4/6'
track_id = 4

embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)

# Calculate cosine similarities for each sampling
time_points_30_min, cosine_similarities_30_min = calculate_cosine_similarity_cell(embedding_dataset_30_min, fov_name, track_id)
time_points_no_track, cosine_similarities_no_track = calculate_cosine_similarity_cell(embedding_dataset_no_track, fov_name, track_id)
time_points_any_time, cosine_similarities_any_time = calculate_cosine_similarity_cell(embedding_dataset_any_time, fov_name, track_id)

# %% Plot cosine similarities over time for all three conditions

plt.figure(figsize=(10, 6))

plt.plot(time_points_no_track, cosine_similarities_no_track, marker='o', label='classical contrastive (no tracking)')
plt.plot(time_points_any_time, cosine_similarities_any_time, marker='o', label='cell aware')
plt.plot(time_points_30_min, cosine_similarities_30_min, marker='o', label='cell & time aware (interval 30 min)')


plt.xlabel("Time Delay (t)")
plt.ylabel("Cosine Similarity with First Time Point")
plt.title("Cosine Similarity Over Time for Infected Cell")

#plt.savefig('infected_cell_example.svg', format='svg')
#plt.savefig('infected_cell_example.pdf', format='pdf')

plt.grid(True)
plt.legend()


plt.show()
# %%

# %% import statements

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks, load_annotation
from sklearn.metrics.pairwise import cosine_similarity

# %% Paths to datasets
features_path_30_min = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr")
feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")
#features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_1chan_128patch_32projDim/1chan_128patch_63ckpt_FebTest.zarr")

# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
#embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)

# %% Compute displacements for both datasets (using Euclidean distance and Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements

mean_displacement_30_min, std_displacement_30_min = compute_displacement_mean_std(embedding_dataset_30_min, max_tau, use_cosine=False, use_dissimilarity=True)
mean_displacement_no_track, std_displacement_no_track = compute_displacement_mean_std(embedding_dataset_no_track, max_tau, use_cosine=False, use_dissimilarity=True)
#mean_displacement_any_time, std_displacement_any_time = compute_displacement_mean_std(embedding_dataset_any_time, max_tau, use_cosine=False)

mean_displacement_30_min_cosine, std_displacement_30_min_cosine = compute_displacement_mean_std(embedding_dataset_30_min, max_tau, use_cosine=True, use_dissimilarity=True)
mean_displacement_no_track_cosine, std_displacement_no_track_cosine = compute_displacement_mean_std(embedding_dataset_no_track, max_tau, use_cosine=True, use_dissimilarity=True)
#mean_displacement_any_time_cosine, std_displacement_any_time_cosine = compute_displacement_mean_std(embedding_dataset_any_time, max_tau, use_cosine=True)
# %% Plot 1: Euclidean Displacements
plt.figure(figsize=(10, 6))

taus = list(mean_displacement_30_min.keys())
mean_values_30_min = list(mean_displacement_30_min.values())
std_values_30_min = list(std_displacement_30_min.values())

mean_values_no_track = list(mean_displacement_no_track.values())
std_values_no_track = list(std_displacement_no_track.values())

# mean_values_any_time = list(mean_displacement_any_time.values())
# std_values_any_time = list(std_displacement_any_time.values())

# Plotting Euclidean displacements
plt.plot(taus, mean_values_30_min, marker='o', label='Cell & Time Aware (30 min interval)')
plt.fill_between(taus, np.array(mean_values_30_min) - np.array(std_values_30_min), np.array(mean_values_30_min) + np.array(std_values_30_min), 
                 color='gray', alpha=0.3, label='Std Dev (30 min interval)')

plt.plot(taus, mean_values_no_track, marker='o', label='Classical Contrastive (No Tracking)')
plt.fill_between(taus, np.array(mean_values_no_track) - np.array(std_values_no_track), np.array(mean_values_no_track) + np.array(std_values_no_track), 
                 color='blue', alpha=0.3, label='Std Dev (No Tracking)')

plt.xlabel('Time Shift (τ)')
plt.ylabel('Displacement')
plt.title('Embedding Displacement Over Time')
plt.grid(True)
plt.legend()

# plt.savefig('embedding_displacement_euclidean.svg', format='svg')
# plt.savefig('embedding_displacement_euclidean.pdf', format='pdf')

# Show the Euclidean plot
plt.show()

# %% Plot 2: Cosine Displacements
plt.figure(figsize=(10, 6))

# Plotting Cosine displacements
mean_values_30_min_cosine = list(mean_displacement_30_min_cosine.values())
std_values_30_min_cosine = list(std_displacement_30_min_cosine.values())

mean_values_no_track_cosine = list(mean_displacement_no_track_cosine.values())
std_values_no_track_cosine = list(std_displacement_no_track_cosine.values())

plt.plot(taus, mean_values_30_min_cosine, marker='o', label='Cell & Time Aware (30 min interval)')
plt.fill_between(taus, np.array(mean_values_30_min_cosine) - np.array(std_values_30_min_cosine), np.array(mean_values_30_min_cosine) + np.array(std_values_30_min_cosine), 
                 color='orange', alpha=0.3, label='Std Dev (30 min interval, Cosine)')

plt.plot(taus, mean_values_no_track_cosine, marker='o', label='Classical Contrastive (No Tracking)')
plt.fill_between(taus, np.array(mean_values_no_track_cosine) - np.array(std_values_no_track_cosine), np.array(mean_values_no_track_cosine) + np.array(std_values_no_track_cosine), 
                 color='red', alpha=0.3, label='Std Dev (No Tracking)')

plt.xlabel('Time Shift (τ)')
plt.ylabel('Cosine Similarity')
plt.title('Embedding Displacement Over Time')
plt.grid(True)
plt.legend()

# Show the Cosine plot
plt.show()
# %%

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from viscy.representation.embedding_writer import read_embedding_dataset

# %% Paths to datasets
features_path_30_min = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr")
feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")

# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)

# %% Compute displacements for both datasets (using Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements

# Compute displacements for Cell & Time Aware (30 min interval) using Cosine similarity
displacement_per_tau_aware_cosine = compute_displacement(embedding_dataset_30_min, max_tau, use_cosine=True, use_dissimilarity=True)

# Compute displacements for Classical Contrastive (No Tracking) using Cosine similarity
displacement_per_tau_contrastive_cosine = compute_displacement(embedding_dataset_no_track, max_tau, use_cosine=True, use_dissimilarity=True)

# %% Prepare data for violin plot
# Prepare the data in a long-form DataFrame for the violin plot
def prepare_violin_data(taus, displacement_aware, displacement_contrastive):
    # Create a list to hold the data
    data = []

    # Populate the data for Cell & Time Aware
    for tau in taus:
        displacements_aware = displacement_aware.get(tau, [])
        for displacement in displacements_aware:
            data.append({'Time Shift (τ)': tau, 'Displacement': displacement, 'Sampling': 'Cell & Time Aware (30 min interval)'})

    # Populate the data for Classical Contrastive
    for tau in taus:
        displacements_contrastive = displacement_contrastive.get(tau, [])
        for displacement in displacements_contrastive:
            data.append({'Time Shift (τ)': tau, 'Displacement': displacement, 'Sampling': 'Classical Contrastive (No Tracking)'})

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    return df

# Assuming 'displacement_per_tau_aware_cosine' and 'displacement_per_tau_contrastive_cosine' hold the displacements as dictionaries
taus = list(displacement_per_tau_aware_cosine.keys())

# Prepare the violin plot data
df = prepare_violin_data(taus, displacement_per_tau_aware_cosine, displacement_per_tau_contrastive_cosine)

# Create a violin plot using seaborn
plt.figure(figsize=(12, 8))
sns.violinplot(
    x='Time Shift (τ)', 
    y='Displacement', 
    hue='Sampling', 
    data=df, 
    palette='Set2',  
    scale='width',   
    bw=0.2,          
    inner=None,      
    split=True,      
    cut=0            
)

# Add labels and title
plt.xlabel('Time Shift (τ)', fontsize=14)
plt.ylabel('Cosine Dissimilarity', fontsize=14)
plt.title('Cosine Dissimilarity Distribution', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)  # Lighter grid lines for less distraction
plt.legend(title='Sampling', fontsize=12, title_fontsize=14)

plt.ylim(0, 0.5)  

# Save the violin plot as SVG and PDF
plt.savefig('violin_plot_cosine_similarity.svg', format='svg')
plt.savefig('violin_plot_cosine_similarity.pdf', format='pdf')

# Show the plot
plt.show()
# %%
