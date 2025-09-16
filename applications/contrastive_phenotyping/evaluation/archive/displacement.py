# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
   calculate_normalized_euclidean_distance_cell,
   compute_displacement_mean_std_full,
)

# %% paths 

features_path_30_min = Path(
   "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")

features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr")

data_path = Path(
   "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)

tracks_path = Path(
   "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

# %% Load embedding datasets for all three sampling
fov_name = '/B/4/6'
track_id = 52

embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)

#%%
# Calculate displacement for each sampling
time_points_30_min, cosine_similarities_30_min = calculate_normalized_euclidean_distance_cell(embedding_dataset_30_min, fov_name, track_id)
time_points_no_track, cosine_similarities_no_track = calculate_normalized_euclidean_distance_cell(embedding_dataset_no_track, fov_name, track_id)
time_points_any_time, cosine_similarities_any_time = calculate_normalized_euclidean_distance_cell(embedding_dataset_any_time, fov_name, track_id)

# %% Plot displacement over time for all three conditions

plt.figure(figsize=(10, 6))

plt.plot(time_points_no_track, cosine_similarities_no_track, marker='o', label='classical contrastive (no tracking)')
plt.plot(time_points_any_time, cosine_similarities_any_time, marker='o', label='cell aware')
plt.plot(time_points_30_min, cosine_similarities_30_min, marker='o', label='cell & time aware (interval 30 min)')

plt.xlabel("Time Delay (t)", fontsize=10)
plt.ylabel("Normalized Euclidean Distance with First Time Point", fontsize=10)
plt.title("Normalized Euclidean Distance (Features) Over Time for Infected Cell", fontsize=12)

plt.grid(True)
plt.legend(fontsize=10)

#plt.savefig('4_euc_dist_full.svg', format='svg')
plt.show()


# %% Paths to datasets
features_path_30_min = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr")
feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")

embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)


# %%
max_tau = 10  

mean_displacement_30_min_euc, std_displacement_30_min_euc = compute_displacement_mean_std_full(embedding_dataset_30_min, max_tau)
mean_displacement_no_track_euc, std_displacement_no_track_euc = compute_displacement_mean_std_full(embedding_dataset_no_track, max_tau)

# %% Plot 2: Cosine Displacements
plt.figure(figsize=(10, 6))

taus = list(mean_displacement_30_min_euc.keys())

mean_values_30_min_euc = list(mean_displacement_30_min_euc.values())
std_values_30_min_euc = list(std_displacement_30_min_euc.values())

plt.plot(taus, mean_values_30_min_euc, marker='o', label='Cell & Time Aware (30 min interval)', color='green')
plt.fill_between(taus, 
                 np.array(mean_values_30_min_euc) - np.array(std_values_30_min_euc), 
                 np.array(mean_values_30_min_euc) + np.array(std_values_30_min_euc),
                 color='green', alpha=0.3, label='Std Dev (30 min interval)')

mean_values_no_track_euc = list(mean_displacement_no_track_euc.values())
std_values_no_track_euc = list(std_displacement_no_track_euc.values())

plt.plot(taus, mean_values_no_track_euc, marker='o', label='Classical Contrastive (No Tracking)', color='blue')
plt.fill_between(taus, 
                 np.array(mean_values_no_track_euc) - np.array(std_values_no_track_euc), 
                 np.array(mean_values_no_track_euc) + np.array(std_values_no_track_euc),
                 color='blue', alpha=0.3, label='Std Dev (No Tracking)')

plt.xlabel('Time Shift (τ)')
plt.ylabel('Euclidean Distance')
plt.title('Embedding Displacement Over Time (Features)')

plt.grid(True)
plt.legend()

plt.show()
