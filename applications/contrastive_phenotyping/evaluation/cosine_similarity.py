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


# %% Paths and parameters.

features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr")

features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_128projDim/2chan_128patch_58ckpt_FebTest.zarr")

data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)

tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

# %%
# Define the FOV name and track ID for the infected cell
fov_name = '/B/4/6'
track_id = 4

# %%
# Function to extract embeddings and calculate cosine similarities
def calculate_cosine_similarity(embedding_dataset):
    # Filter the dataset for the specific infected cell
    filtered_data = embedding_dataset.where(
        (embedding_dataset['fov_name'] == fov_name) & 
        (embedding_dataset['track_id'] == track_id), 
        drop=True
    )
    
    # Extract the feature embeddings and time points
    features = filtered_data['features'].values  # (sample, features)
    time_points = filtered_data['t'].values      # (sample,)
    
    # Get the first time point's embedding
    first_time_point_embedding = features[0].reshape(1, -1)
    
    # Calculate cosine similarity between each time point and the first time point
    cosine_similarities = []
    for i in range(len(time_points)):
        similarity = cosine_similarity(
            first_time_point_embedding, features[i].reshape(1, -1)
        )
        cosine_similarities.append(similarity[0][0])
    
    return time_points, cosine_similarities

# %% Load embedding datasets for all three sampling

embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)
embedding_dataset_any_time = read_embedding_dataset(features_path_any_time)

# Calculate cosine similarities for each sampling
time_points_30_min, cosine_similarities_30_min = calculate_cosine_similarity(embedding_dataset_30_min)
time_points_no_track, cosine_similarities_no_track = calculate_cosine_similarity(embedding_dataset_no_track)
time_points_any_time, cosine_similarities_any_time = calculate_cosine_similarity(embedding_dataset_any_time)

# %% Plot cosine similarities over time for all three conditions

plt.figure(figsize=(10, 6))

plt.plot(time_points_no_track, cosine_similarities_no_track, marker='o', label='classical contrastive (no tracking)')
plt.plot(time_points_any_time, cosine_similarities_any_time, marker='o', label='cell aware')
plt.plot(time_points_30_min, cosine_similarities_30_min, marker='o', label='cell & time aware (interval 30 min)')


plt.xlabel("Time (30 min intervals)")
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

# %% Read embedding datasets
embedding_dataset_30_min = read_embedding_dataset(features_path_30_min)
embedding_dataset_no_track = read_embedding_dataset(feature_path_no_track)

# %% Function to compute the norm of differences between embeddings at t and t + tau
def compute_displacement(embedding_dataset, max_tau=10, use_cosine=False):
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset['fov_name'].values
    track_ids = embedding_dataset['track_id'].values
    timepoints = embedding_dataset['t'].values
    embeddings = embedding_dataset['features'].values
    
    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)
    
    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]
        
        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau
            
            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name) & (track_ids == track_id) & (timepoints == future_time)
            )[0]
            
            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]
                
                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(current_embedding.reshape(1, -1), future_embedding.reshape(1, -1))[0][0]
                    displacement = 1 - similarity  # Cosine dissimilarity (1 - cosine similarity)
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)
                
                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)
    
    # Compute mean and std displacement for each tau by averaging the displacements
    mean_displacement_per_tau = {tau: np.mean(displacements) for tau, displacements in displacement_per_tau.items()}
    std_displacement_per_tau = {tau: np.std(displacements) for tau, displacements in displacement_per_tau.items()}
    
    return mean_displacement_per_tau, std_displacement_per_tau

# %% Compute displacements for both datasets (using Euclidean distance and Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements

# Assuming 'embedding_dataset_30_min' and 'embedding_dataset_no_track' are already loaded with your data.
mean_displacement_30_min, std_displacement_30_min = compute_displacement(embedding_dataset_30_min, max_tau, use_cosine=False)
mean_displacement_no_track, std_displacement_no_track = compute_displacement(embedding_dataset_no_track, max_tau, use_cosine=False)

mean_displacement_30_min_cosine, std_displacement_30_min_cosine = compute_displacement(embedding_dataset_30_min, max_tau, use_cosine=True)
mean_displacement_no_track_cosine, std_displacement_no_track_cosine = compute_displacement(embedding_dataset_no_track, max_tau, use_cosine=True)

# %% Plot 1: Euclidean Displacements
plt.figure(figsize=(10, 6))

taus = list(mean_displacement_30_min.keys())
mean_values_30_min = list(mean_displacement_30_min.values())
std_values_30_min = list(std_displacement_30_min.values())

mean_values_no_track = list(mean_displacement_no_track.values())
std_values_no_track = list(std_displacement_no_track.values())

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

plt.savefig('embedding_displacement_euclidean.svg', format='svg')
plt.savefig('embedding_displacement_euclidean.pdf', format='pdf')

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
plt.ylabel('Cosine Dissimilarity')
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

# %% Function to compute the norm of differences between embeddings at t and t + tau
# %% Function to compute the norm of differences between embeddings at t and t + tau
def compute_displacement(embedding_dataset, max_tau=10, use_cosine=True):
    # Get the arrays of (fov_name, track_id, t, and embeddings)
    fov_names = embedding_dataset['fov_name'].values
    track_ids = embedding_dataset['track_id'].values
    timepoints = embedding_dataset['t'].values
    embeddings = embedding_dataset['features'].values
    
    # Dictionary to store displacements for each tau
    displacement_per_tau = defaultdict(list)
    
    # Iterate over all entries in the dataset
    for i in range(len(fov_names)):
        fov_name = fov_names[i]
        track_id = track_ids[i]
        current_time = timepoints[i]
        current_embedding = embeddings[i]
        
        # For each time point t, compute displacements for t + tau
        for tau in range(1, max_tau + 1):
            future_time = current_time + tau
            
            # Find if future_time exists for the same (fov_name, track_id)
            matching_indices = np.where(
                (fov_names == fov_name) & (track_ids == track_id) & (timepoints == future_time)
            )[0]
            
            if len(matching_indices) == 1:
                # Get the embedding at t + tau
                future_embedding = embeddings[matching_indices[0]]
                
                if use_cosine:
                    # Compute cosine similarity
                    similarity = cosine_similarity(current_embedding.reshape(1, -1), future_embedding.reshape(1, -1))[0][0]
                    displacement = 1 - similarity  # Cosine dissimilarity (1 - cosine similarity)
                else:
                    # Compute the Euclidean distance, elementwise square on difference
                    displacement = np.sum((current_embedding - future_embedding) ** 2)
                
                # Store the displacement for the given tau
                displacement_per_tau[tau].append(displacement)
    
    return displacement_per_tau

# %% Compute displacements for both datasets (using Cosine similarity)
max_tau = 10  # Maximum time shift (tau) to compute displacements

# Compute displacements for Cell & Time Aware (30 min interval) using Cosine similarity
displacement_per_tau_aware_cosine = compute_displacement(embedding_dataset_30_min, max_tau, use_cosine=True)

# Compute displacements for Classical Contrastive (No Tracking) using Cosine similarity
displacement_per_tau_contrastive_cosine = compute_displacement(embedding_dataset_no_track, max_tau, use_cosine=True)

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
    palette='Set2',  # Use a clearer color palette
    scale='width',   # Make the violins width proportional
    bw=0.2,          # Adjust the bandwidth to make the violins wider and smoother
    inner=None,      # Remove the inner quartile boxes
    split=True,      # Split the violins for both classes
    cut=0            # Cut off the violins to prevent the long tails
)

# Add labels and title
plt.xlabel('Time Shift (τ)', fontsize=14)
plt.ylabel('Cosine Dissimilarity', fontsize=14)
plt.title('Cosine Dissimilarity Distribution', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)  # Lighter grid lines for less distraction
plt.legend(title='Sampling', fontsize=12, title_fontsize=14)

plt.ylim(0, 0.5)  # Set the y-axis limit based on cosine dissimilarity (0 to 2 is a reasonable range for cosine dissimilarity)

# Save the violin plot as SVG and PDF
plt.savefig('violin_plot_cosine_dissimilarity.svg', format='svg')
plt.savefig('violin_plot_cosine_dissimilarity.pdf', format='pdf')

# Show the plot
plt.show()
# %%
