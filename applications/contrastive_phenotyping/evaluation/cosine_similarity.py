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

# %% Paths and parameters.

features_path_30_min = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

feature_path_no_track = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/febtest_predict.zarr")

features_path_any_time = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr")

data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

# %%
# Define the FOV name and track ID for the infected cell
fov_name = '/B/4/6'
track_id = 50

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

plt.plot(time_points_no_track, cosine_similarities_no_track, marker='o', label='no tracking')
plt.plot(time_points_any_time, cosine_similarities_any_time, marker='o', label='time any')
plt.plot(time_points_30_min, cosine_similarities_30_min, marker='o', label='time interval 30 min')

plt.xlabel("Time (30 min intervals)")
plt.ylabel("Cosine Similarity with First Time Point")
plt.title("Cosine Similarity Over Time for Infected Cell")
plt.grid(True)
plt.legend()
plt.show()
# %%
