# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.distance import (
    calculate_normalized_euclidean_distance_cell,
    compute_displacement_mean_std_full,
    compute_dynamic_smoothness_metrics,
)

# %% Paths to datasets for different intervals
feature_paths = {
    "7 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_7mins.zarr",
    "21 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_21mins.zarr",
    "28 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_28mins.zarr",
    "56 min interval": "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_56mins.zarr",
}

no_track_path = "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_classical.zarr"
cell_aware_path = "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_cellaware.zarr"

# Parameters
max_tau = 69

metrics = {}

features_path_no_track = Path(no_track_path)
embedding_dataset_no_track = read_embedding_dataset(features_path_no_track)

mean_displacement_no_track, std_displacement_no_track = compute_displacement_mean_std_full(embedding_dataset_no_track, max_tau)
dynamic_range_no_track, smoothness_no_track = compute_dynamic_smoothness_metrics(mean_displacement_no_track)

metrics["No Tracking"] = {
    "dynamic_range": dynamic_range_no_track,
    "smoothness": smoothness_no_track,
    "mean_displacement": mean_displacement_no_track,
    "std_displacement": std_displacement_no_track,
}

print("Metrics for No Tracking:")
print(f"  Dynamic Range: {dynamic_range_no_track}")
print(f"  Smoothness: {smoothness_no_track}")

features_path_cell_aware = Path(cell_aware_path)
embedding_dataset_cell_aware = read_embedding_dataset(features_path_cell_aware)

mean_displacement_cell_aware, std_displacement_cell_aware = compute_displacement_mean_std_full(embedding_dataset_cell_aware, max_tau)
dynamic_range_cell_aware, smoothness_cell_aware = compute_dynamic_smoothness_metrics(mean_displacement_cell_aware)

metrics["Cell Aware"] = {
    "dynamic_range": dynamic_range_cell_aware,
    "smoothness": smoothness_cell_aware,
    "mean_displacement": mean_displacement_cell_aware,
    "std_displacement": std_displacement_cell_aware,
}

print("Metrics for Cell Aware:")
print(f"  Dynamic Range: {dynamic_range_cell_aware}")
print(f"  Smoothness: {smoothness_cell_aware}")

for label, path in feature_paths.items():
    features_path = Path(path)
    embedding_dataset = read_embedding_dataset(features_path)
    
    mean_displacement, std_displacement = compute_displacement_mean_std_full(embedding_dataset, max_tau)
    dynamic_range, smoothness = compute_dynamic_smoothness_metrics(mean_displacement)
    
    metrics[label] = {
        "dynamic_range": dynamic_range,
        "smoothness": smoothness,
        "mean_displacement": mean_displacement,
        "std_displacement": std_displacement,
    }
    
    plt.figure(figsize=(10, 6))
    taus = list(mean_displacement.keys())
    mean_values = list(mean_displacement.values())
    std_values = list(std_displacement.values())

    plt.plot(taus, mean_values, marker='o', label=f'{label}', color='green')
    plt.fill_between(taus,
                     np.array(mean_values) - np.array(std_values),
                     np.array(mean_values) + np.array(std_values),
                     color='green', alpha=0.3, label=f'Std Dev ({label})')

    mean_values_no_track = list(metrics["No Tracking"]["mean_displacement"].values())
    std_values_no_track = list(metrics["No Tracking"]["std_displacement"].values())

    plt.plot(taus, mean_values_no_track, marker='o', label='Classical Contrastive (No Tracking)', color='blue')
    plt.fill_between(taus,
                     np.array(mean_values_no_track) - np.array(std_values_no_track),
                     np.array(mean_values_no_track) + np.array(std_values_no_track),
                     color='blue', alpha=0.3, label='Std Dev (No Tracking)')

    plt.xlabel('Time Shift (τ)')
    plt.ylabel('Euclidean Distance')
    plt.title(f'Embedding Displacement Over Time ({label})')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Metrics for {label}:")
    print(f"  Dynamic Range: {dynamic_range}")
    print(f"  Smoothness: {smoothness}")
# %%
