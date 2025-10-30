# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.smoothness import compute_embeddings_smoothness

# %%
# FEATURES

# openphenom_features_path = Path("/home/jason/projects/contrastive_phenotyping/data/open_phenom/features/open_phenom_features.csv")
# imagenet_features_path = Path("/home/jason/projects/contrastive_phenotyping/data/imagenet/features/imagenet_features.csv")
dynaclr_features_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_11_07_A549_SEC61_DENV/4-phenotyping/dtw_evaluation/SAM2/sam2_sensor_only.zarr"
)
dinov3_features_path = Path(
    "/home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/DINOV3/embeddings_convnext_tiny_phase_only_2.zarr"
)

# LOADING DATASETS
# openphenom_features = read_embedding_dataset(openphenom_features_path)
# imagenet_features = read_embedding_dataset(imagenet_features_path)
dynaclr_embedding_dataset = read_embedding_dataset(dynaclr_features_path)
dinov3_embedding_dataset = read_embedding_dataset(dinov3_features_path)
# %%
# Compute the smoothness of the features
DISTANCE_METRIC = "cosine"
feature_paths = {
    # "dynaclr": dynaclr_features_path,
    "dinov3": dinov3_features_path,
}
cmap = plt.get_cmap("tab10")  # or use "Set2", "tab20", etc.
labels = list(feature_paths.keys())
interval_colors = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}
# Print and check each path
for label, path in feature_paths.items():
    print(f"{label} color: {interval_colors[label]}")
    assert Path(path).exists(), f"Path {path} does not exist"

output_dir = Path("./smoothness_metrics")
output_dir.mkdir(parents=True, exist_ok=True)

results = {}
for label, path in feature_paths.items():
    results[label] = {}
    print(f"\nProcessing - {label}")
    embedding_dataset = read_embedding_dataset(Path(path))

    # Compute displacements
    stats, distributions, _ = compute_embeddings_smoothness(
        embedding_dataset=embedding_dataset,
        distance_metric=DISTANCE_METRIC,
        verbose=True,
    )

    # Plot the piecewise distances
    plt.figure()
    sns.histplot(
        distributions["adjacent_frame_distribution"],
        bins=30,
        kde=True,
        color="cyan",
        alpha=0.5,
        stat="density",
    )
    sns.histplot(
        distributions["random_frame_distribution"],
        bins=30,
        kde=True,
        color="red",
        alpha=0.5,
        stat="density",
    )
    plt.xlabel(f"{DISTANCE_METRIC} Distance")
    plt.ylabel("Density")
    # Add vertical lines for the peaks
    plt.axvline(x=stats["adjacent_frame_peak"], color="cyan", linestyle="--", alpha=0.8)
    plt.axvline(x=stats["random_frame_peak"], color="red", linestyle="--", alpha=0.8)
    plt.tight_layout()
    plt.legend(["Adjacent Frame", "Random Sample", "Adjacent Peak", "Random Peak"])
    plt.savefig(output_dir / f"{label}_smoothness.pdf", dpi=300)
    plt.savefig(output_dir / f"{label}_smoothness.png", dpi=300)
    plt.close()

    # metrics to csv
    scalar_metrics = {
        "adjacent_frame_mean": stats["adjacent_frame_mean"],
        "adjacent_frame_std": stats["adjacent_frame_std"],
        "adjacent_frame_median": stats["adjacent_frame_median"],
        "adjacent_frame_peak": stats["adjacent_frame_peak"],
        "random_frame_mean": stats["random_frame_mean"],
        "random_frame_std": stats["random_frame_std"],
        "random_frame_median": stats["random_frame_median"],
        "random_frame_peak": stats["random_frame_peak"],
        "smoothness_score": stats["smoothness_score"],
        "dynamic_range": stats["dynamic_range"],
    }
    # Create DataFrame with single row
    stats_df = pd.DataFrame(scalar_metrics, index=[0])
    stats_df.to_csv(output_dir / f"{label}_smoothness_stats.csv", index=False)
