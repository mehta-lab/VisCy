# %%
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from viscy.representation.evaluation.distance import (
    compute_embedding_distances,
    analyze_and_plot_distances,
)

# plt.style.use("../evaluation/figure.mplstyle")

if __name__ == "__main__":
    # Define models as a dictionary with meaningful keys
    prediction_paths = {
        "ntxent_classical": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_100ckpt_classical_ntxent.zarr"
        ),
        "triplet_classical": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_101ckpt_classical_triplet.zarr"
        ),
        "triplet_cellaware": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_101ckpt_cellAware_triplet.zarr"
        ),
        "triplet_timeaware": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_102ckpt_timeAware_triplet.zarr"
        ),  
        "ntxent_timeaware": Path(
            "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"
        ),
    }

    # output_folder to save the distributions as .csv
    output_folder = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2024_02_04_test/metrics"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Evaluate each model
    for model_name, prediction_path in tqdm(
        prediction_paths.items(), desc="Evaluating models"
    ):
        print(f"\nAnalyzing model: {prediction_path.stem} (Loss: {model_name})")

        # Compute and save distributions
        distributions_df = compute_embedding_distances(
            prediction_path=prediction_path,
            output_path=output_folder / f"{model_name}_distance.csv",
            distance_metric="euclidean",
            verbose=True,
        )

        # Analyze distributions and create plots
        metrics = analyze_and_plot_distances(
            distributions_df,
            output_file_path=output_folder / f"{model_name}_distance_plot.pdf",
            overwrite=True,
        )

        # Print statistics
        print("\nAdjacent Frame Distance Statistics:")
        print(f"{'Mean:':<15} {metrics['dissimilarity_mean']:.3f}")
        print(f"{'Std:':<15} {metrics['dissimilarity_std']:.3f}")
        print(f"{'Median:':<15} {metrics['dissimilarity_median']:.3f}")
        print(f"{'Peak:':<15} {metrics['dissimilarity_peak']:.3f}")
        print(f"{'P1:':<15} {metrics['dissimilarity_p1']:.3f}")
        print(f"{'P99:':<15} {metrics['dissimilarity_p99']:.3f}")

        # Print random sampling statistics
        print("\nRandom Sampling Statistics:")
        print(f"{'Mean:':<15} {metrics['random_mean']:.3f}")
        print(f"{'Std:':<15} {metrics['random_std']:.3f}")
        print(f"{'Median:':<15} {metrics['random_median']:.3f}")
        print(f"{'Peak:':<15} {metrics['random_peak']:.3f}")

        # Print dynamic range
        print("\nComparison Metrics:")
        print(f"{'Dynamic Range:':<15} {metrics['dynamic_range']:.3f}")

        # Print distribution sizes
        print("\nDistribution Sizes:")
        print(
            f"{'Adjacent Frame:':<15} {len(distributions_df['adjacent_frame']):,d} samples"
        )
        print(f"{'Random:':<15} {len(distributions_df['random_sampling']):,d} samples")

# %% plot ratio of mean of random to adjacent frame for all models

plt.figure(figsize=(10, 6))
for model_name, prediction_path in tqdm(
    prediction_paths.items(), desc="Plotting models"
):
    distributions_df = pd.read_csv(output_folder / f"{model_name}_distance.csv")
    plt.plot(distributions_df['random_sampling'] / distributions_df['adjacent_frame'], label=model_name)
plt.legend()
plt.show()

# %% violin plot of random to adjacent frame for all models

# plot ratio of random to adjacent frame distances as violin plot
plt.figure(figsize=(12, 6))
ratio_data = []
labels = []

for model_name, prediction_path in tqdm(
    prediction_paths.items(), desc="Processing models"
):
    distributions_df = pd.read_csv(output_folder / f"{model_name}_distance.csv")
    ratios = distributions_df['random_sampling'] / distributions_df['adjacent_frame']
    ratio_data.append(ratios)
    labels.append(model_name)

# Set font sizes
plt.rcParams.update({'font.size': 24})  # Double the default font size

# Create violin plot with reduced width
plt.violinplot(ratio_data, showmeans=True, showmedians=True, widths=0.7)  # Reduced width from default 0.8
plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
plt.ylabel('Ratio (Random / Adjacent Frame Distance)')
plt.title('Distribution of Distance Ratios by Model')
plt.tight_layout()
plt.show()

# show the plot again with ylim from 0 to 10
plt.violinplot(ratio_data, showmeans=True, showmedians=True)
plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
plt.ylabel('Ratio (Random / Adjacent Frame Distance)')
plt.ylim(0, 10)
plt.title('Distribution of Distance Ratios by Model')
plt.tight_layout()
plt.show()

# %%
