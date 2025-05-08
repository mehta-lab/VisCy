# %%
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from viscy.representation.evaluation.distance import (
    compute_embedding_distances,
    analyze_and_plot_distances,
)

plt.style.use("../evaluation/figure.mplstyle")


# %%
if __name__ == "__main__":
    # Define models as a dictionary with meaningful keys
    prediction_paths = {
        "ntxent_sensor_phase": Path(
            "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_98ckpt_rev6_2.zarr"
        ),
        "triplet_sensor_phase": Path(
            "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/log_alfi_triplet_time_intervals/prediction/ALFI_cellaware.zarr"
        ),
    }

    # output_folder to save the distributions as .csv
    output_folder = Path(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/cosine_dissimilarity_distributions_alfi"
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
            output_path=output_folder,
            distance_metric="cosine",
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

# %%
