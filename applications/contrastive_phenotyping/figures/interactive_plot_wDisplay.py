"""Interactive visualization of phenotype data."""

import logging
from pathlib import Path

import pandas as pd
from natsort import natsorted

from viscy.visualization import EmbeddingVisualizationApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_phenotype_annotations(annotation_path: str | Path) -> dict:
    """Load phenotype annotations and create FOV-track mapping.

    Parameters
    ----------
    annotation_path : str | Path
        Path to the phenotype annotations CSV file.

    Returns
    -------
    dict
        Dictionary mapping phenotype IDs to FOV-track dictionaries.
    """
    annotations_df = pd.read_csv(annotation_path)
    unique_observed_phenotypes = natsorted(
        annotations_df["Observed phenotype"].unique()
    )
    unique_fovs = natsorted(annotations_df["FOV"].unique())

    # Create a dictionary where keys are phenotypes and values are the fov_tracks_dict
    phenotype_dict = {}

    for phenotype_id in unique_observed_phenotypes:
        fov_tracks_dict = {}
        # make a dictionary of the FOVs and track_ids that have the phenotype
        for _, row in annotations_df.iterrows():
            if row["Observed phenotype"] == phenotype_id:
                fov = row["FOV"]
                track_id = row["Track_id"]
                # Initialize the list for this FOV if it doesn't exist
                if fov not in fov_tracks_dict:
                    fov_tracks_dict[fov] = []
                fov_tracks_dict[fov].append(track_id)

        # Store the fov_tracks_dict for this phenotype
        phenotype_dict[phenotype_id] = fov_tracks_dict
        logging.debug(f"\nPhenotype: {phenotype_id}")
        logging.debug("FOVs and tracks:", fov_tracks_dict)

    return phenotype_dict, unique_fovs


def create_visualization_app(
    phenotype_id: int,
    phenotype_dict: dict,
    unique_fovs: list,
    debug: bool = False,
) -> EmbeddingVisualizationApp:
    """Create the visualization app for a specific phenotype.

    Parameters
    ----------
    phenotype_id : int
        The phenotype ID to visualize.
    phenotype_dict : dict
        Dictionary mapping phenotype IDs to FOV-track dictionaries.
    unique_fovs : list
        List of unique FOV names.
    debug : bool, optional
        Whether to run in debug mode, by default False

    Returns
    -------
    EmbeddingVisualizationApp
        The configured visualization app.
    """
    fov_tracks_dict = phenotype_dict[phenotype_id]

    if not debug:
        # Add empty lists for FOVs not in the phenotype
        for fov in unique_fovs:
            if fov not in fov_tracks_dict:
                fov_tracks_dict[fov] = []

        app = EmbeddingVisualizationApp(
            data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr",
            tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr",
            features_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr",
            channels_to_display=["Phase3D", "raw GFP EX488 EM525-45"],
            fov_tracks=fov_tracks_dict,
            z_range=(25, 40),
            yx_patch_size=(192, 192),
            num_PC_components=8,
            num_loading_workers=16,
            cache_path=f"/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/analysis/pca_analysis/2024_11_07_phenotype_{phenotype_id}_cache.pkl",
        )
    else:
        # Debug configuration
        fov_tracks_dict = {
            "/0/6/000000": [1, 5, 6, 7, 9, 14, 15],
        }
        app = EmbeddingVisualizationApp(
            data_path="/hpc/projects/organelle_phenotyping/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/registered_chunked.zarr",
            tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr",
            features_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr",
            channels_to_display=[
                "Phase3D",
                "MultiCam_GFP_mCherry_BF-Prime BSI Express",
            ],
            fov_tracks=fov_tracks_dict,
            z_range=(31, 36),
            yx_patch_size=(128, 128),
            num_PC_components=8,
            num_loading_workers=16,
        )

    return app


def main():
    """Main function to run the visualization app."""
    # Configuration
    SELECTED_OBSERVED_PHENOTYPE = 2
    DEBUG = True

    # Load phenotype annotations
    annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/phenotype_observations.csv"
    phenotype_dict, unique_fovs = load_phenotype_annotations(annotation_path)

    try:
        # Create and run the visualization app
        app = create_visualization_app(
            SELECTED_OBSERVED_PHENOTYPE,
            phenotype_dict,
            unique_fovs,
            debug=DEBUG,
        )
        app.preload_images()
        app.run(debug=DEBUG)

    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
