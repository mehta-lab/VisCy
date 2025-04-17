"""Interactive visualization of phenotype data."""

# %%
import logging
from pathlib import Path

import pandas as pd
from natsort import natsorted
from numpy.random import seed

from viscy.representation.evaluation.visualization import EmbeddingVisualizationApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed(42)


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


# %%
def main():
    """Main function to run the visualization app."""
    # Configuration
    DEBUG = True
    VIZ_DATASET = "organelle"  # ["organelle", "phenotype", "microglia"]

    if VIZ_DATASET == "organelle":
        SELECTED_OBSERVED_PHENOTYPE = 2
        annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/phenotype_observations.csv"
        phenotype_dict, unique_fovs = load_phenotype_annotations(annotation_path)
        fov_tracks_dict = phenotype_dict[SELECTED_OBSERVED_PHENOTYPE]

        fov_tracks_dict = {
            "/C/2/000000": list(range(300)),
            "/C/2/000001": list(range(300)),
            # "/C/1/001000": list(range(100)),
        }
        if not DEBUG:
            # Add empty lists for FOVs not in the phenotype
            for fov in unique_fovs:
                if fov not in fov_tracks_dict:
                    phenotype_dict[fov] = []

        viz_config = {
            "data_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr",
            "tracks_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr",
            "features_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr",
            "channels_to_display": ["Phase3D", "raw GFP EX488 EM525-45"],
            # "fov_tracks": phenotype_dict[SELECTED_OBSERVED_PHENOTYPE],
            "fov_tracks": fov_tracks_dict,
            "z_range": (25, 40),
            "yx_patch_size": (192, 192),
            "num_PC_components": 8,
        }
    elif VIZ_DATASET == "phenotype":
        viz_config = {
            "data_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized.zarr",
            "tracks_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized_tracks.zarr",
            "features_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/time_interval_1_microglia_test_fovs.zarr",
            "channels_to_display": ["Phase3D"],
            "fov_tracks": {
                "/0/6/000000": [1, 5, 6, 7, 9, 14, 15],
            },
            "yx_patch_size": (128, 128),
            "num_PC_components": 8,
        }
    elif VIZ_DATASET == "microglia":
        viz_config = {
            "data_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized.zarr",
            "tracks_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/test_dataset/test_fovs_20191107_1209_1_GW23_blank_bg_stabilized_tracks.zarr",
            "features_path": "/hpc/projects/organelle_phenotyping/dynamorph/4-phenotyping/time_interval_1_microglia_test_fovs.zarr",
            "channels_to_display": ["Phase3D"],
            "fov_tracks": {
                "/B/2/0": list(range(50)),
            },
            "yx_patch_size": (128, 128),
            "num_PC_components": 8,
        }

    # Create and run the visualization app
    try:
        # Create and run the visualization app
        app = EmbeddingVisualizationApp(**viz_config)
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

# %%
