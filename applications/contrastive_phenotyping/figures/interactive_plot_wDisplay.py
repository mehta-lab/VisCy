import logging

import pandas as pd
from natsort import natsorted

from viscy.representation.evaluation.visualization import EmbeddingVisualizationApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# %%
if __name__ == "__main__":

    """
    1  Uninfected - survives

    2 Infected remodeling

    3 Infected (visually from the viral sensor) partial remodeling and recovery

    4 No remodeling (no condensation) and death. Viral sensor is not

    5 Death in uninfected
    """
    SELECTED_OBSERVED_PHENOTYPE = 1

    annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/phenotype_observations.csv"
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
    fov_tracks_dict = phenotype_dict[SELECTED_OBSERVED_PHENOTYPE]

    # %%
    DEBUG = True
    try:
        if not DEBUG:
            # add the fovs with empty list to the fov_tracks_dict missing from the unique_fovs
            for fov in unique_fovs:
                if fov not in fov_tracks_dict:
                    fov_tracks_dict[fov] = []
            app = ImageDisplayApp(
                data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr",
                tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr",
                features_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr",
                channels_to_display=["Phase3D", "raw GFP EX488 EM525-45"],
                fov_tracks=fov_tracks_dict,
                z_range=(25, 40),
                yx_patch_size=(192, 192),
                num_PC_components=8,
            )

        else:
            # Example of using multiple FOVs with specific track IDs for each
            fov_tracks_dict = {
                "/0/6/000000": [1, 5, 6, 7, 9, 14, 15],
                # "/0/3/002000": "all",  # Use all tracks for this FOV
            }
            app = ImageDisplayApp(
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
            )
        app.preload_images()
        app.run(debug=True)  # Debug mode without reloader

    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete")
