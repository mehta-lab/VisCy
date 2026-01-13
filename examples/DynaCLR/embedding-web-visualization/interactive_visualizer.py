"""Interactive visualization of phenotype data."""

import logging
from pathlib import Path

from numpy.random import seed

from viscy.representation.evaluation.visualization import EmbeddingVisualizationApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed(42)


def main():
    """Main function to run the visualization app."""

    # Config for the visualization app
    # TODO: Update the paths to the downloaded data. By default the data is downloaded to ~/data/dynaclr/demo
    Path.home() / "data/dynaclr/demo"
    output_path = None
    viz_config = {
        "data_path": "/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61_TOMM20_G3BP1/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/rechunked.zarr",  # TODO add path to data
        "tracks_path": "/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61_TOMM20_G3BP1/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/tracking.zarr",  # TODO add path to tracks
        "features_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/dynaclr_v3/model_2024_08_14_ZIKV_pal17_48h/timeaware/dynaclrv3_phaseOnly_tau1_temp0p5_ckpt34.zarr",  # TODO add path to features
        "channels_to_display": ["Phase3D", "mCherry EX561 EM600-37"],
        "fov_tracks": "all",  # Use "all" to cache all FOVs (recommended for PHATE visualization)
        # Or specify specific FOVs:
        # "fov_tracks": {
        #     "A/1/000000": "all",  # Load all tracks from this FOV
        #     "A/2/000000": "all",
        # },
        "yx_patch_size": (160, 160),
        "z_range": (15, 30),
        "num_PC_components": 8,
        "output_dir": output_path,
    }

    # Create and run the visualization app
    try:
        # Create and run the visualization app
        app = EmbeddingVisualizationApp(**viz_config)
        app.preload_images()
        app.run(debug=True)

    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
