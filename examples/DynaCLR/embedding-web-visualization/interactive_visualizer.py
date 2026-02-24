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
        "data_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/4-phenotyping/0-train-test/2025_08_26_A549_SEC61_TOMM20_ZIKV.zarr",  # TODO add path to data
        "tracks_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/1-preprocess/label-free/3-track/2025_08_26_A549_SEC61_TOMM20_ZIKV_cropped.zarr",  # TODO add path to tracks
        "features_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/4-phenotyping/1-predictions/organellebox-v3/organelle_160patch_104ckpt_ver3max.zarr",  # TODO add path to features
        # "channels_to_display": ["Phase3D", "mCherry EX561 EM600-37"],
        "channels_to_display": ["GFP EX488 EM525-45"],
        # "fov_tracks": "all",  # Use "all" to cache all FOVs (recommended for PHATE visualization)
        # Or specify specific FOVs:
        "fov_tracks": {
            "B/2/000000": "all",
            "B/2/000001": "all",
            "A/2/000000": "all",
            "A/2/000001": "all",
        },
        "yx_patch_size": (160, 160),
        "z_range": (0, 1),
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
