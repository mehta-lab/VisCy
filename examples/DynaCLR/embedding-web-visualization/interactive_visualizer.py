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
    # TODO: Update the paths to the downloaded data
    # NOTE: selecting random FOVs for the demo
    download_root = Path("/path/to/downloaded/data")
    viz_config = {
        "data_path": download_root / "registered_test.zarr",  # TODO add path to data
        "tracks_path": download_root / "track_test.zarr",  # TODO add path to tracks
        "features_path": download_root
        / "precomputed_embeddings/infection_160patch_94ckpt_rev6_dynaclr.zarr",  # TODO add path to features
        "channels_to_display": ["Phase3D", "RFP"],
        "fov_tracks": {
            "/A/3/9": list(range(50)),
            "/B/4/9": list(range(50)),
        },
        "yx_patch_size": (160, 160),
        "num_PC_components": 8,
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
