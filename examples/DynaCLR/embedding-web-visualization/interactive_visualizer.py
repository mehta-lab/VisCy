import logging
from pathlib import Path

import click
from numpy.random import seed

from viscy.representation.visualization.app import EmbeddingVisualizationApp
from viscy.representation.visualization.settings import VizConfig
from viscy.utils.cli_utils import yaml_to_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed(42)


@click.command()
@click.option(
    "--config-filepath",
    "-c",
    required=True,
    help="Path to YAML configuration file.",
)
def main(config_filepath):
    """Main function to run the visualization app."""

    # Load and validate configuration from YAML file
    viz_config = yaml_to_model(yaml_path=config_filepath, model=VizConfig)

    # Use configured paths, with fallbacks to current defaults if not specified
    output_dir = viz_config.output_dir or str(Path(__file__).parent / "output")
    cache_path = viz_config.cache_path

    logger.info(f"Using output directory: {output_dir}")
    logger.info(f"Using cache path: {cache_path}")

    # Create and run the visualization app
    try:
        app = EmbeddingVisualizationApp(
            viz_config=viz_config,
            cache_path=cache_path,
            num_loading_workers=16,
            output_dir=output_dir,
        )
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
