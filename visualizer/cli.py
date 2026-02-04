"""
Command-line interface and entry point for PHATE Track Viewer.

This module provides the main entry point for launching the visualization
application from the command line with support for configuration files
and command-line argument overrides.

Functions
---------
parse_args : Parse command-line arguments
main : Application entry point
"""

import argparse
import logging
from pathlib import Path

from .app import create_app, run_app
from .config_loader import load_config_from_json, load_config_from_yaml
from .example_config import CONFIG

try:
    from . import __version__
except ImportError:
    __version__ = "unknown"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="PHATE Track Viewer - Interactive visualization for cell tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use example config from code
  python -m visualizer.cli

  # Load from YAML config
  python -m visualizer.cli --config config.yaml

  # Load from JSON config
  python -m visualizer.cli --config config.json

  # Override port and enable debug mode
  python -m visualizer.cli --config config.yaml --port 8080 --debug

  # Override color mode
  python -m visualizer.cli --config config.yaml --color-mode time

  # Combined overrides
  python -m visualizer.cli --config config.yaml --port 8080 --debug --color-mode dataset
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML or JSON configuration file",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Server port (overrides config)",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode (overrides config)",
    )

    parser.add_argument(
        "--color-mode",
        choices=["annotation", "time", "track_id", "dataset"],
        help="Default coloring mode (overrides config)",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"visualizer {__version__}",
    )

    return parser.parse_args()


def main():
    """
    Application entry point.

    Parses command-line arguments, loads configuration from file or uses
    example configuration, applies CLI overrides, and launches the Dash server.
    """
    args = parse_args()

    if args.config:
        config_path = Path(args.config)

        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return

        if config_path.suffix in [".yaml", ".yml"]:
            logger.info(f"Loading configuration from YAML: {config_path}")
            config = load_config_from_yaml(config_path)
        elif config_path.suffix == ".json":
            logger.info(f"Loading configuration from JSON: {config_path}")
            config = load_config_from_json(config_path)
        else:
            logger.error(
                f"Unsupported config file format: {config_path.suffix}. "
                "Supported formats: .yaml, .yml, .json"
            )
            return
    else:
        logger.info("Using example configuration from code")
        config = CONFIG

    if args.port or args.debug or args.color_mode:
        logger.info("Applying command-line overrides to configuration")
        config = config._replace(
            port=args.port if args.port else config.port,
            debug=args.debug if args.debug else config.debug,
            default_color_mode=args.color_mode
            if args.color_mode
            else config.default_color_mode,
        )

    logger.info(f"Loaded {len(config.datasets)} dataset(s) from configuration")
    logger.info(f"Starting server on port {config.port} (debug={config.debug})")

    app = create_app(config)

    run_app(app, debug=config.debug, port=config.port)


if __name__ == "__main__":
    main()
