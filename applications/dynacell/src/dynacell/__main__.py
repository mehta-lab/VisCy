"""Lightning CLI entry point for the Dynacell application.

Usage
-----
cd applications/dynacell/examples/configs
dynacell fit -c unetvit3d/fit.yml
python -m dynacell fit --config unetvit3d/fit.yml
"""

from viscy_utils.cli import main


def main_cli():
    """Console script entry point for ``dynacell`` command."""
    main()


if __name__ == "__main__":
    main()
