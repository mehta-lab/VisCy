"""Lightning CLI entry point for the translation application.

Usage
-----
python -m viscy_translation fit --config fit.yml
python -m viscy_translation predict --config predict.yml
"""

from viscy_utils.cli import main

if __name__ == "__main__":
    main()
