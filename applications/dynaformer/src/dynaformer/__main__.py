"""Lightning CLI entry point for the Dynaformer application.

Usage
-----
python -m dynaformer fit --config fucci/fit_unext2.yml
python -m dynaformer predict --config fucci/predict.yml
"""

from viscy_utils.cli import main

if __name__ == "__main__":
    main()
