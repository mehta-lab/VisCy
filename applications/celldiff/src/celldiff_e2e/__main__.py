"""Lightning CLI entry point for the CellDiff application.

Usage
-----
python -m celldiff_e2e fit --config fit_unetvit3d.yml
python -m celldiff_e2e predict --config predict_unetvit3d.yml
"""

from viscy_utils.cli import main

if __name__ == "__main__":
    main()
