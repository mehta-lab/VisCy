"""Lightning CLI entry point for the CellDiff application.

Usage
-----
python -m celldiff fit --config fit.yml
python -m celldiff predict --config predict.yml
"""

from viscy_utils.cli import main

if __name__ == "__main__":
    main()
