"""Lightning CLI entry point for the Cytoland application.

Usage
-----
python -m cytoland fit --config vscyto3d/finetune.yml
python -m cytoland predict --config vscyto3d/predict.yml
"""

from viscy_utils.cli import main

if __name__ == "__main__":
    main()
