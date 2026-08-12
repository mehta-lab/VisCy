#!/usr/bin/env python
"""Write pooled control-MAD/PCA80 coordinates into embedding Zarr stores.

Run from the VisCy repository root:

    uv run python applications/dynaclr/scripts/evaluation/write_normalized_pca80.py \
        --config applications/dynaclr/configs/evaluation/recipes/mmd_pooled.yaml
"""

from dynaclr.evaluation.mmd.export_representation import main

if __name__ == "__main__":
    main()
