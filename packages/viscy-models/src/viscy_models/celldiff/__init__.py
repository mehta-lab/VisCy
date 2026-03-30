"""CellDiff models for virtual staining microscopy.

Provides deterministic and flow-matching 3D U-Net architectures with
Vision Transformer bottlenecks.

Requires optional dependencies: ``pip install viscy-models[celldiff]``
"""

from viscy_models.celldiff.celldiff_net import CELLDiffNet
from viscy_models.celldiff.unet_vit_3d import UNetViT3D

__all__ = ["CELLDiffNet", "UNetViT3D"]
