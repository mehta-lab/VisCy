"""Visualization utilities (model-agnostic)."""

from viscy_utils.visualization.occlusion import (
    occlusion_saliency,
    saliency_to_rgb,
)
from viscy_utils.visualization.pca_rgb import pca_rgb_from_patches

__all__ = ["pca_rgb_from_patches", "occlusion_saliency", "saliency_to_rgb"]
