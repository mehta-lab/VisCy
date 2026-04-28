"""Single-cell feature orchestrator calling all feature modules."""

import numpy as np

from viscy_phenotyping.features_cp_measure import (
    cp_granularity_features,
    cp_intensity_features,
    cp_sizeshape_features,
    cp_texture_features,
)
from viscy_phenotyping.features_density import density_features
from viscy_phenotyping.features_gradient import gradient_features
from viscy_phenotyping.features_radial import concentric_uniformity_features, radial_distribution_features
from viscy_phenotyping.features_shape import shape_features
from viscy_phenotyping.features_structure import structure_features
from viscy_phenotyping.features_texture import texture_features

__all__ = ["compute_cell_features"]


def compute_cell_features(
    img_patch: np.ndarray,
    label_patch: np.ndarray,
    cell_id: int,
    channel_names: list[str],
) -> dict[str, float]:
    """Compute all image-based phenotyping features for a single cell.

    Parameters
    ----------
    img_patch : np.ndarray, shape (C, Y, X)
        Multi-channel fluorescence patch (already cropped to patch_size).
    label_patch : np.ndarray, shape (Y, X)
        Integer nuclear label patch. ``cell_id`` selects this cell's mask.
    cell_id : int
        Label ID of the target cell in ``label_patch``.
    channel_names : list[str]
        Names of channels in ``img_patch`` (used as feature prefixes).

    Returns
    -------
    dict[str, float]
        Flat dict of all features. Per-channel features are prefixed
        ``{channel_name}_`` (spaces replaced with underscores).
        Nuclear shape features (Problem 6) have no prefix.
    """
    mask = label_patch == cell_id
    if not mask.any():
        return {}

    features: dict[str, float] = {}

    # Problem 6: nuclear shape — mask only, no channel prefix
    features.update(shape_features(mask))

    # cp-measure: MeasureObjectSizeShape — mask only, no channel prefix
    features.update({"cp_" + k: v for k, v in cp_sizeshape_features(mask).items()})

    for ch_idx, ch_name in enumerate(channel_names):
        ch_img = img_patch[ch_idx].astype(np.float32)
        prefix = ch_name.replace(" ", "_") + "_"

        # Problem 1: radial distribution (nuclear centroid from mask; profile over full patch)
        features.update({prefix + k: v for k, v in radial_distribution_features(ch_img, mask).items()})
        # Problem 3: concentric ring uniformity (nuclear centroid from mask; profile over full patch)
        features.update({prefix + k: v for k, v in concentric_uniformity_features(ch_img, mask).items()})
        # Problem 2: texture / homogeneity (full patch)
        features.update({prefix + k: v for k, v in texture_features(ch_img).items()})
        # Problem 4: packing density (full patch)
        features.update({prefix + k: v for k, v in density_features(ch_img).items()})
        # Problem 5: edge count / strand continuity (full patch)
        features.update({prefix + k: v for k, v in structure_features(ch_img).items()})
        # Problem 7: gradient changes (full patch; mask used only for signal_to_background)
        features.update({prefix + k: v for k, v in gradient_features(ch_img, mask).items()})

        # cp-measure: MeasureObjectIntensity, MeasureTexture, MeasureGranularity
        features.update({prefix + "cp_" + k: v for k, v in cp_intensity_features(ch_img, mask).items()})
        features.update({prefix + "cp_" + k: v for k, v in cp_texture_features(ch_img, mask).items()})
        features.update({prefix + "cp_" + k: v for k, v in cp_granularity_features(ch_img, mask).items()})

    return features
