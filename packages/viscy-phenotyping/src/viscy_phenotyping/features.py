"""Nuclear morphology feature extraction from segmentation label images."""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

__all__ = [
    "NUCLEAR_MORPHOLOGY_PROPERTIES_2D",
    "NUCLEAR_MORPHOLOGY_PROPERTIES_3D",
    "extract_nuclear_morphology",
]

# 2-D nuclear mask properties — includes eccentricity, perimeter, orientation.
NUCLEAR_MORPHOLOGY_PROPERTIES_2D = (
    "label",
    "area",
    "eccentricity",
    "equivalent_diameter_area",
    "extent",
    "euler_number",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "perimeter",
    "solidity",
)

# 3-D nuclear mask properties — eccentricity/perimeter/orientation are 2-D only.
NUCLEAR_MORPHOLOGY_PROPERTIES_3D = (
    "label",
    "area",
    "equivalent_diameter_area",
    "extent",
    "euler_number",
    "major_axis_length",
    "minor_axis_length",
    "solidity",
    "inertia_tensor_eigvals",
)


def extract_nuclear_morphology(
    label_image: np.ndarray,
    label_ids: np.ndarray,
) -> pd.DataFrame:
    """Extract morphological features for requested label IDs from a label image.

    Parameters
    ----------
    label_image : np.ndarray
        Integer label image with shape ``(Y, X)`` or ``(Z, Y, X)``.
        Background must be 0; each unique nonzero integer identifies one nucleus.
    label_ids : np.ndarray
        1-D array of integer label IDs to measure. IDs absent from
        ``label_image`` are silently dropped from the output.

    Returns
    -------
    pd.DataFrame
        One row per found label ID. Columns depend on dimensionality:

        *2-D*: ``label``, ``area``, ``eccentricity``, ``equivalent_diameter_area``,
        ``extent``, ``euler_number``, ``major_axis_length``, ``minor_axis_length``,
        ``orientation``, ``perimeter``, ``solidity``, ``aspect_ratio``

        *3-D*: same minus 2-D-only columns, plus ``inertia_eigval_{0,1,2}``.
    """
    properties = (
        NUCLEAR_MORPHOLOGY_PROPERTIES_2D if label_image.ndim == 2 else NUCLEAR_MORPHOLOGY_PROPERTIES_3D
    )

    masked = np.where(np.isin(label_image, label_ids), label_image, 0).astype(label_image.dtype)
    props = regionprops_table(masked, properties=properties)
    df = pd.DataFrame(props)

    if label_image.ndim == 3:
        rename = {
            col: f"inertia_eigval_{col.split('-')[-1]}"
            for col in df.columns
            if col.startswith("inertia_tensor_eigvals")
        }
        df = df.rename(columns=rename)

    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"].clip(lower=1e-6)

    return df
