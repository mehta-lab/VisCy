import warnings as warning

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skimage import measure
from skimage.feature import graycomatrix, graycoprops


def extract_features_zyx(
    labels_zyx: ArrayLike,
    intensity_zyx: ArrayLike = None,
    frangi_zyx: ArrayLike = None,
    spacing: tuple = (1.0, 1.0),
    properties: list = None,
    extra_properties: list = None,
):
    """
    Extract morphological and intensity features from labeled organelles.

    Handles both 2D (Z=1) and 3D (Z>1) data automatically
    For 2D data, processes the single Z-slice. For 3D data, performs max projection
    along Z axis before feature extraction.

    Based on:
    Lefebvre, A.E.Y.T., Sturm, G., Lin, TY. et al.
    Nellie (2025) https://doi.org/10.1038/s41592-025-02612-7

    Parameters
    ----------
    labels_zyx : ndarray
        Labeled segmentation mask with shape (Z, Y, X).
        Each unique integer value represents a different organelle instance.
    intensity_zyx : ndarray, optional
        Original intensity image with shape (Z, Y, X) for computing
        intensity-based features. If None, only morphological features computed.
    frangi_image : ndarray, optional
        Frangi vesselness response with shape (Z, Y, X) for computing
        tubularity/filament features.
    spacing : tuple
        Physical spacing in same units (e.g., µm).
        For 2D (Z=1): (pixel_size_y, pixel_size_x)
        For 3D (Z>1): (pixel_size_z, pixel_size_y, pixel_size_x)
    properties : list of str, optional
        List of standard regionprops features to compute. If None, uses default set.
        Available: 'label', 'area', 'perimeter', 'axis_major_length',
        'axis_minor_length', 'solidity', 'extent', 'orientation',
        'equivalent_diameter_area', 'convex_area', 'eccentricity',
        'mean_intensity', 'min_intensity', 'max_intensity'
    extra_properties : list of str, optional
        Additional features beyond standard regionprops. Options:
        - 'moments_hu': Hu moments (shape descriptors, 7 features)
        - 'texture': Haralick texture features (4 features: contrast, homogeneity, energy, correlation)
        - 'aspect_ratio': Major axis / minor axis ratio
        - 'circularity':  area / perimeter
        - 'frangi_intensity': Mean/min/max/sum/std of Frangi vesselness
        - 'feret_diameter_max': Maximum Feret diameter (expensive)
        - 'sum_intensity': Sum of intensity values
        - 'std_intensity': Standard deviation of intensity values

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame where each row represents one labeled object with columns
        for each computed feature. Always includes 'label' and 'channel' columns.

    Examples
    --------
    >>> # Basic morphology only
    >>> df = extract_features_zyx(labels_zyx)

    >>> # With intensity features
    >>> df = extract_features_zyx(labels_zyx, intensity_zyx=intensity)

    >>> # Custom property selection
    >>> df = extract_features_zyx(
    ...     labels_zyx,
    ...     intensity_zyx=intensity,
    ...     properties=['label', 'area', 'mean_intensity'],
    ...     extra_properties=['aspect_ratio', 'circularity']
    ... )

    >>> # Full feature set including Frangi
    >>> df = extract_features_zyx(
    ...     labels_zyx,
    ...     intensity_zyx=intensity,
    ...     frangi_image=vesselness,
    ...     extra_properties=['moments_hu', 'texture', 'frangi_intensity']
    ... )
    """

    if intensity_zyx is not None:
        assert (
            intensity_zyx.shape == labels_zyx.shape
        ), "Image and labels must have same shape"

    Z, _, _ = labels_zyx.shape

    # Default properties if not specified
    if properties is None:
        properties = [
            "label",
            "area",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "extent",
            "orientation",
            "equivalent_diameter_area",
            "convex_area",
            "eccentricity",
        ]
        # Add intensity features if image provided
        if intensity_zyx is not None:
            properties.extend(["mean_intensity", "min_intensity", "max_intensity"])

    if extra_properties is None:
        extra_properties = []

    # Determine 2D vs 3D mode and set appropriate spacing
    spacing_2d = spacing if len(spacing) == 2 else spacing[-2:]

    if Z == 1:
        # Squeeze Z dimension for 2D processing
        labels_processed = labels_zyx[0]  # Shape: (Y, X)
        intensity_processed = intensity_zyx[0] if intensity_zyx is not None else None
        frangi_processed = frangi_zyx[0] if frangi_zyx is not None else None
    else:
        # Use max projection along Z for 3D -> 2D
        labels_processed = np.max(labels_zyx, axis=0)  # Shape: (Y, X)
        intensity_processed = (
            np.max(intensity_zyx, axis=0) if intensity_zyx is not None else None
        )
        frangi_processed = (
            np.max(frangi_zyx, axis=0) if frangi_zyx is not None else None
        )

    # Check if we have any objects to process
    if labels_processed.max() == 0:
        warning.warn(f"Warning: No objects found")

    # Compute base regionprops features (those that support spacing)
    props_with_spacing = [p for p in properties if p not in ["moments_hu"]]

    try:
        props_dict = measure.regionprops_table(
            labels_processed,
            intensity_image=intensity_processed,
            properties=tuple(props_with_spacing),
            spacing=spacing_2d,
        )
        df = pd.DataFrame(props_dict)
    except Exception as e:
        warning.warn(f"Error computing base regionprops: {e}")

    # Add Hu moments separately (without spacing)
    if "moments_hu" in properties or "moments_hu" in extra_properties:
        try:
            hu_props = measure.regionprops_table(
                labels_processed, properties=("label", "moments_hu"), spacing=(1, 1)
            )
            hu_df = pd.DataFrame(hu_props)
            # Rename columns to be clearer
            hu_rename = {f"moments_hu-{i}": f"hu_moment_{i}" for i in range(7)}
            hu_df = hu_df.rename(columns=hu_rename)
            df = df.merge(hu_df, on="label", how="left")
        except Exception as e:
            warning.warn(f"Could not compute Hu moments: {e}")

    # Add derived metrics
    if "aspect_ratio" in extra_properties:
        minor_axis = df["axis_minor_length"].replace(0, 1)  # Avoid division by zero
        df["aspect_ratio"] = df["axis_major_length"] / minor_axis

    if "circularity" in extra_properties:
        perimeter_sq = df["perimeter"] ** 2
        df["circularity"] = np.divide(
            4 * np.pi * df["area"],
            perimeter_sq,
            out=np.ones_like(perimeter_sq),
            where=perimeter_sq != 0,
        )

    # Add expensive/iterative features
    if any(
        prop in extra_properties
        for prop in ["texture", "feret_diameter_max", "frangi_intensity"]
    ):
        regions = measure.regionprops(
            labels_processed, intensity_image=intensity_processed
        )
        extra_features = []

        for region in regions:
            features = {"label": region.label}

            # Haralick texture features
            if "texture" in extra_properties and intensity_processed is not None:
                min_r, min_c, max_r, max_c = region.bbox
                region_intensity = (
                    intensity_processed[min_r:max_r, min_c:max_c] * region.image
                )

                # Normalize to uint8
                min_val, max_val = region_intensity.min(), region_intensity.max()
                if max_val > min_val:
                    region_uint8 = (
                        (region_intensity - min_val) / (max_val - min_val) * 255
                    ).astype(np.uint8)
                else:
                    region_uint8 = np.zeros_like(region_intensity, dtype=np.uint8)

                try:
                    glcm = graycomatrix(
                        region_uint8,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True,
                    )
                    features["texture_contrast"] = graycoprops(glcm, "contrast")[0, 0]
                    features["texture_homogeneity"] = graycoprops(glcm, "homogeneity")[
                        0, 0
                    ]
                    features["texture_energy"] = graycoprops(glcm, "energy")[0, 0]
                    features["texture_correlation"] = graycoprops(glcm, "correlation")[
                        0, 0
                    ]
                except Exception:
                    features["texture_contrast"] = np.nan
                    features["texture_homogeneity"] = np.nan
                    features["texture_energy"] = np.nan
                    features["texture_correlation"] = np.nan

            # Feret diameter
            if "feret_diameter_max" in extra_properties:
                features["feret_diameter_max"] = region.feret_diameter_max

            # Frangi intensity features
            if "frangi_intensity" in extra_properties and frangi_processed is not None:
                min_r, min_c, max_r, max_c = region.bbox
                region_frangi = frangi_processed[min_r:max_r, min_c:max_c][region.image]

                if region_frangi.size > 0:
                    features["frangi_mean_intensity"] = np.mean(region_frangi)
                    features["frangi_min_intensity"] = np.min(region_frangi)
                    features["frangi_max_intensity"] = np.max(region_frangi)
                    features["frangi_sum_intensity"] = np.sum(region_frangi)
                    features["frangi_std_intensity"] = np.std(region_frangi)
                else:
                    features["frangi_mean_intensity"] = np.nan
                    features["frangi_min_intensity"] = np.nan
                    features["frangi_max_intensity"] = np.nan
                    features["frangi_sum_intensity"] = np.nan
                    features["frangi_std_intensity"] = np.nan

            extra_features.append(features)

        if extra_features:
            extra_df = pd.DataFrame(extra_features)
            df = df.merge(extra_df, on="label", how="left")

    # Add sum and std intensity if we have intensity image
    if intensity_processed is not None and (
        "sum_intensity" in extra_properties or "std_intensity" in extra_properties
    ):
        regions = measure.regionprops(
            labels_processed, intensity_image=intensity_processed
        )
        sum_std_features = []

        for region in regions:
            min_r, min_c, max_r, max_c = region.bbox
            region_pixels = intensity_processed[min_r:max_r, min_c:max_c][region.image]

            features = {"label": region.label}
            if region_pixels.size > 0:
                if "sum_intensity" in extra_properties:
                    features["sum_intensity"] = np.sum(region_pixels)
                if "std_intensity" in extra_properties:
                    features["std_intensity"] = np.std(region_pixels)
            else:
                if "sum_intensity" in extra_properties:
                    features["sum_intensity"] = np.nan
                if "std_intensity" in extra_properties:
                    features["std_intensity"] = np.nan

            sum_std_features.append(features)

        if sum_std_features:
            sum_std_df = pd.DataFrame(sum_std_features)
            df = df.merge(sum_std_df, on="label", how="left")

    return df
