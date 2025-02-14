from typing import TypedDict

import numpy as np
import pandas as pd
from mahotas.features import haralick, zernike_moments
from numpy.typing import ArrayLike
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops


class IntensityFeatures(TypedDict):
    """
    Intensity features extracted from a single cell.
    """

    mean: float
    std: float
    min: float
    max: float


class TextureFeatures(TypedDict):
    """
    Texture features extracted from a single cell
    """

    spectral_entropy: float
    contrast: float
    entropy: float
    homogeneity: float
    dissimilarity: float


class MorphologyFeatures(TypedDict):
    """
    Morphology features extracted from a single cell
    """

    area: float
    perimeter: float
    circularity: float
    eccentricity: float


class SymmetryDescriptor(TypedDict):
    """
    Symmetry descriptor extracted from a single cell
    """

    zerknike_moments: ArrayLike
    radial_intenstiy_gradient: ArrayLike


class TrackFeatures(TypedDict):
    """Track features extracted from a single track"""

    instantaneous_velocity: float


class CellFeatures:
    """
    Cell features extracted from a single cell image patch

    Parameters:
        image: Image patch with shape (Y,X)
        segmentation_mask: Segmentation mask with shape (C,Y,X)

    Attributes:
        intensity_features: Intensity features
        texture_features: Texture features
        morphology_features: Morphology features
        symmetry_descriptor: Symmetry descriptor

    """

    def __init__(
        self,
        image: ArrayLike,
        segmentation_mask: ArrayLike = None,
    ):
        self.image = image
        self.segmentation_mask = segmentation_mask

        # Feature dictionaries
        self.intensity_features: IntensityFeatures | None = None
        self.texture_features: TextureFeatures | None = None
        self.morphology_features: dict[str, dict] | None = None
        self.symmetry_descriptor: SymmetryDescriptor | None = None

    def compute_intensity_features(self) -> IntensityFeatures:
        """Compute intensity features."""
        return {
            "mean": np.mean(self.image),
            "std": np.std(self.image),
            "min": np.min(self.image),
            "max": np.max(self.image),
        }

    def compute_texture_features(self) -> TextureFeatures:
        """
        Compute texture features
        """
        texture_features = {}

        texture_features["spectral_entropy"] = self._compute_spectral_entropy()
        # TODO: Add contrast, entropy, homogeneity, dissimilarity
        # self._compute_contrast()
        # self._compute_entropy()
        # self._compute_homogeneity()
        # self._compute_dissimilarity()

        raise NotImplementedError

    def _compute_spectral_entropy(self):
        """Compute spectral entropy"""
        # TODO: Implement spectral entropy

        raise NotImplementedError

    def compute_morphology_features(
        self,
        segmentation_channel_names: list[str],
        properties: list[str] = ["area", "perimeter", "circularity", "eccentricity"],
    ) -> dict[str, dict]:
        """
        Compute morphology features for each segmentation channel.

        Args:
            segmentation_channel_names: Names of channels in segmentation mask
            properties: List of regionprops properties to compute

        Returns:
            Dictionary mapping channel names to their morphology features
        """
        assert self.segmentation_mask is not None, "Segmentation mask is required"
        assert (
            len(segmentation_channel_names) == self.segmentation_mask.shape[0]
        ), "Number of channel names must match number of channels in mask"

        morphology_features = {}
        for idx, channel_name in enumerate(segmentation_channel_names):
            channel_mask = self.segmentation_mask[idx]
            regionprops_dict = regionprops(channel_mask)[0]

            # Extract only requested properties
            morphology_features[channel_name] = {
                prop: getattr(regionprops_dict, prop) for prop in properties
            }

        self.morphology_features = morphology_features
        return morphology_features

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all available features."""
        self.intensity_features = self.compute_intensity_features()
        self.texture_features = self.compute_texture_features()
        self.morphology_features = self.compute_morphology_features()
        return self.to_df()

    def to_df(self) -> pd.DataFrame:
        """Convert all features to a pandas DataFrame."""
        features_dict = {}
        if self.intensity_features:
            features_dict.update(self.intensity_features)
        if self.texture_features:
            features_dict.update(self.texture_features)
        if self.morphology_features:
            features_dict.update(self.morphology_features)

        return pd.DataFrame(features_dict)


class DynamicFeatures:
    """
    Dyanamic track based features extracted from a single track

    Parameters:
        tracking_df: Tracking dataframe

    Attributes:
        track_features: Track features
    """

    def __init__(self, tracking_df: pd.DataFrame):
        self.tracking_df = tracking_df
        self.track_features: TrackFeatures | None = None

    def compute_instantaneous_velocity(self) -> float:
        """Compute instantaneous velocity"""

        raise NotImplementedError

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all available features."""
        self.track_features = self.compute_instantaneous_velocity()
        return self.to_df()

    def to_df(self) -> pd.DataFrame:
        """Convert all features to a pandas DataFrame."""
        return pd.DataFrame(self.track_features)
