from typing import TypedDict

import mahotas as mh
import numpy as np
import pandas as pd
import scipy.stats
from numpy import fft
from numpy.typing import ArrayLike
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops


def normalize_image(image: ArrayLike) -> ArrayLike:
    """Normalize the image to 0-255 range."""
    return (image - image.min()) / (image.max() - image.min()) * 255


class IntensityFeatures(TypedDict):
    """Intensity features extracted from a single cell."""

    mean_intensity: float
    std_dev: float
    min_intensity: float
    max_intensity: float
    kurtosis: float
    skewness: float
    spectral_entropy: float
    iqr: float
    weighted_intensity_gradient: float


class TextureFeatures(TypedDict):
    """Texture features extracted from a single cell."""

    spectral_entropy: float
    contrast: float
    entropy: float
    homogeneity: float
    dissimilarity: float
    texture: float


class MorphologyFeatures(TypedDict):
    """Morphology features extracted from a single cell."""

    area: float
    perimeter: float
    perimeter_area_ratio: float
    eccentricity: float
    intensity_localization: float
    masked_intensity: float
    masked_area: float


class SymmetryDescriptor(TypedDict):
    """
    Symmetry descriptor extracted from a single cell
    """

    zernike_std: float
    zernike_mean: float
    radial_intensity_gradient: float


class TrackFeatures(TypedDict):
    """Track features extracted from a single track.

    Contains velocity-based features computed from the track's motion.
    """

    instantaneous_velocity: list[float]  # Array of velocities at each timepoint
    mean_velocity: float
    max_velocity: float
    min_velocity: float
    std_velocity: float


class DisplacementFeatures(TypedDict):
    """Displacement-based features extracted from a single track."""

    total_distance: float
    net_displacement: float
    directional_persistence: float


class AngularFeatures(TypedDict):
    """Angular features extracted from a single track."""

    mean_angular_velocity: float
    max_angular_velocity: float
    std_angular_velocity: float


class CellFeatures:
    """Cell features extracted from a single cell image patch."""

    def __init__(self, image: ArrayLike, segmentation_mask: ArrayLike = None):
        self.image = image
        self.segmentation_mask = segmentation_mask
        self.intensity_features = None
        self.texture_features = None
        self.morphology_features = None
        self.symmetry_descriptor = None

    def _compute_kurtosis(self):
        """Compute the kurtosis of the image."""
        normalized_image = normalize_image(self.image)
        kurtosis = scipy.stats.kurtosis(normalized_image, fisher=True, axis=None)
        return kurtosis

    def _compute_skewness(self):
        """Compute the skewness of the image."""
        normalized_image = normalize_image(self.image)
        skewness = scipy.stats.skew(normalized_image, axis=None)
        return skewness

    def _compute_glcm_features(self):
        """
        Compute the contrast, dissimilarity and homogeneity of the image
        Both sensor and phase texture changes when infected, smooth in sensor, and rough in phase
        :param np.array image: input image
        :return: contrast, dissimilarity, homogeneity
        """

        # Normalize the input image from 0 to 255
        image = (self.image - np.min(self.image)) * (
            255 / (np.max(self.image) - np.min(self.image))
        )
        image = image.astype(np.uint8)

        # Compute the GLCM
        distances = [1]  # Distance between pixels
        angles = [45]  # Angle in radians

        glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

        # Compute GLCM properties - average across all angles
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

        return contrast, dissimilarity, homogeneity

    def _compute_iqr(self):
        """
        Compute the interquartile range of pixel intensities
        Observed to increase when cell is infected
        :param np.array image: input image
        :return: interquartile range of pixel intensities
        """

        # Compute the interquartile range of pixel intensities
        iqr = np.percentile(self.image, 75) - np.percentile(self.image, 25)

        return iqr

    def _compute_weighted_intensity_gradient(self):
        """Compute the radial gradient profile and its slope.

        Args:
            image (np.ndarray): Input image

        Returns:
            float: Slope of the azimuthally averaged radial gradient profile
        """
        # Get image dimensions
        h, w = self.image.shape
        center_y, center_x = h // 2, w // 2

        # Create meshgrid of coordinates
        y, x = np.ogrid[:h, :w]

        # Calculate radial distances from center
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Calculate gradients in x and y directions
        gy, gx = np.gradient(self.image)

        # Calculate magnitude of gradient
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Weight gradient by intensity
        weighted_gradient = gradient_magnitude * self.image

        # Calculate maximum radius (to edge of image)
        max_radius = int(min(h // 2, w // 2))

        # Initialize arrays for radial profile
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        # Bin pixels by radius
        for i in range(h):
            for j in range(w):
                radius = int(r[i, j])
                if radius < max_radius:
                    radial_profile[radius] += weighted_gradient[i, j]
                    counts[radius] += 1

        # Average by counts (avoiding division by zero)
        valid_mask = counts > 0
        radial_profile[valid_mask] /= counts[valid_mask]

        # Calculate slope using linear regression
        x = np.arange(max_radius)[valid_mask]
        y = radial_profile[valid_mask]
        slope = np.polyfit(x, y, 1)[0]

        return slope

    def _compute_spectral_entropy(self):
        """
        Compute the spectral entropy of the image
        High frequency components are observed to increase in phase and reduce in sensor when cell is infected
        :param np.array image: input image
        :return: spectral entropy
        """

        # Compute the 2D Fourier Transform
        f_transform = fft.fft2(self.image)

        # Compute the power spectrum
        power_spectrum = np.abs(f_transform) ** 2

        # Compute the probability distribution
        power_spectrum += 1e-10  # Avoid log(0) issues
        prob_distribution = power_spectrum / np.sum(power_spectrum)

        # Compute the spectral entropy
        entropy = -np.sum(prob_distribution * np.log(prob_distribution))

        return entropy

    def _compute_texture_features(self):
        """Compute the texture features of the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            float: Mean of the texture features
        """
        # rescale image to 0 to 255 and convert to uint8
        image_rescaled = (
            (self.image - self.image.min())
            / (self.image.max() - self.image.min())
            * 255
        )
        texture_features = mh.features.haralick(image_rescaled.astype("uint8")).ptp(0)
        return np.mean(texture_features)

    def _compute_perimeter_area_ratio(self):
        """Compute the perimeter of the nuclear segmentations found inside the patch."""
        total_perimeter = 0
        total_area = 0

        # Use regionprops to analyze each labeled region
        regions = regionprops(self.segmentation_mask)

        if not regions:  # If no regions found
            return 0, 0, 0

        # Sum up perimeter and area for all regions
        for region in regions:
            total_perimeter += region.perimeter
            total_area += region.area

        average_area = total_area / len(regions)
        average_perimeter = total_perimeter / len(regions)

        return average_perimeter, average_area, total_perimeter / total_area

    def _compute_nucleus_eccentricity(self):
        """Compute the eccentricity of the nucleus.

        Returns:
            float: Average eccentricity of the nuclei in the image
        """
        # Use regionprops to analyze each labeled region
        regions = regionprops(self.segmentation_mask)

        if not regions:  # If no regions found
            return 0.0

        # Calculate mean eccentricity across all regions
        eccentricities = [region.eccentricity for region in regions]
        return float(np.mean(eccentricities))

    def _compute_Eucledian_distance_transform(self):
        """Compute the Euclidean distance transform of a binary mask.

        Args:
            image (np.ndarray): Binary mask (0s and 1s)

        Returns:
            np.ndarray: Distance transform where each pixel value represents
                    the Euclidean distance to the nearest non-zero pixel
        """
        # Ensure the image is binary
        binary_mask = (self.segmentation_mask > 0).astype(np.uint8)

        # Compute the distance transform using scikit-image
        from scipy.ndimage import distance_transform_edt

        dist_transform = distance_transform_edt(binary_mask)

        return dist_transform

    def _compute_intensity_localization(self):
        """Compute localization of fluor using Eucledian distance transformation and fluor intensity"""
        # compute EDT of mask
        edt = self._compute_Eucledian_distance_transform()
        # compute the intensity weighted center of the fluor
        intensity_weighted_center = np.sum(self.image * edt) / (np.sum(edt) + 1e-10)
        return intensity_weighted_center

    def _compute_area(self, sigma=0.6):
        """Create a binary mask using morphological operations
        Sensor area will increase when infected due to expression in nucleus
        :param np.array input_image: generate masks from this 3D image
        :param float sigma: Gaussian blur standard deviation, increase in value increases blur
        :return: area of the sensor mask & mean intensity inside the sensor area
        """

        input_image_blur = gaussian(self.image, sigma=sigma)

        thresh = threshold_otsu(input_image_blur)
        mask = self.image >= thresh

        # Apply sensor mask to the image
        masked_image = self.image * mask

        # Compute the mean intensity inside the sensor area
        masked_intensity = np.mean(masked_image)

        return masked_intensity, np.sum(mask)

    def _compute_zernike_moments(self):
        """Compute the Zernike moments of the image"""
        zernike_moments = mh.features.zernike_moments(self.image, 32)
        return zernike_moments

    def _compute_radial_intensity_gradient(self):
        """
        Compute the radial intensity gradient of the image
        The sensor relocalizes inside the nucleus, which is center of the image when cells are infected
        Expected negative gradient when infected and zero to positive gradient when not infected
        :param np.array image: input image
        :return: radial intensity gradient
        """
        # get the slope radial_intensity_values
        from scipy.stats import linregress

        # normalize the image
        normalized_image = (self.image - np.min(self.image)) / (
            np.max(self.image) - np.min(self.image)
        )

        # compute the intensity gradient from center to periphery
        y, x = np.indices(normalized_image.shape)
        center = np.array(normalized_image.shape) / 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), normalized_image.ravel())
        nr = np.bincount(r.ravel())
        radial_intensity_values = tbin / nr

        radial_intensity_gradient = linregress(
            range(len(radial_intensity_values)), radial_intensity_values
        )

        return radial_intensity_gradient[0]

    def compute_intensity_features(self):
        """Compute intensity features."""
        self.intensity_features = IntensityFeatures(
            mean_intensity=float(np.mean(self.image)),
            std_dev=float(np.std(self.image)),
            min_intensity=float(np.min(self.image)),
            max_intensity=float(np.max(self.image)),
            kurtosis=self._compute_kurtosis(),
            skewness=self._compute_skewness(),
            spectral_entropy=self._compute_spectral_entropy(),
            iqr=self._compute_iqr(),
            weighted_intensity_gradient=self._compute_weighted_intensity_gradient(),
        )

    def compute_texture_features(self):
        """Compute texture features."""
        contrast, dissimilarity, homogeneity = self._compute_glcm_features()
        self.texture_features = TextureFeatures(
            spectral_entropy=self._compute_spectral_entropy(),
            contrast=contrast,
            entropy=self._compute_spectral_entropy(),  # Note: This could be redundant
            homogeneity=homogeneity,
            dissimilarity=dissimilarity,
            texture=self._compute_texture_features(),
        )

    def compute_morphology_features(self):
        """Compute morphology features."""
        assert self.segmentation_mask is not None, "Segmentation mask is required"

        masked_intensity, masked_area = self._compute_area()
        perimeter, area, ratio = self._compute_perimeter_area_ratio()
        self.morphology_features = MorphologyFeatures(
            area=area,
            perimeter=perimeter,
            perimeter_area_ratio=ratio,
            eccentricity=self._compute_nucleus_eccentricity(),
            intensity_localization=self._compute_intensity_localization(),
            masked_intensity=masked_intensity,
            masked_area=masked_area,
        )
    
    def compute_symmetry_descriptor(self):
        """Compute the symmetry descriptor of the image"""
        self.symmetry_descriptor = SymmetryDescriptor(
            zernike_std=np.std(self._compute_zernike_moments()),
            zernike_mean=np.mean(self._compute_zernike_moments()),
            radial_intensity_gradient=self._compute_radial_intensity_gradient(),
        )

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all features."""
        # Compute intensity features
        self.compute_intensity_features()

        # Compute texture features
        self.compute_texture_features()

        # Compute symmetry descriptor
        self.compute_symmetry_descriptor()

        if self.segmentation_mask is not None:
            self.compute_morphology_features()

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
        if self.symmetry_descriptor:
            features_dict.update(self.symmetry_descriptor)
        return pd.DataFrame([features_dict])


class DynamicFeatures:
    """Dynamic track based features extracted from tracks.

    Parameters:
        tracking_df: Tracking dataframe containing at least the track_id, t, x, y columns

    Attributes:
        tracking_df: Original tracking dataframe
        track_features: Computed track/velocity features
        displacement_features: Computed displacement features
        angular_features: Computed angular features
    """

    def __init__(self, tracking_df: pd.DataFrame):
        self.tracking_df = tracking_df
        self.track_features = None
        self.displacement_features = None
        self.angular_features = None

        # Verify required columns exist
        required_cols = ["track_id", "t", "x", "y"]
        missing_cols = [col for col in required_cols if col not in tracking_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. Check tracking_df"
            )

    def _compute_instantaneous_velocity(self, track_id: str) -> np.ndarray:
        """Compute the instantaneous velocity for all timepoints in a track.

        Args:
            track_id: ID of the track to compute velocities for

        Returns:
            np.ndarray: Array of instantaneous velocities for each timepoint
        """
        # Get track data sorted by time
        track_data = self.tracking_df[
            self.tracking_df["track_id"] == track_id
        ].sort_values("t")

        # TODO: decide if we want to return nans or zeros
        if len(track_data) < 2:
            return np.array([0.0])  # Return zero velocity for single-point tracks

        # Calculate displacements between consecutive points
        dx = np.diff(track_data["x"].values)
        dy = np.diff(track_data["y"].values)
        dt = np.diff(track_data["t"].values)

        # Compute distances
        distances = np.sqrt(dx**2 + dy**2)

        # Compute velocities (avoid division by zero)
        velocities = np.zeros(len(track_data))
        velocities[1:] = distances / np.maximum(dt, 1e-6)

        return velocities

    def _compute_displacement(self, track_id: str) -> tuple[float, float, float]:
        """Compute displacement-based features.

        Args:
            track_id: ID of the track to compute displacement for

        Returns:
            tuple: (total_distance, net_displacement, directional_persistence)
        """
        track_data = self.tracking_df[
            self.tracking_df["track_id"] == track_id
        ].sort_values("t")

        if len(track_data) < 2:
            return 0.0, 0.0, 0.0

        # Compute total distance
        dx = np.diff(track_data["x"].values)
        dy = np.diff(track_data["y"].values)
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)

        # Compute net displacement
        start_point = track_data.iloc[0][["x", "y"]].values
        end_point = track_data.iloc[-1][["x", "y"]].values
        net_displacement = np.sqrt(np.sum((end_point - start_point) ** 2))

        # Compute directional persistence
        directional_persistence = (
            net_displacement / total_distance if total_distance > 0 else 0.0
        )

        return total_distance, net_displacement, directional_persistence

    def _compute_angular_velocity(self, track_id: str) -> tuple[float, float, float]:
        """Compute angular velocity features.

        Args:
            track_id: ID of the track to compute angular velocity for

        Returns:
            tuple: (mean_angular_velocity, max_angular_velocity, std_angular_velocity)
        """
        track_data = self.tracking_df[
            self.tracking_df["track_id"] == track_id
        ].sort_values("t")

        if len(track_data) < 3:  # Need at least 3 points to compute angle changes
            return 0.0, 0.0, 0.0

        # Compute vectors between consecutive points
        dx = np.diff(track_data["x"].values)
        dy = np.diff(track_data["y"].values)
        dt = np.diff(track_data["t"].values)

        # Compute angles between consecutive vectors
        vectors = np.column_stack([dx, dy])
        angles = np.zeros(len(vectors) - 1)
        for i in range(len(vectors) - 1):
            v1, v2 = vectors[i], vectors[i + 1]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
            )
            angles[i] = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Compute angular velocities (change in angle over time)
        angular_velocities = angles / (dt[1:] + 1e-10)

        return (
            float(np.mean(angular_velocities)),
            float(np.max(angular_velocities)),
            float(np.std(angular_velocities)),
        )

    def compute_all_features(self, track_id: str) -> pd.DataFrame:
        """Compute all dynamic features for a given track.

        Args:
            track_id: ID of the track to compute features for

        Returns:
            pd.DataFrame: DataFrame containing all computed features
        """
        # Compute velocity features
        velocities = self._compute_instantaneous_velocity(track_id)
        self.velocity_features = TrackFeatures(
            instantaneous_velocity=velocities.tolist(),
            mean_velocity=float(np.mean(velocities)),
            max_velocity=float(np.max(velocities)),
            min_velocity=float(np.min(velocities)),
            std_velocity=float(np.std(velocities)),
        )

        # Compute displacement features
        total_dist, net_disp, dir_persist = self._compute_displacement(track_id)
        self.displacement_features = DisplacementFeatures(
            total_distance=total_dist,
            net_displacement=net_disp,
            directional_persistence=dir_persist,
        )

        # Compute angular features
        mean_ang, max_ang, std_ang = self._compute_angular_velocity(track_id)
        self.angular_features = AngularFeatures(
            mean_angular_velocity=mean_ang,
            max_angular_velocity=max_ang,
            std_angular_velocity=std_ang,
        )

        return self.to_df()

    def to_df(self) -> pd.DataFrame:
        """Convert all features to a pandas DataFrame."""
        features_dict = {}
        if self.velocity_features:
            features_dict.update(self.velocity_features)
        if self.displacement_features:
            features_dict.update(self.displacement_features)
        if self.angular_features:
            features_dict.update(self.angular_features)
        return pd.DataFrame([features_dict])
