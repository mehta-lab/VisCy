from typing import TypedDict

import mahotas as mh
import numpy as np
import pandas as pd
import scipy.stats
from numpy import fft
from numpy.typing import ArrayLike
from scipy.ndimage import distance_transform_edt
from scipy.stats import linregress
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops


class IntensityFeatures(TypedDict):
    """Intensity-based features extracted from a single cell."""

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
    """Texture-based features extracted from a single cell."""

    spectral_entropy: float
    contrast: float
    entropy: float
    homogeneity: float
    dissimilarity: float
    texture: float


class MorphologyFeatures(TypedDict):
    """Morphological features extracted from a single cell."""

    area: float
    perimeter: float
    perimeter_area_ratio: float
    eccentricity: float
    intensity_localization: float
    masked_intensity: float
    masked_area: float


class SymmetryDescriptor(TypedDict):
    """Symmetry-based features extracted from a single cell."""

    zernike_std: float
    zernike_mean: float
    radial_intensity_gradient: float


class TrackFeatures(TypedDict):
    """Velocity-based features extracted from a single track."""

    instantaneous_velocity: list[float]
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
    """Class for computing various features from a single cell image patch.

    This class provides methods to compute intensity, texture, morphological,
    and symmetry features from a cell image and its segmentation mask.

    Parameters
    ----------
    image : ArrayLike
        Input image array of the cell.
    segmentation_mask : ArrayLike, optional
        Binary mask of the cell segmentation, by default None.

    Attributes
    ----------
    image : ArrayLike
        Input image array.
    segmentation_mask : ArrayLike
        Binary segmentation mask.
    intensity_features : IntensityFeatures
        Computed intensity features.
    texture_features : TextureFeatures
        Computed texture features.
    morphology_features : MorphologyFeatures
        Computed morphological features.
    symmetry_descriptor : SymmetryDescriptor
        Computed symmetry features.
    """

    def __init__(self, image: ArrayLike, segmentation_mask: ArrayLike | None = None):
        self.image = image
        self.segmentation_mask = segmentation_mask
        self.image_normalized = rescale_intensity(self.image, out_range=(0, 1))

        # Initialize feature containers
        self.intensity_features = None
        self.texture_features = None
        self.morphology_features = None
        self.symmetry_descriptor = None

        self._eps = 1e-10

    def _compute_kurtosis(self):
        """Compute the kurtosis of the image.

        Returns
        -------
        kurtosis: float
            Kurtosis of the image intensity distribution (scale-invariant).
            Returns nan for constant arrays.
        """
        if np.std(self.image) == 0:
            return np.nan
        return scipy.stats.kurtosis(self.image, fisher=True, axis=None)

    def _compute_skewness(self):
        """Compute the skewness of the image.

        Returns
        -------
        skewness: float
            Skewness of the image intensity distribution (scale-invariant).
            Returns nan for constant arrays.
        """
        if np.std(self.image) == 0:
            return np.nan
        return scipy.stats.skew(self.image, axis=None)

    def _compute_glcm_features(self):
        """Compute GLCM-based texture features from the image.

        Converts normalized image to uint8 for GLCM computation.
        """
        # Convert 0-1 normalized image to uint8 (0-255)
        image_uint8 = (self.image_normalized * 255).astype(np.uint8)

        glcm = graycomatrix(image_uint8, [1], [45], symmetric=True, normed=True)

        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

        return contrast, dissimilarity, homogeneity

    def _compute_iqr(self):
        """Compute the interquartile range of pixel intensities.

        The IQR is observed to increase when a cell is infected,
        providing a measure of intensity distribution spread.

        Returns
        -------
        iqr: float
            Interquartile range of pixel intensities.
        """
        iqr = np.percentile(self.image, 75) - np.percentile(self.image, 25)

        return iqr

    def _compute_weighted_intensity_gradient(self):
        """Compute the weighted radial intensity gradient profile.

        Calculates the slope of the azimuthally averaged radial gradient
        profile, weighted by intensity. This provides information about
        how intensity changes with distance from the cell center.

        Returns
        -------
        slope: float
            Slope of the weighted radial intensity gradient profile.
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
        """Compute the spectral entropy of the image.

        Spectral entropy measures the complexity of the image's frequency
        components. High frequency components are observed to increase in
        phase and reduce in sensor when a cell is infected.

        Returns
        -------
        entropy: float
            Spectral entropy of the image.
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
        """Compute Haralick texture features from the image.

        Converts normalized image to uint8 for Haralick computation.
        """
        # Convert 0-1 normalized image to uint8 (0-255)
        image_uint8 = (self.image_normalized * 255).astype(np.uint8)
        texture_features = mh.features.haralick(image_uint8)
        return np.mean(np.ptp(texture_features, axis=0))

    def _compute_perimeter_area_ratio(self):
        """Compute the perimeter of the nuclear segmentations found inside the patch.

        This function calculates the average perimeter, average area, and their ratio
        for all nuclear segmentations in the patch.

        Returns
        -------
        average_perimeter, average_area, ratio: tuple
            Tuple containing:
            - average_perimeter : float
                Average perimeter of all regions in the patch
            - average_area : float
                Average area of all regions
            - ratio : float
                Ratio of total perimeter to total area
        """
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

        Eccentricity measures how much the nucleus deviates from
        a perfect circle, with 0 being perfectly circular and 1
        being a line segment.

        Returns
        -------
        eccentricity: float
            Eccentricity of the nucleus (0 to 1).
        """
        # Use regionprops to analyze each labeled region
        regions = regionprops(self.segmentation_mask)

        if not regions:  # If no regions found
            return 0.0

        # Calculate mean eccentricity across all regions
        eccentricities = [region.eccentricity for region in regions]
        return float(np.mean(eccentricities))

    def _compute_Eucledian_distance_transform(self):
        """Compute the Euclidean distance transform of the segmentation mask.

        This transform computes the distance from each pixel to the
        nearest background pixel, providing information about the
        spatial distribution of the cell.

        Returns
        -------
        dist_transform: ndarray
            Distance transform of the segmentation mask.
        """
        # Ensure the image is binary
        binary_mask = (self.segmentation_mask > 0).astype(np.uint8)

        # Compute the distance transform using scikit-image
        dist_transform = distance_transform_edt(binary_mask)

        return dist_transform

    def _compute_intensity_localization(self):
        """Compute localization of fluor using Eucledian distance transformation and fluor intensity.

        This function computes the intensity-weighted center of the fluor
        using the Euclidean distance transform of the segmentation mask.
        The intensity-weighted center is calculated as the sum of the
        product of the image intensity and the distance transform,
        divided by the sum of the distance transform.

        Returns
        -------
        intensity_weighted_center: float
            Intensity-weighted center of the fluor.
        """
        # compute EDT of mask
        edt = self._compute_Eucledian_distance_transform()
        # compute the intensity weighted center of the fluor
        intensity_weighted_center = np.sum(self.image * edt) / (np.sum(edt) + self._eps)
        return intensity_weighted_center

    def _compute_area(self, sigma=0.6):
        """Create a binary mask using morphological operations.

        This function creates a binary mask from the input image using Gaussian blur
        and Otsu thresholding. The sensor area will increase when infected due to
        expression in nucleus.

        Parameters
        ----------
        sigma : float
            Gaussian blur standard deviation. Increasing this value increases the blur,
            by default 0.6

        Returns
        -------
        masked_intensity, masked_area: tuple
            Tuple containing:
            - masked_intensity : float
                Mean intensity inside the sensor area
            - masked_area : float
                Area of the sensor mask in pixels
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
        """Compute the Zernike moments of the image.

        Zernike moments are a set of orthogonal moments that capture
        the shape of the image. They are invariant to translation, rotation,
        and scale.

        Returns
        -------
        zernike_moments: np.ndarray
            Zernike moments of the image.
        """
        zernike_moments = mh.features.zernike_moments(self.image, 32)
        return zernike_moments

    def _compute_radial_intensity_gradient(self):
        """Compute the radial intensity gradient of the image.

        Uses 0-1 normalized image directly for gradient calculation.
        """
        # Use 0-1 normalized image directly
        y, x = np.indices(self.image_normalized.shape)
        center = np.array(self.image_normalized.shape) / 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)

        tbin = np.bincount(r.ravel(), self.image_normalized.ravel())
        nr = np.bincount(r.ravel())
        radial_intensity_values = tbin / nr

        radial_intensity_gradient = linregress(
            range(len(radial_intensity_values)), radial_intensity_values
        )

        return radial_intensity_gradient[0]

    def compute_intensity_features(self):
        """Compute intensity features.

        This function computes various intensity-based features from the input image.
        It calculates the mean, standard deviation, minimum, maximum, kurtosis,
        skewness, spectral entropy, interquartile range, and weighted intensity gradient.

        Returns
        -------
        IntensityFeatures
            Dictionary containing all computed intensity features.
        """
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
        """Compute texture features.

        This function computes texture features from the input image.
        It calculates the spectral entropy, contrast, entropy, homogeneity,
        dissimilarity, and texture features.

        Returns
        -------
        TextureFeatures
            Dictionary containing all computed texture features.
        """
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
        """Compute morphology features.

        This function computes morphology features from the input image.
        It calculates the area, perimeter, perimeter-to-area ratio,
        eccentricity, intensity localization, masked intensity, and masked area.

        Returns
        -------
        MorphologyFeatures
            Dictionary containing all computed morphology features.

        Raises
        ------
        AssertionError
            If segmentation mask is None or empty
        """
        if self.segmentation_mask is None:
            raise AssertionError("Segmentation mask is required")

        if np.sum(self.segmentation_mask) == 0:
            raise AssertionError("Segmentation mask is empty")

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
        """Compute the symmetry descriptor of the image.

        This function computes the symmetry descriptor of the image.
        It calculates the Zernike moments, Zernike mean, and radial intensity gradient.

        Returns
        -------
        SymmetryDescriptor
            Dictionary containing all computed symmetry descriptor features.
        """
        self.symmetry_descriptor = SymmetryDescriptor(
            zernike_std=np.std(self._compute_zernike_moments()),
            zernike_mean=np.mean(self._compute_zernike_moments()),
            radial_intensity_gradient=self._compute_radial_intensity_gradient(),
        )

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all features.

        This function computes all features from the input image.
        It calculates the intensity, texture, symmetry descriptor,
        and morphology features.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed features.
        """
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
        """Convert all features to a pandas DataFrame.

        This function combines all computed features (intensity, texture,
        morphology, and symmetry features) into a single pandas DataFrame.
        The features are organized in a flat structure where each column
        represents a different feature.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed features with the following columns:
            - Intensity features (if computed)
            - Texture features (if computed)
            - Morphology features (if computed)
            - Symmetry descriptor (if computed)

        Notes
        -----
        Only features that have been computed (non-None) will be included
        in the output DataFrame. The DataFrame will have a single row
        containing all the features.
        """
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
    """Compute dynamic features from cell tracking data.

    This class provides methods to compute various dynamic features from cell
    tracking data, including velocity, displacement, and angular features.
    These features are useful for analyzing cell movement patterns and behavior.

    Parameters
    ----------
    tracking_df : pandas.DataFrame
        DataFrame containing cell tracking data with track_id, t, x, y columns

    Attributes
    ----------
    tracking_df : pandas.DataFrame
        The input tracking dataframe containing cell position data over time
    track_features : TrackFeatures or None
        Computed velocity-based features including mean, max, min velocities
        and their standard deviation
    displacement_features : DisplacementFeatures or None
        Computed displacement features including total distance traveled,
        net displacement, and directional persistence
    angular_features : AngularFeatures or None
        Computed angular features including mean, max, and standard deviation
        of angular velocities

    Raises
    ------
    ValueError
        If the tracking dataframe is missing any of the required columns
        (track_id, t, x, y)
    """

    def __init__(self, tracking_df: pd.DataFrame):
        self.tracking_df = tracking_df
        self.track_features = None
        self.displacement_features = None
        self.angular_features = None

        self._eps = 1e-10
        # Verify required columns exist
        required_cols = ["track_id", "t", "x", "y"]
        missing_cols = [col for col in required_cols if col not in tracking_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Verify numeric types for coordinates
        for col in ["t", "x", "y"]:
            if not np.issubdtype(tracking_df[col].dtype, np.number):
                raise ValueError(f"Column {col} must be numeric")

    def _compute_instantaneous_velocity(self, track_id: str) -> np.ndarray:
        """Compute the instantaneous velocity for all timepoints in a track.

        Parameters
        ----------
        track_id : str
            ID of the track to compute velocities for

        Returns
        -------
        velocities : np.ndarray
            Array of instantaneous velocities for each timepoint
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
        velocities[1:] = distances / np.maximum(dt, self._eps)

        return velocities

    def _compute_displacement(self, track_id: str) -> tuple[float, float, float]:
        """Compute displacement-based features for a track.

        This function calculates various displacement metrics for a given track,
        including total distance traveled, net displacement, and directional
        persistence. These metrics help characterize the movement pattern of
        the tracked cell.

        Parameters
        ----------
        track_id : str
            ID of the track to compute displacement features for

        Returns
        -------
        total_distance, net_displacement, directional_persistence: tuple
            Tuple containing:
            - total_distance : float
                Total distance traveled by the cell along its path
            - net_displacement : float
                Straight-line distance between start and end positions
            - directional_persistence : float
                Ratio of net displacement to total distance (0 to 1),
                where 1 indicates perfectly straight movement
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
        """Compute angular velocity features for a track.

        This function calculates the angular velocity statistics for a given track,
        including mean, maximum, and standard deviation of angular velocities.
        Angular velocity is computed as the change in angle between consecutive
        movement vectors over time.

        Parameters
        ----------
        track_id : str
            ID of the track to compute angular velocity for

        Returns
        -------
        mean_angular_velocity, max_angular_velocity, std_angular_velocity: tuple
            Tuple containing:
            - mean_angular_velocity
            - max_angular_velocity
            - std_angular_velocity
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
        angular_velocities = angles / (dt[1:] + self._eps)

        return (
            float(np.mean(angular_velocities)),
            float(np.max(angular_velocities)),
            float(np.std(angular_velocities)),
        )

    def compute_all_features(self, track_id: str) -> pd.DataFrame:
        """Compute all dynamic features for a given track.

        This function computes a comprehensive set of dynamic features for a track,
        including velocity, displacement, and angular features. These features
        characterize the movement patterns and behavior of the tracked cell.

        Parameters
        ----------
        track_id : str
            ID of the track to compute features for

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed features:
            - Velocity features: instantaneous, mean, max, min velocities and std
            - Displacement features: total distance, net displacement, persistence
            - Angular features: mean, max, and std of angular velocities
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
        """Convert all features to a pandas DataFrame.

        This function combines all computed features (velocity, displacement,
        and angular features) into a single pandas DataFrame. The features
        are organized in a flat structure where each column represents a
        different feature.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed features with the following columns:
            - Velocity features
            - Displacement features
            - Angular features
        """
        features_dict = {}
        if self.velocity_features:
            features_dict.update(self.velocity_features)
        if self.displacement_features:
            features_dict.update(self.displacement_features)
        if self.angular_features:
            features_dict.update(self.angular_features)
        return pd.DataFrame([features_dict])
