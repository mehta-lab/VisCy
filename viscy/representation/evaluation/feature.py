from typing import TypedDict

import numpy as np
import pandas as pd
from mahotas.features import haralick, zernike_moments
from numpy.typing import ArrayLike
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
import mahotas as mh
import cv2
import scipy.stats
from numpy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu

def normalize_image(image: ArrayLike) -> ArrayLike:
    """Normalize the image to 0-255 range."""
    return (image - image.min()) / (image.max() - image.min()) * 255

class IntensityFeatures(TypedDict):
    """Intensity features extracted from a single cell."""
    mean: float
    std: float
    min: float
    max: float
    kurtosis: float
    skewness: float
    spectral_entropy: float
    iqr: float
    weighted_intensity_gradient: float

    def __init__(self, image: ArrayLike):
        super().__init__()
        self.image = image
        self.update(self._compute_features())

    def _compute_features(self) -> dict:
        """Compute all intensity features."""
        return {
            "mean": self._compute_mean_intensity(),
            "std": self._compute_std_dev(),
            "min": np.min(self.image),
            "max": np.max(self.image),
            "kurtosis": self._compute_kurtosis(),
            "skewness": self._compute_skewness(),
            "spectral_entropy": self._compute_spectral_entropy(),
            "iqr": self._compute_iqr(),
            "weighted_intensity_gradient": self._compute_weighted_intensity_gradient()
        }

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
    
    def _compute_mean_intensity(self):
        """
        Compute the mean pixel intensity
        Expected to vary when cell morphology changes due to infection, divison or death
        :param np.array image: input image
        :return: mean pixel intensity
        """

        # Compute the mean pixel intensity
        mean_intensity = np.mean(self.image)

        return mean_intensity

    def _compute_std_dev(self):
        """
        Compute the standard deviation of pixel intensities
        Expected to vary when cell morphology changes due to infection, divison or death
        :param np.array image: input image
        :return: standard deviation of pixel intensities
        """
        # Compute the standard deviation of pixel intensities
        std_dev = np.std(self.image)

        return std_dev

    def _compute_glcm_features(self):
        """
        Compute the contrast, dissimilarity and homogeneity of the image
        Both sensor and phase texture changes when infected, smooth in sensor, and rough in phase
        :param np.array image: input image
        :return: contrast, dissimilarity, homogeneity
        """

        # Normalize the input image from 0 to 255
        image = (self.image - np.min(self.image)) * (255 / (np.max(self.image) - np.min(self.image)))
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
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Calculate gradients in x and y directions
        gy, gx = np.gradient(self.image)
        
        # Calculate magnitude of gradient
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Weight gradient by intensity
        weighted_gradient = gradient_magnitude * self.image
        
        # Calculate maximum radius (to edge of image)
        max_radius = int(min(h//2, w//2))
        
        # Initialize arrays for radial profile
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        # Bin pixels by radius
        for i in range(h):
            for j in range(w):
                radius = int(r[i,j])
                if radius < max_radius:
                    radial_profile[radius] += weighted_gradient[i,j]
                    counts[radius] += 1
        
        # Average by counts (avoiding division by zero)
        valid_mask = counts > 0
        radial_profile[valid_mask] /= counts[valid_mask]
        
        # Calculate slope using linear regression
        x = np.arange(max_radius)[valid_mask]
        y = radial_profile[valid_mask]
        slope = np.polyfit(x, y, 1)[0]
        
        return slope

class TextureFeatures(TypedDict):
    """Texture features extracted from a single cell."""
    spectral_entropy: float
    contrast: float
    entropy: float
    homogeneity: float
    dissimilarity: float
    texture_mean: float

    def __init__(self, image: ArrayLike):
        super().__init__()
        self.image = image
        self.update(self._compute_features())

    def _compute_features(self) -> dict:
        """Compute all texture features."""
        contrast, dissimilarity, homogeneity = self._compute_glcm_features()
        return {
            "spectral_entropy": self._compute_spectral_entropy(),
            "contrast": contrast,
            "homogeneity": homogeneity,
            "dissimilarity": dissimilarity,
            "texture_mean": self._compute_texture_features()
        }

    def _compute_texture_features(self):
        """Compute the texture features of the image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            float: Mean of the texture features
        """
        # rescale image to 0 to 255 and convert to uint8
        image_rescaled = (self.image - self.image.min()) / (self.image.max() - self.image.min()) * 255
        texture_features = mh.features.haralick(image_rescaled.astype('uint8')).ptp(0)
        return np.mean(texture_features)

class MorphologyFeatures(TypedDict):
    """Morphology features extracted from a single cell."""
    area: float
    perimeter: float
    perimeter_area_ratio: float
    eccentricity: float
    fluor_localization: float
    masked_intensity: float

    def __init__(self, image: ArrayLike):
        super().__init__()
        self.image = image
        self.update(self._compute_features())

    def _compute_features(self) -> dict:
        """Compute all morphology features."""
        perimeter, area, ratio = self._compute_perimeter_area_ratio()
        masked_intensity, area_value = self._compute_area()
        return {
            "area": area,
            "perimeter": perimeter,
            "perimeter_area_ratio": ratio,
            "eccentricity": self._compute_nucleus_eccentricity(),
            "fluor_localization": self._compute_fluor_localization(),
            "masked_intensity": masked_intensity
        }

    def _compute_perimeter_area_ratio(self):
        """Compute the perimeter of the nuclear segmentations found inside the patch.
        
        Args:
            image (np.ndarray): Input image with nuclear segmentation labels
            
        Returns:
            float: Total perimeter of the nuclear segmentations found inside the patch
        """
        total_perimeter = 0
        total_area = 0
        
        # Get the binary mask of each nuclear segmentation labels
        for label in np.unique(self.image):
            if label != 0:  # Skip background
                continue
                
            # Create binary mask for current label
            mask = (self.image == label)
            
            # Convert to proper format for OpenCV
            mask = mask.astype(np.uint8)
            
            # Ensure we have a 2D array
            if mask.ndim > 2:
                # Take the first channel if multi-channel
                mask = mask[:, :, 0] if mask.shape[-1] > 1 else mask.squeeze()
            
            # Ensure we have values 0 and 1 only
            mask = (mask > 0).astype(np.uint8) * 255
            
            # Find contours in the binary mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Add perimeter of all contours for this label
            for contour in contours:
                total_perimeter += cv2.arcLength(contour, closed=True)
            
            # Add area of all contours for this label
            for contour in contours:
                total_area += cv2.contourArea(contour)
        average_area = total_area / (len(np.unique(self.image))-1)
        average_perimeter = total_perimeter / (len(np.unique(self.image))-1)

        return average_perimeter, average_area, total_perimeter / total_area

    def _compute_nucleus_eccentricity(self):
        """Compute the eccentricity of the nucleus.
        
        Args:
            image (np.ndarray): Input image with nuclear segmentation labels
            
        Returns:
            float: Eccentricity of the nucleus
        """
        # convert the label image to a binary image and ensure single channel
        binary_image = (self.image > 0).astype(np.uint8)
        if binary_image.ndim > 2:
            binary_image = binary_image[:,:,0]  # Take first channel if multi-channel
        binary_image = cv2.convertScaleAbs(binary_image * 255)  # Ensure proper OpenCV format
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        eccentricities = []
        
        for contour in contours:
            # Fit an ellipse to the contour (works well for ellipse-like shapes)
            if len(contour) >= 5:  # At least 5 points are required to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                
                # Extract the lengths of the semi-major and semi-minor axes
                major_axis, minor_axis = max(axes), min(axes)
                
                # Calculate the eccentricity using the formula: e = sqrt(1 - (b^2 / a^2))
                eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
                eccentricities.append(eccentricity)
        
        return np.mean(eccentricities)
    
    def _compute_Eucledian_distance_transform(self):
        """Compute the Euclidean distance transform of a binary mask.
        
        Args:
            image (np.ndarray): Binary mask (0s and 1s)
            
        Returns:
            np.ndarray: Distance transform where each pixel value represents 
                    the Euclidean distance to the nearest non-zero pixel
        """
        # Ensure the image is binary
        binary_mask = (self.image > 0).astype(np.uint8)
        
        # Compute the distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        return dist_transform
    
    def _compute_fluor_localization(self):
        """ Compute localization of fluor using Eucledian distance transformation and fluor intensity"""
        # compute EDT of mask
        edt = self._compute_Eucledian_distance_transform()
        # compute the intensity weighted center of the fluor
        intensity_weighted_center = np.sum(self.image * edt) / np.sum(edt)
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

class SymmetryDescriptor(TypedDict):
    """
    Symmetry descriptor extracted from a single cell
    """

    zernike_moments: ArrayLike
    radial_intensity_gradient: ArrayLike

    def __init__(self, image: ArrayLike):
        self.image = image

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
        # normalize the image
        image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))

        # compute the intensity gradient from center to periphery
        y, x = np.indices(self.image.shape)
        center = np.array(self.image.shape) / 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), self.image.ravel())
        nr = np.bincount(r.ravel())
        radial_intensity_values = tbin / nr

        # get the slope radial_intensity_values
        from scipy.stats import linregress

        radial_intensity_gradient = linregress(
            range(len(radial_intensity_values)), radial_intensity_values
        )

        return radial_intensity_gradient[0]


class TrackFeatures(TypedDict):
    """Track features extracted from a single track"""

    instantaneous_velocity: float

    def __init__(self, track_info: pd.DataFrame, row_idx: int):
        self.track_info = track_info
        self.row_idx = row_idx
    
    def compute_instantaneous_velocity(self):
        """Compute the instantaneous velocity of the cell.

        Args:
            track_info (pd.DataFrame): DataFrame containing track information
            row_idx (int): Current row index in the track_info DataFrame
            
        Returns:
            float: Instantaneous velocity of the cell
        """
        # Check if previous timepoint exists
        has_prev = self.row_idx > 0
        # Check if next timepoint exists
        has_next = self.row_idx < len(self.track_info) - 1
        
        if has_prev:
            # Use previous timepoint
            prev_row = self.track_info.iloc[self.row_idx - 1]
            curr_row = self.track_info.iloc[self.row_idx]
            distance = np.sqrt((curr_row["x"] - prev_row["x"])**2 + 
                            (curr_row["y"] - prev_row["y"])**2)
            time_diff = curr_row["t"] - prev_row["t"]
        elif has_next:
            # Use next timepoint if previous doesn't exist
            next_row = self.track_info.iloc[self.row_idx + 1]
            curr_row = self.track_info.iloc[self.row_idx]
            distance = np.sqrt((next_row["x"] - curr_row["x"])**2 + 
                            (next_row["y"] - curr_row["y"])**2)
            time_diff = next_row["t"] - curr_row["t"]
        else:
            # No neighboring timepoints exist
            return 0.0
            
        # Compute velocity (avoid division by zero)
        velocity = distance / max(time_diff, 1e-6)
        return velocity


class CellFeatures:
    """Cell features extracted from a single cell image patch."""

    def __init__(self, image: ArrayLike, segmentation_mask: ArrayLike = None):
        self.image = normalize_image(image)
        self.segmentation_mask = segmentation_mask
        self.intensity_features = None
        self.texture_features = None 
        self.morphology_features = None
        self.symmetry_descriptor = None

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all available features."""
        self.intensity_features = IntensityFeatures(self.image)
        self.texture_features = TextureFeatures(self.image)
        
        if self.segmentation_mask is not None:
            self.morphology_features = MorphologyFeatures(self.image)
            self.symmetry_descriptor = SymmetryDescriptor(self.image)
            
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
        instantaneous_velocity = self.track_features.compute_instantaneous_velocity()
        return instantaneous_velocity

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all available features."""
        self.track_features = self.compute_instantaneous_velocity()
        return self.to_df()

    def to_df(self) -> pd.DataFrame:
        """Convert all features to a pandas DataFrame."""
        return pd.DataFrame(self.track_features)
