import numpy as np
from numpy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu
import mahotas
import cv2

class CellFeatures:
    # This class is used to compute the expression and localization of the physical and molecular features of a cell.
    # Some of this features are inspired by CellProfiler, and use the convention similar to skimage.regionprops.
    
    # Feature set:
    # Intensity features: Mean, Standard Deviation, Median, IQR, Skewness, Kurtosis
    # Texture features: Haralick texture features (), Spectral entropy
    # Symmetry features: Zernike moments (similar to MeanObjectIntensityDistribution in CellProfiler)
    # Shape features (for nuclear mask): Perimeter to Area ratio, Eccentricity

    def __init__(self, image, nuclear_mask = None, cell_mask = None):
        self.image = image
        self.nuclear_mask = nuclear_mask
        self.cell_mask = cell_mask
        
        # Intensity features
        self.mean = np.nan
        self.std = np.nan
        self.median = np.nan
        self.iqr = np.nan
        self.skewness = np.nan
        self.kurtosis = np.nan
        
        # Texture features: computed with mahotas.features.haralick
        self.spectral_entropy = np.nan
        self.contrast = np.nan
        self.dissimilarity = np.nan
        self.homogeneity = np.nan
        self.entropy = np.nan
        self.energy = np.nan
        
        # Shape features of nuclei
        self.perimeter_area_ratio = np.nan
        self.eccentricity = np.nan
        
        # Symmetry features: Zernike moments, computed with mahotas.features.zernike_moments
        self.zernike_moments = np.nan
        self.radial_intensity_gradient = np.nan

  

    # def compute_fourier_descriptors(image):
    #     """
    #     Compute the Fourier descriptors of the image
    #     The sensor or nuclear shape changes when infected, which can be captured by analyzing Fourier descriptors
    #     :param np.array image: input image
    #     :return: Fourier descriptors
    #     """
    #     # Convert contour to complex numbers
    #     contour_complex = image[:, 0] + 1j * image[:, 1]
    #     NOTE: computed from the first two columns of an image - doesn't look correct.

    #     # Compute Fourier descriptors
    #     descriptors = np.fft.fft(contour_complex)

    #     return descriptors

    # def analyze_symmetry(descriptors):
    #     """
    #     Analyze the symmetry of the Fourier descriptors
    #     Symmetry of the sensor or nuclear shape changes when infected
    #     :param np.array descriptors: Fourier descriptors
    #     :return: standard deviation of the descriptors
    #     """
    #     # Normalize descriptors
    #     descriptors = np.abs(descriptors) / np.max(np.abs(descriptors))

    #     return np.std(descriptors)  # Lower standard deviation indicates higher symmetry
    
    def compute_intensity_features(self):
        """Compute the intensity features of the cell."""
        pass

    def compute_texture_features(self):
        """Compute the texture features of the cell."""
        pass
    
    def compute_nuclei_shape_features(self):
        """Compute the shape features of the nuclei."""
        pass

    def compute_symmetry_features(self):
        """Compute the symmetry features of the cell."""
        pass


    def compute_area(input_image, sigma=0.6):
        """Create a binary mask using morphological operations
        Sensor area will increase when infected due to expression in nucleus
        :param np.array input_image: generate masks from this 3D image
        :param float sigma: Gaussian blur standard deviation, increase in value increases blur
        :return: area of the sensor mask & mean intensity inside the sensor area
        """

        input_image_blur = gaussian(input_image, sigma=sigma)

        thresh = threshold_otsu(input_image_blur)
        mask = input_image >= thresh

        # Apply sensor mask to the image
        masked_image = input_image * mask

        # Compute the mean intensity inside the sensor area
        masked_intensity = np.mean(masked_image)

        return masked_intensity, np.sum(mask)

    def compute_spectral_entropy(image):
        """
        Compute the spectral entropy of the image
        High frequency components are observed to increase in phase and reduce in sensor when cell is infected
        :param np.array image: input image
        :return: spectral entropy
        """

        # Compute the 2D Fourier Transform
        f_transform = fft.fft2(image)

        # Compute the power spectrum
        power_spectrum = np.abs(f_transform) ** 2

        # Compute the probability distribution
        power_spectrum += 1e-10  # Avoid log(0) issues
        prob_distribution = power_spectrum / np.sum(power_spectrum)

        # Compute the spectral entropy
        entropy = -np.sum(prob_distribution * np.log(prob_distribution))

        return entropy

    # def compute_glcm_features(image):
    #     # Same as Haralick texture features. Deprecated.
    #     """
    #     Compute the contrast, dissimilarity and homogeneity of the image
    #     Both sensor and phase texture changes when infected, smooth in sensor, and rough in phase
    #     :param np.array image: input image
    #     :return: contrast, dissimilarity, homogeneity
    #     """

    #     # Normalize the input image from 0 to 255
    #     image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))
    #     image = image.astype(np.uint8)

    #     # Compute the GLCM
    #     distances = [1]  # Distance between pixels
    #     angles = [0]  # Angle in radians

    #     glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

    #     # Compute GLCM properties
    #     contrast = graycoprops(glcm, "contrast")[0, 0]
    #     dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    #     homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    #     return contrast, dissimilarity, homogeneity

    def compute_iqr(image):
        """
        Compute the interquartile range of pixel intensities
        Observed to increase when cell is infected
        :param np.array image: input image
        :return: interquartile range of pixel intensities
        """

        # Compute the interquartile range of pixel intensities
        iqr = np.percentile(image, 75) - np.percentile(image, 25)

        return iqr

    def compute_mean_intensity(image):
        """
        Compute the mean pixel intensity
        Expected to vary when cell morphology changes due to infection, divison or death
        :param np.array image: input image
        :return: mean pixel intensity
        """

        # Compute the mean pixel intensity
        mean_intensity = np.mean(image)

        return mean_intensity

    def compute_std_dev(image):
        """
        Compute the standard deviation of pixel intensities
        Expected to vary when cell morphology changes due to infection, divison or death
        :param np.array image: input image
        :return: standard deviation of pixel intensities
        """
        # Compute the standard deviation of pixel intensities
        std_dev = np.std(image)

        return std_dev

    def compute_radial_intensity_gradient(image):
        """
        Compute the radial intensity gradient of the image
        The sensor relocalizes inside the nucleus, which is center of the image when cells are infected
        Expected negative gradient when infected and zero to positive gradient when not infected
        :param np.array image: input image
        :return: radial intensity gradient
        """
        # normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # compute the intensity gradient from center to periphery
        y, x = np.indices(image.shape)
        center = np.array(image.shape) / 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), image.ravel())
        nr = np.bincount(r.ravel())
        radial_intensity_values = tbin / nr

        # get the slope radial_intensity_values
        from scipy.stats import linregress

        radial_intensity_gradient = linregress(
            range(len(radial_intensity_values)), radial_intensity_values
        )

        return radial_intensity_gradient[0]
    
    def compute_weighted_intensity_gradient(image):
        """Compute the radial gradient profile and its slope.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        float: Slope of the azimuthally averaged radial gradient profile
    """
        # Get image dimensions
        h, w = image.shape
        center_y, center_x = h // 2, w // 2

        # Create meshgrid of coordinates
        y, x = np.ogrid[:h, :w]

        # Calculate radial distances from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Calculate gradients in x and y directions
        gy, gx = np.gradient(image)

        # Calculate magnitude of gradient
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Weight gradient by intensity
        weighted_gradient = gradient_magnitude * image

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

    # perimeter of nuclear segmentations found inside the patch
    def compute_perimeter_area_ratio(image):
        """Compute the perimeter of the nuclear segmentations found inside the patch.
        
        Args:
            image (np.ndarray): Input image with nuclear segmentation labels
            
        Returns:
            float: Total perimeter of the nuclear segmentations found inside the patch
        """
        total_perimeter = 0
        total_area = 0
        
        # Get the binary mask of each nuclear segmentation labels
        for label in np.unique(image):
            if label != 0:  # Skip background
                continue
                
            # Create binary mask for current label
            mask = (image == label)
            
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

        return total_perimeter / total_area

    def texture_features(image):
        """Compute the texture features of the image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            float: Mean of the texture features
        """
        # rescale image to 0 to 255 and convert to uint8
        image_rescaled = (image - image.min()) / (image.max() - image.min()) * 255
        texture_features = mahotas.features.haralick(image_rescaled.astype('uint8')).ptp(0)
        return np.mean(texture_features)

    def nucleus_eccentricity(image):
        """Compute the eccentricity of the nucleus.
        
        Args:
            image (np.ndarray): Input image with nuclear segmentation labels
            
        Returns:
            float: Eccentricity of the nucleus
        """
        # convert the label image to a binary image and ensure single channel
        binary_image = (image > 0).astype(np.uint8)
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

    def Eucledian_distance_transform(image):
        """Compute the Euclidean distance transform of a binary mask.
        
        Args:
            image (np.ndarray): Binary mask (0s and 1s)
            
        Returns:
            np.ndarray: Distance transform where each pixel value represents 
                    the Euclidean distance to the nearest non-zero pixel
        """
        # Ensure the image is binary
        binary_mask = (image > 0).astype(np.uint8)
        
        # Compute the distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        return dist_transform

    def fluor_localization(image, mask):
        """ Compute localization of fluor using Eucledian distance transformation and fluor intensity"""
        # compute EDT of mask
        edt = Eucledian_distance_transform(mask)
        # compute the intensity weighted center of the fluor
        intensity_weighted_center = np.sum(image * edt) / np.sum(edt)
        return intensity_weighted_center
    

class TrackFeatures:
    """Compute the features of a track."""
    def __init__(self, track_info):
        self.track_info = track_info
        self.instantaneous_speed = np.nan
        
    def compute_instantaneous_velocity(track_info, row_idx):
        """Compute the instantaneous velocity of the cell.

        Args:
            track_info (pd.DataFrame): DataFrame containing track information
            row_idx (int): Current row index in the track_info DataFrame
            
        Returns:
            float: Instantaneous velocity of the cell
        """
        # Check if previous timepoint exists
        has_prev = row_idx > 0
        # Check if next timepoint exists
        has_next = row_idx < len(track_info) - 1
        
        if has_prev:
            # Use previous timepoint
            prev_row = track_info.iloc[row_idx - 1]
            curr_row = track_info.iloc[row_idx]
            distance = np.sqrt((curr_row["x"] - prev_row["x"])**2 + 
                            (curr_row["y"] - prev_row["y"])**2)
            time_diff = curr_row["t"] - prev_row["t"]
        elif has_next:
            # Use next timepoint if previous doesn't exist
            next_row = track_info.iloc[row_idx + 1]
            curr_row = track_info.iloc[row_idx]
            distance = np.sqrt((next_row["x"] - curr_row["x"])**2 + 
                            (next_row["y"] - curr_row["y"])**2)
            time_diff = next_row["t"] - curr_row["t"]
        else:
            # No neighboring timepoints exist
            return 0.0
            
        # Compute velocity (avoid division by zero)
        instantaneous_speed = distance / max(time_diff, 1e-6)
        return instantaneous_speed
        
        


