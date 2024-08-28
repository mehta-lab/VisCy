import cv2
import numpy as np
from skimage import color
from numpy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu, gaussian

class FeatureExtractor:

    def __init__(self):
        pass

    def compute_fourier_descriptors(image):
        
        # Convert contour to complex numbers
        contour_complex = image[:, 0] + 1j * image[:, 1]
        
        # Compute Fourier descriptors
        descriptors = np.fft.fft(contour_complex)
        
        return descriptors

    def analyze_symmetry(descriptors):
        # Normalize descriptors
        descriptors = np.abs(descriptors) / np.max(np.abs(descriptors))
        # Check symmetry (for a perfect circle, descriptors should be quite uniform)
        return np.std(descriptors)  # Lower standard deviation indicates higher symmetry

    def compute_area(input_image, sigma=0.6):
        """Create a binary mask using morphological operations
        :param np.array input_image: generate masks from this 3D image
        :param float sigma: Gaussian blur standard deviation, increase in value increases blur
        :return: volume mask of input_image, 3D np.array
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
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

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

    def compute_glcm_features(image):

        # Normalize the input image from 0 to 255
        image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))
        image = image.astype(np.uint8)

        # Compute the GLCM
        distances = [1]  # Distance between pixels
        angles = [0]  # Angle in radians
        
        glcm = graycomatrix(image, distances, angles, symmetric=True, normed=True)

        # Compute GLCM properties
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

        return contrast, dissimilarity, homogeneity

    # def detect_edges(image):

    #     # Apply Canny edge detection
    #     edges = cv2.Canny(image, 100, 200)

    #     return edges

    def compute_iqr(image):

        # Compute the interquartile range of pixel intensities
        iqr = np.percentile(image, 75) - np.percentile(image, 25)

        return iqr
    
    def compute_mean_intensity(image):
        
        # Compute the mean pixel intensity
        mean_intensity = np.mean(image)
        
        return mean_intensity
    
    def compute_std_dev(image):
        
        # Compute the standard deviation of pixel intensities
        std_dev = np.std(image)
        
        return std_dev
    
