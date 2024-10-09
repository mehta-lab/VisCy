import numpy as np
from numpy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu


class FeatureExtractor:
    # FIXME: refactor into a separate module with standalone functions

    def __init__(self):
        pass

    def compute_fourier_descriptors(image):
        """
        Compute the Fourier descriptors of the image
        The sensor or nuclear shape changes when infected, which can be captured by analyzing Fourier descriptors
        :param np.array image: input image
        :return: Fourier descriptors
        """
        # Convert contour to complex numbers
        contour_complex = image[:, 0] + 1j * image[:, 1]

        # Compute Fourier descriptors
        descriptors = np.fft.fft(contour_complex)

        return descriptors

    def analyze_symmetry(descriptors):
        """
        Analyze the symmetry of the Fourier descriptors
        Symmetry of the sensor or nuclear shape changes when infected
        :param np.array descriptors: Fourier descriptors
        :return: standard deviation of the descriptors
        """
        # Normalize descriptors
        descriptors = np.abs(descriptors) / np.max(np.abs(descriptors))

        return np.std(descriptors)  # Lower standard deviation indicates higher symmetry

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

    def compute_glcm_features(image):
        """
        Compute the contrast, dissimilarity and homogeneity of the image
        Both sensor and phase texture changes when infected, smooth in sensor, and rough in phase
        :param np.array image: input image
        :return: contrast, dissimilarity, homogeneity
        """

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
