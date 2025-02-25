import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import pytest
import numpy as np
from viscy.representation.evaluation.feature import CellFeatures

class TestCellFeatures:

    def create_constant_image(self, size=(100, 100), value=255):
        """
        Create a constant image
        """
        return np.full(size, value, dtype=np.float32)
    
    def create_small_range_image(self, size=(100, 100), min_val=-0.1, max_val=0.1):
        """
        Create an image with a small range of values
        """
        return np.random.uniform(min_val, max_val, size).astype(np.float32)
    
    def create_large_range_image(self, size=(100, 100), min_val=-100, max_val=1000):
        """
        Create an image with a large range of values
        """
        return np.random.uniform(min_val, max_val, size).astype(np.int32)
    
    def create_binary_image(self, size=(100, 100), threshold=128):
        """
        Create a binary image
        """
        return np.random.randint(0, 2, size)
    
    def test_constant_image_features_are_zero(self):
        """Test that statistical features of a constant image are zero/constant"""
        constant_image = self.create_constant_image()
        binary_image = self.create_binary_image()

        features = CellFeatures(constant_image, binary_image)
        cell_features = features.compute_all_features()
        
        # All statistical measures should be 0 for constant image
        zero_features = ['std_dev', 'iqr', 
                        'dissimilarity', 'contrast', 
                        'texture', 'radial_intensity_gradient']
        nan_features = ['kurtosis', 'skewness']
        positive_features = ['spectral_entropy', 'zernike_std', 'zernike_mean']
        constant_features = ['mean_intensity', 'masked_intensity', 'intensity_localization']
        
        for feature in zero_features:
            assert cell_features[feature].values[0] == 0, f"{feature} should be 0 for constant image"

        for feature in nan_features:
            assert np.isnan(cell_features[feature].values[0]), f"{feature} should be nan for constant image"

        for feature in positive_features:
            assert abs(cell_features[feature].values[0]) > 0, f"{feature} should be positive for constant image"

        for feature in constant_features:
            assert np.isclose(cell_features[feature].values[0], 255), f"{feature} should be 255 for constant image"

        assert cell_features['masked_area'].values[0]>=0, "masked_area should be non-negative"

    def test_small_range_image_features_are_valid(self):
        """Test that features of small range image are within expected bounds"""
        small_range_image = self.create_small_range_image()
        binary_image = self.create_binary_image()

        features = CellFeatures(small_range_image, binary_image)
        cell_features = features.compute_all_features()

        # Statistical features should be positive but small
        positive_features = ['mean_intensity', 'masked_intensity', 'std_dev', 'kurtosis', 'skewness', 'iqr', 'spectral_entropy', 
                           'dissimilarity', 'contrast', 'texture', 'zernike_std', 'zernike_mean',
                           'radial_intensity_gradient', 'intensity_localization', 'masked_area']
        integer_features = ['masked_area']
        float_features = list(set(positive_features) - set(integer_features))

        for feature in positive_features:
            assert abs(cell_features[feature].values[0]) >= 0, f"{feature} should be positive"
            
            if feature in float_features:
                assert np.issubdtype(type(cell_features[feature].values[0]), np.floating), f"{feature} should be float"
            elif feature in integer_features:
                assert np.issubdtype(type(cell_features[feature].values[0]), np.integer), f"{feature} should be integer"

    def test_cellfeatures_large_range_image(self):
        """Test that features of large range image are within expected bounds"""
        large_range_image = self.create_large_range_image()
        binary_image = self.create_binary_image()

        features = CellFeatures(large_range_image, binary_image)
        cell_features = features.compute_all_features()

        positive_features = ['mean_intensity', 'masked_intensity', 'std_dev', 'kurtosis', 'skewness', 'iqr', 
                             'spectral_entropy', 'dissimilarity', 'contrast', 'texture', 
                             'zernike_std', 'zernike_mean', 'radial_intensity_gradient',
                             'intensity_localization']
        # Test all features are within expected bounds
        for feature in positive_features:
            assert abs(cell_features[feature].values[0]) >= 0, f"{feature} should be non-negative"
            assert isinstance(cell_features[feature].values[0], float), f"{feature} should be float"
