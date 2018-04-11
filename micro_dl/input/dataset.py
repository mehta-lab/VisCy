"""Dataset classes"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class BaseDataSet(metaclass=ABCMeta):
    """Base class for input and target images"""

    def __init__(self, df_metadata, batch_size, augmentations=None,
                 isotropic=False, shuffle=True, random_seed=42):
        """Init

        Isotropic here refers to same row, col, slice dimensions and NOT
        isotropic voxel size!

        :param pd.DataFrame df_metadata: pd.df with each row containing
         relevant information for one training pair
        :param int batch_size: num of datasets in each batch
        :param dict augmentations: dictionary with allowed augmentations as
         keys and distortion amount as values
        :param bool isotropic: resample to have same dimension along row, col,
         slice. Default=False.
        :param bool shuffle: shuffle data for each epoch
        """

        self.df_metadata = df_metadata
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.isotropic = isotropic
        self.shuffle = shuffle
        num_samples = len(self.df_metadata)
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Next

        https://www.tensorflow.org/performance/performance_guide#use_nchw_imag.
        will use nchw format. shape: [batch_size, num_channels, x, y, z]
        """
        if self.current > self.num_samples:
            raise StopIteration
        if self.current == 0:
            row_idx = range(self.num_samples)
            if self.shuffle:
                row_idx = shuffle(row_idx)

        if (self.current + self.batch_size) >= self.num_samples:
            end_idx = self.num_samples
        else:
            end_idx = self.current + self.batch_size

        input_image = []
        target_image = []
        for idx in range(self.current, end_idx, 1):
            cur_row = self.df_metadata[row_idx[idx]]
            cur_input = self._get_input_volume(cur_row['fpaths_input'])
            cur_target = self._get_input_volume(cur_row['fpaths_target'])
            input_image.append(cur_input)
            target_image.append(cur_target)
        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        self.current = self.current + self.batch_size
        return input_image, target_image

    @abstractmethod
    def _get_input_volume(self, input_fnames):
        """Read one input dataset"""

        raise NotImplementedError

    @abstractmethod
    def _get_target_volume(self, target_fnames):
        """Read one target dataset"""

        raise NotImplementedError

