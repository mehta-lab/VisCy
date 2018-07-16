"""Dataset classes"""
import keras
import numpy as np
import pandas as pd


class BaseDataSet(keras.utils.Sequence):
    """Base class for input and target images

    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    https://github.com/aleju/imgaug
    """

    def __init__(self, input_fnames, target_fnames, batch_size,
                 shuffle=True, augmentations=None, random_seed=42):
        """Init

        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param int batch_size: num of datasets in each batch
        :param bool shuffle: shuffle data for each epoch
        :param dict augmentations: options for image augmentation
        :param int random_seed: initialize the random number generator with
         this seed
        """

        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.batch_size = batch_size

        self.shuffle = shuffle
        num_samples = len(self.input_fnames)
        self.num_samples = num_samples
        self.augmentations = augmentations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.on_epoch_end()

    def __len__(self):
        """Gets the number of batches per epoch"""

        n_batches = int(self.num_samples / self.batch_size)
        return n_batches

    def _augment_image(self, input_image, target_image, mask_image=None):
        """Augment images"""

        return NotImplementedError

    def __getitem__(self, index):
        """Get a batch of data

        https://www.tensorflow.org/performance/performance_guide#use_nchw_imag.
        will use nchw format. shape: [batch_size, num_channels, z, y, x]

        :param int index: batch index
        :return: np.ndarrays input_image and target_image of shape
         [batch_size, num_channels, z, y, x]
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_image = []
        target_image = []
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_input = self._get_volume(cur_input_fnames.split(','))
            cur_input = (cur_input - np.mean(cur_input)) / np.std(cur_input)
            cur_target = self._get_volume(cur_target_fnames.split(','))
            # If target is boolean (segmentation masks), convert to float
            if cur_target.dtype == bool:
                cur_target = cur_target.astype(np.float64)
            else:
                # Only normalize target if we're dealing with regression
                cur_target = (cur_target - np.mean(cur_target)) /\
                             np.std(cur_target)
            # _augment_image(cur_input, cur_target)
            input_image.append(cur_input)
            target_image.append(cur_target)
        input_image = np.stack(input_image)
        target_image = np.stack(target_image)

        return input_image, target_image

    def on_epoch_end(self):
        """Update indices and shuffle after each epoch"""

        self.row_idx = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.row_idx)

    def _get_volume(self, fname_list):
        """Read a volume from fname_list

        :param list fname_list: list of file names of input/target images
        :return: np.ndarray of stacked images
        """

        image_volume = []
        for fname in fname_list:
            cur_channel = np.load(fname)
            image_volume.append(cur_channel)

        image_volume = np.stack(image_volume)
        return image_volume


class DataSetWithMask(BaseDataSet):
    """DataSet class that returns input, target images and sample weights"""

    def __init__(self, input_fnames, target_fnames, mask_fnames, batch_size,
                 shuffle=True, augmentations=None, random_seed=42):
        """Init

        https://stackoverflow.com/questions/44747288/keras-sample-weight-array-error
        https://gist.github.com/andreimouraviev/2642384705034da92d6954dd9993fb4d

        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param pd.Series mask_fnames: pd.Series with each row containing
         mask filenames
        :param int batch_size: num of datasets in each batch
        :param bool shuffle: shuffle data for each epoch
        :param int random_seed: initialize the random number generator with
         this seed
        """

        super().__init__(input_fnames, target_fnames, batch_size,
                         shuffle, augmentations, random_seed)
        self.mask_fnames = mask_fnames

    def __getitem__(self, index):
        """Get a batch of data

        :param int index: batch index
        :return: np.ndarrays input_image and target_image of shape
         [batch_size, num_channels, z, y, x] and mask_image of shape
         [batch_size, z, y, x]
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_image = []
        target_image = []
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_mask_fnames = self.mask_fnames.iloc[self.row_idx[idx]]
            cur_input = super()._get_volume(cur_input_fnames.split(','))

            cur_target = super()._get_volume(cur_target_fnames.split(','))

            # the mask is based on sum of flurophore images
            cur_mask = super()._get_volume(cur_mask_fnames.split(','))
            cur_target = np.concatenate((cur_target, cur_mask), axis=0)

            input_image.append(cur_input)
            target_image.append(cur_target)

        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image
