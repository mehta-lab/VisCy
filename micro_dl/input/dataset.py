"""Dataset classes"""
import keras
import numpy as np
import os
import pandas as pd


class BaseDataSet(keras.utils.Sequence):
    """Base class for input and target images

    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    https://github.com/aleju/imgaug
    """

    def __init__(self,
                 tile_dir,
                 input_fnames,
                 target_fnames,
                 dataset_config,
                 batch_size):
        """Init

        The images could be normalized at the image level during tiling
        (default, normalize=False). If images were not normalized during tiling
        set the normalize flag to normalize at the tile-level
        Works for either data format (channels_first or channels_last).
        Tiles will be loaded as is, so it's important that you've made sure
        your data format matches your preprocessing config.
        TODO: Test with multiple channel tiles

        :param str tile_dir: directory containing training image tiles
        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param dict dataset_config: Dataset part of the main config file
        :param int batch_size: num of datasets in each batch
        """
        self.tile_dir = tile_dir
        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.num_samples = len(self.input_fnames)
        self.batch_size = batch_size

        # Check if model task (regression or segmentation) is specified
        self.model_task = 'regression'
        if 'model_task' in dataset_config:
            self.model_task = dataset_config['model_task']
            assert self.model_task in {'regression', 'segmentation'}, \
                "Model task must be either 'segmentation' or 'regression'"

        self.augmentations = False
        if 'augmentations' in dataset_config:
            self.augmentations = dataset_config['augmentations']
        assert isinstance(self.augmentations, bool),\
            'augmentation parameter should be boolean'

        self.normalize = False
        if 'normalize' in dataset_config:
            self.normalize = dataset_config['normalize']
        assert isinstance(self.normalize, bool),\
            'normalize parameter should be boolean'

        self.shuffle = True
        if 'shuffle' in dataset_config:
            self.shuffle = dataset_config['shuffle']
        assert isinstance(self.shuffle, bool),\
            'shuffle parameter should be boolean'

        random_seed = None
        if 'random_seed' in dataset_config:
            random_seed = dataset_config['random_seed']

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.on_epoch_end()

    def __len__(self):
        """Gets the number of batches per epoch"""

        n_batches = int(np.ceil(self.num_samples / self.batch_size))
        return n_batches

    def _augment_image(self, input_image, aug_idx):
        """Adds image augmentation among 6 possible options

        :param np.array input_image: input image to be transformed
        :param int aug_idx: integer specifying the transformation to apply.
         0 - Image as is
         1 - flip LR (horizontally) about axis 1 (y)
         2 - flip UD (vertically) about axis 0 (x)
         3 - rotate 90 degrees in the xy-plane in the x toward y direction
         4 - rotate 180 degrees in the xy-plane in the x toward y direction
         5 - rotate 270 degrees in the xy-plane in the x toward y direction
        :param str data_format: channels_first or _last. Data is always loaded
        as channels_first so channels_last operations may be obsolete
        :return np.array image after transformation is applied
        """
        # We need to flip over different dimensions depending on data format
        add_dim = 0
        # Get tile data format from shape
        if len(input_image.shape) == 3 and input_image.shape[0] <= 3:
            add_dim = 1

        if aug_idx == 0:
            return input_image
        elif aug_idx == 1:
            # flip about axis=1 (which is row in numpy, hence about y)
            trans_image = np.flip(input_image, 1 + add_dim)
        elif aug_idx == 2:
            # flip about axis=0 (which is cols in numpy, hence about x)
            trans_image = np.flip(input_image, 0 + add_dim)
        elif aug_idx == 3:
            # rot in plane defined by axis=(0, 1) or (1,2)
            trans_image = np.rot90(
                input_image,
                k=1,
                axes=(0 + add_dim, 1 + add_dim),
            )
        elif aug_idx == 4:
            # rot in plane defined by axis=(0, 1) or (1,2)
            trans_image = np.rot90(
                input_image,
                k=2,
                axes=(0 + add_dim, 1 + add_dim),
            )
        elif aug_idx == 5:
            # rot in plane defined by axis=(0, 1) or (1,2)
            trans_image = np.rot90(
                input_image,
                k=3,
                axes=(0 + add_dim, 1 + add_dim),
            )
        else:
            msg = '{} not in allowed aug_idx: 0-5'.format(aug_idx)
            raise ValueError(msg)
        return trans_image

    def _get_volume(self, fname_list, aug_idx=0):
        """
        Read tiles from fname_list and stack them into an image volume.

        :param list fname_list: list of file names of input/target images
        :param int aug_idx: type of augmentation to be applied (if any)
        :return: np.ndarray of stacked images
        """

        image_volume = []
        for fname in fname_list:
            cur_tile = np.load(os.path.join(self.tile_dir, fname))
            if self.augmentations:
                cur_tile = self._augment_image(cur_tile, aug_idx)
            image_volume.append(cur_tile)
        # Stack images channels first
        return np.stack(image_volume)

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
        aug_idx = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            # Select select int randomly that will represent augmentation type
            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)

            cur_input = self._get_volume(cur_input_fnames.split(','),
                                         aug_idx)
            cur_target = self._get_volume(cur_target_fnames.split(','),
                                          aug_idx)
            # If target is boolean (segmentation masks), convert to float
            if cur_target.dtype == bool:
                cur_target = cur_target.astype(np.float64)
            if self.normalize:
                cur_input = (cur_input - np.mean(cur_input)) /\
                             np.std(cur_input)
                # Only normalize target if we're dealing with regression
                if self.model_task is not 'segmentation' and self.normalize:
                    cur_target = (cur_target - np.mean(cur_target)) /\
                                 np.std(cur_target)
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


class DataSetWithMask(BaseDataSet):
    """DataSet class that returns input, target images and sample weights"""

    def __init__(self,
                 tile_dir,
                 input_fnames,
                 target_fnames,
                 mask_fnames,
                 dataset_config,
                 batch_size):
        """Init

        https://stackoverflow.com/questions/44747288/keras-sample-weight-array-error
        https://gist.github.com/andreimouraviev/2642384705034da92d6954dd9993fb4d

        :param str tile_dir: directory containing training image tiles
        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param pd.Series mask_fnames: pd.Series with each row containing
         mask filenames
        :param dict dataset_config: Dataset part of the main config file
        :param int batch_size: num of datasets in each batch
        :param bool shuffle: shuffle data for each epoch
        """

        super().__init__(tile_dir,
                         input_fnames,
                         target_fnames,
                         dataset_config,
                         batch_size)
        self.mask_fnames = mask_fnames
        # list label_weights: weight for each label
        self.label_weights = None
        if 'label_weights' in dataset_config:
            self.label_weights = dataset_config['label_weights']

    def __getitem__(self, index):
        """Get a batch of data

        :param int index: batch index
        :return: np.ndarrays input_image and target_image of shape
         [batch_size, num_channels, z, y, x] and mask_image of shape
         [batch_size, z, y, x] for data format channels_first,
         otherwise [..., y, x, z]
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_image = []
        target_image = []
        aug_idx = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_mask_fnames = self.mask_fnames.iloc[self.row_idx[idx]]

            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)
            cur_input = super()._get_volume(cur_input_fnames.split(','),
                                            aug_idx)
            cur_target = super()._get_volume(cur_target_fnames.split(','),
                                             aug_idx)

            # If target is boolean (segmentation masks), convert to float

            if cur_target.dtype == bool:
                cur_target = cur_target.astype(np.float64)
            if self.normalize:
                cur_input = (cur_input - np.mean(cur_input)) /\
                             np.std(cur_input)
                # Only normalize target if we're dealing with regression
                if self.model_task is not 'segmentation':
                    cur_target = (cur_target - np.mean(cur_target)) /\
                                 np.std(cur_target)

            # the mask is based on sum of channel images
            cur_mask = super()._get_volume(cur_mask_fnames.split(','),
                                           aug_idx)
            if self.label_weights is not None:
                wtd_mask = np.zeros(cur_mask.shape)
                for label_idx in range(len(self.label_weights)):
                    wtd_mask += (cur_mask == label_idx) * \
                                self.label_weights[label_idx]
                cur_mask = wtd_mask
            cur_target = np.concatenate((cur_target, cur_mask), axis=0)

            input_image.append(cur_input)
            target_image.append(cur_target)

        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image
