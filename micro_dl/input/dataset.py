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
                 batch_size,
                 image_format='zyx'):
        """Init

        The images could be normalized at the image level during tiling
        (default, normalize=False). If images were not normalized during tiling
        set the normalize flag to normalize at the tile-level
        Works for tile shapes with z first (zyx) or last (xyz).
        Tiles will be loaded as is, and shape order will be determined
        from the preprocessing config file which is saved in your training
        data dir as preprocessing_info.json.
        TODO: Test with multiple channel tiles

        :param str tile_dir: directory containing training image tiles
        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param dict dataset_config: Dataset part of the main config file
        :param int batch_size: num of datasets in each batch
        :param str image_format: Tile shape order: 'xyz' or 'zyx'
        """
        self.tile_dir = tile_dir
        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.num_samples = len(self.input_fnames)
        self.batch_size = batch_size
        assert image_format in {'xyz', 'zyx'},\
            "Image format should be xyz or zyx, not {}".format(image_format)
        self.image_format = image_format

        # Check if model task (regression or segmentation) is specified
        self.model_task = 'regression'
        if 'model_task' in dataset_config:
            self.model_task = dataset_config['model_task']
            assert self.model_task in {'regression', 'segmentation'}, \
                "Model task must be either 'segmentation' or 'regression'"

        # Whether or not to do augmentations
        self.augmentations = False
        if 'augmentations' in dataset_config:
            self.augmentations = dataset_config['augmentations']
        assert isinstance(self.augmentations, bool),\
            'augmentation parameter should be boolean'

        # Whether to do zscore normalization on tile level
        self.normalize = False
        if 'normalize' in dataset_config:
            self.normalize = dataset_config['normalize']
        assert isinstance(self.normalize, bool),\
            'normalize parameter should be boolean'

        # Whether to shuffle indices at the end of each epoch
        self.shuffle = True
        if 'shuffle' in dataset_config:
            self.shuffle = dataset_config['shuffle']
        assert isinstance(self.shuffle, bool),\
            'shuffle parameter should be boolean'

        # Whether to only use a fraction of training data each epoch
        self.num_epoch_samples = self.num_samples
        if 'train_fraction' in dataset_config:
            train_fraction = dataset_config['train_fraction']
            assert 0. < train_fraction <= 1.,\
                'Train fraction should be {0,1}, not {}'.format(train_fraction)
            # You must shuffle if only using a fraction of the training data
            self.shuffle = True
            self.num_epoch_samples = int(self.num_samples * train_fraction)
        self.steps_per_epoch = int(np.ceil(self.num_epoch_samples /
                                           self.batch_size))
        # Declare row indices, will to an inital shuffle at the end of init`
        self.row_idx = np.arange(self.num_samples)

        # Whether to remove singleton dimensions from tiles (e.g. 2D models)
        self.squeeze = False
        if 'squeeze' in dataset_config:
            self.squeeze = dataset_config['squeeze']
        assert isinstance(self.squeeze, bool),\
            'squeeze parameter should be boolean'

        # Whether to use fixed random seed (only recommended for testing)
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

    def get_steps_per_epoch(self):
        """
        Returns steps per epoch which is number of training samples per
        epoch divided by batch size.

        :return int steps_per_epoch: Steps per epoch
        """
        return self.steps_per_epoch

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
        :return np.array image after transformation is applied
        """
        # We need to flip over different dimensions depending on data format
        add_dim = 0
        if self.image_format == 'zyx' and not self.squeeze:
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

    def _get_volume(self, fname_list, normalize=True, aug_idx=0):
        """
        Read tiles from fname_list and stack them into an image volume.

        :param list fname_list: list of file names of input/target images
        :param bool normalize: Whether to zscore normalize tiles
        :param int aug_idx: type of augmentation to be applied (if any)
        :return: np.ndarray of stacked images
        """
        image_volume = []
        for fname in fname_list:
            cur_tile = np.load(os.path.join(self.tile_dir, fname))
            if self.augmentations:
                cur_tile = self._augment_image(cur_tile, aug_idx)
            if self.squeeze:
                cur_tile = np.squeeze(cur_tile)
            image_volume.append(cur_tile)
        # Stack images channels first
        image_volume = np.stack(image_volume)
        if image_volume.dtype == bool:
            image_volume = image_volume.astype(np.float64)
        elif normalize:
            image_volume = (image_volume - np.mean(image_volume)) / \
                           np.std(image_volume)
        return image_volume

    def __getitem__(self, index):
        """
        Get a batch of data. If using a fraction of the training data, shuffle
        if automatically set to True to make sure you have a chance of accessing
        all training data and not just the first indices.

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

        norm_output = self.model_task is not 'segmentation' and self.normalize

        input_image = []
        target_image = []
        aug_idx = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            # Select select int randomly that will represent augmentation type
            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)

            cur_input = self._get_volume(
                fname_list=cur_input_fnames.split(','),
                normalize=self.normalize,
                aug_idx=aug_idx,
            )
            cur_target = self._get_volume(
                fname_list=cur_target_fnames.split(','),
                normalize=norm_output,
                aug_idx=aug_idx,
            )
            input_image.append(cur_input)
            target_image.append(cur_target)

        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image

    def on_epoch_end(self):
        """Update indices and shuffle after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.row_idx)

