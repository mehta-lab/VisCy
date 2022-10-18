"""Dataset classes"""
import warnings
import cv2
from tensorflow import keras
import numpy as np
import os
from scipy import ndimage
import micro_dl.utils.normalize as norm


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=1, col_axis=2, channel_axis=0,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 3D numpy array - a 2D image with one or more channels.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows (aka Y axis) in the input image.
                  Direction: left to right.
        col_axis: Index of axis for columns (aka X axis) in the input image.
                  Direction: top to bottom.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError("'row_axis', 'col_axis', and 'channel_axis'"
                         " must be distinct")

    # TODO: shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(
            f"Invalid axis' indices: {actual_indices - valid_indices}")

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError("Channels are allowed and the first and last dimensions.")

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class BaseDataSet(keras.utils.Sequence):
    warnings.warn('Warning: tf-dependent BaseDataSet to be replaced with GunPowder in 2.1.0')
    """
    Base class for input and target images

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

        :param str tile_dir: directory containing training image tiles
        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param dict dataset_config: Dataset part of the main config file
            Can contain a subset augmentations with args see line 186
        :param int batch_size: num of datasets in each batch
        :param str image_format: Tile shape order: 'xyz' or 'zyx'
        """
        self.tile_dir = tile_dir
        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.num_samples = len(self.input_fnames)
        self.batch_size = batch_size
        assert image_format in {'xyz', 'zyx'}, \
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
        self.zoom_range = (1, 1)
        self.rotate_range = 0
        self.mean_jitter = 0
        self.std_jitter = 0
        self.noise_std = 0
        self.blur_range = (0, 0)
        self.shear_range = 0
        if 'augmentations' in dataset_config:
            self.augmentations = True
            if 'zoom_range' in dataset_config['augmentations']:
                self.zoom_range = dataset_config['augmentations']['zoom_range']
            if 'rotate_range' in dataset_config['augmentations']:
                self.rotate_range = dataset_config['augmentations']['rotate_range']
            if 'intensity_jitter' in dataset_config['augmentations']:
                self.mean_jitter, self.std_jitter = dataset_config['augmentations']['intensity_jitter']
            if 'noise_std' in dataset_config['augmentations']:
                self.noise_std = dataset_config['augmentations']['noise_std']
            if 'blur_range' in dataset_config['augmentations']:
                self.blur_range = dataset_config['augmentations']['blur_range']
            if 'shear_range' in dataset_config['augmentations']:
                self.shear_range = dataset_config['augmentations']['shear_range']

        # Whether to do zscore normalization on tile level
        self.normalize = False
        if 'normalize' in dataset_config:
            self.normalize = dataset_config['normalize']
        assert isinstance(self.normalize, bool), \
            'normalize parameter should be boolean'

        # Whether to shuffle indices at the end of each epoch
        self.shuffle = True
        if 'shuffle' in dataset_config:
            assert isinstance(dataset_config['shuffle'], bool), \
                'shuffle parameter should be boolean'
            self.shuffle = dataset_config['shuffle']

        # Whether to only use a fraction of training data each epoch
        self.num_epoch_samples = self.num_samples
        if 'train_fraction' in dataset_config:
            train_fraction = dataset_config['train_fraction']
            assert 0. < train_fraction <= 1., \
                'Train fraction should be [0,1], not {}'.format(train_fraction)
            # You must shuffle if only using a fraction of the training data
            self.shuffle = True
            self.num_epoch_samples = int(self.num_samples * train_fraction)
        self.steps_per_epoch = int(np.ceil(
            self.num_epoch_samples / self.batch_size,
        ))
        # Declare row indices, will to an inital shuffle at the end of init`
        self.row_idx = np.arange(self.num_samples)

        # Whether to remove singleton dimensions from tiles (e.g. 2D models)
        self.squeeze = False
        if 'squeeze' in dataset_config:
            assert isinstance(dataset_config['squeeze'], bool), \
                'squeeze parameter should be boolean'
            self.squeeze = dataset_config['squeeze']

        # Whether to use fixed random seed (only recommended for testing)
        self.random_seed = None
        if 'random_seed' in dataset_config:
            self.random_seed = dataset_config['random_seed']
        np.random.seed(self.random_seed)

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

    def _augment_image(self,
                       input_image,
                       aug_idx,
                       zoom=1,
                       theta=0,
                       mean_offset=0,
                       std_scale=1,
                       noise_std=0,
                       blur_img=False,
                       blur_sigma=0,
                       shear=0):
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
        if aug_idx == 0:
            return input_image
        elif aug_idx == 1:
            # flip about y (if zxy) or x if (zyx))
            trans_image = np.flip(input_image, -1)
        elif aug_idx == 2:
            # flip about x (if zxy) or y if (zyx))
            trans_image = np.flip(input_image, -2)
        elif aug_idx == 3:
            # rot in plane defined by axis=(-2,-1)
            trans_image = np.rot90(
                input_image,
                k=1,
                axes=(-2, -1),
            )
        elif aug_idx == 4:
            # rot in plane defined by axis=(-2,-1)
            trans_image = np.rot90(
                input_image,
                k=2,
                axes=(-2, -1),
            )
        elif aug_idx == 5:
            # rot in plane defined by axis=(-2,-1)
            trans_image = np.rot90(
                input_image,
                k=3,
                axes=(-2, -1),
            )
        else:
            msg = '{} not in allowed aug_idx: 0-5'.format(aug_idx)
            raise ValueError(msg)
        if blur_img:
            trans_image = cv2.GaussianBlur(trans_image, ksize=(0, 0), sigmaX=blur_sigma)
        if noise_std != 0:
            trans_image = trans_image + np.random.normal(scale=noise_std, size=trans_image.shape)
        if not (mean_offset == 0 and std_scale == 1):
            trans_image = norm.unzscore(trans_image, mean_offset, std_scale)
        trans_image = apply_affine_transform(trans_image, zx=zoom, theta=theta,
                                             zy=zoom, shear=shear, fill_mode='constant',
                                             cval=0., order=1)
        return trans_image

    def _get_volume(self,
                    fname_list,
                    normalize=True,
                    aug_idx=0,
                    zoom=1,
                    theta=0,
                    mean_offset=0,
                    std_scale=1,
                    noise_std=0,
                    blur_img=False,
                    blur_sigma=0,
                    shear=0,
                    ):
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
                cur_tile = self._augment_image(cur_tile,
                                               aug_idx,
                                               zoom=zoom,
                                               theta=theta,
                                               mean_offset=mean_offset,
                                               std_scale=std_scale,
                                               noise_std=noise_std,
                                               blur_img=blur_img,
                                               blur_sigma=blur_sigma,
                                               shear=shear)
            if self.squeeze:
                cur_tile = np.squeeze(cur_tile)
            image_volume.append(cur_tile)
        # Stack images channels first
        image_volume = np.stack(image_volume)
        if image_volume.dtype == bool:
            image_volume = image_volume.astype(np.float32)
        elif normalize:
            image_volume = norm.zscore(image_volume)
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

        norm_output = self.model_task != 'segmentation' and self.normalize

        input_image = []
        target_image = []
        aug_idx = 0
        zoom = 1
        theta = 0
        mean_offset = 0
        std_scale = 1
        noise_std = 0
        blur_img = False
        blur_sigma = 0
        shear = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            # Select select int randomly that will represent augmentation type
            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)
                zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                theta = np.random.uniform(-self.rotate_range, self.rotate_range)
                mean_offset = np.random.uniform(-self.mean_jitter, self.mean_jitter)
                std_scale = 1 + np.random.uniform(-self.std_jitter, self.std_jitter)
                noise_std = np.random.uniform(0, self.noise_std)
                shear = np.random.uniform(-self.shear_range, self.shear_range)
                if not (self.blur_range[0] == 0 and self.blur_range[1] == 0):
                    blur_img = np.random.choice([True, False], 1)[0]
                    blur_sigma = np.random.uniform(self.blur_range[0], self.blur_range[1])
            # only apply intensity jitter to input
            cur_input = self._get_volume(
                fname_list=cur_input_fnames.split(','),
                normalize=self.normalize,
                aug_idx=aug_idx,
                zoom=zoom,
                theta=theta,
                mean_offset=mean_offset,
                std_scale=std_scale,
                noise_std=noise_std,
                blur_img=blur_img,
                blur_sigma=blur_sigma,
                shear=shear,
            )
            cur_target = self._get_volume(
                fname_list=cur_target_fnames.split(','),
                normalize=norm_output,
                aug_idx=aug_idx,
                zoom=zoom,
                theta=theta,
                shear=shear,
                mean_offset=0,
                std_scale=1,
                noise_std=0,
                blur_img=False,
                blur_sigma=0,
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
