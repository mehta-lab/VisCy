"""Dataset class / generator for inference only"""

import keras
import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class InferenceDataSet(keras.utils.Sequence):
    """Dataset class for model inference"""

    def __init__(self,
                 image_dir,
                 dataset_config,
                 network_config,
                 split_col_ids,
                 image_format='zyx',
                 mask_dir=None,
                 flat_field_dir=None):
        """Init

        :param str image_dir: dir containing images AND NOT TILES!
        :param dict dataset_config: dict with dataset related params
        :param dict network_config: dict with network related params
        :param tuple split_col_ids: How to split up the dataset for inference:
         for frames_meta: (str split column name, list split row indices)
        :param str image_format: xyz or zyx format
        :param str/None mask_dir: If inference targets are masks stored in a
         different directory than the image dir. Assumes the directory contains
         a frames_meta.csv containing mask channels (which will be target channels
          in the inference config) z, t, p indices matching the ones in image_dir
        :param str flat_field_dir: Directory with flat field images
        """
        self.image_dir = image_dir
        self.target_dir = image_dir
        self.frames_meta = aux_utils.read_meta(self.image_dir)
        self.flat_field_dir = flat_field_dir
        if mask_dir is not None:
            self.target_dir = mask_dir
            # Append mask meta to frames meta
            mask_meta = aux_utils.read_meta(mask_dir)
            self.frames_meta = self.frames_meta.append(
                mask_meta,
                ignore_index=True,
            )
        # Use only indices selected for inference
        (split_col, split_ids) = split_col_ids
        meta_ids = self.frames_meta[split_col].isin(split_ids)
        self.frames_meta = self.frames_meta[meta_ids]

        assert image_format in {'xyz', 'zyx'}, \
            "Image format should be xyz or zyx, not {}".format(image_format)
        self.image_format = image_format

        # Check if model task (regression or segmentation) is specified
        self.model_task = 'regression'
        if 'model_task' in dataset_config:
            self.model_task = dataset_config['model_task']
            assert self.model_task in {'regression', 'segmentation'}, \
                "Model task must be either 'segmentation' or 'regression'"
        # the raw input images have to be normalized (z-score typically)
        self.normalize = True if self.model_task == 'regression' else False

        self.input_channels = dataset_config['input_channels']
        self.target_channels = dataset_config['target_channels']
        # get a subset of frames meta for only one channel to easily
        # extract indices (pos, time, slice) to iterate over
        df_idx = (self.frames_meta['channel_idx'] == self.target_channels[0])
        self.iteration_meta = self.frames_meta.copy()
        self.iteration_meta = self.iteration_meta[df_idx]

        self.depth = 1
        self.target_depth = 1
        # adjust slice margins if stacktostack or stackto2d
        network_cls = network_config['class']
        if network_cls in ['UNetStackTo2D', 'UNetStackToStack']:
            self.depth = network_config['depth']
            self.adjust_slice_indices()

        # if Unet2D 4D tensor, remove the singleton dimension, else 5D
        self.squeeze = False
        if network_cls == 'UNet2D':
            self.squeeze = True

        self.im_3d = False
        if network_cls == 'UNet3D':
            self.im_3d = True

        self.data_format = 'channels_first'
        if 'data_format' in network_config:
            self.data_format = network_config['data_format']

        # check if sorted values look right
        self.iteration_meta = self.iteration_meta.sort_values(
            ['pos_idx',  'slice_idx'],
            ascending=[True, True],
        )
        self.iteration_meta = self.iteration_meta.reset_index(drop=True)
        self.num_samples = len(self.iteration_meta)

    def adjust_slice_indices(self):
        """
        Adjust slice indices if model is UNetStackTo2D or UNetStackToStack.
        These networks will have a depth > 1.
        Adjust iteration_meta only as we'll need all the indices to load
        stack with depth > 1.
        """
        margin = self.depth // 2
        # Drop indices above margin
        max_slice_idx = self.iteration_meta['slice_idx'].max() + 1
        drop_idx = list(range(max_slice_idx - margin, max_slice_idx))
        df_drop_idx = self.iteration_meta.index[
            self.iteration_meta['slice_idx'].isin(drop_idx),
        ]
        self.iteration_meta.drop(df_drop_idx, inplace=True)
        # Drop indices below margin
        df_drop_idx = self.iteration_meta.index[
            self.iteration_meta['slice_idx'].isin(list(range(margin)))
        ]
        self.iteration_meta.drop(df_drop_idx, inplace=True)

    def get_iteration_meta(self):
        """
        Get the dataframe containing indices for one channel for
        inference iterations.

        :return pandas Dataframe iteration_meta: Metadata and indices for
         first target channel
        """
        return self.iteration_meta

    def __len__(self):
        """
        Get the total number of samples inference is performed on.

        :return int num_samples: Number of inference samples
        """
        return self.num_samples

    def _get_image(self,
                   input_dir,
                   cur_row,
                   channel_ids,
                   depth,
                   normalize):
        """
        Assemble one input or target tensor

        :param str input_dir: Directory containing images or targets
        :param pd.Series cur_row: Current row in frames_meta
        :param int/list channel_ids: Channel indices
        :param int depth: Stack depth
        :param bool normalize: If image should be normalized
        :return np.array (3D / 4D) im_stack: Image stack
        """
        im_stack = []
        for channel_idx in channel_ids:
            flat_field_im = None
            if self.flat_field_dir is not None and normalize:
                flat_field_fname = os.path.join(
                    self.flat_field_dir,
                    'flat-field_channel-{}.npy'.format(channel_idx)
                )
                flat_field_im = np.load(flat_field_fname)
            # Load image with given indices
            im = image_utils.preprocess_imstack(
                frames_metadata=self.frames_meta,
                input_dir=input_dir,
                depth=depth,
                time_idx=cur_row['time_idx'],
                channel_idx=channel_idx,
                slice_idx=cur_row['slice_idx'],
                pos_idx=cur_row['pos_idx'],
                flat_field_im=flat_field_im,
                normalize_im=normalize,
            )
            # Crop image to nearest factor of two in xy
            im = image_utils.crop2base(im)  # crop_z=self.im_3d)
            # Make sure image format is right and squeeze for 2D models
            if self.image_format == 'zyx' and len(im.shape) > 2:
                im = np.transpose(im, [2, 0, 1])
            if self.squeeze:
                im = np.squeeze(im)
            im_stack.append(im)
        # stack for channel dimension
        if self.data_format == 'channels_first':
            im_stack = np.stack(im_stack)
        else:
            im_stack = np.stack(im_stack, axis=self.n_dims - 2)
        # Make sure all images have the same dtype
        im_stack = im_stack.astype(np.float32)
        return im_stack

    def __getitem__(self, index):
        """
        Get a batch of data, input and target stacks, for inference.

        :param int index: Iteration index (looped through linearly in inference)
        :return np.array input_stack: Input image stack with dimensionality
            matching model
        :return np.array target_stack: Target image stack for model inference
        """
        # Get indices for current inference iteration
        cur_row = self.iteration_meta.iloc[index]
        # Get input and target stacks for inference
        input_stack = self._get_image(
            input_dir=self.image_dir,
            cur_row=cur_row,
            channel_ids=self.input_channels,
            depth=self.depth,
            normalize=True,
        )
        target_stack = self._get_image(
            input_dir=self.target_dir,
            cur_row=cur_row,
            channel_ids=self.target_channels,
            depth=self.target_depth,
            normalize=self.normalize,
        )
        # Add batch dimension
        input_stack = np.expand_dims(input_stack, axis=0)
        target_stack = np.expand_dims(target_stack, axis=0)
        return input_stack, target_stack
