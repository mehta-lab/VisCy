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
                 inference_config,
                 dataset_config,
                 network_config,
                 split_col_ids,
                 preprocess_config=None,
                 image_format='zyx',
                 mask_dir=None,
                 flat_field_dir=None,
                 crop2base=True):
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
        normalize_im = 'stack'
        if preprocess_config is not None:
            if 'normalize' in preprocess_config:
                if 'normalize_im' in preprocess_config['normalize']:
                    normalize_im = preprocess_config['normalize']['normalize_im']
            elif 'normalize_im' in preprocess_config:
                normalize_im = preprocess_config['normalize_im']
            elif 'normalize_im' in preprocess_config['tile']:
                normalize_im = preprocess_config['tile']['normalize_im']

        self.normalize_im = normalize_im
        # assume input and target channels are the same as training if not specified
        self.input_channels = dataset_config['input_channels']
        self.target_channels = dataset_config['target_channels']
        slice_ids = self.frames_meta['slice_idx'].unique()
        pos_ids = self.frames_meta['pos_idx'].unique()
        time_ids = self.frames_meta['time_idx'].unique()
        # overwrite default parameters from train config
        if 'dataset' in inference_config:
            if 'input_channels' in inference_config['dataset']:
                self.input_channels = inference_config['dataset']['input_channels']
            if 'target_channels' in inference_config['dataset']:
                self.target_channels = inference_config['dataset']['target_channels']
            if 'slice_ids' in inference_config['dataset']:
                slice_ids = inference_config['dataset']['slice_ids']
            if 'pos_ids' in inference_config['dataset']:
                pos_ids = inference_config['dataset']['pos_ids']
            if 'time_ids' in inference_config['dataset']:
                time_ids = inference_config['dataset']['time_ids']
        if not set(self.target_channels) <= set(self.frames_meta['channel_idx'].unique()):
            ValueError('target channels are out of range. Add "mask" to config if target channel is mask')
        # get a subset of frames meta for only one channel to easily
        # extract indices (pos, time, slice) to iterate over
        self.inf_frames_meta = aux_utils.get_sub_meta(
            self.frames_meta,
            time_ids=time_ids,
            pos_ids=pos_ids,
            slice_ids=slice_ids,
            channel_ids=self.target_channels)

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
        self.inf_frames_meta = self.inf_frames_meta.sort_values(
            ['pos_idx',  'slice_idx'],
            ascending=[True, True],
        )
        self.inf_frames_meta = self.inf_frames_meta.reset_index(drop=True)
        self.num_samples = len(self.inf_frames_meta)
        self.crop2base = crop2base

    def adjust_slice_indices(self):
        """
        Adjust slice indices if model is UNetStackTo2D or UNetStackToStack.
        These networks will have a depth > 1.
        Adjust inf_frames_meta only as we'll need all the indices to load
        stack with depth > 1.
        """
        margin = self.depth // 2
        # Drop indices on both margins
        max_slice_idx = self.inf_frames_meta['slice_idx'].max() + 1
        min_slice_idx = self.inf_frames_meta['slice_idx'].min()
        drop_idx = list(range(max_slice_idx - margin, max_slice_idx)) + \
                   list(range(min_slice_idx, min_slice_idx + margin))
        df_drop_idx = self.inf_frames_meta.index[
            self.inf_frames_meta['slice_idx'].isin(drop_idx),
        ]
        self.inf_frames_meta.drop(df_drop_idx, inplace=True)
        # Drop indices below margin
        df_drop_idx = self.inf_frames_meta.index[
            self.inf_frames_meta['slice_idx'].isin(list(range(margin)))
        ]
        self.inf_frames_meta.drop(df_drop_idx, inplace=True)

    def get_iteration_meta(self):
        """
        Get the dataframe containing indices for one channel for
        inference iterations.

        :return pandas Dataframe inf_frames_meta: Metadata and indices for
         first target channel
        """
        return self.inf_frames_meta

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
                   normalize_im,
                   is_mask=False):
        """
        Assemble one input or target tensor

        :param str input_dir: Directory containing images or targets
        :param pd.Series cur_row: Current row in frames_meta
        :param int/list channel_ids: Channel indices
        :param int depth: Stack depth
        :param str normalize: normalization options for images
        :return np.array (3D / 4D) im_stack: Image stack
        """
        im_stack = []
        for channel_idx in channel_ids:
            flat_field_im = None
            if self.flat_field_dir is not None:
                assert normalize_im in [None, 'stack'],\
                    "flat field correction currently only supports " \
                    "None or 'stack' option for 'normalize_im'"
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
                normalize_im=normalize_im,
            )
            # Crop image to nearest factor of two in xy
            if self.crop2base:
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
        # binarize the target images for segmentation task
        if is_mask:
            im_stack = im_stack > 0
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
        cur_row = self.inf_frames_meta.iloc[index]
        # binarize the target images for segmentation task
        is_mask = False
        if self.model_task == 'segmentation':
            is_mask = True
        # Get input and target stacks for inference
        input_stack = self._get_image(
            input_dir=self.image_dir,
            cur_row=cur_row,
            channel_ids=self.input_channels,
            depth=self.depth,
            normalize_im=self.normalize_im,
        )
        target_stack = self._get_image(
            input_dir=self.target_dir,
            cur_row=cur_row,
            channel_ids=self.target_channels,
            depth=self.target_depth,
            normalize_im=None,
            is_mask=is_mask,
        )
        # Add batch dimension
        input_stack = np.expand_dims(input_stack, axis=0)
        target_stack = np.expand_dims(target_stack, axis=0)
        return input_stack, target_stack
