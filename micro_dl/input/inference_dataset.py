"""Dataset class / generator for inference only"""

import keras
import numpy as np
import os
import pandas as pd

from micro_dl.utils.image_utils import crop2base
from micro_dl.utils.tile_utils import preprocess_imstack


class InferenceDataset(keras.utils.Sequence):
    """Dataset class for model inference"""

    def __init__(self,
                 image_dir,
                 dataset_config,
                 network_config,
                 df_meta,
                 image_format='zyx',
                 flat_field_dir=None):
        """Init

        :param str image_dir: dir containing images AND NOT TILES!
        :param dict dataset_config: dict with dataset related params
        :param dict network_config: dict with network related params
        :param pd.Dataframe df_meta: dataframe with time, channel, pos and
         slice indices
        :param str image_format: xyz or zyx format
        :param str flat_field_dir: dir with flat field images
        """

        # no augmentation needed for inference
        self.augmentations = False
        self.shuffle = False
        self.flat_field_dir = flat_field_dir

        self.image_dir = image_dir
        assert image_format in {'xyz', 'zyx'}, \
            "Image format should be xyz or zyx, not {}".format(image_format)
        self.image_format = image_format

        # Check if model task (regression or segmentation) is specified
        self.model_task = 'regression'
        if 'model_task' in dataset_config:
            self.model_task = dataset_config['model_task']
            assert self.model_task in {'regression', 'segmentation'}, \
                "Model task must be either 'segmentation' or 'regression'"

        self.depth = 1
        # adjust slice margins if stacktostack or stackto2d
        network_cls = network_config['class']
        if network_cls in ['UNetStackTo2D', 'UNetStackToStack']:
            self.depth = network_config['network']['depth']
            df_meta = self.adjust_slice_indices(df_meta, self.depth)

        self.df_meta = df_meta

        # if Unet2D 4D tensor, remove the singleton dimension, else 5D
        self.squeeze = False
        if network_cls == 'UNet2D':
            self.squeeze = True

        self.im_3d = False
        if network_cls == 'UNet3D':
            self.im_3d = True

        self.data_format = network_config['data_format']

        self.input_channels = dataset_config['input_channels']
        self.target_channels = dataset_config['target_channels']
        df_idx = (df_meta['channel_idx'] ==
                  dataset_config['target_channels'][0])
        df_iteration_meta = df_meta[df_idx]
        # check if sorted values look right
        df_iteration_meta = df_iteration_meta.sort_values(
            ['pos_idx',  'slice_idx'], ascending=[True, True]
        )
        df_iteration_meta = df_iteration_meta.reset_index(drop=True)
        self.df_iteration_meta = df_iteration_meta
        self.num_samples = len(self.df_iteration_meta)

    @staticmethod
    def adjust_slice_indices(df_meta, depth):
        """Adjust slice indices if stackto2d or stacktostack

        :param pd.Dataframe df_meta: dataframe with info for all slices
        :param int depth: depth of stack for UNetStackto2D and UNetStackToStack
        :return pd.Dataframe df_meta: with rows corr. to end slices removed
        """

        # these networks will have a depth > 1
        margin = depth // 2
        df_drop_idx = df_meta.index[
            df_meta['slice_idx'].isin(list(range(margin)))
        ]
        df_meta.drop(df_meta.index[df_drop_idx], inplace=True)
        max_slice_idx = df_meta['slice_idx'].max()
        drop_idx = list(range(max_slice_idx - margin, max_slice_idx))
        df_drop_idx = df_meta.index[df_meta['slice_idx'].isin(drop_idx)]
        df_meta.drop(df_meta.index[df_drop_idx], inplace=True)
        return df_meta

    def get_df_iteration_meta(self):
        """Get the iteration dataframe"""

        return self.df_iteration_meta

    def __len__(self):
        """Gets the number of batches per epoch"""

        return self.num_samples

    def _get_image(self, cur_row, channel_ids, normalize):
        """Assemble one input/target tensor

        :param pd.Series cur_row:
        :param int/list channel_ids:
        :param bool normalize:
        :return np.array (3D / 4D) cur_stack:
        """

        cur_slice_idx = cur_row['slice_idx']
        cur_time_idx = cur_row['time_idx']
        cur_pos_idx = cur_row['pos_idx']
        cur_stack = []
        for ch_idx in channel_ids:
            cur_flat_field_im = None
            if self.flat_field_dir is not None:
                cur_flat_field_fname = os.path.join(
                    self.flat_field_dir,
                    'flat-field_channel-{}.npy'.format(ch_idx)
                )
                cur_flat_field_im = np.load(cur_flat_field_fname)

            cur_image = preprocess_imstack(
                frames_metadata=self.df_meta,
                input_dir=self.image_dir,
                depth=self.depth,
                time_idx=cur_time_idx,
                channel_idx=ch_idx,
                slice_idx=cur_slice_idx,
                pos_idx=cur_pos_idx,
                flat_field_im=cur_flat_field_im,
                normalize_im=normalize
            )

            # Crop image to nearest factor of two in xy
            cur_image = crop2base(cur_image)  # crop_z=self.im_3d)

            if self.image_format == 'zyx':
                cur_image = np.transpose(cur_image, [2, 0, 1])
            if self.squeeze:
                cur_image = np.squeeze(cur_image)
            cur_stack.append(cur_image)

        # stack for channel dimension
        if self.data_format == 'channels_first':
            cur_stack = np.stack(cur_stack)
        else:
            cur_stack = np.stack(cur_stack, axis=self.n_dims - 2)
        return cur_stack

    def __getitem__(self, index):
        """Get a batch of data"""

        input_stack = []
        target_stack = []

        cur_row = self.df_iteration_meta.iloc[index]
        cur_input = self._get_image(cur_row,
                                    self.input_channels,
                                    normalize=True)
        # the raw input images have to be normalized (z-score typically)
        normalize = True if self.model_task == 'regression' else False
        cur_target = self._get_image(cur_row, self.target_channels, normalize)
        input_stack.append(cur_input)
        target_stack.append(cur_target)
        # stack for batch dimension
        input_stack = np.stack(input_stack)
        target_stack = np.stack(target_stack)
        return input_stack, target_stack
