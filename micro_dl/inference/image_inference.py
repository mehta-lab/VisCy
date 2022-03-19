"""Model inference at the image/volume level"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd

from micro_dl.input.inference_dataset import InferenceDataSet
import micro_dl.inference.model_inference as inference
from micro_dl.inference.evaluation_metrics import MetricsEstimator
from micro_dl.inference.stitch_predictions import ImageStitcher
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.tile_utils as tile_utils
from micro_dl.utils.train_utils import set_keras_session
import micro_dl.utils.normalize as normalize
import micro_dl.plotting.plot_utils as plot_utils


class ImagePredictor:
    """Infer on larger images"""

    def __init__(self,
                 train_config,
                 inference_config,
                 preprocess_config=None,
                 gpu_id=-1,
                 gpu_mem_frac=None):
        """Init

        :param dict train_config: Training config dict with params related
            to dataset, trainer and network
        :param dict inference_config: Read yaml file with following parameters:
            str model_dir: Path to model directory
            str/None model_fname: File name of weights in model dir (.hdf5).
             If left out, latest weights file will be selected.
            str image_dir: dir containing input images AND NOT TILES!
            str data_split: Which data (train/test/val) to run inference on.
             (default = test)
            dict images:
             str image_format: 'zyx' or 'xyz'
             str/None flat_field_dir: flatfield directory
             str im_ext: For writing predictions e.g. '.png' or '.npy' or '.tiff'
             FOR 3D IMAGES USE NPY AS PNG AND TIFF ARE CURRENTLY NOT SUPPORTED.
             list crop_shape: center crop the image to a specified shape before
             tiling for inference
             str suffix: Any extra string you want to include at the end of the name
             str name_format: image name format. 'cztp' or 'sms'
            dict metrics:
             list metrics_list: list of metrics to estimate. available
             metrics: [ssim, corr, r2, mse, mae}]
             list metrics_orientations: xy, xyz, xz or yz
              (see evaluation_metrics.py for description of orientations)
            dict masks: dict with keys
             str mask_dir: Mask directory containing a frames_meta.csv containing
             mask channels (which will be target channels in the inference config)
             z, t, p indices matching the ones in image_dir. Mask dirs are often
             generated or have frames_meta added to them during preprocessing.
             str mask_type: 'target' for segmentation, 'metrics' for weighted
             int mask_channel: mask channel as in training
            dict tile: dict with params for tiling/stitching with keys:
             num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
             int num_slices: in case of 3D, the full volume will not fit in GPU
              memory, specify the number of slices to use and this will depend on
              the network depth, for ex 8 for a network of depth 4.
             list inf_shape: inference on a center sub volume.
             list tile_shape: shape of tile for tiling along xyz.
             int/list num_overlap: int for tile_z, list for tile_xyz
             str overlap_operation: e.g. 'mean'
        :param dict preprocess_config: parameters from proprocess config yaml file
        :param int gpu_id: GPU number to use. -1 for debugging (no GPU)
        :param float/None gpu_mem_frac: Memory fractions to use corresponding
            to gpu_ids
        """
        # Use model_dir from inference config if present, otherwise use train
        if 'model_dir' in inference_config:
            model_dir = inference_config['model_dir']
        else:
            model_dir = train_config['trainer']['model_dir']

        if 'model_fname' in inference_config:
            model_fname = inference_config['model_fname']
        else:
            # If model filename not listed, grab latest one
            fnames = [f for f in os.listdir(inference_config['model_dir'])
                      if f.endswith('.hdf5')]
            assert len(fnames) > 0, 'No weight files found in model dir'
            fnames = natsort.natsorted(fnames)
            model_fname = fnames[-1]

        self.config = train_config
        self.model_dir = model_dir
        self.image_dir = inference_config['image_dir']


        # Set default for data split, determine column name and indices
        data_split = 'test'
        if 'data_split' in inference_config:
            data_split = inference_config['data_split']
        assert data_split in ['train', 'val', 'test', 'all'], \
            'data_split not in [train, val, test, all]'
        split_col_ids = self._get_split_ids(data_split)

        self.data_format = self.config['network']['data_format']
        assert self.data_format in {'channels_first', 'channels_last'}, \
            "Data format should be channels_first/last"
        self.input_depth = 1
        if 'depth' in self.config['network']:
            self.input_depth = self.config['network']['depth']
        flat_field_dir = None
        images_dict = inference_config['images']
        if 'flat_field_dir' in images_dict:
            flat_field_dir = images_dict['flat_field_dir']

        # Set defaults
        self.image_format = 'zyx'
        if 'image_format' in images_dict:
            self.image_format = images_dict['image_format']
        self.pred_chan_names = None
        if 'pred_chan_names' in images_dict:
            self.pred_chan_names = images_dict['pred_chan_names']
        self.image_ext = '.png'
        if 'image_ext' in images_dict:
            self.image_ext = images_dict['image_ext']
        self.suffix = None
        if 'suffix' in images_dict:
            self.suffix = images_dict['suffix']
        self.name_format = 'ctzp'
        if 'name_format' in images_dict:
            self.name_format = images_dict['name_format']
        # Create image subdirectory to write predicted images
        self.pred_dir = os.path.join(self.model_dir, 'predictions')
        if 'save_to_image_dir' in inference_config:
            if inference_config['save_to_image_dir']:
                self.pred_dir = os.path.join(self.image_dir, os.path.basename(model_dir))
        os.makedirs(self.pred_dir, exist_ok=True)

        self.save_figs = True
        if 'save_figs' in inference_config:
            self.save_figs = inference_config['save_figs']

        # Check if model task (regression or segmentation) is specified
        self.model_task = 'regression'
        if 'model_task' in self.config['dataset']:
            self.model_task = self.config['dataset']['model_task']
            assert self.model_task in {'regression', 'segmentation'}, \
                "Model task must be either 'segmentation' or 'regression'"

        # Handle masks as either targets or for masked metrics
        self.masks_dict = None
        self.mask_metrics = False
        self.mask_dir = None
        self.mask_meta = None
        mask_dir = None
        if 'masks' in inference_config:
            self.masks_dict = inference_config['masks']
            assert 'mask_channel' in self.masks_dict , 'mask_channel is needed'
            assert 'mask_dir' in self.masks_dict, 'mask_dir is needed'
            self.mask_dir = self.masks_dict['mask_dir']
            self.mask_meta = aux_utils.read_meta(self.mask_dir)
            assert 'mask_type' in self.masks_dict, \
                'mask_type (target/metrics) is needed'
            if self.masks_dict['mask_type'] == 'metrics':
                assert self.model_task == 'regression', \
                    'masked metrics are for regression tasks only'
                # Compute weighted metrics
                self.mask_metrics = True
            else:
                assert self.model_task == 'segmentation', \
                    'masks can only be target for segmentation tasks'
                mask_dir = self.mask_dir

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

        # Handle 3D volume inference settings
        self.num_overlap = 0
        self.stitch_inst = None
        self.tile_option = None
        self.z_dim = 2
        self.crop_shape = None
        if 'crop_shape' in images_dict:
            self.crop_shape = images_dict['crop_shape']
        crop2base = True
        if 'tile' in inference_config:
            self.tile_params = inference_config['tile']
            self._assign_3d_inference()
            crop2base = False
            # Make image ext npy default for 3D
        # Create dataset instance
        self.dataset_inst = InferenceDataSet(
            image_dir=self.image_dir,
            inference_config=inference_config,
            dataset_config=self.config['dataset'],
            network_config=self.config['network'],
            preprocess_config=preprocess_config,
            split_col_ids=split_col_ids,
            image_format=images_dict['image_format'],
            mask_dir=mask_dir,
            flat_field_dir=flat_field_dir,
            crop2base=crop2base,
        )
        # create an instance of MetricsEstimator
        self.inf_frames_meta = self.dataset_inst.get_iteration_meta()
        self.target_channels = self.dataset_inst.target_channels
        if self.pred_chan_names is not None:
            assert len(self.target_channels) == len(self.pred_chan_names), \
                '"pred_chan_names" and "target_channels" have to have the same lengths'
        else:
            self.pred_chan_names = [None] * len(self.target_channels)
        assert not self.inf_frames_meta.empty, 'inference metadata is empty.'
        # Handle metrics config settings
        self.metrics_inst = None
        self.metrics_dict = None
        if 'metrics' in inference_config:
            self.metrics_dict = inference_config['metrics']
        if self.metrics_dict is not None:
            assert 'metrics' in self.metrics_dict,\
                'Must specify with metrics to use'
            self.metrics_inst = MetricsEstimator(
                metrics_list=self.metrics_dict['metrics'],
                masked_metrics=self.mask_metrics,
            )
            self.metrics_orientations = ['xy']
            available_orientations = ['xy', 'xyz', 'xz', 'yz']
            if 'metrics_orientations' in self.metrics_dict:
                self.metrics_orientations = \
                    self.metrics_dict['metrics_orientations']
                assert set(self.metrics_orientations).\
                    issubset(available_orientations),\
                    'orientation not in [xy, xyz, xz, yz]'
            self.df_xy = pd.DataFrame()
            self.df_xyz = pd.DataFrame()
            self.df_xz = pd.DataFrame()
            self.df_yz = pd.DataFrame()

        # Set session if not debug
        if gpu_id >= 0:
            self.sess = set_keras_session(
                gpu_ids=gpu_id,
                gpu_mem_frac=gpu_mem_frac,
            )
        # create model and load weights
        self.model = inference.load_model(
            network_config=self.config['network'],
            model_fname=os.path.join(self.model_dir, model_fname),
            predict=True,
        )

    def _get_split_ids(self, data_split='test'):
        """
        Get the indices for data_split

        :param str data_split: in [train, val, test]
        :return list inference_ids: Indices for inference given data split
        :return str split_col: Dataframe column name, which was split in training
        """
        split_col = self.config['dataset']['split_by_column']
        frames_meta = aux_utils.read_meta(self.image_dir)
        inference_ids = np.unique(frames_meta[split_col]).tolist()
        if data_split == 'all':
            return split_col, inference_ids

        try:
            split_fname = os.path.join(self.model_dir, 'split_samples.json')
            split_samples = aux_utils.read_json(split_fname)
            inference_ids = split_samples[data_split]
        except FileNotFoundError as e:
            print("No split_samples file. "
                  "Will predict all images in dir.")

        return split_col, inference_ids

    def _assign_3d_inference(self):
        """
        Assign inference options for 3D volumes

        tile_z - 2d/3d predictions on full xy extent, stitch predictions along
            z axis
        tile_xyz - 2d/3d prediction on sub-blocks, stitch predictions along xyz
        infer_on_center - infer on center block
        """
        # assign zdim if not Unet2D
        if self.image_format == 'zyx':
            self.z_dim = 2 if self.data_format == 'channels_first' else 1
        elif self.image_format == 'xyz':
            self.z_dim = 4 if self.data_format == 'channels_first' else 3

        if 'num_slices' in self.tile_params and self.tile_params['num_slices'] > 1:
            self.tile_option = 'tile_z'
            assert self.tile_params['num_slices'] >= self.input_depth, \
                'inference num of slices < num of slices used for training. ' \
                'Inference on reduced num of slices gives sub optimal results' \
                'Train slices: {}, inference slices: {}'.format(
                    self.input_depth, self.tile_params['num_slices'],
                )
            num_slices = self.tile_params['num_slices']

            assert self.config['network']['class'] == 'UNet3D', \
                'currently stitching predictions available for 3D models only'
            network_depth = len(
                self.config['network']['num_filters_per_block']
            )
            min_num_slices = 2 ** (network_depth - 1)
            assert num_slices >= min_num_slices, \
                'Insufficient number of slices {} for the network ' \
                'depth {}'.format(num_slices, network_depth)
            self.num_overlap = self.tile_params['num_overlap'] \
                if 'num_overlap' in self.tile_params else 0
        elif 'tile_shape' in self.tile_params:
            if self.config['network']['class'] == 'UNet3D':
                self.tile_option = 'tile_xyz'
            else:
                self.tile_option = 'tile_xy'
            self.num_overlap = self.tile_params['num_overlap'] \
                if 'num_overlap' in self.tile_params else [0, 0, 0]
        elif 'inf_shape' in self.tile_params:
            self.tile_option = 'infer_on_center'
            self.num_overlap = 0

        # create an instance of ImageStitcher
        if self.tile_option in ['tile_z', 'tile_xyz', 'tile_xy']:
            overlap_dict = {
                'overlap_shape': self.num_overlap,
                'overlap_operation': self.tile_params['overlap_operation']
            }
            self.stitch_inst = ImageStitcher(
                tile_option=self.tile_option,
                overlap_dict=overlap_dict,
                image_format=self.image_format,
                data_format=self.data_format
            )

    def _get_sub_block_z(self,
                         input_image,
                         start_z_idx,
                         end_z_idx):
        """Get the sub block along z given start and end slice indices

        :param np.array input_image: 5D tensor with the entire 3D volume
        :param int start_z_idx: start slice for the current block
        :param int end_z_idx: end slice for the current block
        :return np.array cur_block: sub block / volume
        """

        if self.image_format == 'xyz' and \
                self.data_format == 'channels_first':
            cur_block = input_image[:, :, :, :, start_z_idx: end_z_idx]
        elif self.image_format == 'xyz' and \
                self.data_format == 'channels_last':
            cur_block = input_image[:, :, :, start_z_idx: end_z_idx, :]
        elif self.image_format == 'zyx' and \
                self.data_format == 'channels_first':
            cur_block = input_image[:, :, start_z_idx: end_z_idx, :, :]
        elif self.image_format == 'zyx' and \
                self.data_format == 'channels_last':
            cur_block = input_image[:, start_z_idx: end_z_idx, :, :, :]
        return cur_block

    def _predict_sub_block_z(self, input_image):
        """Predict sub blocks along z

        :param np.array input_image: 5D tensor with the entire 3D volume
        :return list pred_ims - list of predicted sub blocks
         list start_end_idx - list of tuples with start and end z indices
        """

        pred_ims = []
        start_end_idx = []
        num_z = input_image.shape[self.z_dim]
        num_slices = self.tile_params['num_slices']
        num_blocks = np.ceil(
            num_z / (num_slices - self.num_overlap)
        ).astype('int')
        for block_idx in range(num_blocks):
            start_idx = block_idx * (num_slices - self.num_overlap)
            end_idx = start_idx + num_slices
            if end_idx >= num_z:
                end_idx = num_z
                start_idx = end_idx - num_slices
            cur_block = self._get_sub_block_z(input_image,
                                              start_idx,
                                              end_idx)
            pred_block = inference.predict_large_image(
                model=self.model,
                input_image=cur_block,
            )
            # reduce predictions from 5D to 3D for simplicity
            pred_ims.append(np.squeeze(pred_block))
            start_end_idx.append((start_idx, end_idx))
        return pred_ims, start_end_idx

    def _predict_sub_block_xy(self,
                              input_image,
                              crop_indices):
        """Predict sub blocks along xyz

        :param np.array input_image: 5D tensor with the entire 3D volume
        :param list crop_indices: list of crop indices: min/max xyz
        :return list pred_ims - list of predicted sub blocks
        """
        pred_ims = []
        assert self.image_format == 'zyx', \
            'predict_sub_block_xy only supports zyx format'
        for idx, crop_idx in enumerate(crop_indices):
            print('Running inference on tile {}/{}'.format(idx, len(crop_indices)))
            if self.data_format == 'channels_first':
                if len(input_image.shape) == 5: # bczyx
                    cur_block = input_image[:, :, :, crop_idx[0]: crop_idx[1],
                                            crop_idx[2]: crop_idx[3]]
                else: # bcyx
                    cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3]]
            else:
                if len(input_image.shape) == 5: # bzyxc
                    cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                            crop_idx[2]: crop_idx[3],
                                            :]
                else: # byxc
                    cur_block = input_image[:, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3],
                                :]

            pred_block = inference.predict_large_image(
                model=self.model,
                input_image=cur_block,
            )
            # remove b & z dimention, prediction of 2D and 2.5D has only single z
            if self.data_format == 'channels_first':
                if len(pred_block.shape) == 5:  # bczyx
                    pred_block = pred_block[0, :, 0, ...]
                else: # bcyx
                    pred_block = pred_block[0, :, ...]
            else:
                if len(pred_block.shape) == 5:  # bzyxc
                    pred_block = pred_block[0, 0, ...]
                else:  # byxc
                    pred_block = pred_block[0, ...]
            pred_ims.append(pred_block)
        return pred_ims

    def _predict_sub_block_xyz(self,
                               input_image,
                               crop_indices):
        """Predict sub blocks along xyz

        :param np.array input_image: 5D tensor with the entire 3D volume
        :param list crop_indices: list of crop indices: min/max xyz
        :return list pred_ims - list of predicted sub blocks
        """
        pred_ims = []
        for crop_idx in crop_indices:
            if self.data_format == 'channels_first':
                cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                        crop_idx[2]: crop_idx[3],
                                        crop_idx[4]: crop_idx[5]]
            else:
                cur_block = input_image[:, crop_idx[0]: crop_idx[1],
                                        crop_idx[2]: crop_idx[3],
                                        crop_idx[4]: crop_idx[5], :]

            pred_block = inference.predict_large_image(
                model=self.model,
                input_image=cur_block,
            )
            # retain the full 5D tensor to experiment for multichannel case
            pred_ims.append(pred_block)
        return pred_ims

    def unzscore(self,
                 im_pred,
                 im_target,
                 meta_row):

        if self.normalize_im is not None:
            if self.normalize_im in ['dataset', 'volume', 'slice']:
                zscore_median = meta_row['zscore_median']
                zscore_iqr =  meta_row['zscore_iqr']
            else:
                zscore_median = np.nanmean(im_target)
                zscore_iqr = np.nanstd(im_target)
            im_pred = normalize.unzscore(im_pred, zscore_median, zscore_iqr)
        return im_pred

    def save_pred_image(self,
                        im_input,
                        im_target,
                        im_pred,
                        meta_row,
                        pred_chan_name=None,
                        ):
        """
        Save predicted images with image extension given in init.

        :param np.array im_pred: 2D / 3D predicted image
        :param int time_idx: time index
        :param int target_channel_idx: target / predicted channel index
        :param int pos_idx: FOV / position index
        :param int slice_idx: slice index
        :param str chan_name: channel name
        """
        # Write prediction image
        if self.name_format == 'cztp':
            im_name = aux_utils.get_im_name(
                time_idx=meta_row['time_idx'],
                channel_idx=meta_row['channel_idx'],
                slice_idx=meta_row['slice_idx'],
                pos_idx=meta_row['pos_idx'],
                ext=self.image_ext,
                extra_field=self.suffix,
            )
        else:
            if pred_chan_name is None:
                pred_chan_name = meta_row['channel_name']
            im_name = aux_utils.get_sms_im_name(
                time_idx=meta_row['time_idx'],
                channel_name=pred_chan_name,
                slice_idx=meta_row['slice_idx'],
                pos_idx=meta_row['pos_idx'],
                ext=self.image_ext,
                extra_field=self.suffix,
            )
        file_name = os.path.join(self.pred_dir, im_name)
        if self.model_task == 'regression':
            im_pred = np.clip(im_pred, 0, 65535)
            im_pred = im_pred.astype(np.uint16)
        else:
            # assuming segmentation output is probability maps
            im_pred = im_pred.astype(np.float32)
        if self.image_ext in ['.png', '.tif']:
            if self.image_ext == '.png':
                assert im_pred.dtype == np.uint16,\
                    'PNG format does not support float type. ' \
                    'Change file extension as ".tif" or ".npy" instead'
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif self.image_ext == '.npy':
            np.save(file_name, np.squeeze(im_pred), allow_pickle=True)
        else:
            raise ValueError(
                'Unsupported file extension: {}'.format(self.image_ext),
            )

        if self.save_figs:
            # save predicted images assumes 2D
            fig_dir = os.path.join(self.pred_dir, 'figures')
            os.makedirs(self.pred_dir, exist_ok=True)
            if self.input_depth > 1:
                im_input = im_input[..., self.input_depth // 2, :, :]
                im_target = im_target[..., 0, :, :]
                im_pred = im_pred[..., 0, :, :]
            plot_utils.save_predicted_images(
                input_batch=im_input,
                target_batch=im_target,
                pred_batch=im_pred,
                output_dir=fig_dir,
                output_fname=im_name[:-4],
                ext='jpg',
                clip_limits=1,
                font_size=15
            )

    def estimate_metrics(self,
                         target,
                         prediction,
                         pred_fnames,
                         mask):
        """
        Estimate evaluation metrics
        The row of metrics gets added to metrics_est.df_metrics

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param list pred_fnames: File names (str) for saving model predictions
        :param np.array mask: foreground/ background mask
        """
        kw_args = {'target': target,
                   'prediction': prediction,
                   'pred_name': pred_fnames[0]}

        if mask is not None:
            kw_args['mask'] = mask

        if 'xy' in self.metrics_orientations:
            # If not using pos idx as test, prediction names will
            # have to be looped through
            if len(pred_fnames) > 1:
                mask_i = None
                for i, pred_name in enumerate(pred_fnames):
                    if mask is not None:
                        mask_i = mask[..., i]
                    self.metrics_inst.estimate_xy_metrics(
                        target=target[..., i],
                        prediction=prediction[..., i],
                        pred_name=pred_name,
                        mask=mask_i,
                    )
                    self.df_xy = self.df_xy.append(
                        self.metrics_inst.get_metrics_xy()
                    )
            else:
                # 3D image or separate positions for each row
                self.metrics_inst.estimate_xy_metrics(**kw_args)
                self.df_xy = self.df_xy.append(
                    self.metrics_inst.get_metrics_xy()
                )
        if 'xyz' in self.metrics_orientations:
            self.metrics_inst.estimate_xyz_metrics(**kw_args)
            self.df_xyz = self.df_xyz.append(
                self.metrics_inst.get_metrics_xyz()
            )
        if 'xz' in self.metrics_orientations:
            self.metrics_inst.estimate_xz_metrics(**kw_args)
            self.df_xz = self.df_xz.append(
                self.metrics_inst.get_metrics_xz()
            )
        if 'yz' in self.metrics_orientations:
            self.metrics_inst.estimate_yz_metrics(**kw_args)
            self.df_yz = self.df_yz.append(
                self.metrics_inst.get_metrics_yz()
            )

    def get_mask(self, cur_row, transpose=False):
        """Get mask, either from image or mask dir

        :param pd.Series/dict cur_row: row containing indices
        :param bool transpose: Changes image format from xyz to zxy
        :return np.array mask: Mask
        """
        mask_idx = aux_utils.get_meta_idx(
            self.mask_meta,
            time_idx=cur_row['time_idx'],
            channel_idx=self.masks_dict['mask_channel'],
            slice_idx=cur_row['slice_idx'],
            pos_idx=cur_row['pos_idx'],
        )
        mask_fname = self.mask_meta.loc[mask_idx, 'file_name']
        mask = image_utils.read_image(
            os.path.join(self.mask_dir, mask_fname),
        )
        # Need metrics mask to be cropped the same way as inference dataset
        mask = image_utils.crop2base(mask)
        if self.crop_shape is not None:
            mask = image_utils.center_crop_to_shape(
                mask,
                self.crop_shape,
                self.image_format,
            )
        # moves z from last axis to first axis
        if transpose and len(mask.shape) > 2:
            mask = np.transpose(mask, [2, 0, 1])
        return mask

    def predict_2d(self, chan_slice_meta):
        """
        Run prediction on 2D or 2.5D on indices given by metadata row.

        :param list meta_row_ids: Inference meta rows
        :return np.array pred_stack: Prediction
        :return np.array target_stack: Target
        :return np.array/list mask_stack: Mask for metrics (empty list if
         not using masked metrics)
        """
        pred_stack = []
        target_stack = []
        mask_stack = []
        # going through z for given p & t
        for z_idx in chan_slice_meta['slice_idx'].unique():
            chan_meta = chan_slice_meta[chan_slice_meta['slice_idx'] == z_idx]
            cur_input, cur_target = \
                self.dataset_inst.__getitem__(chan_meta.index[0])
            if self.crop_shape is not None:
                cur_input = image_utils.center_crop_to_shape(
                    cur_input,
                    self.crop_shape,
                    self.image_format,
                )
                cur_target = image_utils.center_crop_to_shape(
                    cur_target,
                    self.crop_shape,
                    self.image_format,
                )
            if self.tile_option == 'tile_xy':
                print('tiling input...')
                step_size = (np.array(self.tile_params['tile_shape']) -
                             np.array(self.num_overlap))

                # TODO tile_image works for 2D/3D imgs, modify for multichannel
                if self.data_format == 'channels_first':
                    cur_input_1chan = cur_input[0, 0, ...]
                else:
                    cur_input_1chan = cur_input[0, ..., 0]
                _, crop_indices = tile_utils.tile_image(
                    input_image=np.squeeze(cur_target),
                    tile_size=self.tile_params['tile_shape'],
                    step_size=step_size,
                    return_index=True
                )
                print('crop_indices:', crop_indices)
                pred_block_list = self._predict_sub_block_xy(
                    cur_input,
                    crop_indices,
                )
                pred_image = self.stitch_inst.stitch_predictions(
                    cur_target[0].shape,
                    pred_block_list,
                    crop_indices,
                )
                # add batch dimension
                pred_image = pred_image[np.newaxis, ...]
            else:
                pred_image = inference.predict_large_image(
                    model=self.model,
                    input_image=cur_input,
                )
            for i, chan_idx in enumerate(self.target_channels):
                meta_row = chan_meta.loc[chan_meta['channel_idx'] == chan_idx, :].squeeze()
                if self.model_task == 'regression':
                    pred_image[:, i, ...] = self.unzscore(pred_image[:, i, ...],
                                                   cur_target[:, i, ...],
                                                   meta_row)
                # save prediction
                self.save_pred_image(
                    im_input=cur_input,
                    im_target=cur_target[:, i:i+1, ...],
                    im_pred=pred_image[:, i:i+1, ...],
                    meta_row=meta_row,
                    pred_chan_name=self.pred_chan_names[i]
                )

            # get mask
            if self.mask_metrics:
                cur_mask = self.get_mask(meta_row[0])
                mask_stack.append(cur_mask)
            # add to vol
            pred_stack.append(pred_image)
            target_stack.append(cur_target.astype(np.float32))
        pred_stack = np.concatenate(pred_stack, axis=0) #zcyx
        target_stack = np.concatenate(target_stack, axis=0)
        # Stack images and transpose (metrics assumes cyxz format)
        if self.image_format == 'zyx':
            if self.input_depth > 1:
                pred_stack = pred_stack[:, :, 0, :, :]
                target_stack = target_stack[:, :, 0, :, :]
            pred_stack = np.transpose(pred_stack, [1, 2, 3, 0])
            target_stack = np.transpose(target_stack, [1, 2, 3, 0])
        if self.mask_metrics:
            mask_stack = np.concatenate(mask_stack, axis=0)
            if self.image_format == 'zyx':
                mask_stack = np.transpose(mask_stack, [1, 2, 3, 0])
        return pred_stack, target_stack, mask_stack

    def predict_3d(self, iteration_rows):
        """
        Run prediction in 3D on images with 3D shape.

        :param list iteration_rows: Inference meta rows
        :return np.array pred_stack: Prediction
        :return np.array target_stack: Target
        :return np.array/list mask_stack: Mask for metrics
        """
        crop_indices = None
        assert len(iteration_rows) == 1, \
            'more than one matching row found for position ' \
            '{}'.format(iteration_rows.pos_idx)
        cur_input, cur_target = \
            self.dataset_inst.__getitem__(iteration_rows[0])
        # If crop shape is defined in images dict
        if self.crop_shape is not None:
            cur_input = image_utils.center_crop_to_shape(
                cur_input,
                self.crop_shape,
            )
            cur_target = image_utils.center_crop_to_shape(
                cur_target,
                self.crop_shape,
            )
        inf_shape = None
        if self.tile_option == 'infer_on_center':
            inf_shape = self.tile_params['inf_shape']
            center_block = image_utils.center_crop_to_shape(cur_input, inf_shape)
            cur_target = image_utils.center_crop_to_shape(cur_target, inf_shape)
            pred_image = inference.predict_large_image(
                model=self.model,
                input_image=center_block,
            )
        elif self.tile_option == 'tile_z':
            pred_block_list, start_end_idx = \
                self._predict_sub_block_z(cur_input)
            pred_image = self.stitch_inst.stitch_predictions(
                np.squeeze(cur_input).shape,
                pred_block_list,
                start_end_idx
            )
        elif self.tile_option == 'tile_xyz':
            step_size = (np.array(self.tile_params['tile_shape']) -
                         np.array(self.num_overlap))
            if crop_indices is None:
                # TODO tile_image works for 2D/3D imgs, modify for multichannel
                _, crop_indices = tile_utils.tile_image(
                    input_image=np.squeeze(cur_input),
                    tile_size=self.tile_params['tile_shape'],
                    step_size=step_size,
                    return_index=True
                )
            pred_block_list = self._predict_sub_block_xyz(
                cur_input,
                crop_indices,
            )
            pred_image = self.stitch_inst.stitch_predictions(
                np.squeeze(cur_input).shape,
                pred_block_list,
                crop_indices,
            )
        pred_image = np.squeeze(pred_image).astype(np.float32)
        target_image = np.squeeze(cur_target).astype(np.float32)
        if self.model_task == 'regression':
            pred_image = self.unzscore(pred_image,
                                       cur_target,
                                       iteration_rows[0])
        # save prediction
        cur_row = self.inf_frames_meta.iloc[iteration_rows[0]]
        self.save_pred_image(
            im_pred=pred_image,
            time_idx=cur_row['time_idx'],
            target_channel_idx=cur_row['channel_idx'],
            pos_idx=cur_row['pos_idx'],
            slice_idx=cur_row['slice_idx'],
        )
        # 3D uses zyx, estimate metrics expects xyz
        if self.image_format == 'zyx':
            pred_image = np.transpose(pred_image, [1, 2, 0])
            target_image = np.transpose(target_image, [1, 2, 0])
        # get mask
        mask_image = None
        if self.masks_dict is not None:
            mask_image = self.get_mask(cur_row, transpose=True)
            if inf_shape is not None:
                mask_image = image_utils.center_crop_to_shape(
                    mask_image,
                    inf_shape,
                )
            if self.image_format == 'zyx':
                mask_image = np.transpose(mask_image, [1, 2, 0])
        return pred_image, target_image, mask_image

    def run_prediction(self):
        """Run prediction for entire 2D image or a 3D stack"""
        id_df = self.inf_frames_meta[['time_idx', 'pos_idx']].drop_duplicates()
        for id_row in id_df.to_numpy():
            time_idx, pos_idx = id_row
            print('Running inference on time {} position {}'.format(time_idx, pos_idx))
            chan_slice_meta = self.inf_frames_meta[
                (self.inf_frames_meta['time_idx'] == time_idx) &
                (self.inf_frames_meta['pos_idx'] == pos_idx)
                ]
            if self.config['network']['class'] == 'UNet3D':
                pred_image, target_image, mask_image = self.predict_3d(
                    chan_slice_meta,
                )
            else:
                pred_image, target_image, mask_image = self.predict_2d(
                    chan_slice_meta,
                )

            for i, chan_idx in enumerate(self.target_channels):
                pred_fnames = []
                slice_ids = chan_slice_meta.loc[chan_slice_meta['channel_idx'] == chan_idx, 'slice_idx'].to_list()
                for z_idx in slice_ids:
                    pred_fname = aux_utils.get_im_name(
                        time_idx=time_idx,
                        channel_idx=chan_idx,
                        slice_idx=z_idx,
                        pos_idx=pos_idx,
                        ext='',
                    )
                    pred_fnames.append(pred_fname)
                if self.metrics_inst is not None:
                    print('Computing metrics on time {} position {}'.format(time_idx, pos_idx))
                    if not self.mask_metrics:
                        mask_image = None

                    self.estimate_metrics(
                        target=target_image[i],
                        prediction=pred_image[i],
                        pred_fnames=pred_fnames,
                        mask=mask_image,
                    )
            del pred_image, target_image

        # Save metrics csv files
        if self.metrics_inst is not None:
            metrics_mapping = {
                'xy': self.df_xy,
                'xz': self.df_xz,
                'yz': self.df_yz,
                'xyz': self.df_xyz,
            }
            for orientation in self.metrics_orientations:
                metrics_df = metrics_mapping[orientation]
                df_name = 'metrics_{}.csv'.format(orientation)
                metrics_df.to_csv(
                    os.path.join(self.pred_dir, df_name),
                    sep=',',
                    index=False,
                )
