"""Model inference at the image/volume level"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from micro_dl.input.inference_dataset import InferenceDataSet
from micro_dl.inference.evaluation_metrics import MetricsEstimator
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils
import micro_dl.utils.tile_utils as tile_utils
import micro_dl.utils.normalize as normalize
import micro_dl.plotting.plot_utils as plot_utils


class ImagePredictor:
    """
    Inference on larger images.
    Methods in this class provide functionality for performing inference on 
    large (x,y > 256,256) images through direct inference in 2 and 3 dimensions,
    or by tiling in 2 or 3 dimensions along xy and z axes. 
    
    Inference is performed by first initalizing an InferenceDataset object which 
    loads samples and performs normalization according to the tile-specific 
    normalization values acquired during preprocessing. 
    
    Actual inference is performed by a trained '.pt' pytorch model loaded into a
    TorchPredictor object provided through torch_predictor.
    
    After inference is performed on samples acquired from InferenceDataset object,
    dynamic range is return by denormalizing the outputs if the model task is reg-
    ression.
    
    Metrics are calculated by use of a MetricEstimator object, which calculates, 
    and indexes metrics for successive images.
    
    """

    def __init__(self,
                 train_config,
                 inference_config,
                 torch_predictor,
                 preprocess_config=None):
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
        :param TorchPredictor torch_predictor: predictor object which handles
            transporting data to a PyTorch model for inference.
        """
        # create model and load weights
        self.torch_predictor = torch_predictor
        
        # Use model_dir from torch config.
        if 'model_dir' in self.torch_predictor.network_config:
            model_dir = self.torch_predictor.network_config['model_dir'] 
        else:
            raise AssertionError('model_dir must be specified in torch config for inference')
        
        self.config = train_config
        self.model_dir = model_dir
        self.image_dir = inference_config['image_dir']

        
        if 'save_folder_name' in inference_config:
            if inference_config['save_folder_name'][0] == '/':
                self.pred_dir = inference_config['save_folder_name']
            else:
                self.pred_dir = os.path.join(Path(self.model_dir).parent,
                                             inference_config['save_folder_name'])
        else:
            print(f'No save_folder_name specified... saving predictions to '\
                'image_dir: \n\t{self.image_dir}')
            self.pred_dir = os.path.join(self.image_dir, os.path.basename(model_dir))
        os.makedirs(self.pred_dir, exist_ok=True)

        # Set default for data split, determine column name and indices
        #TODO remove these parameters from config. drop support for variability
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
        #TODO remove some of these parameters from config
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
            assert 'mask_channel' in self.masks_dict, 'mask_channel is needed'
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
        #TODO standardize this parameter into the master config file.
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
        # TODO remove! dropping support for 3d inference
        self.num_overlap = 0
        self.stitch_inst = None
        self.tile_option = None
        self.z_dim = 2
        self.crop_shape = None
        if 'crop_shape' in images_dict:
            self.crop_shape = images_dict['crop_shape']
        crop2base = True
        self.tile_params = None
        if 'tile' in inference_config:
            self.tile_params = inference_config['tile']
            self._assign_3d_inference()
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
        #TODO in 2.1.0 this metadata will be stored in .zarr keys
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
            assert 'metrics' in self.metrics_dict, 'Must specify with metrics to use'
            self.metrics_inst = MetricsEstimator(
                metrics_list=self.metrics_dict['metrics'],
                masked_metrics=self.mask_metrics,
            )
            self.metrics_orientations = ['xy']
            available_orientations = ['xy', 'xyz', 'xz', 'yz']
            if 'metrics_orientations' in self.metrics_dict:
                self.metrics_orientations = self.metrics_dict['metrics_orientations']
                assert set(self.metrics_orientations).issubset(available_orientations), \
                    'orientation not in [xy, xyz, xz, yz]'
        self.df_xy = pd.DataFrame()
        self.df_xyz = pd.DataFrame()
        self.df_xz = pd.DataFrame()
        self.df_yz = pd.DataFrame()
        

    def _get_split_ids(self, data_split='test'):
        """
        Get the indices for data_split. Used to determine which images to run
        inference on to avoid validating on training data.
        
        Indices returned refer to full-image sample FoV slices, rather than
        spatial indices of image tiles.

        :param str data_split: in [train, val, test]
        :return list inference_ids: Indices for inference given data split
        :return str split_col: Dataframe column name, which was split in training
        """
        #TODO handle config parameter in master config
        split_col = self.config['dataset']['split_by_column']
        frames_meta = aux_utils.read_meta(self.image_dir)
        inference_ids = np.unique(frames_meta[split_col]).tolist()
        if data_split == 'all':
            return split_col, inference_ids

        try:
            split_fname = os.path.join(Path(self.torch_predictor.network_config['model_dir']).parent,
                                       'split_samples.json')
            split_samples = aux_utils.read_json(split_fname)
            inference_ids = split_samples[data_split]
        except FileNotFoundError as e:
            print("No split_samples file. "
                  "Will predict all images in dir.")

        return split_col, inference_ids

    def _assign_3d_inference(self):
        raise DeprecationWarning('3d inference no longer supported in 2.0.0')

    def _get_sub_block_z(self,
                         input_image,
                         start_z_idx,
                         end_z_idx):
        raise DeprecationWarning('3d inference no longer supported in 2.0.0')

    def _predict_sub_block_z(self, input_image):
        raise DeprecationWarning('3d inference no longer supported in 2.0.0')

    def _predict_sub_block_xy(self,
                              input_image,
                              crop_indices):
        
        """
        Predict sub blocks along xy, specifically when generating a 2d
        prediction by spatial tiling and stitching.

        :param np.array input_image: 5D tensor with the entire 3D image stack
        :param list crop_indices: list of crop indices: min/max xyz
        :return list pred_ims - list of predicted sub blocks
        """
        pred_ims = []
        assert self.image_format == 'zyx', \
            'predict_sub_block_xy only supports zyx format'
        for idx, crop_idx in enumerate(crop_indices):
            print('Running inference on tile {}/{}'.format(idx, len(crop_indices)))
            if self.data_format == 'channels_first':
                if len(input_image.shape) == 5:  # bczyx
                    cur_block = input_image[:, :, :, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3]]
                else:  # bcyx
                    cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3]]
            else:
                if len(input_image.shape) == 5:  # bzyxc
                    cur_block = input_image[:, :, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3],
                                :]
                else:  # byxc
                    cur_block = input_image[:, crop_idx[0]: crop_idx[1],
                                crop_idx[2]: crop_idx[3],
                                :]
            pred_block = self.torch_predictor.predict_large_image(
                input_image=cur_block
            )
            # remove b & z dimention, prediction of 2D and 2.5D has only single z
            if self.data_format == 'channels_first':
                if len(pred_block.shape) == 5:  # bczyx
                    pred_block = pred_block[0, :, 0, ...]
                else:  # bcyx
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
        raise DeprecationWarning('3d inference no longer supported in 2.0.0')

    def unzscore(self,
                 im_pred,
                 im_target,
                 meta_row):
        """
        Revert z-score normalization applied during preprocessing. Necessary
        before computing SSIM.
        
        Used for models tasked with regression to reintroduce dynamic range
        into the model predictions.

        :param im_pred: Prediction image, normalized image for un-zscore
        :param im_target: Target image to compute stats from
        :param pd.DataFrame meta_row: Metadata row for image
        :return im_pred: image at its original scale
        """
        if self.normalize_im is not None:
            if self.normalize_im in ['dataset', 'volume', 'slice'] \
                and ('zscore_median' in meta_row and
                     'zscore_iqr' in meta_row):
                zscore_median = meta_row['zscore_median']
                zscore_iqr = meta_row['zscore_iqr']
            else:
                zscore_median = np.nanmean(im_target)
                zscore_iqr = np.nanstd(im_target)
            im_pred = normalize.unzscore(im_pred, zscore_median, zscore_iqr)
        return im_pred

    def save_pred_image(self,
                        im_input,
                        im_target,
                        im_pred,
                        metric,
                        meta_row,
                        pred_chan_name=np.nan,
                        ):
        """
        Save predicted images with image extension given in init. 
        
        Note: images and predictions stored as float values are 
        compressed into uint16 before figure generation. Some loss
        of information may occur during compression for extremely
        low-value data.

        :param np.array im_input: Input image
        :param np.array im_target: Target image
        :param np.array im_pred: 2D / 3D predicted image
        :param pd.series metric: xy similarity metrics between prediction and target
        :param pd.DataFrame meta_row: Row of meta dataframe containing sample
        :param str/NaN pred_chan_name: Predicted channel name
        """
        if pd.isnull(pred_chan_name):
            if 'channel_name' in meta_row:
                pred_chan_name = meta_row['channel_name']

        if pd.isnull(pred_chan_name):
            im_name = aux_utils.get_im_name(
                time_idx=meta_row['time_idx'],
                channel_idx=meta_row['channel_idx'],
                slice_idx=meta_row['slice_idx'],
                pos_idx=meta_row['pos_idx'],
                ext=self.image_ext,
                extra_field=self.suffix,
            )
        else:
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
            # Assuming segmentation output is probability maps
            im_pred = im_pred.astype(np.float32)
        # Check file format
        if self.image_ext in ['.png', '.tif']:
            if self.image_ext == '.png':
                assert im_pred.dtype == np.uint16, \
                    'PNG format does not support float type. ' \
                    'Change file extension as ".tif" or ".npy" instead'
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif self.image_ext == '.npy':
            # TODO: add support for saving prediction of 3D slices
            np.save(file_name, np.squeeze(im_pred), allow_pickle=True)
        else:
            raise ValueError(
                'Unsupported file extension: {}'.format(self.image_ext),
            )
        if self.save_figs and len(im_target.shape) == 2:
            # save predicted images assumes 2D
            fig_dir = os.path.join(self.pred_dir, 'figures')
            os.makedirs(self.pred_dir, exist_ok=True)
            # for every target image channel a new overlay image is saved
            plot_utils.save_predicted_images(
                input_imgs=im_input,
                target_img=im_target,
                pred_img=im_pred,
                metric=metric,
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
        """
        Get mask, read either from image or mask dir

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
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        # moves z from last axis to first axis
        if transpose and len(mask.shape) > 2:
            mask = np.transpose(mask, [2, 0, 1])
        return mask

    def predict_2d(self, chan_slice_meta):
        """
        Run prediction on 2D or 2.5D on indices given by metadata row.
        
        Reads in images from the inference dataset object, which performs normalization on
        the images based on values calculated in preprocessing and stored in intensity_meta.csv.
        
        Prediction is done over an entire image or over each tile and stitched together,
        as specified by data/model structure.
        
        For regression models, post-processes images by reversing the z-score normalization
        to reintroduce dynamic range removed in normalization.

        :param pd.DataFrame chan_slice_meta: Inference meta rows
        :return np.array pred_stack: Prediction
        :return np.array target_stack: Target
        :return np.array/list mask_stack: Mask for metrics (empty list if
         not using masked metrics)
        """
        input_stack = []
        pred_stack = []
        target_stack = []
        mask_stack = []
        # going through z slices for given position and time
        with tqdm(total=len(chan_slice_meta['slice_idx'].unique()),
                  desc='z-stack prediction', leave=False) as pbar:
            for z_idx in chan_slice_meta['slice_idx'].unique():
                # get input image and target
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
                
                # pass to network for prediction
                if self.tile_option == 'tile_xy':
                    step_size = (np.array(self.tile_params['tile_shape']) -
                                np.array(self.num_overlap))

                    _, crop_indices = tile_utils.tile_image(
                        input_image=np.squeeze(cur_target),
                        tile_size=self.tile_params['tile_shape'],
                        step_size=step_size,
                        return_index=True
                    )
                    pred_block_list = self._predict_sub_block_xy(
                        cur_input,
                        crop_indices,
                    )
                    pred_image = self.stitch_inst.stitch_predictions(
                        cur_target[0].shape,
                        pred_block_list,
                        crop_indices,
                    )
                else:
                    pred_image = self.torch_predictor.predict_large_image(
                        input_image=cur_input
                        )
                
                # add batch dimension
                if len(pred_image.shape) < 4:
                    pred_image = pred_image[np.newaxis, ...]
                
                # if regression task, undo normalization to recover dynamic range
                for i, chan_idx in enumerate(self.target_channels):
                    meta_row = chan_meta.loc[chan_meta['channel_idx'] == chan_idx, :].squeeze()
                    if self.model_task == 'regression':
                        if self.input_depth > 1:
                            pred_image[:, i, 0, ...] = self.unzscore(
                                pred_image[:, i, 0, ...],
                                cur_target[:, i, 0, ...],
                                meta_row,
                            )
                        else:
                            pred_image[:, i, ...] = self.unzscore(
                                pred_image[:, i, ...],
                                cur_target[:, i, ...],
                                meta_row,
                            )

                # get mask
                if self.mask_metrics:
                    cur_mask = self.get_mask(chan_meta)
                    # add batch dimension
                    cur_mask = cur_mask[np.newaxis, ...]
                    mask_stack.append(cur_mask)
                # add to vol
                input_stack.append(cur_input)
                pred_stack.append(pred_image)
                target_stack.append(cur_target.astype(np.float32))
                pbar.update(1)

        input_stack = np.concatenate(input_stack, axis=0)
        pred_stack = np.concatenate(pred_stack, axis=0)  #zcyx
        target_stack = np.concatenate(target_stack, axis=0)
        # Stack images and transpose (metrics assumes cyxz format)
        if self.image_format == 'zyx':
            input_stack = input_stack[:, :, self.input_depth // 2, :, :]
            pred_stack = pred_stack[:, :, 0, :, :]
            target_stack = target_stack[:, :, 0, :, :]
            
            input_stack = np.transpose(input_stack,  [1, 2, 3, 0])
            pred_stack = np.transpose(pred_stack, [1, 2, 3, 0])
            target_stack = np.transpose(target_stack, [1, 2, 3, 0])
        if self.mask_metrics:
            mask_stack = np.concatenate(mask_stack, axis=0)
            if self.image_format == 'zyx':
                mask_stack = np.transpose(mask_stack, [1, 2, 3, 0])

        return pred_stack, target_stack, mask_stack, input_stack
    
    def predict_3d(self, iteration_rows):
        raise DeprecationWarning('3d inference no longer supported in 2.0.0') 
            
    def run_prediction(self):
        """
        Run prediction for entire set of 2D images or a 3D image stacks
        
        Prediction procedure depends on format of input and desired prediction (2D or 3D).
        Generates metrics, if specified in parent object initiation, for each prediction
        channel, channel-wise.
        
        Saves image predictions by individual channel and prediction metrics in separate
        files. z*** position specified in saved prediction name represents prediction channel.
        
        """
        id_df = self.inf_frames_meta[['time_idx', 'pos_idx']].drop_duplicates()
        pbar = tqdm(id_df.to_numpy())
        #for each image, one per row of df
        for id_row_idx, id_row in enumerate(id_df.to_numpy()):
            #run prediction on image
            time_idx, pos_idx = id_row
            pbar.set_description('time {} position {}'.format(time_idx, pos_idx))
            chan_slice_meta = self.inf_frames_meta[
                (self.inf_frames_meta['time_idx'] == time_idx) &
                (self.inf_frames_meta['pos_idx'] == pos_idx)
                ]
            
            if self.config['network']['class'] == 'UNet3D':
                pred_image, target_image, mask_image, input_image = self.predict_3d(
                    chan_slice_meta,
                )
            else:
                pred_image, target_image, mask_image, input_image = self.predict_2d(
                    chan_slice_meta,
                )
            
            #separate predictions by channel (type)
            for c, chan_idx in enumerate(self.target_channels):
                pred_names = []
                slice_ids = chan_slice_meta.loc[chan_slice_meta['channel_idx'] == chan_idx,
                                                'slice_idx'].to_list()
                for z_idx in slice_ids:
                    pred_name = aux_utils.get_im_name(
                        time_idx=time_idx,
                        channel_idx=chan_idx,
                        slice_idx=z_idx,
                        pos_idx=pos_idx,
                        ext='',
                    )
                    pred_names.append(pred_name)
                
                #generate metrics channel-wise
                if self.metrics_inst is not None:
                    if not self.mask_metrics:
                        mask = None
                    else:
                        mask = mask_image[c, ...]
                    self.estimate_metrics(
                        target=target_image[c, ...],
                        prediction=pred_image[c, ...],
                        pred_fnames=pred_names,
                        mask=mask,
                    )
                with tqdm(total=len(chan_slice_meta['slice_idx'].unique()), desc='z-stack saving',
                          leave=False) as pbar_s:
                    for z, z_idx in enumerate(chan_slice_meta['slice_idx'].unique()):
                        meta_row = chan_slice_meta[
                            (chan_slice_meta['channel_idx'] == chan_idx) &
                            (chan_slice_meta['slice_idx'] == z_idx)].squeeze()
                        metrics = None
                        if self.metrics_inst is not None:
                            metrics_mapping = {
                                'xy': self.df_xy,
                                'xz': self.df_xz,
                                'yz': self.df_yz,
                                'xyz': self.df_xyz,
                            }
                            # Only one orientation can be added to the plot
                            metrics_df = metrics_mapping[self.metrics_orientations[0]]
                            metrics = metrics_df.loc[
                                metrics_df['pred_name'].str.contains(pred_names[z], case=False),
                            ]
                        if self.config['network']['class'] == 'UNet3D':
                            assert self.image_ext == '.npy', \
                                "Must save as numpy to get all 3D data"
                            input = input_image
                            target = target_image[c, ...]
                            pred = pred_image[c, ...]
                        else:
                            input = input_image[..., z]
                            target = target_image[c, ..., z]
                            pred = pred_image[c, ..., z]
                        self.save_pred_image(
                            im_input=input,
                            im_target=target,
                            im_pred=pred,
                            metric=metrics,
                            meta_row=meta_row,
                            pred_chan_name=self.pred_chan_names[c]
                        )
                        pbar_s.update(1)
            del pred_image, target_image
            pbar.update(1)

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
        
        #output save dirs
        print(f'Predictions and figures saved at: {self.pred_dir}')
