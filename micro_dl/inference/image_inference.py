"""Model inference at the image/volume level"""
import cv2
import glob
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
from micro_dl.utils.train_utils import set_keras_session
import micro_dl.utils.tile_utils as tile_utils


class ImagePredictor:
    """Infer on larger images"""

    def __init__(self,
                 train_config,
                 inference_config,
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
            dict inference_3d: dict with params for 3D inference with keys:
             num_slices, inf_shape, tile_shape, num_overlap, overlap_operation.
             int num_slices: in case of 3D, the full volume will not fit in GPU
              memory, specify the number of slices to use and this will depend on
              the network depth, for ex 8 for a network of depth 4. inf_shape -
              inference on a center sub volume.
             list tile_shape: shape of tile for tiling along xyz.
             int/list num_overlap: int for tile_z, list for tile_xyz
             str overlap_operation: e.g. 'mean'
        :param int gpu_id: GPU number to use. -1 for debugging (no GPU)
        :param float/None gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
         TODO: add accuracy and dice coeff to metrics list
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
        assert data_split in ['train', 'val', 'test'], \
            'data_split not in [train, val, test]'
        split_col_ids = self._get_split_ids(data_split)

        self.data_format = self.config['network']['data_format']
        assert self.data_format in {'channels_first', 'channels_last'}, \
            "Data format should be channels_first/last"

        flat_field_dir = None
        images_dict = inference_config['images']
        if 'flat_field_dir' in images_dict:
            flat_field_dir = images_dict['flat_field_dir']

        # Set defaults
        self.image_format = 'zyx'
        if 'image_format' in images_dict:
            self.image_format = images_dict['image_format']
        self.image_ext = '.png'
        if 'image_ext' in images_dict:
            self.image_ext = images_dict['image_ext']

        # Create image subdirectory to write predicted images
        self.pred_dir = os.path.join(self.model_dir, 'predictions')
        os.makedirs(self.pred_dir, exist_ok=True)

        # Handle masks as either targets or for masked metrics
        self.masks_dict = None
        self.mask_metrics = False
        self.mask_dir = None
        self.mask_meta = None
        if 'masks' in inference_config:
            self.masks_dict = inference_config['masks']
        if self.masks_dict is not None:
            assert 'mask_channel' in self.masks_dict , 'mask_channel is needed'
            assert 'mask_dir' in self.masks_dict, 'mask_dir is needed'
            self.mask_dir = self.masks_dict['mask_dir']
            self.mask_meta = aux_utils.read_meta(self.mask_dir)
            assert 'mask_type' in self.masks_dict, \
                'mask_type (target/metrics) is needed'
            if self.masks_dict['mask_type'] == 'metrics':
                # Compute weighted metrics
                self.mask_metrics = True

        # Create dataset instance
        self.dataset_inst = InferenceDataSet(
            image_dir=self.image_dir,
            dataset_config=self.config['dataset'],
            network_config=self.config['network'],
            split_col_ids=split_col_ids,
            image_format=images_dict['image_format'],
            mask_dir=self.mask_dir,
            flat_field_dir=flat_field_dir,
        )
        # create an instance of MetricsEstimator
        self.iteration_meta = self.dataset_inst.get_iteration_meta()

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

        # Handle 3D volume inference settings
        self.num_overlap = 0
        self.stitch_inst = None
        self.tile_option = None
        self.z_dim = 2
        self.crop_shape = None
        if 'crop_shape' in images_dict:
            self.crop_shape = images_dict['crop_shape']
        if 'inference_3d' in inference_config:
            self.params_3d = inference_config['inference_3d']
            self._assign_3d_inference()
            # Make image ext npy default for 3D
            self.image_ext = '.npy'

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
        try:
            split_fname = os.path.join(self.model_dir, 'split_samples.json')
            split_samples = aux_utils.read_json(split_fname)
            inference_ids = split_samples[data_split]
        except FileNotFoundError as e:
            print("No split_samples file. "
                  "Will predict all images in dir.")
            frames_meta = aux_utils.read_meta(self.image_dir)
            inference_ids = np.unique(frames_meta[split_col]).tolist()
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

        if 'num_slices' in self.params_3d and self.params_3d['num_slices'] > 1:
            self.tile_option = 'tile_z'
            train_num_slices = self.config['network']['depth']
            assert self.params_3d['num_slices'] >= train_num_slices, \
                'inference num of slies < num of slices used for training. ' \
                'Inference on reduced num of slices gives sub optimal results'
            num_slices = self.params_3d['num_slices']

            assert self.config['network']['class'] == 'UNet3D', \
                'currently stitching predictions available for 3D models only'
            network_depth = len(
                self.config['network']['num_filters_per_block']
            )
            min_num_slices = 2 ** (network_depth - 1)
            assert num_slices >= min_num_slices, \
                'Insufficient number of slices {} for the network ' \
                'depth {}'.format(num_slices, network_depth)
            self.num_overlap = self.params_3d['num_overlap'] \
                if 'num_overlap' in self.params_3d else 0
        elif 'tile_shape' in self.params_3d:
            self.tile_option = 'tile_xyz'
            self.num_overlap = self.params_3d['num_overlap'] \
                if 'num_overlap' in self.params_3d else [0, 0, 0]
        elif 'inf_shape' in self.params_3d:
            self.tile_option = 'infer_on_center'
            self.num_overlap = 0

        # create an instance of ImageStitcher
        if self.tile_option in ['tile_z', 'tile_xyz']:
            overlap_dict = {
                'overlap_shape': self.num_overlap,
                'overlap_operation': self.params_3d['overlap_operation']
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
        num_slices = self.params_3d['num_slices']
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
                input_image=cur_block
            )
            # reduce predictions from 5D to 3D for simplicity
            pred_ims.append(np.squeeze(pred_block))
            start_end_idx.append((start_idx, end_idx))
        return pred_ims, start_end_idx

    def _predict_sub_block_xyz(self,
                               input_image,
                               crop_indices):
        """Predict sub blocks along xyz

        :param np.array input_image: 5D tensor with the entire 3D volume
        :param list crop_indices: list of crop indices
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

    def save_pred_image(self,
                        predicted_image,
                        time_idx,
                        target_channel_idx,
                        pos_idx,
                        slice_idx):
        """
        Save predicted images with image extension given in init.

        :param np.array predicted_image: 2D / 3D predicted image
        :param int time_idx: time index
        :param int target_channel_idx: target / predicted channel index
        :param int pos_idx: FOV / position index
        :param int slice_idx: slice index
        """
        # Write prediction image
        im_name = aux_utils.get_im_name(
            time_idx=time_idx,
            channel_idx=target_channel_idx,
            slice_idx=slice_idx,
            pos_idx=pos_idx,
            ext=self.image_ext,
        )
        file_name = os.path.join(self.pred_dir, im_name)
        if self.image_ext == '.png':
            # Convert to uint16 for now
            im_pred = predicted_image.astype(np.float)
            if im_pred.max() > im_pred.min():
                im_pred = np.iinfo(np.uint16).max * \
                          (im_pred - im_pred.min()) / \
                          (im_pred.max() - im_pred.min())
            else:
                im_pred = im_pred / im_pred.max() * np.iinfo(np.uint16).max
            im_pred = im_pred.astype(np.uint16)
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif self.image_ext == '.tif':
            # Convert to float32 and remove batch dimension
            im_pred = predicted_image.astype(np.float32)
            cv2.imwrite(file_name, np.squeeze(im_pred))
        elif self.image_ext == '.npy':
            np.save(file_name, predicted_image, allow_pickle=True)
        else:
            raise ValueError(
                'Unsupported file extension: {}'.format(self.image_ext),
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
        # moves z from last axis to first axis
        if transpose and len(mask.shape) > 2:
            mask = np.transpose(mask, [2, 0, 1])
        if self.crop_shape is not None:
            mask = image_utils.center_crop_to_shape(mask, self.crop_shape)
        return mask

    def predict_2d(self, iteration_rows):
        """
        Run prediction on 2D or 2.5D on indices given by metadata row.

        :param list iteration_rows: Inference meta rows
        :return np.array pred_stack: Prediction
        :return np.array target_stack: Target
        :return np.array/list mask_stack: Mask for metrics (empty list if
         not using masked metrics)
        """
        pred_stack = []
        target_stack = []
        mask_stack = []
        for row_idx in iteration_rows:
            cur_input, cur_target = \
                self.dataset_inst.__getitem__(row_idx)
            if self.crop_shape is not None:
                cur_input = image_utils.center_crop_to_shape(
                    cur_input,
                    self.crop_shape,
                )
                cur_target = image_utils.center_crop_to_shape(
                    cur_target,
                    self.crop_shape,
                )
            pred_image = inference.predict_large_image(
                model=self.model,
                input_image=cur_input,
            )
            # Squeeze prediction for writing
            pred_image = np.squeeze(pred_image)
            # save prediction
            cur_row = self.iteration_meta.iloc[row_idx]
            self.save_pred_image(
                predicted_image=pred_image,
                time_idx=cur_row['time_idx'],
                target_channel_idx=cur_row['channel_idx'],
                pos_idx=cur_row['pos_idx'],
                slice_idx=cur_row['slice_idx']
            )
            # get mask
            if self.mask_metrics:
                cur_mask = self.get_mask(cur_row)
                mask_stack.append(cur_mask)
            # add to vol
            pred_stack.append(pred_image)
            target_stack.append(np.squeeze(cur_target))
        # Stack images and transpose (metrics assumes xyz format)
        pred_stack = np.transpose(np.stack(pred_stack), [1, 2, 0])
        target_stack = np.transpose(np.stack(target_stack), [1, 2, 0])
        if self.mask_metrics:
            mask_stack = np.transpose(np.stack(mask_stack), [1, 2, 0])
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
        if self.crop_shape is not None:
            cur_input = image_utils.center_crop_to_shape(
                cur_input,
                self.crop_shape,
            )
            cur_target = image_utils.center_crop_to_shape(
                cur_target,
                self.crop_shape,
            )
        if self.tile_option == 'infer_on_center':
            inf_shape = self.params_3d['inf_shape']
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
            step_size = (np.array(self.params_3d['tile_shape']) -
                         np.array(self.num_overlap))
            if crop_indices is None:
                # TODO tile_image works for 2D/3D imgs, modify for multichannel
                _, crop_indices = tile_utils.tile_image(
                    input_image=np.squeeze(cur_input),
                    tile_size=self.params_3d['tile_shape'],
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
        pred_image = np.squeeze(pred_image)
        target_image = np.squeeze(cur_target)
        # save prediction
        cur_row = self.iteration_meta.iloc[iteration_rows[0]]
        self.save_pred_image(
            predicted_image=pred_image,
            time_idx=cur_row['time_idx'],
            target_channel_idx=cur_row['channel_idx'],
            pos_idx=cur_row['pos_idx'],
            slice_idx=cur_row['slice_idx'],
        )
        # get mask
        mask_image = None
        if self.masks_dict is not None:
            mask_image = self.get_mask(cur_row, transpose=True)
            mask_image = np.transpose(mask_image, [1, 2, 0])
        # 3D uses zyx, estimate metrics expects xyz
        pred_image = np.transpose(pred_image, [1, 2, 0])
        target_image = np.transpose(target_image, [1, 2, 0])

        return pred_image, target_image, mask_image

    def run_prediction(self):
        """Run prediction for entire 2D image or a 3D stack"""

        pos_ids = self.iteration_meta['pos_idx'].unique()
        for idx, pos_idx in enumerate(pos_ids):
            print('Inference idx {}/{}'.format(idx, len(pos_ids)))
            iteration_rows = self.iteration_meta.index[
                self.iteration_meta['pos_idx'] == pos_idx,
            ].values
            if self.tile_option is None:
                # 2D, 2.5D
                pred_image, target_image, mask_image = self.predict_2d(
                    iteration_rows,
                )
            else:  # 3D
                pred_image, target_image, mask_image = self.predict_3d(
                    iteration_rows,
                )
            pred_fnames = []
            for row_idx in iteration_rows:
                cur_row = self.iteration_meta.iloc[row_idx]
                pred_fname = aux_utils.get_im_name(
                    time_idx=cur_row['time_idx'],
                    channel_idx=cur_row['channel_idx'],
                    slice_idx=cur_row['slice_idx'],
                    pos_idx=cur_row['pos_idx'],
                    ext='',
                )
                pred_fnames.append(pred_fname)
            if self.metrics_inst is not None:
                if not self.mask_metrics:
                    mask_image = None
                self.estimate_metrics(
                    target=target_image,
                    prediction=pred_image,
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
