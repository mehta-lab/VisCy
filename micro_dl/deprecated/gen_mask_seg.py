"""Generate masks to be used as target images for segmentation"""
import cv2
import glob
import natsort
import numpy as np
import os
import pandas as pd
import pickle

import micro_dl.utils.masks
import micro_dl.utils.tile_utils as tile_utils
from micro_dl.plotting.plot_utils import save_mask_overlay
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as image_utils


class MaskCreator:
    """Creates masks for segmentation"""

    def __init__(self,
                 input_dir,
                 input_channel_id,
                 output_dir,
                 output_channel_id,
                 timepoint_id=0,
                 correct_flat_field=True,
                 focal_plane_idx=0,
                 plot_masks=False):
        """Init

        :param str input_dir: base input dir at the level of individual sample
         images (or the level above timepoint dirs)
        :param int/list/tuple input_channel_id: channel_ids for which masks
         have to be generated
        :param str output_dir: output dir with full path. It is the base dir
         for tiled images
        :param int/list/tuple output_channel_id: channel_ids to be assigned to
         the created masks. Must match the len(input_channel_id), i.e.
         mask(input_channel_id[0])->output_channel_id[0]
        :param int/list/tuple timepoint_id: timepoints to consider
        :param bool correct_flat_field: indicator to apply flat field
         correction
        :param str meta_fname: fname that contains the
         meta info at the sample image level. If None, read from the default
         dir structure
        :param int focal_plane_idx: focal plane acquisition to use
        :param bool plot_masks: Plot input, masks and overlays
        """

        assert os.path.exists(input_dir), 'input_dir does not exist'
        assert os.path.exists(output_dir), 'output_dir does not exist'
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.correct_flat_field = correct_flat_field

        meta_fname = glob.glob(os.path.join(input_dir, "*info.csv"))
        assert len(meta_fname) == 1,\
            "Can't find info.csv file in {}".format(input_dir)
        study_metadata = pd.read_csv(meta_fname[0])

        self.study_metadata = study_metadata
        self.plot_masks = plot_masks

        avail_tp_channels = aux_utils.validate_metadata_indices(study_metadata,
                                                timepoint_ids=timepoint_id,
                                                channel_ids=input_channel_id)

        msg = 'timepoint_id is not available'
        assert timepoint_id in avail_tp_channels['timepoints'], msg
        if isinstance(timepoint_id, int):
            timepoint_id = [timepoint_id]
        self.timepoint_id = timepoint_id
        # Convert channel to int if there's only one value present
        if isinstance(input_channel_id, (list, tuple)):
            if len(input_channel_id) == 1:
                input_channel_id = input_channel_id[0]
        msg = 'input_channel_id is not available'
        assert input_channel_id in avail_tp_channels['channels'], msg

        msg = 'output_channel_id is already present'
        assert output_channel_id not in avail_tp_channels['channels'], msg
        if isinstance(input_channel_id, (list, tuple)):
            msg = 'input and output channel ids are not of same length'
            assert len(input_channel_id) == len(output_channel_id), msg
        else:
            input_channel_id = [input_channel_id]
            output_channel_id = [output_channel_id]

        self.input_channel_id = input_channel_id
        self.output_channel_id = output_channel_id
        self.focal_plane_idx = focal_plane_idx

    def create_masks_for_stack(self, str_elem_radius=3):
        """Create masks for sample images and save to disk

        :param int str_elem_radius: size of the disk to be used for
         morphological operations
        """

        for tp_idx in self.timepoint_id:
            for ch_idx, ch in enumerate(self.input_channel_id):
                row_idx = aux_utils.get_row_idx(
                    self.study_metadata,
                    tp_idx,
                    ch,
                    self.focal_plane_idx,
                )
                ch_meta = self.study_metadata[row_idx]
                if self.correct_flat_field:
                    #  read flat field image
                    cur_flat_field = np.load(os.path.join(
                        self.input_dir, 'flat_field_images',
                        'flat-field_channel-{}.npy'.format(ch)
                    ))
                #  create mask dir
                mask_dir = os.path.join(
                    self.input_dir,
                    'timepoint_{}'.format(tp_idx),
                    'channel_{}'.format(self.output_channel_id[ch_idx])
                )
                os.makedirs(mask_dir, exist_ok=True)
                # make a dir for storing collages
                collage_dir = os.path.join(self.input_dir,
                                           'timepoint_{}'.format(tp_idx),
                                           'mask_{}'.format(ch)
                                           )
                # Create collage directory if it doesn't exist
                os.makedirs(collage_dir, exist_ok=True)
                # Generate masks for all files in meta csv
                for _, meta_row in ch_meta.iterrows():
                    sample_fname = meta_row['fname']
                    if sample_fname[-3:] == 'npy':
                        cur_image = np.load(meta_row['fname'])
                    else:
                        cur_image = cv2.imread(sample_fname,
                                               cv2.IMREAD_ANYDEPTH)

                    if self.correct_flat_field:
                        cur_image = image_utils.apply_flat_field_correction(
                            cur_image, flat_field_image=cur_flat_field
                        )
                    mask = micro_dl.utils.masks.create_otsu_mask(
                        cur_image, str_elem_size=str_elem_radius
                    )
                    _, fname = os.path.split(sample_fname)
                    mask_fname = os.path.join(mask_dir, fname)
                    np.save(mask_fname, mask,
                            allow_pickle=True, fix_imports=True)
                    #  save a collage to check the quality of masks for the
                    #  current set of parameters
                    if fname[-3:] == 'npy':
                        op_fname = os.path.join(collage_dir, fname).split('.')[0]
                    else:
                        op_fname = os.path.join(collage_dir, fname)
                    if self.plot_masks:
                        save_mask_overlay(cur_image, mask, op_fname)

    def tile_mask_stack(self,
                        input_mask_dir,
                        tile_index_fname=None,
                        tile_size=None,
                        step_size=None,
                        isotropic=False):
        """
        Tiles a stack of masks

        :param str/list input_mask_dir: input_mask_dir with full path
        :param str tile_index_fname: fname with full path for the pickle file
         which contains a dict with fname as keys and crop indices as values.
         Needed when tiling using a volume fraction constraint (i.e. check for
         minimum foreground in tile)
        :param list/tuple tile_size: as named
        :param list/tuple step_size: as named
        :param bool isotropic: indicator for making the tiles have isotropic
         shape (only for 3D)
        """

        if tile_index_fname:
            msg = 'tile index file does not exist'
            assert (os.path.exists(tile_index_fname) and
                    os.path.isfile(tile_index_fname)), msg
            with open(tile_index_fname, 'rb') as f:
                crop_indices_dict = pickle.load(f)
        else:
            msg = 'tile_size and step_size are needed'
            assert tile_size is not None and step_size is not None, msg
            msg = 'tile and step sizes should have same length'
            assert len(tile_size) == len(step_size), msg

        if not isinstance(input_mask_dir, list):
            input_mask_dir = [input_mask_dir]

        for ch_idx, cur_dir in enumerate(input_mask_dir):
            # Split dir name and remove last / if present
            sep_strs = cur_dir.split(os.sep)
            if len(sep_strs[-1]) == 0:
                sep_strs.pop(-1)
            cur_tp = int(sep_strs[-2].split('_')[-1])
            cur_ch = int(sep_strs[-1].split('_')[-1])
            #  read all mask npy files
            mask_fnames = glob.glob(os.path.join(cur_dir, '*.npy'))
            # Sort file names, the assumption is that the csv is sorted
            mask_fnames = natsort.natsorted(mask_fnames)
            cropped_meta = []
            output_dir = os.path.join(
                self.output_dir, 'timepoint_{}'.format(cur_tp),
                'channel_{}'.format(self.output_channel_id[ch_idx])
            )
            os.makedirs(output_dir, exist_ok=True)
            for cur_mask_fname in mask_fnames:
                _, fname = os.path.split(cur_mask_fname)
                sample_num = int(fname.split('_')[1][1:])
                cur_mask = np.load(cur_mask_fname)
                if tile_index_fname:
                    cropped_image_data = tile_utils.crop_at_indices(
                        input_image=cur_mask,
                        crop_indices=crop_indices_dict[fname],
                        isotropic=isotropic
                    )
                else:
                    cropped_image_data = tile_utils.tile_image(
                        input_image=cur_mask,
                        tile_size=tile_size,
                        step_size=step_size,
                        isotropic=isotropic
                    )
                # save the stack
                for id_img_tuple in cropped_image_data:
                    rcsl_idx = id_img_tuple[0]
                    img_fname = 'n{}_{}.npy'.format(sample_num, rcsl_idx)
                    cropped_img = id_img_tuple[1]
                    cropped_img_fname = os.path.join(output_dir, img_fname)
                    np.save(cropped_img_fname,
                            cropped_img,
                            allow_pickle=True,
                            fix_imports=True,
                    )
                    cropped_meta.append(
                            (cur_tp, cur_ch, sample_num, self.focal_plane_idx,
                             cropped_img_fname)
                        )
                    aux_utils.save_tile_meta(
                        cropped_meta,
                        cur_channel=cur_ch,
                        tiled_dir=self.output_dir,
                    )
