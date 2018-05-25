"""Classes for handling microscopy data in lif format

Uses dir structure:
base_output_dir
 |-image_volume, image_volumes_info.csv
    |-tp0
        |-channel0
 |-img_512_512_8_.., cropped_images_info.csv
    |-tp-0
        |-channel0: contains all npy files for cropped images from channel0
        |-channel1: contains all npy files for cropped images from channel1..
        and so on
"""
from abc import ABCMeta, abstractmethod
import bioformats as bf
import javabridge as jv
import logging
import numpy as np
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening

import micro_dl.utils.image_utils as image_utils


class BasePreProcessor(metaclass=ABCMeta):
    """Base class for volume splitting and cropping"""

    def __init__(self, base_output_dir, verbose=0):
        """Init

        :param str base_output_dir: base folder for storing the individual
         image and cropped volumes
        :param int verbose: specifies the logging level: NOTSET:0, DEBUG:10,
         INFO:20, WARNING:30, ERROR:40, CRITICAL:50
        """

        self.base_output_dir = base_output_dir
        self.volume_dir = os.path.join(self.base_output_dir, 'image_volumes')
        log_levels = [0, 10, 20, 30, 40, 50]
        if verbose in log_levels:
            self.verbose = verbose
        else:
            self.verbose = 10
        self.logger = self._init_logger()

    def _init_logger(self):
        """Initialize logger for pre-processing

        Logger outputs to console and log_file
        """

        logger = logging.getLogger('preprocessing')
        logger.setLevel(self.verbose)
        logger.propagate = False

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.verbose)
        logger.addHandler(stream_handler)
        
        logger_fname = os.path.join(self.base_output_dir, 'preprocessing.log')
        file_handler = logging.FileHandler(logger_fname)
        file_handler.setLevel(self.verbose)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    @abstractmethod
    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Save each image as a numpy array"""

        raise NotImplementedError

    def _log_info(self, msg):
        """Log info"""

        if self.verbose > 0:
            self.logger.info(msg)

    def save_images(self, img_fname, mask_channels=None, focal_plane_idx=None):
        """Saves the individual images as a npy file

        2D might have more acquisitions +/- focal plane, (usually 3 images).
        focal_plane_idx corresponds to the plane to consider. Mid-plane is the
        one in focus and the +/- on either side would be blurred. For 2D
        acquisitions, this is stored along the Z dimension. How is this handled
        for 3D acquisitions?

        :param str img_fname: fname with full path of the Lif file
        :param int/list mask_channels: channels from which masks have to be
         generated
        :param int focal_plane_idx: focal plane to consider
        """

        if not os.path.exists(img_fname):
            raise FileNotFoundError(
                "LIF file doesn't exist at:", img_fname
            )
        os.makedirs(self.volume_dir, exist_ok=True)

        jv.start_vm(class_path=bf.JARS, max_heap_size='8G')
        metadata = bf.get_omexml_metadata(img_fname)
        omexml_object = bf.OMEXML(metadata)
        num_channels = omexml_object.image().Pixels.channel_count
        num_samples = omexml_object.get_image_count()
        num_timepoints = omexml_object.image().Pixels.SizeT
        num_pix_z = omexml_object.image().Pixels.SizeZ
        size_x_um = omexml_object.image().Pixels.PhysicalSizeX
        size_y_um = omexml_object.image().Pixels.PhysicalSizeY
        size_z_um = omexml_object.image().Pixels.PhysicalSizeZ

        reader = bf.ImageReader(img_fname, perform_init=True)

        records = []
        for timepoint_idx in range(num_timepoints):
            timepoint_dir = os.path.join(self.volume_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)

            for channel_idx in range(num_channels):
                channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                for sample_idx in range(15, 412): # num_samples
                    cur_records = self.save_each_image(
                        reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um
                    )
                    records.extend(cur_records)
                msg = 'Wrote files for tp:{}, channel:{}'.format(
                    timepoint_idx, channel_idx
                )
                self._log_info(msg)
        df = pd.DataFrame.from_records(
            records,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns']
            )
        metadata_fname = os.path.join(self.volume_dir,
                                      'image_volumes_info.csv')
        df.to_csv(metadata_fname, sep=',')
        jv.kill_vm()

        num_timepoints = 1
        if mask_channels is not None:
            timepoints = list(range(num_timepoints))
            self.gen_mask(timepoints, mask_channels, focal_plane_idx)

    def flat_field_corr(self, focal_plane_idx=None):
        """Estimates flat field correction image"""

        meta_fname = os.path.join(self.volume_dir, 'image_volumes_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            self.logger.error('cannot read individual image info:' + str(e))
            raise

        all_channels = volume_metadata['channel_num'].unique()
        flat_field_dir = os.path.join(self.base_output_dir,
                                      'flat_field_images')
        os.makedirs(flat_field_dir, exist_ok=True)
        for tp_idx in volume_metadata['timepoint'].unique():
            for channel_idx in all_channels:
                row_idx = self.get_row_idx(volume_metadata, tp_idx,
                                           channel_idx, focal_plane_idx)
                channel_metadata = volume_metadata[row_idx]
                for idx, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    n_dim = len(cur_image.shape)
                    if n_dim == 3:
                        cur_image = np.mean(cur_image, axis=2)
                    if idx == 0:
                        summed_image = cur_image.astype('float64')
                    else:
                        summed_image += cur_image
                mean_image = summed_image / len(row_idx)
                # Compute flatfield from from mean image of stack
                # TODO (Jenny): it currently samples median values
                # from a mean images, not very statistically meaningful
                # but easier than computing median of image stack
                flatfield = image_utils.get_flatfield(mean_image)
                corrected_image = mean_image / flatfield
                fname = 'flat-field_tp-{}_channel-{}.npy'.format(tp_idx,
                                                                 channel_idx)
                cur_fname = os.path.join(flat_field_dir, fname)
                np.save(cur_fname, corrected_image,
                        allow_pickle=True, fix_imports=True)

    def save_flat_field_corrected_images(self):
        """Saves flat field corrected images

        For the current set, no structure in flat field or averaged images,
        ignore for now
        """
        raise NotImplementedError

    def gen_mask(self, timepoint_ids, channel_ids, focal_plane_idx=None):
        """Generates a binary mask based on summation of channels"""
        
        meta_fname = os.path.join(self.volume_dir, 'image_volumes_info.csv')
        try:
            volume_metadata = pd.read_csv(meta_fname)
        except IOError as e:
            self.logger.error('cannot read individual image info:', str(e))
            raise

        if isinstance(channel_ids, int):
            channel_ids = [channel_ids]
        ch_str = '-'.join(map(str, channel_ids))

        for tp_idx in timepoint_ids:
            row_idx = self.get_row_idx(volume_metadata,tp_idx, channel_ids[0],
                                       focal_plane_idx)
            metadata = volume_metadata[row_idx]
            tp_dir = os.path.join(self.volume_dir,
                                  'timepoint_{}'.format(tp_idx))
            mask_dir = os.path.join(tp_dir, 'mask_{}'.format(ch_str))
            os.makedirs(mask_dir, exist_ok=True)

            fnames = [os.path.split(row['fname'])[1]
                      for _, row in metadata.iterrows()]
            for fname in fnames:
                for idx, channel in enumerate(channel_ids):
                    cur_fname = os.path.join(
                        tp_dir, 'channel_{}'.format(channel), fname
                    )
                    if idx == 0:
                        summed_image = np.load(cur_fname)
                    else:
                         summed_image += np.load(cur_fname)
                thr = threshold_otsu(summed_image, nbins=512)
                str_elem = disk(5)
                thr_image = binary_opening(summed_image>=thr, str_elem)
                mask = binary_fill_holes(thr_image)
                np.save(os.path.join(mask_dir, fname), mask,
                        allow_pickle=True, fix_imports=True)

    @abstractmethod
    def get_row_idx(self, volume_metadata, timepoint_idx,
                    channel_idx, focal_plane_idx=None):
        """Get the indices for images with timepoint_idx and channel_idx"""

        raise NotImplementedError

    def crop_images(self, tile_size, step_size, isotropic=False,
                    channel_ids=-1, focal_plane_idx=None,
                    mask_channel_ids=None, min_fraction=None):
        """Crop image volumes in the specified channels

        Isotropic here refers to the same dimension/shape along x,y,z and not
        really isotropic resolution in mm.

        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In
         case of no overlap, the step size is tile_size. If overlap,
         step_size < tile_size
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param list channel_ids: crop volumes in the given channels.
         default=-1, crop all channels
        :param int focal_plane_idx: focal plane to consider
        :param list/int mask_channel_ids: channels from which masks have to be
         generated
        :param float min_fraction: minimum volume fraction of the ROI to retain
         a tile
        """

        volume_metadata = pd.read_csv(os.path.join(self.volume_dir,
                                                   'image_volumes_info.csv'))
        available_channels = volume_metadata['channel_num'].unique()
        if isinstance(channel_ids, int) and channel_ids == -1:
            channel_ids = available_channels

        channel_indicator = [c in available_channels for c in channel_ids]
        assert np.all(channel_indicator)

        if mask_channel_ids is not None:
            assert min_fraction > 0.0
            if isinstance(channel_ids, int):
                channel_ids = [channel_ids]
            ch_str = '-'.join(map(str, mask_channel_ids))
            mask_dir_name = 'mask_{}'.format(ch_str)

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        cropped_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                          str_step_size)
        cropped_dir = os.path.join(self.base_output_dir, cropped_dir_name)
        os.makedirs(cropped_dir, exist_ok=True)

        for timepoint_idx in volume_metadata['timepoint'].unique():
            timepoint_dir = os.path.join(cropped_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)
            if mask_channel_ids is not None:
                mask_dir = os.path.join(self.volume_dir,
                                        'timepoint_{}'.format(timepoint_idx),
                                        mask_dir_name)
                cropped_mask_dir = os.path.join(timepoint_dir, mask_dir_name)
                os.makedirs(cropped_mask_dir, exist_ok=True)
                crop_indices_dict = image_utils.get_crop_indices(
                    mask_dir, min_fraction, cropped_mask_dir, tile_size,
                    step_size, isotropic
                )
            for channel_idx in channel_ids:
                row_idx = self.get_row_idx(volume_metadata, timepoint_idx,
                                           channel_idx, focal_plane_idx)
                channel_metadata = volume_metadata[row_idx]
                channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                metadata = []
                for _, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    if mask_channel_ids is not None:
                        _, fname = os.path.split(sample_fname)
                        cropped_image_data = image_utils.crop_at_indices(
                            cur_image, crop_indices_dict[fname], isotropic
                        )
                    else:
                        cropped_image_data = image_utils.crop_image(
                            input_image=cur_image, tile_size=tile_size,
                            step_size=step_size, isotropic=isotropic
                        )
                    for id_img_tuple in cropped_image_data:
                        xyz_idx = id_img_tuple[0]
                        img_fname = 'n{}_{}'.format(row['sample_num'], xyz_idx)
                        cropped_img = id_img_tuple[1]
                        cropped_img_fname = os.path.join(
                            channel_dir, '{}.npy'.format(img_fname)
                        )
                        np.save(cropped_img_fname, cropped_img,
                                allow_pickle=True, fix_imports=True)
                        metadata.append((row['timepoint'], row['channel_num'],
                                         row['sample_num'], row['slice_num'],
                                         cropped_img_fname))
                msg = 'Cropped images for channel:{}'.format(channel_idx)
                self._log_info(msg)
                fname_header = 'fname_{}'.format(channel_idx)
                cur_df = pd.DataFrame.from_records(
                    metadata,
                    columns=['timepoint', 'channel_num', 'sample_num',
                             'slice_num', fname_header]
                )
                metadata_fname = os.path.join(cropped_dir,
                                              'cropped_images_info.csv')
                if channel_idx == 0:
                    df = cur_df
                else:
                    df = pd.read_csv(metadata_fname, sep=',', index_col=0)
                    df[fname_header] = cur_df[fname_header]
                df.to_csv(metadata_fname, sep=',')


class LifPreProcessor2D(BasePreProcessor):
    """Saves the individual images as a npy file

    In some acquisitions there are 3 z images corresponding to different focal
    planes (focal plane might not be the correct term here!). Using z=0 for the
    recent experiment
    """

    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Saves the each individual image as a npy file

        Have to decide when to reprocess the file and when not to. Currently
        doesn't check if the file has already been processed.
        :param str img_fname: fname with full path of the lif image
        """
        records = []
        # exclude the first 14 due to artifacts and some have one z
        # (series 15, 412) instead of 3
        for z_idx in range(num_pix_z):
            cur_fname = os.path.join(
                channel_dir, 'image_n{}_z{}.npy'.format(sample_idx, z_idx)
            )
            # image voxels are 16 bits
            img = reader.read(c=channel_idx, z=z_idx, t=timepoint_idx,
                              series=sample_idx, rescale=False)
            np.save(cur_fname, img, allow_pickle=True, fix_imports=True)
            msg = 'Generated file:{}'.format(cur_fname)
            self._log_info(msg)
            # add wavelength info perhaps?
            records.append((timepoint_idx, channel_idx, sample_idx, z_idx,
                            cur_fname, size_x_um, size_y_um, size_z_um))
        return records

    def get_row_idx(self, volume_metadata, timepoint_idx,
                    channel_idx, focal_plane_idx=None):
        """Get the indices for images with timepoint_idx and channel_idx"""

        row_idx = ((volume_metadata['timepoint'] == timepoint_idx) &
                   (volume_metadata['channel_num'] == channel_idx) &
                   (volume_metadata['slice_num'] == focal_plane_idx))
        return row_idx


class LifPreProcessor3D(BasePreProcessor):
    """Class for splitting and cropping lif images"""

    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Saves the individual image volumes as a npy file"""

        records = []
        cur_vol_fname = os.path.join(channel_dir,
                                     'image_n{}.npy'.format(sample_idx))
        for z_idx in range(num_pix_z):
            img[:, :, z_idx] = reader.read(c=channel_idx, z=z_idx,
                                           t=timepoint_idx, series=sample_idx,
                                           rescale=False)
        np.save(cur_vol_fname, img, allow_pickle=True, fix_imports=True)
        # add wavelength info perhaps?
        records.append((channel_idx, sample_idx, timepoint_idx, cur_vol_fname,
                        size_x_um, size_y_um, size_z_um))
        msg = 'Wrote files for channel:{}'.format(channel_idx)
        self._log_info(msg)
        return records

    def get_row_idx(self, volume_metadata, timepoint_idx,
                    channel_idx, focal_plane_idx=None):
        """Get the indices for images with timepoint_idx and channel_idx"""

        row_idx = ((volume_metadata['timepoint'] == timepoint_idx) &
                   (volume_metadata['channel_num'] == channel_idx))
        return row_idx
