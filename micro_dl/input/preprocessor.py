"""Classes for handling microscopy data in lif format

Uses dir structure:
base_output_dir
 |-image_volume, image_volumes_info.csv
    |-tp0
        |-channel0
 |-img_512_512_8_sc_norm, cropped_images_info.csv
    |-tp-0
        |-channel0: contains all npy files for images from channel0
        |-channel1: contains all npy files for images from channel1..and so on
"""
from abc import ABCMeta, abstractmethod
import bioformats as bf
import javabridge as jv
import logging
import numpy as np
import os
import pandas as pd
from micro_dl.utils.image_utils import crop_image, normalize_zscore


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

    def _log_info(self, msg):
        """Log info"""

        if self.verbose > 0:
            self.logger.info(msg)

    def save_images(self, img_fname):
        """Saves the individual images as a npy file"""

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
                for sample_idx in range(num_samples): # 15, 412):
                    cur_records = save_one_image(reader, num_pix_z, channel_dir,
                                                 timepoint_idx, channel_idx,
                                                 sample_idx, size_x_um,
                                                 size_y_um, size_z_um)
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


    @abstractmethod
    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Save each image as a numpy array"""

        raise NotImplementedError

    def crop_images(self, tile_size, step_size, normalize,
                           isotropic=True, channel_ids=-1):
        """Crop image volumes in the specified channels

        Isotropic here refers to the same dimension/shape along x,y,z and not
        really isotropic resolution in mm

        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In
         case of no overlap, the step size is tile_size. If overlap,
         step_size < tile_size
        :param str normalize: type of normalization allowed
        :param bool isotropic: if 3D, make the grid/shape isotropic
        :param list channel_ids: crop volumes in the given channels.
         default=-1, crop all channels
        """

        volume_metadata = pd.read_csv(os.path.join(self.volume_dir,
                                                   'image_volumes_info.csv'))
        available_channels = volume_metadata['channel_num'].unique()
        if isinstance(channel_ids, int) and channel_ids == -1:
            channel_ids = available_channels

        channel_indicator = [c in available_channels for c in channel_ids]
        assert np.all(channel_indicator)

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        cropped_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                          str_step_size)
        if normalize:
            cropped_dir_name = '{}_{}'.format(cropped_dir_name, normalize)
        cropped_dir = os.path.join(self.base_output_dir, cropped_dir_name)
        os.makedirs(cropped_dir, exist_ok=True)

        for timepoint_idx in volume_metadata['timepoint'].unique():
            timepoint_dir = os.path.join(cropped_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)
            for channel_idx in channel_ids:
                row_idx = get_row_idx(volume_metadata, timepoint_idx,
                                      channel_idx)
                channel_metadata = volume_metadata[row_idx]
                channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                metadata = []
                for _, row in channel_metadata.iterrows():
                    sample_fname = row['fname']
                    cur_image = np.load(sample_fname)
                    cropped_image_data = crop_image(cur_image, tile_size,
                                                    step_size)
                    for id_img_tuple in cropped_image_data:
                        xyz_idx = id_img_tuple[0]
                        img_fname = 'n{}_{}'.format(row['sample_num'], xyz_idx)
                        cropped_img = id_img_tuple[1]
                        if normalize:
                            # other normalizations to be added
                            cropped_img = normalize_zscore(cropped_img)
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
                    cur_records,
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

    @abstractmethod
    def get_row_idx(self, volume_metadata, timepoint_idx, channel_idx):
        """Get the indices for images with timepoint_idx and channel_idx"""

        raise NotImplementedError


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
        # (series 5, 412) instead of 3
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

    def get_row_idx(self, volume_metadata, timepoint_idx, channel_idx):
        """Get the indices for images with timepoint_idx and channel_idx"""

        row_idx = ((volume_metadata['timepoint'] == timepoint_idx) &
                   (volume_metadata['channel_num'] == channel_idx) &
                   (volume_metadata['slice_num'] == 0))
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

    def get_row_idx(self, volume_metadata, timepoint_idx, channel_idx):
        """Get the indices for images with timepoint_idx and channel_idx"""

        row_idx = ((volume_metadata['timepoint'] == timepoint_idx) &
                   (volume_metadata['channel_num'] == channel_idx))
        return row_idx