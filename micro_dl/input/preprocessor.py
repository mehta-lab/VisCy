"""Classes for handling microscopy data in lif format

Uses dir structure:
base_output_dir
 |-img_512_512_8_sc_norm
  |-data_config.yml (contains info on crop size, scale, norm, layers used
    metadata.csv (fpath_label_0, fpath_label_1..., fpath_label_n)
     fpath_label_0: with fname of the format
     ch0_sample0_xinit-xend_yinit_yend_zinit_zend.npy
     |-channel0: contains all npy files for images from channel0
     |-channel1: contains all npy files for images from channel1....and so on
"""
from abc import ABCMeta, abstractmethod
import bioformats as bf
import javabridge as jv
import logging
import numpy as np
import os
import pandas as pd
from micro_dl.utils.image_utils import crop_3d, normalize_zscore


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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.verbose)
        logger.addHandler(file_handler)
        return logger

    @abstractmethod
    def save_image_volumes(self, img_fname):
        """Saves the individual image volumes as a npy file"""

        raise NotImplementedError

    @abstractmethod
    def crop_image_volumes(self, tile_size, step_size, normalize,
                           channel_ids=-1):
        """Crop image volumes in the specified channels"""

        raise NotImplementedError


class LifPreProcessor(BasePreProcessor):
    """Class for splitting and cropping lif images"""

    def save_image_volumes(self, img_fname):
        """Saves the individual image volumes as a npy file

        Have to decide when to reprocess the file and when not to. Currently
        doesn't check if the file has already been processed.
        :param str img_fname: fname with full path of the lif image
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
        size_x = omexml_object.image().Pixels.SizeX
        size_y = omexml_object.image().Pixels.SizeY
        size_z = omexml_object.image().Pixels.SizeZ
        reader = bf.ImageReader(img_fname, perform_init=True)

        records = []
        for channel_idx in range(num_channels):
            if self.verbose > 0:
                self.logger.info(
                    'Reading images for channel:{}'.format(channel_idx)
                )
            channel_dir = os.path.join(self.volume_dir,
                                       'channel_{}'.format(channel_idx))
            os.makedirs(channel_dir, exist_ok=True)
            for sample_idx in range(num_samples):
                for timepoint_idx in range(num_timepoints):

                    cur_vol_fname = os.path.join(
                        channel_dir,
                        'image_n{}_tp{}.npy'.format(sample_idx, timepoint_idx)
                    )

                    if not os.path.exists(cur_vol_fname):
                        # image voxels are 16 bits
                        img = np.empty(shape=(size_x, size_y, size_z),
                                       dtype=np.uint16)

                        for z_idx in range(size_z):
                            img[:, :, z_idx] = reader.read(
                                c=channel_idx, z=z_idx, t=timepoint_idx,
                                series=sample_idx, rescale=False
                            )
                        np.save(cur_vol_fname, img,
                                allow_pickle=True, fix_imports=True)
                        if self.verbose > 1:
                            self.logger.info(
                                'Generated file:{}'.format(cur_vol_fname)
                            )
                    # add resolution and wavelength info perhaps?
                    records.append((channel_idx, sample_idx,
                                    timepoint_idx, cur_vol_fname))
            if self.verbose:
                self.logger.info(
                    'Wrote files for channel:{}'.format(channel_idx)
                )

        df = pd.DataFrame.from_records(
            records,
            columns=['channel_num', 'sample_num', 'timepoint', 'fname']
        )
        metadata_fname = os.path.join(self.volume_dir,
                                      'image_volumes_info.csv')
        df.to_csv(metadata_fname, sep=',')
        jv.kill_vm()

    def crop_image_volumes(self, tile_size, step_size, normalize,
                           channel_ids=-1):
        """Crop image volumes in the specified channels

        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In
         case of no overlap, the step size is tile_size. If overlap,
         step_size < tile_size
        :param str normalize: type of normalization allowed
        :param list channel_ids: crop volumes in the given channels.
         default=-1, crop all channels
        """

        volume_metadata = pd.read_csv(os.path.join(self.volume_dir,
                                                   'image_volumes_info.csv'))
        available_channels = volume_metadata['channel_num'].unique()
        if isinstance(channel_ids, int) and channel_ids==-1:
            channel_ids = available_channels

        channel_indicator = [c in available_channels for c in channel_ids]
        assert np.all(channel_indicator)

        assert normalize == 'zscore', 'only z-score normalization is available'

        str_tile_size = '-'.join([str(val) for val in tile_size])
        str_step_size = '-'.join([str(val) for val in step_size])
        cropped_dir_name = 'image_tile_{}_step_{}'.format(str_tile_size,
                                                          str_step_size)
        if normalize:
            cropped_dir_name = '{}_{}'.format(cropped_dir_name, normalize)
        cropped_dir = os.path.join(self.base_output_dir, cropped_dir_name)
        os.makedirs(cropped_dir, exist_ok=True)

        for channel_idx in channel_ids:
            row_idx = volume_metadata['channel_num']==channel_idx
            channel_metadata = volume_metadata[row_idx]
            channel_dir = os.path.join(cropped_dir,
                                       'channel_{}'.format(channel_idx))
            os.makedirs(channel_dir, exist_ok=True)
            metadata = []
            for _, row in channel_metadata.iterrows():
                sample_fname = row['fname']
                cur_image = np.load(sample_fname)
                cropped_image_data = crop_3d(cur_image, tile_size, step_size)
                for id_img_tuple in cropped_image_data:
                    cropped_img = id_img_tuple[1]
                    if normalize:
                        # other normalizations to be added
                        cropped_img = normalize_zscore(cropped_img)
                    cropped_img_fname = os.path.join(
                        channel_dir, '{}.npy'.format(id_img_tuple[0])
                    )
                    np.save(cropped_img_fname, cropped_img,
                            allow_pickle=True, fix_imports=True)
                    metadata.append((row['channel_num'], row['sample_num'],
                                     row['timepoint'], cropped_img_fname))
            if self.verbose:
                self.logger.info(
                    'Cropped images for channel:{}'.format(channel_idx)
                )
            fname_header = 'fname_{}'.format(channel_idx)
            cur_df = pd.DataFrame.from_records(
                metadata,
                columns=['channel_num', 'sample_num',
                         'timepoint', fname_header]
            )
            metadata_fname = os.path.join(cropped_dir,
                                          'cropped_images_info.csv')
            if channel_idx == 0:
                df = cur_df
            else:
                df = pd.read_csv(metadata_fname, sep=',', index_col=0)
                df[fname_header] = cur_df[fname_header]
            df.to_csv(metadata_fname, sep=',')
