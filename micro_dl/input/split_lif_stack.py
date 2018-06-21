"""Split Lif stack into individual images

Uses dir structure:
base_output_dir
 |-split_images, split_images_info.csv
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
import numpy as np
import os
import pandas as pd

from micro_dl.utils.aux_utils import init_logger


class LifStackSplitter(metaclass=ABCMeta):
    """Base class for volume splitting and cropping"""

    def __init__(self, lif_fname, base_output_dir, verbose=0):
        """Init.

        :param str lif_fname: fname with full path of the Lif file
        :param str base_output_dir: base folder for storing the individual
         image and cropped volumes
        :param int verbose: specifies the logging level: NOTSET:0, DEBUG:10,
         INFO:20, WARNING:30, ERROR:40, CRITICAL:50
        """

        self.lif_fname = lif_fname
        self.base_output_dir = base_output_dir
        self.split_dir = os.path.join(self.base_output_dir, 'split_images')
        log_levels = [0, 10, 20, 30, 40, 50]
        if verbose in log_levels:
            self.verbose = verbose
        else:
            self.verbose = 10
        self.logger = self._init_logger()

    def _init_logger(self):
        """Initialize logger for pre-processing.

        Logger outputs to console and log_file
        """

        logger_fname = os.path.join(self.base_output_dir, 'lif_splitter.log')
        logger = init_logger('lif_splitter', logger_fname, self.verbose)
        return logger

    @abstractmethod
    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Save each image as a numpy array."""

        raise NotImplementedError

    def _log_info(self, msg):
        """Log info.

        :param str msg: message to be logged
        """

        if self.verbose > 0:
            self.logger.info(msg)

    def save_images(self):
        """Saves the individual images as a npy file

        2D might have more acquisitions +/- focal plane, (usually 3 images).
        focal_plane_idx corresponds to the plane to consider. Mid-plane is the
        one in focus and the +/- on either side would be blurred. For 2D
        acquisitions, this is stored along the Z dimension. How is this handled
        for 3D acquisitions?
        """

        if not os.path.exists(self.lif_fname):
            raise FileNotFoundError(
                "LIF file doesn't exist at:", self.lif_fname
            )
        os.makedirs(self.split_dir, exist_ok=True)

        jv.start_vm(class_path=bf.JARS, max_heap_size='8G')
        metadata = bf.get_omexml_metadata(self.lif_fname)
        omexml_object = bf.OMEXML(metadata)
        num_channels = omexml_object.image().Pixels.channel_count
        num_samples = omexml_object.get_image_count()
        num_timepoints = omexml_object.image().Pixels.SizeT
        num_pix_z = omexml_object.image().Pixels.SizeZ
        size_x_um = omexml_object.image().Pixels.PhysicalSizeX
        size_y_um = omexml_object.image().Pixels.PhysicalSizeY
        size_z_um = omexml_object.image().Pixels.PhysicalSizeZ

        reader = bf.ImageReader(self.lif_fname, perform_init=True)

        records = []
        for timepoint_idx in range(num_timepoints):
            timepoint_dir = os.path.join(self.split_dir,
                                         'timepoint_{}'.format(timepoint_idx))
            os.makedirs(timepoint_dir, exist_ok=True)

            for channel_idx in range(num_channels):
                channel_dir = os.path.join(timepoint_dir,
                                           'channel_{}'.format(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                for sample_idx in range(15, 412):  # num_samples
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
        metadata_fname = os.path.join(self.split_dir,
                                      'split_images_info.csv')
        df.to_csv(metadata_fname, sep=',')
        jv.kill_vm()


class LifStackSplitter2D(LifStackSplitter):
    """Saves the individual images as a npy file

    In some acquisitions there are 3 z images corresponding to different focal
    planes (focal plane might not be the correct term here!). Using z=0 for the
    recent experiment
    """

    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Saves the each individual image as a npy file.

        Have to decide when to reprocess the file and when not to. Currently
        doesn't check if the file has already been processed.
        :param bf.ImageReader reader: fname with full path of the lif image
        :param int num_pix_z: number of focal_plane acquisitions
        :param str channel_dir: dir to save the split images
        :param int timepoint_idx: timepoint to split
        :param int channel_idx: channel to split
        :param int sample_idx: sample to split
        :param float size_x_um: voxel resolution along x in microns
        :param float size_y_um: voxel resolution along y in microns
        :param float size_z_um: voxel resolution along focal_plane in microns
        :return: list of tuples of metadata
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


class LifStackSplitter3D(LifStackSplitter):
    """Class for splitting and cropping lif images."""

    def save_each_image(self, reader, num_pix_z, channel_dir, timepoint_idx,
                        channel_idx, sample_idx, size_x_um, size_y_um,
                        size_z_um):
        """Saves the individual image volumes as a npy file.

        :param bf.ImageReader reader: fname with full path of the lif image
        :param int num_pix_z: number of focal_plane acquisitions
        :param str channel_dir: dir to save the split images
        :param int timepoint_idx: timepoint to split
        :param int channel_idx: channel to split
        :param int sample_idx: sample to split
        :param float size_x_um: voxel resolution along x in microns
        :param float size_y_um: voxel resolution along y in microns
        :param float size_z_um: voxel resolution along z in microns
        :return: list of tuples of metadata
        """

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
