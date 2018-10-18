"""Classes for handling microscopy data in image file format, NOT LIF!

Uses dir structure:
input_dir
 |-image_volume, image_volumes_info.csv
    |-tp0
        |-channel0
 |-img_512_512_8_.., cropped_images_info.csv
    |-tp-0
        |-channel0: contains all npy files for cropped images from channel0
        |-channel1: contains all npy files for cropped images from channel1..
        and so on
"""
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import re

from micro_dl.utils.aux_utils import init_logger


class ImageValidator:
    """Class for verifying image folder structure and writing metadata"""

    def __init__(self, input_dir, meta_name, verbose=0):
        """
        :param str input_dir: Input directory, containing time directories,
            which in turn contain all channels (inputs and target) directories
        :param str meta_name: Name of csv file containing image paths and metadata
            which will be written in input_dir
        :param int verbose: specifies the logging level: NOTSET:0, DEBUG:10,
         INFO:20, WARNING:30, ERROR:40, CRITICAL:50
        """

        self.input_dir = input_dir
        self.time_dirs = self._get_subdirectories(self.input_dir)
        assert len(self.time_dirs) > 0,\
            "Input dir must contain at least one timepoint folder"
        # Check to make sure first timepoint folder contains channel folders
        self.channel_dirs = self._get_subdirectories(
            os.path.join(self.input_dir, self.time_dirs[0]))
        assert len(self.channel_dirs) > 0, \
            "Must be at least one channel folder"
        # Metadata will be written in input folder
        self.meta_name = os.path.join(self.input_dir,
                                      meta_name)
        # Validate and instantiate logging
        log_levels = [0, 10, 20, 30, 40, 50]
        if verbose in log_levels:
            self.verbose = verbose
        else:
            self.verbose = 10
        self.logger = self._init_logger()

    def _init_logger(self):
        """
        Initialize logger for pre-processing

        Logger outputs to console and log_file
        """

        logger_fname = os.path.join(self.input_dir, 'preprocessing.log')
        logger = init_logger('preprocessing', logger_fname, self.verbose)
        return logger

    def _log_info(self, msg):
        """Log info"""

        if self.verbose > 0:
            self.logger.info(msg)

    def _get_subdirectories(self, dir_name):
        subdirs = [subdir_name
                for subdir_name in
                    os.listdir(dir_name)
                    if os.path.isdir(os.path.join(dir_name, subdir_name))
                ]
        return natsort.natsorted(subdirs)

    def folder_validator(self):
        """
        Input directory should contain subdirectories consisting of timepoints,
        which in turn should contain channel folders numbered 0, ...
        This function makes sure images have matching shapes and unique indices
        in each folder and writes a csv containing relevant image information.

        :return list of ints channel_nrbs: Channel numbers determined by searching
            input_dir subfolder names for ints
        :return list of ints im_indices: Unique image indices. Must be matching
            in all the subfolders of input_dir
        """
        # Make sure all input directories contain images with the same indices and shape
        # Collect all timepoint indices
        time_indices = []
        for dir_name in self.time_dirs:
            time_indices.append(self.get_idx_from_dir(dir_name))
        # Collect all channel indices from first timepoint
        channel_indices = []
        for dir_name in self.channel_dirs:
            channel_indices.append(self.get_idx_from_dir(dir_name))
        # Collect all image indices from first channel directory
        im_shape, im_indices, _ = self.image_validator(os.path.join(
            self.input_dir,
            self.time_dirs[0],
            self.channel_dirs[0]))

        # Skipping these records for now
        z_idx = 0
        size_x_um = 0
        size_y_um = 0
        size_z_um = 0

        # Make sure image shapes and indices match across channels
        # and write csv containing relevant metadata
        nbr_idxs = len(im_indices)
        records = []
        for time_idx, time_dir in zip(time_indices, self.time_dirs):
            for channel_idx, channel_dir in zip(channel_indices, self.channel_dirs):
                cur_dir = os.path.join(
                    self.input_dir,
                    time_dir,
                    channel_dir)
                assert os.path.exists(cur_dir), \
                    "Directory doesn't exist: {}".format(cur_dir)
                cur_shape, cur_indices, cur_names = self.image_validator(cur_dir)
                # Assert image shape and indices match
                idx_overlap = set(im_indices).intersection(cur_indices)
                assert len(idx_overlap) == nbr_idxs, \
                    "Index mismatch in folder {}".format(cur_dir)
                assert im_shape == cur_shape, \
                    "Image shape mismatch in folder {}".format(cur_dir)
                for cur_idx, cur_name in zip(cur_indices, cur_names):
                    full_name = os.path.join(self.input_dir, time_dir, channel_dir, cur_name)
                    records.append((time_idx,
                                    channel_idx,
                                    cur_idx,
                                    z_idx,
                                    full_name,
                                    size_x_um,
                                    size_y_um,
                                    size_z_um))
        # Create pandas dataframe
        df = pd.DataFrame.from_records(
            records,
            columns=['timepoint', 'channel_num', 'sample_num', 'slice_num',
                     'fname', 'size_x_microns', 'size_y_microns',
                     'size_z_microns']
        )
        df.to_csv(self.meta_name, sep=',')
        self._log_info("Writing metadata in: {}".format(self.input_dir,
                                                        'image_volumes_info.csv'))
        self._log_info("found timepoints: {}".format(time_indices))
        self._log_info("found channels: {}".format(channel_indices))
        self._log_info("found image indices: {}".format(im_indices))

    def _get_sorted_names(self, image_dir):
        """
        Get image names in directory and sort them by their indices

        :param str image_dir: Image directory name

        :return list of strs im_names: Image names sorted according to indices
        """
        im_names = [f for f in os.listdir(image_dir) if not f.startswith('.')]
        # Sort image names according to indices
        return natsort.natsorted(im_names)

    def _read_or_catch(self, dir_name, im_name):
        """
        Checks file extension for npy and load array if true. Otherwise
        readd regular image (png, tif, jpg, see OpenCV for supported files)
        of any bit depth.

        :param str dir_name: Directory name
        :param str im_name: Image name

        :return array im: image

        :throws IOError if image can't be opened
        """
        if im_name[-3:] == 'npy':
            im = np.load(os.path.join(dir_name, im_name))
        else:
            try:
                im = cv2.imread(os.path.join(dir_name, im_name), cv2.IMREAD_ANYDEPTH)
            except IOError as e:
                print(e)
        return im

    def image_validator(self, image_dir):
        """
        Make sure all images in a directory have unique indexing and the same
        shape.

        :param str image_dir: Directory containing opencv readable images

        :return tuple im_shape: image shape if all images have the same shape
        :return list im_indices: Unique indices for the images
        :return list im_names: list of fnames for images in a channel dir

        :throws IOError: If images can't be read
        """
        im_names = self._get_sorted_names(image_dir)
        assert len(im_names) > 1, "Only one or less images in directory " + image_dir
        # Read first image to determine shape
        im = self._read_or_catch(image_dir, im_names[0])
        im_shape = im.shape
        # Determine indexing
        idx0 = re.findall("\d+", im_names[0])
        idx1 = re.findall("\d+", im_names[1])
        assert len(idx0) == len(idx1), \
            "Different numbers of indices in file names {} {}".format(
                im_names[0], im_names[1])
        potential_idxs = np.zeros(len(idx0))
        for idx, (i, j) in enumerate(zip(idx0, idx1)):
            potential_idxs[idx] = abs(int(j) - int(i))
        idx_pos = np.where(potential_idxs > 0)[0]
        # There should only be one index (varying integer) in filenames
        assert len(idx_pos) == 1, ("Unclear indexing,"
                                   "more than one varying int in file names")
        # Loop through all images
        # check that shape is constant and collect indices
        im_indices = np.zeros(len(im_names), dtype=int)
        for i, im_name in enumerate(im_names):
            im = self._read_or_catch(image_dir, im_name)
            assert im.shape == im_shape, "Mismatching image shape in " + im_name
            im_indices[i] = int(re.findall("\d+", im_name)[idx_pos[0]])

        # Make sure there's a unique index for each image
        assert len(im_indices) == len(np.unique(im_indices)), \
            "Images don't have unique indexing"
        msg = '{} contains indices: {}'.format(image_dir, im_indices)
        self._log_info(msg)
        return im_shape, im_indices, im_names

    def get_idx_from_dir(self, dir_name):
        """
        Get directory index, assuming it's an int in the last part of the
        image directory name.

        :param str dir_name: Directory name containing one int

        :return int idx_nbr: Directory index
        """
        strs = dir_name.split("/")
        pos = -1
        if len(strs[pos]) == 0 and len(strs) > 1:
            pos = -2

        idx_nbr = re.findall("\d+", strs[pos])
        assert len(idx_nbr) == 1, ("Couldn't find index in {}".format(dir_name))
        return int(idx_nbr[0])
