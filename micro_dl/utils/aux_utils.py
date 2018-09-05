"""Auxiliary utility functions"""
import inspect
import importlib
import logging
import numpy as np
import os
import pandas as pd


def import_class(module_name, cls_name):
    """Imports a class specified in yaml dynamically

    REFACTOR THIS!!

    :param str module_name: modules such as input, utils, train etc
    :param str cls_name: class to find
    """

    full_module_name = ".".join(('micro_dl', module_name))
    try:
        module = importlib.import_module(full_module_name)
        obj = getattr(module, cls_name)

        if inspect.isclass(obj):
            return obj
    except ImportError:
        raise


def get_row_idx(study_metadata, timepoint_idx,
                channel_idx, focal_plane_idx=None):
    """Get the indices for images with timepoint_idx and channel_idx

    :param pd.DataFrame study_metadata: DF with columns timepoint,
     channel_num, sample_num, slice_num, fname, size_x_microns, size_y_microns,
     size_z_microns]
    :param int timepoint_idx: get info for this tp
    :param int channel_idx: get info for this channel
    :param int focal_plane: get info for this focal plane (2D)
    """

    if focal_plane_idx is not None:
        row_idx = ((study_metadata['timepoint'] == timepoint_idx) &
                   (study_metadata['channel_num'] == channel_idx) &
                   (study_metadata['slice_num'] == focal_plane_idx))
    else:
        row_idx = ((study_metadata['timepoint'] == timepoint_idx) &
                   (study_metadata['channel_num'] == channel_idx))
    return row_idx


def validate_tp_channel(study_metadata, timepoint_ids=None, channel_ids=None):
    """Check the availability of provided tp and channels

    :param pd.DataFrame study_metadata: DF with columns timepoint,
     channel_num, sample_num, slice_num, fname, size_x_microns, size_y_microns,
     size_z_microns]
    :param int/list timepoint_ids: check availability of these tps in
     study_metadata
    :param int/list channel_ids: check availability of these channels in
     study_metadata
    """

    tp_channels_ids = {}
    if timepoint_ids is not None:
        if np.issubdtype(type(timepoint_ids), np.integer):
            if timepoint_ids == -1:
                timepoint_ids = study_metadata['timepoint'].unique()
            else:
                timepoint_ids = [timepoint_ids]
        all_tps = study_metadata['timepoint'].unique()
        tp_indicator = [tp in all_tps for tp in timepoint_ids]
        assert np.all(tp_indicator), 'timepoint not available'
        tp_channels_ids['timepoints'] = timepoint_ids

    if channel_ids is not None:
        if np.issubdtype(type(channel_ids), np.integer):
            if channel_ids == -1:
                channel_ids = study_metadata['channel_num'].unique()
            else:
                channel_ids = [channel_ids]
        all_channels = study_metadata['channel_num'].unique()
        channel_indicator = [c in all_channels for c in channel_ids]
        assert np.all(channel_indicator), 'channel not available'
        tp_channels_ids['channels'] = channel_ids

    return tp_channels_ids


def init_logger(logger_name, log_fname, log_level):
    """Creates a logger instance

    :param str logger_name: name of the logger instance
    :param str log_fname: fname with full path of the log file
    :param int log_level: specifies the logging level: NOTSET:0, DEBUG:10,
    INFO:20, WARNING:30, ERROR:40, CRITICAL:50
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_fname)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    return logger


def save_tile_meta(cropped_meta,
                   cur_channel,
                   tiled_dir):
    """Save meta data for cropped images

    :param list cropped_meta: list of tuples holding meta info for cropped
     images
    :param int cur_channel: channel being cropped
    :param str tiled_dir: dir to save meta data
    """

    fname_header = 'fname_{}'.format(cur_channel)
    cur_df = pd.DataFrame.from_records(
        cropped_meta,
        columns=['timepoint', 'channel_num', 'sample_num',
                 'slice_num', fname_header]
    )
    metadata_fname = os.path.join(tiled_dir, 'tiled_images_info.csv')
    if cur_channel == 0:
        df = cur_df
    else:
        df = pd.read_csv(metadata_fname, sep=',', index_col=0)
        df[fname_header] = cur_df[fname_header]
    df.to_csv(metadata_fname, sep=',')


def validate_config(config_dict, params):
    """Check if the required params are present in config

    :param dict config_dict: dictionary with params as keys
    :param list params: list of strings with expected params
    :return: list with bool values indicating if param is present or not
    """

    param_indicator = np.zeros(len(params), dtype='bool')
    for idx, exp_param in enumerate(params):
        cur_indicator = (exp_param in config_dict) and \
                        (config_dict[exp_param] is not None)
        param_indicator[idx] = cur_indicator
    return param_indicator


def get_channel_axis(data_format):
    """Get the channel axis given the data format

    :param str data_format: as named. [channels_last, channel_first]
    :return int channel_axis
    """

    assert data_format in ['channels_first', 'channels_last'], \
        'Invalid data format %s' % data_format
    if data_format == 'channel_first':
        channel_axis = 1
    else:
        channel_axis = -1
    return channel_axis
