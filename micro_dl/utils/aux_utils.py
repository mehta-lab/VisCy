"""Auxiliary utility functions"""
import glob
import inspect
import importlib
import json
import logging
import natsort
import numpy as np
import os
import pandas as pd
import yaml


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


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return: dict config: Configuration parameters
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def get_row_idx(frames_metadata, time_idx,
                channel_idx, slice_idx=-1):
    """Get the indices for images with timepoint_idx and channel_idx

    :param pd.DataFrame frames_metadata: DF with columns time_idx,
     channel_idx, slice_idx, file_name]
    :param int time_idx: get info for this timepoint
    :param int channel_idx: get info for this channel
    :param int slice_idx: get info for this focal plane (2D)
    """
    if slice_idx > -1:
        row_idx = ((frames_metadata['time_idx'] == time_idx) &
                   (frames_metadata['channel_idx'] == channel_idx) &
                   (frames_metadata['slice_idx'] == slice_idx))
    else:
        row_idx = ((frames_metadata['time_idx'] == time_idx) &
                   (frames_metadata['channel_idx'] == channel_idx))
    return row_idx


def get_meta_idx(metadata_df,
                 time_idx,
                 channel_idx,
                 slice_idx,
                 pos_idx):
    """
    Get row index in metadata dataframe given variable indices

    :param dataframe metadata_df: Dataframe with column names given below
    :param int time_idx: Timepoint index
    :param int channel_idx: Channel index
    :param int slice_idx: Slize (z) index
    :param int pos_idx: Position (FOV) index
    :return: int pos_idx: Row position matching indices above
    """
    frame_idx = metadata_df.index[
        (metadata_df['channel_idx'] == channel_idx) &
        (metadata_df['time_idx'] == time_idx) &
        (metadata_df["slice_idx"] == slice_idx) &
        (metadata_df["pos_idx"] == pos_idx)].tolist()
    return frame_idx[0]


def get_im_name(time_idx=None,
                channel_idx=None,
                slice_idx=None,
                pos_idx=None,
                extra_field=None,
                int2str_len=3):
    im_name = "im"
    if channel_idx is not None:
        im_name += "_c" + str(channel_idx).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(slice_idx).zfill(int2str_len)
    if time_idx is not None:
        im_name += "_t" + str(time_idx).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(pos_idx).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ".npy"
    return im_name


def sort_meta_by_channel(frames_metadata):
    """
    Rearrange metadata dataframe from all channels being listed in the same column
    to moving file names for each channel to separate columns.

    :param dataframe frames_metadata: Metadata with one column named 'file_name'
    :return dataframe sorted_metadata: Metadata with separate file_name_X for
        channel X.
    """

    metadata_ids = validate_metadata_indices(
        frames_metadata,
        time_ids=-1,
        channel_ids=-1,
        slice_ids=-1,
        pos_ids=-1)

    channel_ids = metadata_ids["channel_ids"]
    # Get all metadata for first channel
    sorted_metadata = frames_metadata[
        frames_metadata["channel_idx"] == channel_ids[0]
    ].reset_index()

    # Loop through the rest of the channels and concat filenames
    for c in channel_ids[1:]:
        col_name = "file_name_{}".format(c)
        channel_meta = frames_metadata[frames_metadata["channel_idx"] == c]
        # TODO: Assert that all indices are the same for safety here
        # It should be taken care of by preprocessing, unless someone runs it
        # several times with different settings
        channel_meta = pd.Series(channel_meta["file_name"].tolist(),
                                 name=col_name)
        sorted_metadata = pd.concat([sorted_metadata, channel_meta], axis=1)

    # Rename file name
    sorted_metadata = sorted_metadata.rename(
        index=str,
        columns={"file_name": "file_name_{}".format(channel_ids[0])})
    sorted_metadata = sorted_metadata.drop(["index", "Unnamed: 0"], axis=1)
    return sorted_metadata


def validate_metadata_indices(frames_metadata,
                              time_ids=None,
                              channel_ids=None,
                              slice_ids=None,
                              pos_ids=None):
    """
    Check the availability of indices provided timepoints, channels, positions
    and slices for all data.
    If input ids are None, the indices for that parameter will not be
    evaluated. If input ids are -1, all indices for that parameter will
    be returned.

    :param pd.DataFrame frames_metadata: DF with columns time_idx,
     channel_idx, slice_idx, pos_idx, file_name]
    :param int/list time_ids: check availability of these timepoints in
     frames_metadata
    :param int/list channel_ids: check availability of these channels in
     frames_metadata
    :param int/list pos_ids: Check availability of positions in metadata
    :param int/list slice_ids: Check availability of z slices in metadata
    :return dict metadata_ids: All indices found given input
    :raise AssertionError: If not all channels, timepoints, positions
        or slices are present
    """
    meta_id_names = [
        "channel_ids",
        "slice_ids",
        "time_ids",
        "pos_ids",
    ]
    id_list = [
        channel_ids,
        slice_ids,
        time_ids,
        pos_ids,
    ]
    col_names = [
        "channel_idx",
        "slice_idx",
        "time_idx",
        "pos_idx",
    ]
    metadata_ids = {}
    for meta_id_name, ids, col_name in zip(meta_id_names, id_list, col_names):
        if ids is not None:
            if np.issubdtype(type(ids), np.integer):
                if ids == -1:
                    ids = frames_metadata[col_name].unique()
                else:
                    ids = [ids]
            all_ids = frames_metadata[col_name].unique()
            id_indicator = [i in all_ids for i in ids]
            assert np.all(id_indicator),\
                'Indices for {} available'.format(col_name)
            metadata_ids[meta_id_name] = ids

    return metadata_ids


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


def read_meta(input_dir, meta_fname='frames_meta.csv'):
    """
    Read metadata file, which is assumed to be named 'frames_meta.csv'
    in given directory

    :param str input_dir: Directory containing data and metadata
    :return dataframe frames_metadata: Metadata for all frames
    :raise IOError: If metadata file isn't present
    """
    meta_fname = glob.glob(os.path.join(input_dir, meta_fname))
    assert len(meta_fname) == 1, \
        "Can't find info.csv file in {}".format(input_dir)
    try:
        frames_metadata = pd.read_csv(meta_fname[0])
    except IOError as e:
        e.args += 'cannot read split image info'
        raise
    return frames_metadata


def save_tile_meta(tiles_meta,
                   cur_channel,
                   tiled_dir):
    """
    Save meta data for tiled images

    :param list tiles_meta: List of tuples holding meta info for tiled
        images
    :param int cur_channel: Channel being tiled
    :param str tiled_dir: Directory to save meta data in
    """
    fname_header = 'fname_{}'.format(cur_channel)
    cur_df = pd.DataFrame.from_records(
        tiles_meta,
        columns=['time_idx', 'channel_idx', 'pos_idx',
                 'slice_idx', fname_header]
    )
    metadata_fname = os.path.join(tiled_dir, 'tiles_meta.csv')
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
    params = np.array(params)
    param_indicator = np.zeros(len(params), dtype='bool')
    for idx, exp_param in enumerate(params):
        cur_indicator = (exp_param in config_dict) and \
                        (config_dict[exp_param] is not None)
        param_indicator[idx] = cur_indicator
    check = np.all(param_indicator)
    msg = 'Params absent in network_config: {}'.\
        format(params[param_indicator == 0])
    return check, msg


def get_channel_axis(data_format):
    """Get the channel axis given the data format

    :param str data_format: as named. [channels_last, channel_first]
    :return int channel_axis
    """
    assert data_format in ['channels_first', 'channels_last'], \
        'Invalid data format %s' % data_format
    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    return channel_axis


def read_json(json_filename):
    """
    Read  JSON file and validate schema

    :param str json_filename: json file name
    :return: dict json_object: JSON object
    :raise FileNotFoundError: if file can't be read
    :raise JSONDecodeError: if file is not in json format
    """
    try:
        with open(json_filename, "r") as read_file:
            try:
                json_object = json.load(read_file)
            except json.JSONDecodeError as jsone:
                print(jsone)
                raise
    except FileNotFoundError as e:
        print(e)
        raise
    return json_object


def write_json(json_dict, json_filename):
    """
    Writes dict as json file.

    :param dict json_dict: Dictionary to be written
    :param str json_filename: Full path file name of json
    """
    json_dump = json.dumps(json_dict)
    with open(json_filename, "w") as write_file:
        write_file.write(json_dump)


def get_sorted_names(dir_name):
    """
    Get image names in directory and sort them by their indices

    :param str dir_name: Image directory name
    :return list of strs im_names: Image names sorted according to indices
    """
    im_names = [f for f in os.listdir(dir_name) if f.startswith('im_')]
    # Sort image names according to indices
    return natsort.natsorted(im_names)


def get_ids_from_imname(im_name, df_names):
    """
    Assumes im_name is im_c***_t***_p***_z***.png, e.g. im_c000_z010_t000_p000.png
    :param str im_name: Image name without path
    :return dict meta_row: One row of metadata given image file name
    """
    meta_row = dict.fromkeys(df_names)
    # Channel name can't be retrieved from image name
    meta_row["channel_name"] = None
    meta_row["channel_idx"] = int(im_name[4:7])
    meta_row["slice_idx"] = int(im_name[9:12])
    meta_row["time_idx"] = int(im_name[14:17])
    meta_row["pos_idx"] = int(im_name[19:22])
    meta_row["file_name"] = im_name
    return meta_row
