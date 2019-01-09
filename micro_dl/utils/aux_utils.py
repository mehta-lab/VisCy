"""Auxiliary utility functions"""
import glob
import inspect
import importlib
import json
import logging
import natsort
import numpy as np
import os
import re
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
                ext='.npy',
                int2str_len=3):
    """
    Create an image name given parameters and extension

    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str extra_field: Any extra string you want to include in the name
    :param str ext: Extension, e.g. '.png'
    :param int int2str_len: Length of string of the converted integers
    :return st im_name: Image file name
    """
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
    im_name += ext
    return im_name


def sort_meta_by_channel(frames_metadata):
    """
    Rearrange metadata dataframe from all channels being listed in the same column
    to moving file names for each channel to separate columns.

    :param dataframe frames_metadata: Metadata with one column named 'file_name'
    :return dataframe sorted_metadata: Metadata with separate file_name_X for
        channel X.
    """

    metadata_ids, tp_dict = validate_metadata_indices(
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
                              pos_ids=None,
                              uniform_structure=True):
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
    :param bool uniform_structure: bool indicator if unequal quantities in any
     of the ids (channel, time, slice, pos)
    :return dict metadata_ids: All indices found given input
    :raise AssertionError: If not all channels, timepoints, positions
        or slices are present
    """
    meta_id_names = [
        'channel_ids',
        'slice_ids',
        'time_ids',
        'pos_ids',
    ]
    id_list = [
        channel_ids,
        slice_ids,
        time_ids,
        pos_ids,
    ]
    col_names = [
        'channel_idx',
        'slice_idx',
        'time_idx',
        'pos_idx',
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

    tp_dict = None
    if not uniform_structure:
        tp_dict = {}
        for tp_idx in metadata_ids['time_ids']:
            ch_dict = {}
            for ch_idx in metadata_ids['channel_ids']:
                pos_dict = {}
                for pos_idx in metadata_ids['pos_ids']:
                    row_idx = ((frames_metadata['time_idx'] == tp_idx) &
                               (frames_metadata['channel_idx'] == ch_idx) &
                               (frames_metadata['pos_idx'] == pos_idx))
                    if np.any(row_idx):
                        cur_slice_ids = \
                            frames_metadata[row_idx]['slice_idx'].unique()
                        pos_dict[pos_idx] = cur_slice_ids
                ch_dict[ch_idx] = pos_dict
            tp_dict[tp_idx] = ch_dict

    return metadata_ids, tp_dict


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
        frames_metadata = pd.read_csv(meta_fname[0], index_col=0)
    except IOError as e:
        e.args += 'cannot read metadata csv file'
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


def adjust_slice_margins(slice_ids, depth):
    """
    Adjusts slice (z) indices to given z depth by removing indices too close
    to boundaries. Assumes that slice indices are contiguous.

    :param list of ints slice_ids: Slice (z) indices
    :param int depth: Number of z slices
    :return: list of ints slice_ids: Slice indices with adjusted margins
    :raises AssertionError if depth is even
    :raises AssertionError if there aren't enough slice ids for given depth
    :raises AssertionError if slices aren't contiguous
    """
    assert depth % 2 == 1, "Depth must be uneven"
    if depth > 1:
        margin = depth // 2
        nbr_slices = len(slice_ids)
        assert nbr_slices > 2 * margin, \
            "Insufficient slices ({}) for max depth {}".format(
                nbr_slices, depth)
        assert slice_ids[-1] - slice_ids[0] + 1 == nbr_slices, \
            "Slice indices are not contiguous"
        # TODO: use itertools.groupby if non-contiguous data is a thing
        # np.unique is sorted so we can just remove first and last ids
        slice_ids = slice_ids[margin:-margin]
    return slice_ids


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


def get_ids_from_imname(im_name, df_names, order="cztp"):
    """
    Assumes im_name is e.g. im_c***_z***_p***_t***.png,
    It doesn't care about the extension or the number of digits each index is
    represented by, it extracts all integers from the image file name and assigns
    them by order. By default it assumes that the order is c, z, t, p.

    :param str im_name: Image name without path
    :param list of strs df_names: Dataframe col names
    :param str order: Order in which c, z, t, p are given in the image (4 chars)
    :return dict meta_row: One row of metadata given image file name
    """
    order_list = list(order)
    assert len(set(order_list)) == 4,\
        "Order needs 4 unique values, not {}".format(order)
    meta_row = dict.fromkeys(df_names)
    # Channel name can't be retrieved from image name
    meta_row["channel_name"] = None
    meta_row["file_name"] = im_name
    # Find all integers in name string
    ints = re.findall(r'\d+', im_name)
    assert len(ints) == 4, "Expected 4 integers, found {}".format(len(ints))
    # Assign indices based on ints and order
    idx_dict = {"c": "channel_idx",
                "z": "slice_idx",
                "t": "time_idx",
                "p": "pos_idx"}
    for i in idx_dict.keys():
        assert i in order_list, "{} not in order".format(i)
    for i, order_char in enumerate(order_list):
        idx_name = idx_dict[order_char]
        meta_row[idx_name] = int(ints[i])
    return meta_row
