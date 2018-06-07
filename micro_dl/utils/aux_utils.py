"""Auxiliary utility functions"""
import inspect
import importlib
import numpy as np


def import_class(module_name, cls_name):
    """Imports a class specified in yaml dynamically

    REFACTOR THIS!!

    :param str module_name: modules such as input, utils, train etc
    :param str cls_name: class to find
    """

    main_module = 'micro_dl'
    try:
        module = importlib.import_module(module_name, main_module)
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
