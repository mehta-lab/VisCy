import glob
import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils


def get_preprocess_config(data_dir):
    # If the parent dir with tile dir, mask dir is passed as data_dir,
    # it should contain a json with directory names
    json_fname = os.path.join(data_dir, 'preprocessing_info.json')
    try:
        preprocessing_info = aux_utils.read_json(json_filename=json_fname)

        # Preprocessing_info is a list of jsons. Use the last json. If a tile
        # (training data) dir is specified and exists in info json use that
        recent_json = preprocessing_info[-1]
        preprocess_config = recent_json['config']
    except FileNotFoundError as e:
        msg = 'No preprocessing config file found in {}. Error {}'.format(data_dir, e)
        raise msg

    return preprocess_config


def validate_mask_meta(mask_dir,
                       input_dir,
                       csv_name=None,
                       mask_channel=None):
    """
    If user provides existing masks, the mask directory should also contain
    a csv file (not named frames_meta.csv which is reserved for output) with
    two column names: mask_name and file_name. Each row should describe the
    mask name and the corresponding file name. Each file_name should exist in
    input_dir and belong to the same channel.
    This function checks that all file names exist in input_dir and writes a
    frames_meta csv containing mask names with indices corresponding to the
    matched file_name. It also assigns a mask channel number for future
    preprocessing steps like tiling.

    :param str mask_dir: Mask directory
    :param str input_dir: Input image directory, to match masks with images
    :param int/None mask_channel: Channel idx assigned to masks
    :return int mask_channel: New channel index for masks for writing tiles
    :raises IOError: If no csv file is present in mask_dir
    :raises IOError: If more than one csv file exists in mask_dir
        and no csv_name is provided to resolve ambiguity
    :raises AssertionError: If csv doesn't consist of two columns named
        'mask_name' and 'file_name'
    :raises IndexError: If unable to match file_name in mask_dir csv with
        file_name in input_dir for any given mask row
    """
    input_meta = aux_utils.read_meta(input_dir)
    if mask_channel is None:
        mask_channel = int(
            input_meta['channel_idx'].max() + 1
        )
    # Make sure there is a csv file file
    if csv_name is not None:
        csv_name = glob.glob(os.path.join(mask_dir, csv_name))
        if len(csv_name) == 1:
            # Use the one existing csv name
            csv_name = csv_name[0]
        else:
            csv_name = None
    # No csv name given, search for it
    if csv_name is None:
        csv_name = glob.glob(os.path.join(mask_dir, '*.csv'))
        if len(csv_name) == 0:
            raise IOError("No csv file present in mask dir")
        else:
            # See if frames_meta is already present, if so, move on
            has_meta = next((s for s in csv_name if 'frames_meta.csv' in s), None)
            if isinstance(has_meta, str):
                # Return existing mask channel from frames_meta
                frames_meta = pd.read_csv(
                    os.path.join(mask_dir, 'frames_meta.csv'),
                )
                mask_channel = np.unique(frames_meta['channel_idx'])
                assert len(mask_channel) == 1,\
                    "Found more than one mask channel: {}".format(mask_channel)
                mask_channel = mask_channel[0]
                return mask_channel
            elif len(csv_name) == 1:
                # Use the one existing csv name
                csv_name = csv_name[0]
            else:
                # More than one csv file in dir
                raise IOError("More than one csv file present in mask dir",
                              "use csv_name to specify which one to use")

    # Read csv with masks and corresponding input file names
    mask_meta = aux_utils.read_meta(input_dir=mask_dir, meta_fname=csv_name)

    assert len(set(mask_meta).difference({'file_name', 'mask_name'})) == 0,\
        "mask csv should have columns mask_name and file_name " +\
        "(corresponding to the file_name in input_dir)"
    # Check that file_name for each mask_name matches files in input_dir
    file_names = input_meta['file_name']
    # Create dataframe that will store all indices for masks
    out_meta = aux_utils.make_dataframe(nbr_rows=mask_meta.shape[0])
    for i, row in mask_meta.iterrows():
        try:
            file_loc = file_names[file_names == row.file_name].index[0]
        except IndexError as e:
            msg = "Can't find image file name match for {}, error {}".format(
                row.file_name, e)
            raise IndexError(msg)
        # Fill dataframe with row indices from matched image in input dir
        out_meta.iloc[i] = input_meta.iloc[file_loc]
        # Write back the mask name
        out_meta.iloc[i]['file_name'] = row.mask_name

    assert len(out_meta.channel_idx.unique()) == 1,\
        "Masks should match one input channel only"
    assert mask_channel not in set(input_meta.channel_idx.unique()),\
        "Mask channel {} already exists in image dir".format(mask_channel)

    # Replace channel_idx new mask channel idx
    out_meta['channel_idx'] = mask_channel

    # Write mask metadata with indices that match input images
    meta_filename = os.path.join(mask_dir, 'frames_meta.csv')
    out_meta.to_csv(meta_filename, sep=",")

    return mask_channel




