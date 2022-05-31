"""Script for preprocessing stack"""
import argparse
import numpy as np
import os
import pandas as pd
import time
import warnings

from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.preprocessing.generate_masks import MaskProcessor
from micro_dl.preprocessing.resize_images import ImageResizer
from micro_dl.preprocessing.tile_3d import ImageTilerUniform3D
from micro_dl.preprocessing.tile_uniform_images import ImageTilerUniform
from micro_dl.preprocessing.tile_nonuniform_images import \
    ImageTilerNonUniform
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    args = parser.parse_args()
    return args


def get_required_params(preprocess_config):
    """
    Create a dictionary with required parameters for preprocessing
    from the preprocessing config. Required parameters are:
        'input_dir': Directory containing input image data
        'output_dir': Directory to write preprocessed data
        'slice_ids': Slice indices
        'time_ids': Time indices
        'pos_ids': Position indices
        'channel_ids': Channel indices
        'uniform_struct': (bool) If images are uniform
        'int2strlen': (int) How long of a string to convert integers to
        'normalize_channels': (list) Containing bools the length of channels
        'num_workers': Number of workers for multiprocessing
        'normalize_im': (str) Normalization scheme
            (stack, dataset, slice, volume)

    :param dict preprocess_config: Preprocessing config
    :return dict required_params: Required parameters
    """
    input_dir = preprocess_config['input_dir']
    output_dir = preprocess_config['output_dir']
    slice_ids = -1
    if 'slice_ids' in preprocess_config:
        slice_ids = preprocess_config['slice_ids']

    time_ids = -1
    if 'time_ids' in preprocess_config:
        time_ids = preprocess_config['time_ids']

    pos_ids = -1
    if 'pos_ids' in preprocess_config:
        pos_ids = preprocess_config['pos_ids']

    channel_ids = -1
    if 'channel_ids' in preprocess_config:
        channel_ids = preprocess_config['channel_ids']

    uniform_struct = True
    if 'uniform_struct' in preprocess_config:
        uniform_struct = preprocess_config['uniform_struct']

    int2str_len = 3
    if 'int2str_len' in preprocess_config:
        int2str_len = preprocess_config['int2str_len']

    num_workers = 4
    if 'num_workers' in preprocess_config:
        num_workers = preprocess_config['num_workers']

    normalize_im = 'stack'
    normalize_channels = -1
    if 'normalize' in preprocess_config:
        if 'normalize_im' in preprocess_config['normalize']:
            normalize_im = preprocess_config['normalize']['normalize_im']
        if 'normalize_channels' in preprocess_config['normalize']:
            normalize_channels = preprocess_config['normalize']['normalize_channels']
            if isinstance(channel_ids, list):
                assert len(channel_ids) == len(normalize_channels), \
                    "Nbr channels {} and normalization {} mismatch".format(
                        channel_ids,
                        normalize_channels,
                    )

    required_params = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'slice_ids': slice_ids,
        'time_ids': time_ids,
        'pos_ids': pos_ids,
        'channel_ids': channel_ids,
        'uniform_struct': uniform_struct,
        'int2strlen': int2str_len,
        'normalize_channels': normalize_channels,
        'num_workers': num_workers,
        'normalize_im': normalize_im,
    }
    return required_params


def flat_field_correct(required_params, block_size):
    """Estimate flat_field_images

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int block_size: Specify block size if different from default (32 pixels)
    :return str flat_field_dir: full path of dir with flat field correction
     images
    """

    flat_field_inst = FlatFieldEstimator2D(
        input_dir=required_params['input_dir'],
        output_dir=required_params['output_dir'],
        channel_ids=required_params['channel_ids'],
        slice_ids=required_params['slice_ids'],
        block_size=block_size,
    )
    flat_field_inst.estimate_flat_field()
    flat_field_dir = flat_field_inst.get_flat_field_dir()
    return flat_field_dir


def resize_images(required_params,
                  scale_factor,
                  num_slices_subvolume,
                  resize_3d,
                  flat_field_dir):
    """Resample images first

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list scale_factor: scale factor for each dimension
    :param int num_slices_subvolume: num of slices to be included in each
     volume. If -1, include all slices in slice_ids
    :param bool resize_3d: indicator for resize 2d or 3d
    :param str flat_field_dir: dir with flat field correction images. if None,
     does not correct for illumination changes
    :return:
     str resize_dir: dir with resized images
     int, list: slice_ids corrected for gaps due to 3d. For ex.
      slice_ids=[0,1,...8] and num_slices_subvolume=3, returned
      slice_ids=[0, 2, 4, 6]
    """

    if isinstance(scale_factor, list):
        scale_factor = np.array(scale_factor)

    if np.all(scale_factor == 1):
        return required_params['input_dir'], required_params['slice_ids']

    resize_inst = ImageResizer(
        input_dir=required_params['input_dir'],
        output_dir=required_params['output_dir'],
        scale_factor=scale_factor,
        channel_ids=required_params['channel_ids'],
        time_ids=required_params['time_ids'],
        slice_ids=required_params['slice_ids'],
        pos_ids=required_params['pos_ids'],
        int2str_len=required_params['int2strlen'],
        num_workers=required_params['num_workers'],
        flat_field_dir=flat_field_dir
    )

    if resize_3d:
        # return slice_ids from resize_volumes to deal with slice_ids=-1
        slice_ids = resize_inst.resize_volumes(num_slices_subvolume)
    else:
        resize_inst.resize_frames()
        slice_ids = required_params['slice_ids']
    resize_dir = resize_inst.get_resize_dir()
    return resize_dir, slice_ids


def generate_masks(required_params,
                   mask_from_channel,
                   flat_field_dir,
                   str_elem_radius,
                   mask_type,
                   mask_channel,
                   mask_ext,
                   mask_dir=None,
                   ):
    """
    Generate masks per image or volume

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
        channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list mask_from_channel: generate masks from sum of these
        channels
    :param str/None flat_field_dir: dir with flat field correction images
    :param int str_elem_radius: structuring element size for morphological
        opening
    :param str mask_type: string to map to masking function. otsu or unimodal
        or borders_weight_loss_map
    :param int/None mask_channel: channel num assigned to mask channel. I
    :param str mask_ext: 'npy' or 'png'. Save the mask as uint8 PNG or
         NPY files
    :param str/None mask_dir: If creating weight maps from mask directory,
        specify mask dir
    :param bool normalize_im: indicator to normalize image based on z-score or not
    :return str mask_dir: Directory with created masks
    :return int mask_channel: Channel number assigned to masks
    """
    assert mask_type in {'otsu', 'unimodal', 'dataset otsu', 'borders_weight_loss_map'},\
        "Supported mask types: 'otsu', 'unimodal', 'dataset otsu', 'borders_weight_loss_map'" +\
        ", not {}".format(mask_type)

    # If generating weights map, input dir is the mask dir
    input_dir = required_params['input_dir']
    if mask_dir is not None:
        input_dir = mask_dir
    # Instantiate channel to mask processor
    mask_processor_inst = MaskProcessor(
        input_dir=input_dir,
        output_dir=required_params['output_dir'],
        channel_ids=mask_from_channel,
        flat_field_dir=flat_field_dir,
        time_ids=required_params['time_ids'],
        slice_ids=required_params['slice_ids'],
        pos_ids=required_params['pos_ids'],
        int2str_len=required_params['int2strlen'],
        uniform_struct=required_params['uniform_struct'],
        num_workers=required_params['num_workers'],
        mask_type=mask_type,
        mask_channel=mask_channel,
        mask_ext=mask_ext,
    )

    correct_flat_field = False
    if flat_field_dir is not None:
        correct_flat_field = True
    mask_processor_inst.generate_masks(
        correct_flat_field=correct_flat_field,
        str_elem_radius=str_elem_radius,
    )
    mask_dir = mask_processor_inst.get_mask_dir()
    mask_channel = mask_processor_inst.get_mask_channel()
    return mask_dir, mask_channel


def generate_zscore_table(required_params,
                          norm_dict,
                          mask_dir):
    """
    Compute z-score parameters and update frames_metadata based on the normalize_im
    :param dict required_params: Required preprocessing parameters
    :param dict norm_dict: Normalization scheme (preprocess_config['normalization'])
    :param str mask_dir: Directory containing masks
    """
    assert 'min_fraction' in norm_dict, \
        "normalization part of config must contain min_fraction"
    frames_metadata = aux_utils.read_meta(required_params['input_dir'])
    ints_metadata = aux_utils.read_meta(
        required_params['input_dir'],
        meta_fname='intensity_meta.csv',
    )
    mask_metadata = aux_utils.read_meta(mask_dir)
    cols_to_merge = ints_metadata.columns[ints_metadata.columns != 'fg_frac']
    ints_metadata = pd.merge(
        ints_metadata[cols_to_merge],
        mask_metadata[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
        how='left',
        on=['pos_idx', 'time_idx', 'slice_idx'],
    )
    _, ints_metadata = meta_utils.compute_zscore_params(
        frames_meta=frames_metadata,
        ints_meta=ints_metadata,
        input_dir=required_params['input_dir'],
        normalize_im=required_params['normalize_im'],
        min_fraction=norm_dict['min_fraction'],
    )
    ints_metadata.to_csv(
        os.path.join(required_params['input_dir'], 'intensity_meta.csv'),
        sep=',',
    )


def tile_images(required_params,
                tile_dict,
                resize_flag,
                flat_field_dir,
                tiles_exist=False,
                ):
    """
    Tile images.

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
     channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param dict tile_dict: dict with tiling related keys: tile_size, step_size,
     image_format, depths, min_fraction. Optional: mask_channel, mask_dir,
     mask_depth, tile_3d
    :param bool resize_flag: indicator if resize related params in preprocess_config
     passed to pre_process()
    :param str/None flat_field_dir: dir with flat field correction images
    :param bool tiles_exist: If tiling weights after other channels, make sure
     previous tiles are not erased
    :return str tile_dir: dir with tiled images
    """
    # Check tile args
    tile_3d = False
    if 'tile_3d' in tile_dict:
        tile_3d = tile_dict['tile_3d']
    tile_dict['tile_3d'] = tile_3d
    hist_clip_limits = None
    if 'hist_clip_limits' in tile_dict:
        hist_clip_limits = tile_dict['hist_clip_limits']
    # Set default minimum fraction to 0
    min_fraction = 0.
    if 'min_fraction' in tile_dict:
        min_fraction = tile_dict['min_fraction']
    # setup tiling keyword arguments
    kwargs = {'input_dir': required_params['input_dir'],
              'output_dir': required_params['output_dir'],
              'normalize_channels': required_params["normalize_channels"],
              'tile_size': tile_dict['tile_size'],
              'step_size': tile_dict['step_size'],
              'depths': tile_dict['depths'],
              'time_ids': required_params['time_ids'],
              'channel_ids': required_params['channel_ids'],
              'slice_ids': required_params['slice_ids'],
              'pos_ids': required_params['pos_ids'],
              'hist_clip_limits': hist_clip_limits,
              'flat_field_dir': flat_field_dir,
              'num_workers': required_params['num_workers'],
              'tile_3d': tile_3d,
              'int2str_len': required_params['int2strlen'],
              'min_fraction': min_fraction,
              'normalize_im': required_params['normalize_im'],
              'tiles_exist': tiles_exist,
              }

    if required_params['uniform_struct']:
        if tile_3d:
            if resize_flag:
                warnings.warn(
                    'If resize_3d was used, slice_idx corresponds to start'
                    'slice of each volume.If slice_ids=-1, the slice_ids'
                    'will be read from frames_meta.csv. Assuming slice_ids'
                    'provided here is fixed for these gaps.', Warning)
            tile_inst = ImageTilerUniform3D(**kwargs)
        else:
            tile_inst = ImageTilerUniform(**kwargs)
    else:
        # currently not supported but should be easy to extend
        tile_inst = ImageTilerNonUniform(**kwargs)

    tile_dir = tile_inst.get_tile_dir()

    # retain tiles with a minimum amount of foreground
    if 'mask_dir' in tile_dict:
        mask_channel = tile_dict['mask_channel']
        mask_dir = tile_dict['mask_dir']
        mask_depth = 1
        if 'mask_depth' in tile_dict:
            mask_depth = tile_dict['mask_depth']

        tile_inst.tile_mask_stack(
            mask_dir=mask_dir,
            mask_channel=mask_channel,
            mask_depth=mask_depth,
        )
    else:
        # retain all tiles
        tile_inst.tile_stack()

    return tile_dir


def save_config(cur_config, runtime):
    """
    Save the current config (cur_config) or append to existing config.

    :param dict cur_config: Current config
    :param float runtime: Run time for preprocessing
    """

    # Read preprocessing.json if exists in input dir
    parent_dir = cur_config['input_dir'].split(os.sep)[:-1]
    parent_dir = os.sep.join(parent_dir)

    prior_config_fname = os.path.join(parent_dir, 'preprocessing_info.json')
    prior_preprocess_config = None
    if os.path.exists(prior_config_fname):
        prior_preprocess_config = aux_utils.read_json(prior_config_fname)

    meta_path = os.path.join(cur_config['output_dir'],
                             'preprocessing_info.json')

    processing_info = [{'processing_time': runtime,
                        'config': cur_config}]
    if prior_preprocess_config is not None:
        prior_preprocess_config.append(processing_info[0])
        processing_info = prior_preprocess_config
    os.makedirs(cur_config['output_dir'], exist_ok=True)
    aux_utils.write_json(processing_info, meta_path)


def pre_process(preprocess_config):
    """
    Preprocess data. Possible options are:

    correct_flat_field: Perform flatfield correction (2D only currently)
    resample: Resize 2D images (xy-plane) according to a scale factor,
        e.g. to match resolution in z. Resize 3d images
    create_masks: Generate binary masks from given input channels
    do_tiling: Split frames (stacked frames if generating 3D tiles) into
    smaller tiles with tile_size and step_size.

    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict preprocess_config: dict with key options:
    [input_dir, output_dir, slice_ids, time_ids, pos_ids
    correct_flat_field, use_masks, masks, tile_stack, tile]
    :param dict required_params: dict with commom params for all tasks
    :raises AssertionError: If 'masks' in preprocess_config contains both channels
     and mask_dir (the former is for generating masks from a channel)
    """
    time_start = time.time()
    required_params = get_required_params(preprocess_config)

    # ------------------Check or create metadata---------------------
    try:
        # Check if metadata is present
        aux_utils.read_meta(required_params['input_dir'])
    except AssertionError as e:
        print(e, "Generating metadata.")
        order = 'cztp'
        name_parser = 'parse_sms_name'
        if 'metadata' in preprocess_config:
            if 'order' in preprocess_config['metadata']:
                order = preprocess_config['metadata']['order']
            if 'name_parser' in preprocess_config['metadata']:
                name_parser = preprocess_config['metadata']['name_parser']
        # Create metadata from file names instead
        meta_utils.frames_meta_generator(
            input_dir=required_params['input_dir'],
            order=order,
            name_parser=name_parser,
        )

    # -----------------Estimate flat field images--------------------
    flat_field_dir = None
    if 'flat_field' in preprocess_config:
        if 'estimate' in preprocess_config['flat_field'] and \
                preprocess_config['flat_field']['estimate']:
            assert 'flat_field_dir' not in preprocess_config['flat_field'], \
                'estimate_flat_field or use images in flat_field_dir.'
            block_size = None
            if 'block_size' in preprocess_config['flat_field']:
                block_size = preprocess_config['flat_field']['block_size']
            flat_field_dir = flat_field_correct(required_params, block_size)
            preprocess_config['flat_field']['flat_field_dir'] = flat_field_dir

        elif 'correct' in preprocess_config['flat_field'] and \
                preprocess_config['flat_field']['correct']:
            flat_field_dir = preprocess_config['flat_field']['flat_field_dir']

    # -------Compute intensities for flatfield corrected images-------
    if required_params['normalize_im'] in ['dataset', 'volume', 'slice']:
        block_size = None
        if 'block_size' in preprocess_config['metadata']:
            block_size = preprocess_config['metadata']['block_size']
        meta_utils.ints_meta_generator(
            input_dir=required_params['input_dir'],
            num_workers=required_params['num_workers'],
            block_size=block_size,
            flat_field_dir=flat_field_dir,
            channel_ids=required_params['channel_ids'],
        )

    # -------------------------Resize images--------------------------
    if 'resize' in preprocess_config:
        scale_factor = preprocess_config['resize']['scale_factor']
        num_slices_subvolume = -1
        if 'num_slices_subvolume' in preprocess_config['resize']:
            num_slices_subvolume = \
                preprocess_config['resize']['num_slices_subvolume']

        resize_dir, slice_ids = resize_images(
            required_params,
            scale_factor,
            num_slices_subvolume,
            preprocess_config['resize']['resize_3d'],
            flat_field_dir,
        )
        # the images are resized after flat field correction
        flat_field_dir = None
        preprocess_config['resize']['resize_dir'] = resize_dir
        required_params['input_dir'] = resize_dir
        required_params['slice_ids'] = slice_ids

    # ------------------------Generate masks-------------------------
    mask_dir = None
    mask_channel = None
    if 'masks' in preprocess_config:
        if 'channels' in preprocess_config['masks']:
            # Generate masks from channel
            assert 'mask_dir' not in preprocess_config['masks'], \
                "Don't specify a mask_dir if generating masks from channel"
            mask_from_channel = preprocess_config['masks']['channels']
            str_elem_radius = 5
            if 'str_elem_radius' in preprocess_config['masks']:
                str_elem_radius = preprocess_config['masks']['str_elem_radius']
            mask_type = 'otsu'
            if 'mask_type' in preprocess_config['masks']:
                mask_type = preprocess_config['masks']['mask_type']
            mask_ext = '.png'
            if 'mask_ext' in preprocess_config['masks']:
                mask_ext = preprocess_config['masks']['mask_ext']

            mask_dir, mask_channel = generate_masks(
                required_params=required_params,
                mask_from_channel=mask_from_channel,
                flat_field_dir=flat_field_dir,
                str_elem_radius=str_elem_radius,
                mask_type=mask_type,
                mask_channel=None,
                mask_ext=mask_ext,
            )
        elif 'mask_dir' in preprocess_config['masks']:
            assert 'channels' not in preprocess_config['masks'], \
                "Don't specify channels to mask if using pre-generated masks"
            mask_dir = preprocess_config['masks']['mask_dir']
            # Get preexisting masks from directory and match to input dir
            mask_meta = meta_utils.mask_meta_generator(
                mask_dir,
            )
            frames_meta = aux_utils.read_meta(required_params['input_dir'])
            # Automatically assign existing masks the next available channel number
            mask_channel = (frames_meta['channel_idx'].max() + 1)
            mask_meta['channel_idx'] = mask_channel
            # Write metadata
            mask_meta_fname = os.path.join(mask_dir, 'frames_meta.csv')
            mask_meta.to_csv(mask_meta_fname, sep=",")
            # mask_channel = preprocess_utils.validate_mask_meta(
            #     mask_dir=mask_dir,
            #     input_dir=required_params['input_dir'],
            #     csv_name=mask_meta_fname,
            #     mask_channel=mask_channel,
            # )
        else:
            raise ValueError("If using masks, specify either mask_channel",
                             "or mask_dir.")
        preprocess_config['masks']['mask_dir'] = mask_dir
        preprocess_config['masks']['mask_channel'] = mask_channel

    # ---------------------Generate z score table---------------------
    if required_params['normalize_im'] in ['dataset', 'volume', 'slice']:
        assert mask_dir is not None, \
            "'dataset', 'volume', 'slice' normalization requires masks"
        generate_zscore_table(
            required_params,
            preprocess_config['normalize'],
            mask_dir,
        )

    # ----------------------Generate weight map-----------------------
    weights_dir = None
    weights_channel = None
    if 'make_weight_map' in preprocess_config and preprocess_config['make_weight_map']:
        # Must have mask dir and mask channel defined to generate weight map
        assert mask_dir is not None,\
            "Must have mask dir to generate weights"
        assert mask_channel is not None,\
            "Must have mask channel to generate weights"
        mask_type = 'borders_weight_loss_map'
        # Mask channel should be highest channel value in dataset at this point
        weights_channel = mask_channel + 1
        # Generate weights
        weights_dir, _ = generate_masks(
            required_params=required_params,
            mask_from_channel=mask_channel,
            flat_field_dir=None,
            str_elem_radius=5,
            mask_type=mask_type,
            mask_channel=weights_channel,
            mask_ext='.npy',
            mask_dir=mask_dir,
        )
        preprocess_config['weights'] = {
            'weights_dir': weights_dir,
            'weights_channel': weights_channel,
        }

    # ------------Tile images, targets, masks, weight maps------------
    if 'tile' in preprocess_config:
        resize_flag = False
        if 'resize' not in preprocess_config:
            resize_flag = True
        # Always tile masks if they exist
        if mask_dir is not None:
            if 'mask_dir' not in preprocess_config['tile']:
                preprocess_config['tile']['mask_dir'] = mask_dir
            if 'mask_channel' not in preprocess_config['tile']:
                preprocess_config['tile']['mask_channel'] = mask_channel
        tile_dir = tile_images(
            required_params=required_params,
            tile_dict=preprocess_config['tile'],
            resize_flag=resize_flag,
            flat_field_dir=flat_field_dir,
        )
        # Tile weight maps as well if they exist
        if 'weights' in preprocess_config:
            weight_params = required_params.copy()
            weight_params["input_dir"] = weights_dir
            weight_params["channel_ids"] = [weights_channel]
            weight_tile_config = preprocess_config['tile'].copy()
            weight_params['normalize_channels'] = [False]
            # Weights depth should be the same as mask depth
            weight_tile_config['depths'] = 1
            weight_tile_config.pop('mask_dir')
            if 'mask_depth' in preprocess_config['tile']:
                weight_tile_config['depths'] = [preprocess_config['tile']['mask_depth']]
            tile_dir = tile_images(
                required_params=weight_params,
                tile_dict=weight_tile_config,
                resize_flag=resize_flag,
                flat_field_dir=None,
                tiles_exist=True,
            )
        preprocess_config['tile']['tile_dir'] = tile_dir

    # Write in/out/mask/tile paths and config to json in output directory
    time_el = time.time() - time_start
    return preprocess_config, time_el


if __name__ == '__main__':
    args = parse_args()
    preprocess_config = aux_utils.read_config(args.config)
    preprocess_config, runtime = pre_process(preprocess_config)
    save_config(preprocess_config, runtime)
