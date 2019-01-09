"""Script for preprocessing stack"""

import argparse
import os
import time

from micro_dl.preprocessing.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.preprocessing.generate_masks import MaskProcessor
from micro_dl.preprocessing.tile_uniform_images import ImageTilerUniform
from micro_dl.preprocessing.tile_nonuniform_images import \
    ImageTilerNonUniform
import micro_dl.utils.aux_utils as aux_utils


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


def pre_process(pp_config):
    """
    Preprocess data. Possible options are:
    correct_flat_field: Perform flatfield correction (2D only currently)
    create_masks: Generate binary masks from given input channels
    do_tiling: Split frames (stacked frames if generating 3D tiles) into
    smaller tiles with tile_size and step_size.
    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict pp_config: dict with key options:
    [input_dir, output_dir, slice_ids, time_ids, pos_ids
    correct_flat_field, use_masks, masks, tile_stack, tile]
    """
    input_dir = pp_config['input_dir']
    output_dir = pp_config['output_dir']

    slice_ids = -1
    if 'slice_ids' in pp_config:
        slice_ids = pp_config['slice_ids']
    time_ids = -1
    if 'time_ids' in pp_config:
        time_ids = pp_config['time_ids']

    uniform_struct = False
    if 'uniform_struct' in pp_config:
        uniform_struct = pp_config['uniform_structure']

    int2str_len = 3
    if 'int2str_len' in pp_config:
        int2str_len = pp_config['int2str_len']

    pos_ids = -1
    if 'pos_ids' in pp_config:
        pos_ids = pp_config['pos_ids']

    num_workers = 4
    if 'num_workers' in pp_config['tile']:
        num_workers = pp_config['num_workers']

    # estimate flat_field images
    correct_flat_field = True if pp_config['correct_flat_field'] else False
    flat_field_dir = None
    if correct_flat_field:
        flat_field_inst = FlatFieldEstimator2D(
            input_dir=input_dir,
            output_dir=output_dir,
            slice_ids=slice_ids,
        )
        flat_field_inst.estimate_flat_field()
        flat_field_dir = flat_field_inst.get_flat_field_dir()

    # Generate masks
    mask_dir = None
    mask_channel = None
    if pp_config['create_masks']:
        mask_processor_inst = MaskProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            channel_ids=pp_config['masks']['channels'],
            flat_field_dir=flat_field_dir,
            time_ids=time_ids,
            slice_ids=slice_ids,
            pos_ids=pos_ids,
            int2str_len=int2str_len,
            uniform_struct=uniform_struct,
            num_workers=num_workers
        )
        str_elem_radius = 5
        if 'str_elem_radius' in pp_config['masks']:
            str_elem_radius = pp_config['masks']['str_elem_radius']
        mask_processor_inst.generate_masks(
            correct_flat_field=correct_flat_field,
            str_elem_radius=str_elem_radius,
        )
        mask_dir = mask_processor_inst.get_mask_dir()
        mask_channel = mask_processor_inst.get_mask_channel()

    # Tile frames
    tile_dir = None
    if pp_config['do_tiling']:
        channel_ids = -1
        if 'channels' in pp_config['tile']:
            channel_ids = pp_config['tile']['channels']

        if uniform_struct:
            tile_inst = ImageTilerUniform(input_dir=input_dir,
                                          output_dir=output_dir,
                                          tile_dict=pp_config['tile'],
                                          time_ids=time_ids,
                                          slice_ids=slice_ids,
                                          channel_ids=channel_ids,
                                          pos_ids=pos_ids,
                                          flat_field_dir=flat_field_dir,
                                          num_workers=num_workers,
                                          int2str_len=int2str_len)
        else:
            tile_inst = ImageTilerNonUniform(input_dir=input_dir,
                                             output_dir=output_dir,
                                             tile_dict=pp_config['tile'],
                                             time_ids=time_ids,
                                             slice_ids=slice_ids,
                                             channel_ids=channel_ids,
                                             pos_ids=pos_ids,
                                             flat_field_dir=flat_field_dir,
                                             num_workers=num_workers,
                                             int2str_len=int2str_len)

        tile_dir = tile_inst.get_tile_dir()
        # and want to tile only the ones with a minimum amount of foreground
        if 'min_fraction' in pp_config['tile'] and pp_config['create_masks']:
            tile_inst.tile_mask_stack(
                min_fraction=pp_config['tile']['min_fraction'],
                mask_dir=mask_dir,
                mask_channel=mask_channel,
            )
        else:
            tile_inst.tile_stack()

    # Write in/out/mask/tile paths and config to json in output directory
    processing_info = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "flat_field_dir": flat_field_dir,
        "mask_dir": mask_dir,
        "mask_channel": mask_channel,
        "tile_dir": tile_dir,
        "config": pp_config,
    }
    meta_path = os.path.join(output_dir, "preprocessing_info.json")
    aux_utils.write_json(processing_info, meta_path)


if __name__ == '__main__':
    args = parse_args()
    config = aux_utils.read_config(args.config)
    pre_process(config)
