"""Script for preprocessing stack"""

import argparse
import os

from micro_dl.input.estimate_flat_field import FlatFieldEstimator2D
from micro_dl.input.generate_masks import MaskProcessor
from micro_dl.input.tile_stack import ImageStackTiler
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
    use_masks: Generate binary masks from given input channels
    tile_stack: Split frames into smaller tiles with tile_size and step_size
    This script will preprocess your dataset, save tiles and associated
    metadata. Then in the train_script, a dataframe for training data
    will be assembled based on the inputs and target you specify.

    :param dict pp_config: dict with key options:
    [input_dir, output_dir, slice_ids, time_ids,
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
    if pp_config['use_masks']:
        mask_processor_inst = MaskProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            channel_ids=pp_config['masks']['channels'],
            flat_field_dir=flat_field_dir,
            time_ids=time_ids,
            slice_ids=slice_ids,
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
    tile_mask_dir = None
    if pp_config['tile_stack']:
        tile_size = pp_config['tile']['tile_size']
        step_size = pp_config['tile']['step_size']
        isotropic = False
        if 'isotropic' in pp_config['tile']:
            isotropic = pp_config['tile']['isotropic']
        channel_ids = -1
        if "channels" in pp_config['tile']:
            channel_ids = pp_config['tile']['channels']
        hist_clip_limits = None
        if 'hist_clip_limits' in pp_config['tile']:
            hist_clip_limits = pp_config['tile']['hist_clip_limits']

        tile_inst = ImageStackTiler(
            input_dir=input_dir,
            output_dir=output_dir,
            tile_size=tile_size,
            step_size=step_size,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
            hist_clip_limits=hist_clip_limits,
            flat_field_dir=flat_field_dir,
            isotropic=isotropic,
        )
        tile_dir = tile_inst.get_tile_dir()
        # If you're using min fraction, it assumes you've generated masks
        # and want to tile only the ones with a minimum amount of foreground
        if 'min_fraction' in pp_config['tile'] and pp_config['use_masks']:
            save_tiled_masks = None
            if 'save_tiled_masks' in pp_config['tile']:
                save_tiled_masks = pp_config['tile']['save_tiled_masks']

            tile_inst.tile_mask_stack(
                min_fraction=pp_config['tile']['min_fraction'],
                mask_dir=mask_dir,
                mask_channel=mask_channel,
                save_tiled_masks=save_tiled_masks,
            )
            tile_mask_dir = tile_inst.get_tile_mask_dir()
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
        "tile_mask_dir": tile_mask_dir,
        "config": pp_config,
    }
    meta_path = os.path.join(output_dir, "preprocessing_info.json")
    aux_utils.write_json(processing_info, meta_path)


if __name__ == '__main__':
    args = parse_args()
    config = aux_utils.read_config(args.config)
    pre_process(config)
