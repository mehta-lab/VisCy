"""Script for preprocessing stack"""

import argparse
import os
import yaml

from micro_dl.input import MaskProcessor, ImageStackTiler
from micro_dl.utils.aux_utils import import_class


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='path to yaml configuration file'
    )
    args = parser.parse_args()
    return args


def read_config(config_fname):
    """Read the config file in yml format

    TODO: validate config!

    :param str config_fname: fname of config yaml with its full path
    :return:
    """

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    return config


def pre_process(pp_config):
    """Split and crop volumes from lif data

    :param dict pp_config: dict with keys [input_fname, base_output_dir,
     split_volumes, crop_volumes]
    """

    # split images
    if pp_config['split_volumes']:
        stack_splitter_cls = pp_config['splitter_class']
        stack_splitter_cls = import_class('input.split_lif_stack', stack_splitter_cls)
        stack_splitter = stack_splitter_cls(
            lif_fname=pp_config['input_fname'],
            base_output_dir=pp_config['base_output_dir'],
            verbose=pp_config['verbose']
        )
        stack_splitter.save_images()

    split_dir = os.path.join(pp_config['base_output_dir'],
                             'split_images')

    # estimate flat_filed_image
    if pp_config['correct_flat_field']:
        correct_flat_field = True
    else:
        correct_flat_field = False

    if 'focal_plane_idx' in pp_config:
        focal_plane_idx = pp_config['focal_plane_idx']
    else:
        focal_plane_idx = None

    if correct_flat_field:
        flat_field_estimator_cls = pp_config['flat_field_class']
        flat_field_estimator_cls = import_class('input.estimate_flat_field',
                                                flat_field_estimator_cls)
        flat_field_estimator = flat_field_estimator_cls(split_dir)
        flat_field_estimator.estimate_flat_field(focal_plane_idx)

    # generate masks
    if pp_config['use_masks']:
        if 'timepoints' in pp_config:
            mask_processor_inst = MaskProcessor(
                split_dir, pp_config['masks']['mask_channels'],
                pp_config['timepoints']
            )
        else:
            mask_processor_inst = MaskProcessor(
                split_dir, pp_config['masks']['mask_channels']
            )

        if 'str_elem_radius' in pp_config['masks']:
            mask_processor_inst.generate_masks(
                focal_plane_idx=focal_plane_idx,
                correct_flat_field=correct_flat_field,
                str_elem_radius=pp_config['masks']['str_elem_radius']
            )
        else:
            mask_processor_inst.generate_masks(
                focal_plane_idx=focal_plane_idx,
                correct_flat_field=correct_flat_field
            )

    # tile stack
    if pp_config['tile_stack']:
        if 'isotropic' in pp_config['tile']:
            isotropic = pp_config['tile']['isotropic']
        else:
            isotropic = False

        cropper_inst = ImageStackTiler(pp_config['base_output_dir'],
                                       pp_config['tile']['tile_size'],
                                       pp_config['tile']['step_size'],
                                       correct_flat_field=correct_flat_field,
                                       isotropic=isotropic)
        if 'hist_clip_limits' in pp_config['tile']:
            hist_clip_limits = pp_config['tile']['hist_clip_limits']
        else:
            hist_clip_limits = None

        if 'min_fraction' in pp_config['tile']:
            cropper_inst.tile_stack_with_vf_constraint(
                mask_channels=pp_config['masks']['mask_channels'],
                min_fraction=pp_config['tile']['min_fraction'],
                save_cropped_masks=pp_config['tile']['save_cropped_masks'],
                isotropic=isotropic, focal_plane_idx=focal_plane_idx,
                hist_clip_limits=hist_clip_limits
            )
        else:
            cropper_inst.tile_stack(focal_plane_idx=focal_plane_idx,
                                    hist_clip_limits=hist_clip_limits)


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    pre_process(config)
