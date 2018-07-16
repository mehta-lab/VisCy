#!/usr/bin/env python
"""Create masks for segmentation"""
import argparse
import os
import pandas as pd
import pickle

import micro_dl.input.gen_masks_seg as gen_mask


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='specify the input dir with full path')
    parser.add_argument('--channel_id', nargs='*', type=int,
                        help='specify the input channel ids')
    parser.add_argument('--mask_id', type=int,
                        help='specify the channel id for masks')
    parser.add_argument('--output_dir', type=str,
                        help='specify the output dir for tiled mask with full path'
    )
    # group = parser.add_mutually_exclusive_group(required=True)
    #  Hmm.. how to specify a bunch of params to a group
    #  mutually_exclusive_group is not ideal here
    #  here tile_size & step_size belong to one group vs tile_index_fname in other
    parser.add_argument('--tile_size', type=list, default=[256, 256],
                       help='specify tile size along each dimension as a list')
    parser.add_argument('--step_size', type=list, default=[256, 256],
                       help='specify step size along each dimension as a list')

    parser.add_argument('--tile_index_fname', type=str,
                        help='path to checkpoint file/directory')

    args = parser.parse_args()
    return args


def create_masks(args):

    # Start mask instance and generate masks
    mask_inst = gen_mask.MaskCreator(input_dir=args.input_dir,
                                     input_channel_id=args.channel_id,
                                     output_dir=args.output_dir,
                                     output_channel_id=args.mask_id,
                                     correct_flat_field=False)

    mask_inst.create_masks_for_stack()
    # Tile mask images and write them to directory containing other tiled data
    # This assumes there's only one mask ID
    timepoints = mask_inst.timepoint_id
    for tp in timepoints:
        mask_dir = os.path.join(
            args.input_dir,
            'timepoint_{}'.format(tp),
            'channel_{}'.format(args.mask_id),
        )
        meta_info = mask_inst.tile_mask_stack(
            input_mask_dir=mask_dir,
            tile_index_fname=args.tile_index_fname,
            tile_size=args.tile_size,
            step_size=args.step_size,
        )



if __name__ == '__main__':
    args = parse_args()
    create_masks(args)
