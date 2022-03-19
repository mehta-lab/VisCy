#!/usr/bin/python

import argparse
import micro_dl.utils.meta_utils as meta_utils


def parse_args():
    """
    Parse command line arguments for directory containing files.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help="Path to folder containing all 2D image frames",
    )
    parser.add_argument(
        '--order',
        type=str,
        default="cztp",
        help="The order in which indices are listed in the image file name",
    )
    parser.add_argument(
        '--name_parser',
        type=str,
        default="parse_idx_from_name",
        help="The function in aux_utils that will parse the file name for indices",
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help="number of workers for multiprocessing",
    )
    parser.add_argument(
        '--normalize_im',
        type=str,
        default='stack',
        help="normalization scheme for images",
    )
    return parser.parse_args()


def main(parsed_args):
    meta_utils.frames_meta_generator(parsed_args.input,
                                     parsed_args.order,
                                     parsed_args.name_parser,
                                     )
    if parsed_args.normalize_im in ['dataset', 'volume', 'slice']:
        meta_utils.ints_meta_generator(parsed_args.input,
                                       parsed_args.order,
                                       parsed_args.name_parser,
                                       parsed_args.num_workers,
                                       )


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)

