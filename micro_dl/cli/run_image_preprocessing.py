#!/usr/bin/python

import argparse
import os

import micro_dl.input.image_validator as image_validator
import micro_dl.input.tile_stack as tile_stack


def parse_args():
    """
    Parse command line arguments for data preprocessing
    prior to training U-Net model

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path to folder containing all time/channel subfolders")
    parser.add_argument('-o', '--output', type=str,
                        help=("Path to base directory where preprocessed data ",
                              "and info csv file will be written"))
    parser.add_argument('--tile_size', nargs='*', type=int, default=[256, 256],
                        help="Split images into tiles of size [height, width (depth)]")
    parser.add_argument('--step_size', nargs='*', type=int, default=[256, 256],
                        help="Split images into patches of size [height, width]")
    parser.add_argument('-v', '--verbose', type=int, default=10)
    return parser.parse_args()


def preprocess(args):
    """
    Split, crop volumes and flatfield correct images in input and target
    directories. Writes output as npy files for faster reading while training.

    :param list args:    parsed args containing
        str input_dir:   path to input main directory containing subfolders
                         for timepoints containing subfolders for channels
        str output_dir:  base path where processed data will be written
        list data_split: fractions of train, validation and test (must sum to 1)
        list tile_size:  shape of image tiles
        list step_size:  shape of step size when making image tiles
        int verbose:     verbosity of preprocess
    """
    # Instantiate preprocessor
    meta_name = 'image_volumes_info.csv'
    validator = image_validator.ImageValidator(
        input_dir=args.input,
        meta_name=meta_name,
        verbose=args.verbose,
    )
    # Check that all indices are unique and write metadata info csv
    # in input base directory
    validator.folder_validator()
    meta_path = os.path.join(args.input, meta_name)

    # TODO (Jenny): Add handling of masks and flatfield correction

    # Tile all images
    # TODO (Jenny): Not dealing with 3D/isotropy for now
    base_output_dir = args.output
    # Create output directory if it doesn't exist already
    os.makedirs(base_output_dir, exist_ok=True)
    isotropic = False
    image_tiler = tile_stack.ImageStackTiler(
        base_output_dir=base_output_dir,
        tile_size=args.tile_size,
        step_size=args.step_size,
        correct_flat_field=False,
        isotropic=isotropic,
        meta_path=meta_path,
    )

    image_tiler.tile_stack(focal_plane_idx=0,
                           hist_clip_limits=None)


if __name__ == '__main__':
    args = parse_args()
    preprocess(args)
