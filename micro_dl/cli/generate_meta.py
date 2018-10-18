#!/usr/bin/python

import argparse
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils


def parse_args():
    """
    Parse command line arguments for directory containing files.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path to folder containing all 2D image frames")
    return parser.parse_args()


def meta_generator(args):
    """
    Generate metadata from file names for preprocessing.
    Will write found data in frames_metadata.csv in input directory.
    Assumed file naming convention is:
    dir_name
    |
    |- im_c***_t***_p***_z***.png
    |- im_c***_t***_p***_z***.png

    c is channel
    t is time
    p is position (FOV)
    z is slice in stack (z)

    :param list args:    parsed args containing
        str input_dir:   path to input directory containing images
    """
    meta_name = 'frames_meta.csv'
    df_names = ["channel_idx",
                "slice_idx",
                "time_idx",
                "channel_name",
                "file_name",
                "pos_idx"]

    # Get all image names
    im_names = aux_utils.get_sorted_names(args.input)
    # Create empty dataframe
    frames_meta = pd.DataFrame(
        index=range(len(im_names)),
        columns=df_names,
    )
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        frames_meta.loc[i] = aux_utils.get_ids_from_imname(
            im_name=im_names[i],
            df_names=df_names,
        )
    # Write metadata
    meta_filename = os.path.join(args.input, meta_name)
    frames_meta.to_csv(meta_filename, sep=",")


if __name__ == '__main__':
    args = parse_args()
    meta_generator(args)