'''
This script corrects for mis-mounted absolute paths in data. You

Metadata of preprocessed data will have absolute paths referring to the data location in the form of
whatever mount was used to acces the data.
For example, metadata might refer to a source image for a tile as '/home/soorya.pradeep/CompMicro/...'
This means that unless you have access to user Soorya Pradeep, the preprocessing script will break
'''

import pandas as pd
import argparse

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mounting',
        type=str,
        help="desired mounting to '/CompMicro' directory",
    )
    
    parser.add_argument(
        '--meta_path',
        type=str,
        help="path to 'frames_meta.csv' metadata file in question",
    )
    
    parser.add_argument(
        '--wrong_mount_length',
        type=str,
        help="length of incorrect mounting before /CompMicro: (ex) /home/soorya.pradeep/CompMicro is 2",
    )
    
    args = parser.parse_args()
    return args

if __name__ == 'main':
    args = parse_args()
    
    path = args.meta_path
    mounting = args.mounting
    wmlength = args.wrong_mounting_length
    
    frames_meta = pd.read_csv(path)

    for i in range(len(frames_meta['dir_name'])):
        new_path = mounting + '/' + '/'.join(frames_meta['dir_name'][i].split('/')[1 + wmlength:])
        frames_meta['dir_name'][i] = new_path
    frames_meta.to_csv(path)

