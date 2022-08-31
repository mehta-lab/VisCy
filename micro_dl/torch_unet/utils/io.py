import os
from PIL import Image
import collections
import re

def unique_tags(directory):
    '''
    Returns list of unique nume tags from data directory
    
    TODO: improve documentation specifics
    '''
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    
    tags = collections.defaultdict(lambda : 0)
    for f in files:
        f_name, f_type = f.split('.')[0], f.split('.')[1]
        if f_type == 'tif':
            suffixes = re.split('_', f_name)

            unique_tag = suffixes[2] + '_' + suffixes[3] + '_' + suffixes[4]
            tags[unique_tag + '.' + f_type] += 1
    return tags