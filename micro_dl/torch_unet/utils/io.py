import os
from PIL import Image
import collections
import re
import time

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

def show_progress_bar(dataloader, current, process = 'training', interval = 1):
    '''
    Utility function to print tensorflow-like progress bar.
    
    :param iterable dataloader: dataloader currently being processed
    :param int current: current index in dataloader
    :param str proces: current process being performed
    :param int interval: interval at which to update progress bar
    '''
    current += 1
    bar_length = 50
    fraction_computed = current/dataloader.__len__()
    
    if current % interval != 0 and fraction_computed < 1:
        return
    
    pointer = '>' if fraction_computed < 1 else '='
    loading_string = '='*int(bar_length*fraction_computed) + '>' + '_'*int(bar_length*(1-fraction_computed))
    output_string = f'\t {process} {current}/{dataloader.__len__()} [{loading_string}] ({int(fraction_computed * 100)}%)'
    
    if fraction_computed <= (dataloader.__len__() - interval)/dataloader.__len__():
        print(output_string, end='\r')
    else:
        print(output_string)
        
    #for smoother output
    time.sleep(0.2)