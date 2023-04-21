"""Utility functions used for training"""
import warnings
import numpy as np
import os
import subprocess

import micro_dl.utils.preprocess_utils as preprocess_utils


def get_image_dir_format(dataset_config):
    """Get dir with input images for generating full path from frames_meta

    If the tiled dir is passed as data dir there will be no
    preprocessing_info.json. If json present use it, else read images from the
    given dir.
    """

    # tile dir pass directly as data_dir
    tile_dir = dataset_config["data_dir"]
    image_format = "zyx"
    try:
        preprocess_config = preprocess_utils.get_preprocess_config(
            os.path.dirname(tile_dir)
        )

        # Get shape order from recent_json
        if "image_format" in preprocess_config["tile"]:
            image_format = preprocess_config["tile"]["image_format"]
    except Exception as e:
        print(
            "Error while reading preprocess config: {}. "
            'Use default image format "zyx"'.format(e)
        )

    return tile_dir, image_format


def check_gpu_availability(gpu_id):
    """
    Check if mem_frac is available in given gpu_id

    :param int/list gpu_id: id of the gpu to be used. Int for single GPU
     training, list for distributed training
    :param list gpu_mem_frac: mem fraction for each GPU in gpu_id
    :return bool gpu_availability: True if all mem_fracs are greater than
        gpu_mem_frac
    :return list curr_mem_frac: list of current memory fractions available
        for gpus in to gpu_id
    """
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    cur_mem_frac = []
    for idx, gpu in enumerate(gpu_id):
        query = (
            "nvidia-smi --id={} --query-gpu=memory.free,memory.total " "--format=csv"
        ).format(gpu)
        sp = subprocess.Popen([query], stdout=subprocess.PIPE, shell=True)
        query_output = sp.communicate()
        query_output = query_output[0].decode("utf8")
        query_output = query_output.split("\n")
        mem = query_output[1].split(",")
        mem = [int(val.replace("MiB", "")) for val in mem]
        cur_mem_frac.append(mem[0] / mem[1])
    return cur_mem_frac


def select_gpu(gpu_ids=None, gpu_mem_frac=None):
    """
    Find the GPU ID with highest available memory fraction.
    If ID is given as input, set the gpu_mem_frac to maximum available,
    or if a memory fraction is given, make sure the given GPU has the desired
    memory fraction available.
    Currently only supports single GPU runs.

    :param int gpu_ids: Desired GPU ID. If None, find GPU with the most memory
        available.
    :param float gpu_mem_frac: Desired GPU memory fraction [0, 1]. If None,
        use maximum available amount of GPU.
    :return int gpu_ids: GPU ID to use.
    :return float cur_mem_frac: GPU memory fraction to use
    :raises NotImplementedError: If gpu_ids is not int
    :raises AssertionError: If requested memory fraction isn't available
    """
    # Check if user has specified a GPU ID to use
    if not isinstance(gpu_ids, type(None)):
        # Currently only supporting one GPU as input
        if not isinstance(gpu_ids, int):
            raise NotImplementedError("Currently gpu id specification must be int")
        if gpu_ids == -1:
            return -1, 0
        cur_mem_frac = check_gpu_availability(gpu_ids)
        cur_mem_frac = cur_mem_frac[0]
        if not isinstance(gpu_mem_frac, type(None)):
            assert (
                cur_mem_frac >= gpu_mem_frac
            ), "Not enough memory available. Requested/current fractions: {0:.4g}/{0:.4g}".format(
                gpu_mem_frac, cur_mem_frac
            )
        print("Using GPU {} with memory fraction {}.".format(gpu_ids, cur_mem_frac))
        return gpu_ids, cur_mem_frac

    # User has not specified GPU ID, find the GPU with most memory available
    sp = subprocess.Popen(
        ["nvidia-smi --query-gpu=index --format=csv"],
        stdout=subprocess.PIPE,
        shell=True,
    )
    gpu_ids = sp.communicate()
    gpu_ids = gpu_ids[0].decode("utf8")
    gpu_ids = gpu_ids.split("\n")
    # If no GPUs are found, run on CPU (debug mode)
    if len(gpu_ids) <= 2:
        print("No GPUs found, run will be slow. Query result: {}".format(gpu_ids))
        return -1, 0
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids[1:-1]]
    cur_mem_frac = check_gpu_availability(gpu_ids)
    # Get the GPU with maximum memory fraction
    max_mem = max(cur_mem_frac)
    idx = cur_mem_frac.index(max_mem)
    gpu_id = gpu_ids[idx]
    # Subtract a little margin to be safe
    max_mem = max_mem - np.finfo(np.float32).eps
    print("Using GPU {} with memory fraction {}.".format(gpu_id, max_mem))
    return gpu_id, max_mem

