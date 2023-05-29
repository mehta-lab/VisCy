import argparse
import datetime
import os
import subprocess

import numpy as np
import torch

import micro_dl.inference.inference as torch_inference_utils
import micro_dl.utils.aux_utils as aux_utils

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
            "nvidia-smi --id={} --query-gpu=memory.free,memory.total "
            "--format=csv"
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
            raise NotImplementedError(
                "Currently gpu id specification must be int"
            )
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
        print(
            "Using GPU {} with memory fraction {}.".format(
                gpu_ids, cur_mem_frac
            )
        )
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
        print(
            "No GPUs found, run will be slow. Query result: {}".format(gpu_ids)
        )
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


def parse_args():
    """
    Parse command line arguments
    In python namespaces are implemented as dictionaries

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "Optional: specify the gpu to use: 0,1,...",
            ", -1 for debugging. Default: pick best GPU",
        ),
    )
    parser.add_argument(
        "--gpu_mem_frac",
        type=float,
        default=None,
        help="Optional: specify gpu memory fraction to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml configuration file",
    )
    args = parser.parse_args()
    return args


def check_save_folder(inference_config, preprocess_config):
    """
    Helper method to ensure that save folder exists.
    If no save folder specified in inference_config, force saving in data
    directory with dynamic name and timestamp.

    :param pd.dataframe inference_config: inference config file (not) containing save_folder_name
    :param pd.dataframe preprocess_config: preprocessing config file containing input_dir
    """

    if "save_folder_name" not in inference_config:
        assert "input_dir" in preprocess_config, (
            "Error in autosaving: 'input_dir'"
            "unspecified in preprocess config"
        )
        now = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace("-", "_")[:-10]
        )
        save_dir = os.path.join(
            preprocess_config["input_dir"], f"../prediction_{now}"
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        preprocess_config["save_folder_name"] = save_dir
        print(
            f"No save folder specified in inference config: automatically saving predictions in : \n\t{save_dir}"
        )


def main(config, gpu, gpu_mem_frac):
    config = aux_utils.read_config(config)

    if gpu is not None:
        # Get GPU ID and memory fraction
        gpu_id, gpu_mem_frac = select_gpu(
            gpu,
            gpu_mem_frac,
        )
        device = torch.device(gpu_id)
    else:
        device = torch.device(config.get("device"))

    # Initialize and run a predictor
    torch_predictor = torch_inference_utils.TorchPredictor(
        config=config,
        device=device,
    )

    torch_predictor.load_model()
    torch_predictor.run_inference()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.gpu, args.gpu_mem_frac)
