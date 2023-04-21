import argparse
import datetime
import os
import torch
import yaml
import sys
import zarr

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.inference.inference as torch_inference_utils
import micro_dl.utils.train_utils as train_utils


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
            "Error in autosaving: 'input_dir'" "unspecified in preprocess config"
        )
        now = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace("-", "_")[:-10]
        )
        save_dir = os.path.join(preprocess_config["input_dir"], f"../prediction_{now}")

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
        gpu_id, gpu_mem_frac = train_utils.select_gpu(
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
