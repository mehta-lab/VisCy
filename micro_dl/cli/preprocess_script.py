"""Script for preprocessing stack"""
import argparse
import iohub.ngff as ngff
import time

from micro_dl.preprocessing.generate_masks import MaskProcessor
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.meta_utils as meta_utils


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml configuration file",
    )
    args = parser.parse_args()
    return args


def pre_process(torch_config):
    """
    Preprocess data. Possible options are:

    normalize: Calculate values for on-the-fly normalization on a FOV &
                dataset level
    create_masks: Generate binary masks from given input channels

    This script will preprocess your dataset, save auxilary data and
    associated metadata for on-the-fly processing during training. Masks
    will be saved both as an additional channel and as an array tracked in
    custom metadata. 

    :param dict torch_config: 'master' torch config with subfields for all steps
                            of data analysis
    :raises AssertionError: If 'masks' in preprocess_config contains both channels
     and mask_dir (the former is for generating masks from a channel)
    """
    time_start = time.time()
    plate = ngff.open_ome_zarr(torch_config["zarr_dir"], layout='hcs', mode='r')
    preprocess_config = torch_config["preprocessing"]

    # ----------------- Generate normalization values -----------------
    if "normalize" in preprocess_config:
        print("Computing Normalization Values: ------------- \n")
        # collect params
        normalize_config = preprocess_config["normalize"]

        norm_num_workers = 4
        if "num_workers" in normalize_config:
            norm_num_workers = normalize_config["num_workers"]

        norm_channel_ids = -1
        if "channel_ids" in normalize_config:
            norm_channel_ids = normalize_config["channel_ids"]

        norm_block_size = 32
        if "block_size" in normalize_config:
            norm_block_size = normalize_config["block_size"]

        meta_utils.generate_normalization_metadata(
            zarr_dir=torch_config["zarr_dir"],
            num_workers=norm_num_workers,
            channel_ids=norm_channel_ids,
            grid_spacing=norm_block_size,
        )

    # ------------------------Generate masks-------------------------
    if "masks" in preprocess_config:
        print("Generating Masks: ------------- \n")
        # collect params
        mask_config = preprocess_config["masks"]

        mask_channel_ids = -1
        if "channel_ids" in mask_config:
            mask_channel_ids = mask_config["channel_ids"]

        mask_time_ids = -1
        if "time_ids" in mask_config:
            mask_time_ids = mask_config["time_ids"]

        mask_pos_ids = -1

        mask_num_workers = 4
        if "num_workers" in mask_config:
            mask_num_workers = mask_config["num_workers"]

        mask_type = "unimodal"
        if "thresholding_type" in mask_config:
            mask_type = mask_config["thresholding_type"]

        overwrite_ok = True
        if "allow_overwrite_old_mask" in mask_config:
            overwrite_ok = mask_config["allow_overwrite_old_mask"]

        structuring_radius = 5
        if "structure_element_radius" in mask_config:
            structuring_radius = mask_config["structure_element_radius"]

        # validate
        if mask_type not in {
            "unimodal",
            "otsu",
            "mem_detection",
            "borders_weight_loss_map",
        }:
            raise ValueError(
                f"Thresholding type {mask_type} must be one of: "
                f"{['unimodal', 'otsu', 'mem_detection', 'borders_weight_loss_map']}"
            )

        # generate masks
        mask_generator = MaskProcessor(
            zarr_dir=torch_config["zarr_dir"],
            channel_ids=mask_channel_ids,
            time_ids=mask_time_ids,
            pos_ids=mask_pos_ids,
            num_workers=mask_num_workers,
            mask_type=mask_type,
            overwrite_ok=overwrite_ok,
        )
        mask_generator.generate_masks(structure_elem_radius=structuring_radius)

    # ----------------------Generate weight map-----------------------
    # TODO: determine if weight map generation should be offered in simultaneity
    #      with binary mask generation

    plate.close()
    return time.time() - time_start


if __name__ == "__main__":
    args = parse_args()
    torch_config = aux_utils.read_config(args.config)
    runtime = pre_process(torch_config)
    print(f"Preprocessing complete. Runtime: {runtime:.2f} seconds")
