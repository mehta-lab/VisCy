# %% script to generate your ground truth directory for viscy prediction evaluation
# After inference, the predictions generated are stored as zarr store.
# Evaluation metrics can be computed by comparison of prediction
# to human proof read ground truth.

import argparse
import os

import imageio as iio
import iohub.ngff as ngff
import numpy as np
from PIL import Image

import viscy.evaluation.evaluation_metrics as metrics
import viscy.utils.aux_utils as aux_utils

# from waveorder.focus import focus_from_transverse_band

# %% read the below details from the config file


def parse_args():
    """
    Parse command line arguments
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


def main(config):
    """
    pick the focus slice from n_pos number of positions, cellpose segment,
    and save as TIFFs.
    also save segmentation input and label-free image as tifs for ground truth curation
    segment fluorescence predictions and store mask as new channel
    """

    torch_config = aux_utils.read_config(config)

    zarr_dir = torch_config["data"]["data_path"]
    pred_dir = torch_config["evaluation_metrics"]["pred_dir"]
    ground_truth_chans = torch_config["data"]["target_channel"]
    labelFree_chan = torch_config["data"]["source_channel"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    z_list = torch_config["evaluation_metrics"]["z_list"]
    cp_model = torch_config["evaluation_metrics"]["cp_model"]
    metric_channel = torch_config["evaluation_metrics"]["metric_channel"]

    # if torch_config["evaluation_metrics"]["NA_det"] is None:
    #     NA_det = 1.3
    #     lambda_illu = 0.4
    #     pxl_sz = 0.103
    # else:
    #     NA_det = torch_config["evaluation_metrics"]["NA_det"]
    #     lambda_illu = torch_config["evaluation_metrics"]["lambda_illu"]
    #     pxl_sz = torch_config["evaluation_metrics"]["pxl_sz"]

    ground_truth_subdir = "ground_truth"
    path_split_head_tail = os.path.split(pred_dir)
    target_zarr_dir, _zarr_name = path_split_head_tail[0]

    if not os.path.exists(os.path.join(target_zarr_dir, ground_truth_subdir)):
        os.mkdir(
            os.path.join(target_zarr_dir, ground_truth_subdir)
        )  # create dir to store single page tifs
        plate = ngff.open_ome_zarr(store_path=zarr_dir, mode="r+")
        chan_names = plate.channel_names

        for position, pos_data in plate.positions():
            im = pos_data.data
            # im = plate.data
            out_shape = im.shape
            # zarr_pos_len = reader.get_num_positions()
            try:
                assert len(PosList) > out_shape[0]
            except AssertionError:
                print(
                    "number of positions listed in config exceeds "
                    "the number of positions in the dataset"
                )
            pos = int(position.split("/")[-1])
            for gt_chan in ground_truth_chans:
                if pos in PosList:
                    idx = PosList.index(pos)
                    target_data = im[0, chan_names.index(gt_chan), ...]
                    Z, Y, X = target_data.shape
                    focus_idx_target = z_list[idx]
                    # focus_idx_target = focus_from_transverse_band(
                    #     target_data, NA_det, lambda_illu, pxl_sz
                    # )
                    target_focus_slice = target_data[
                        focus_idx_target, :, :
                    ]  # FL focus slice image

                    im_target = Image.fromarray(
                        target_focus_slice
                    )  # save focus slice as single page tif
                    save_name = (
                        "_p" + str(format(pos, "03d")) + "_z" + str(focus_idx_target)
                    )
                    im_target.save(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            gt_chan + save_name + ".tif",
                        )
                    )

                    source_focus_slice = im[
                        0, chan_names.index(labelFree_chan[0]), focus_idx_target, :, :
                    ]  # lable-free focus slice image
                    im_source = Image.fromarray(
                        source_focus_slice
                    )  # save focus slice as single page tif
                    im_source.save(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            labelFree_chan[0] + save_name + ".tif",
                        )
                    )  # save for reference

                    cp_mask = metrics.cpmask_array(
                        target_focus_slice, cp_model
                    )  # cellpose segmnetation for binary mask
                    iio.imwrite(
                        os.path.join(
                            target_zarr_dir,
                            ground_truth_subdir,
                            gt_chan + save_name + "_cp_mask.png",
                        ),
                        cp_mask,
                    )  # save binary mask as numpy or png

    # segment prediction and add mask as channel to pred_dir
    pred_plate = ngff.open_ome_zarr(store_path=pred_dir, mode="r+")
    # im_pred = pred_plate.data
    chan_names = pred_plate.channel_names

    predseg_data = ngff.open_ome_zarr(
        os.path.join(target_zarr_dir, metric_channel + "_pred.zarr"),
        layout="hcs",
        mode="w-",
        channel_names=chan_names,
    )
    for position, pos_data in pred_plate.positions():
        row, col, fov = position.split("/")
        new_pos = predseg_data.create_position(row, col, fov)

        if int(fov) in PosList:
            idx = PosList.index(int(fov))
            raw_data = pos_data.data
            target_data = raw_data[:, :, z_list[idx]]
            _, _, Y, X = target_data.shape
            new_pos.create_image("0", target_data[np.newaxis, :])

    chan_no = len(chan_names)
    with ngff.open_ome_zarr(
        os.path.join(target_zarr_dir, metric_channel + "_pred.zarr"), mode="r+"
    ) as dataset:
        for _, position in dataset.positions():
            data = position.data
            new_channel_array = np.zeros((1, 1, Y, X))

            cp_mask = metrics.cpmask_array(
                data[0, chan_names.index(metric_channel), 0, :, :], cp_model
            )
            new_channel_array[0, 0, :, :] = cp_mask

            new_channel_name = metric_channel + "_cp_mask"
            position.append_channel(new_channel_name, resize_arrays=True)
            position["0"][:, chan_no] = new_channel_array


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
