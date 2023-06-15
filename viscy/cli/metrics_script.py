# %% script to generate your ground truth directory for viscy prediction evaluation
# After inference, the predictions generated are stored as zarr store.
# Evaluation metrics can be computed by comparison of prediction to
# human proof read ground truth.

import argparse
import os

import imageio as iio
import iohub.ngff as ngff
import pandas as pd

import viscy.evaluation.evaluation_metrics as metrics
import viscy.utils.aux_utils as aux_utils

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
    pick focus slice mask from pred_zarr from slice number stored on png mask name
    input pred mask & corrected ground truth mask to metrics computation
    store the metrics values as csv file to corresponding positions in list
    Info to be stored:
        1. position no,
        2. eval metrics values
    """

    torch_config = aux_utils.read_config(config)

    pred_dir = torch_config["evaluation_metrics"]["pred_dir"]
    metric_channel = torch_config["evaluation_metrics"]["metric_channel"]
    PosList = torch_config["evaluation_metrics"]["PosList"]
    z_list = torch_config["evaluation_metrics"]["z_list"]
    metrics_list = torch_config["evaluation_metrics"]["metrics"]
    ground_truth_chans = torch_config["data"]["target_channel"]
    ground_truth_subdir = "ground_truth"

    d_pod = [
        "OD_true_positives",
        "OD_false_positives",
        "OD_false_negatives",
        "OD_precision",
        "OD_recall",
        "OD_f1_score",
    ]

    metric_map = {
        "ssim": metrics.ssim_metric,
        "corr": metrics.corr_metric,
        "r2": metrics.r2_metric,
        "mse": metrics.mse_metric,
        "mae": metrics.mae_metric,
        "dice": metrics.dice_metric,
        "IoU": metrics.IOU_metric,
        "VI": metrics.VOI_metric,
        "POD": metrics.POD_metric,
    }

    path_split_head_tail = os.path.split(pred_dir)
    target_zarr_dir = path_split_head_tail[0]
    pred_plate = ngff.open_ome_zarr(
        store_path=os.path.join(target_zarr_dir, metric_channel + "_pred.zarr"),
        mode="r+",
    )
    chan_names = pred_plate.channel_names
    metric_chan_mask = metric_channel + "_cp_mask"
    ground_truth_dir = os.path.join(target_zarr_dir, ground_truth_subdir)

    col_val = metrics_list[:]
    if "POD" in col_val:
        col_val.remove("POD")
        for i in range(len(d_pod)):
            col_val.insert(i + metrics_list.index("POD"), d_pod[i])
    df_metrics = pd.DataFrame(columns=col_val, index=PosList)

    for position, pos_data in pred_plate.positions():
        pos = int(position.split("/")[-1])

        if pos in PosList:
            idx = PosList.index(pos)
            raw_data = pos_data.data
            pred_mask = raw_data[0, chan_names.index(metric_chan_mask)]

            z_slice_no = z_list[idx]
            gt_mask_save_name = (
                ground_truth_chans[0]
                + "_p"
                + str(format(pos, "03d"))
                + "_z"
                + str(z_slice_no)
                + "_cp_mask.png"
            )

            gt_mask = iio.imread(os.path.join(ground_truth_dir, gt_mask_save_name))

            pos_metric_list = []
            for metric_name in metrics_list:
                metric_fn = metric_map[metric_name]
                cur_metric_list = metric_fn(
                    gt_mask,
                    pred_mask[0],
                )
                pos_metric_list = pos_metric_list + cur_metric_list

            df_metrics.loc[pos] = pos_metric_list

    csv_filename = os.path.join(ground_truth_dir, "GT_metrics.csv")
    df_metrics.to_csv(csv_filename)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
