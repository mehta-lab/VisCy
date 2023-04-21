import cv2
import numpy as np
import pandas as pd
import os
import typer


from enum import Enum
import glob
import time
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/shalin.mehta/code/microDL")

import microDL.micro_dl.training.training as training
import micro_dl.utils.aux_utils as aux_utils
import microDL.micro_dl.utils.cli_utils as torch_io_utils
import micro_dl.plotting.plot_utils as plot_utils


def visualize_dataloading(
    torch_config, save_dir, save_type="plot_group", image_extension=".png"
):
    """
    Draws one epoch worth of samples from the dataloader and saves them
    at 'save_dir' as type 'image_extension'. Note that the pipeline
    performs random sampling, so each new pipeline constructed will
    query images from random positions and ROIs

    :param dict torch_config: config with all the necessary parameters ans structure
                        to perform preprocessing and training
    :param str save_dir: directory to save samples images at
    :param str save_type: format to save images in. One of {'group_plot', 'plot', 'image_only'}
    :param str image_extension: image save type, defaults to ".png"
    """
    trainer = training.TorchTrainer(torch_config=torch_config)

    trainer.generate_dataloaders()
    assert len(trainer.train_dataloader) > 0, "No samples found in the train dataloader"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, minibatch in enumerate(trainer.train_dataloader):
        torch_io_utils.show_progress_bar(
            dataloader=trainer.train_dataloader,
            current=i,
            process="Saving image slices",
        )
        sample_dir = os.path.join(save_dir, f"sample_{i}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        input_ = minibatch[0][0][0].cpu().detach().numpy()
        target_ = minibatch[1][0][0].cpu().detach().numpy()

        if save_type == "group_plot":
            for c in range(input_.shape[0]):
                input_imgs = input_[c]
                target_img = target_[c][0]  # pull the first slice
                name = f"data_chan_{c}"

                metric_dict = {
                    "input_max": [np.max(input_)],
                    "input_min": [np.min(input_)],
                    "target_max": [np.max(target_)],
                    "target_min": [np.min(target_)],
                }

                plot_utils.save_predicted_images(
                    input_imgs=input_imgs,
                    target_img=target_img,
                    pred_img=np.zeros((target_img.shape)),
                    metric=pd.DataFrame.from_dict(metric_dict),
                    output_dir=sample_dir,
                    output_fname=name,
                    ext=image_extension[1:],
                )
        else:
            matplotlib = False
            if save_type == "plot":
                matplotlib = True

            for c in range(input_.shape[0]):
                for z in range(input_.shape[1]):
                    slice = input_[c, z]
                    name = f"input_chan_{c}_slice_{z}"
                    save_file = os.path.join(sample_dir, name + image_extension)
                    save_slice(save_file=save_file, slice=slice, matplotlib=matplotlib)

            for c in range(input_.shape[0]):
                for z in range(target_.shape[1]):
                    slice = target_[c, z]
                    name = f"target_chan_{c}_slice_{z}"
                    save_file = os.path.join(sample_dir, name + image_extension)
                    save_slice(save_file=save_file, slice=slice, matplotlib=matplotlib)


def save_slice(save_file, slice, matplotlib):
    """
    Handles saving of slices of different formats, as well as plotting in mat
    plotlib and saving the plot

    :param str save_file: filename to save to
    :param np.ndarray slice: slice to save (must be 2d)
    :param bool matplotlib: Whether to save matplotlib versions or not
    """
    if ".png" in save_file[-5:]:
        if matplotlib:
            title = save_file.split("/")[-1][:-4]
            cmap_gray = "input" in title
            plt.title(title)
            plt.imshow(slice, cmap="gray") if cmap_gray else plt.imshow(slice)
            plt.savefig(save_file)
            plt.close()
        else:
            # Png must be saved as uint16
            slice = np.clip(slice, 0, 65535)
            slice = slice.astype(np.uint16)
            cv2.imwrite(filename=save_file, img=slice)
    elif matplotlib:
        raise ValueError("Matplotlib plots only save-able as png files")
    elif ".tif" in save_file[-5:]:
        cv2.imwrite(filename=save_file, img=slice)
    elif ".npy" in save_file[-5:]:
        np.save(save_file, slice, allow_pickle=True)
    else:
        raise ValueError("Only '.png', '.tif', and '.npy' extensions supported.")


class plot_save_type(str, Enum):
    group_plot = "group_plot"
    plot = "plot"
    image_only = "image_only"


def main(
    config: str = typer.Option(
        ...,
        help="Path to YAML config that specifies dataset and augmentation parameters",
    ),
    save_dir: str = typer.Option(..., help="Directory to save samples images at"),
    extension: str = typer.Option(".png", help="File type of saved images."),
    save_type: plot_save_type = typer.Option(
        plot_save_type.group_plot,
        case_sensitive=False,
        help="Format to save images in.",
    ),
):
    """
    Generates a set of images from the training set to visualize the data with and without augmentation.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    augmented_save_dir = os.path.join(save_dir, "augmented")
    raw_save_dir = os.path.join(save_dir, "raw")

    start = time.time()

    torch_config = aux_utils.read_config(
        config  # "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_11_01_VeroMemNuclStain/data_visualization_12_13/torch_config_25D.yml"
    )

    # visualize augmented data
    print(
        f"Visualizing augmented data, saving as {extension} files"
        f" in {augmented_save_dir}..."
    )
    visualize_dataloading(torch_config, augmented_save_dir, save_type.value, extension)

    # visualize raw data
    print(f"Visualizing raw data, saving as {extension} files" f" in {raw_save_dir}...")
    torch_config["training"].pop("augmentations")
    visualize_dataloading(torch_config, raw_save_dir, save_type.value, extension)

    print(f"Done. Time taken: {time.time() - start}")


if __name__ == "__main__":
    typer.run(main)

# # def parse_args():
#     """Parse command line arguments

#     In python namespaces are implemented as dictionaries
#     :return: namespace containing the arguments passed.
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         type=str,
#         help=(
#             "path to config with all the necessary parameters and structure"
#             " to perform preprocessing and training and some augmentation"
#         ),
#     )

#     parser.add_argument(
#         "--save_dir",
#         type=str,
#         help="directory to save samples images at",
#     )
#     parser.add_argument(
#         "--extension",
#         type=str,
#         default=".png",
#         help="image save type, defaults to '.png'",
#     )
#     parser.add_argument(
#         "--save_type",
#         type=str,
#         default="group_plot",
#         help="format to save images in. One of {'group_plot', 'plot', 'image_only'}",
#     )

#     args = parser.parse_args()
#     return args


#     args = parse_args()
#     save_dir = (
#         args.save_dir
#     )  # "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data_visualization/augmentation_test_12_14/"
#     extension = args.extension  # ".png"
#     save_type = args.save_type  # "plot_group"
#     torch_config = aux_utils.read_config(
#         args.config  # "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_11_01_VeroMemNuclStain/data_visualization_12_13/torch_config_25D.yml"
#     )

#     # setup save locations for both samples
#     augmented_save_dir = os.path.join(save_dir, "augmented")
#     raw_save_dir = os.path.join(save_dir, "raw")

#     # visualize augmented data
#     print(
#         f"Visualizing augmented data, saving as {extension} files"
#         f" in {augmented_save_dir}..."
#     )
#     visualize_dataloading(torch_config, augmented_save_dir, save_type, extension)

#     # visualize raw data
#     print(f"Visualizing raw data, saving as {extension} files" f" in {raw_save_dir}...")
#     torch_config["training"].pop("augmentations")
#     visualize_dataloading(torch_config, raw_save_dir, save_type, extension)

#     print(f"Done. Time taken: {time.time() - start}")
