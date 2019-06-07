"""Utility functions for plotting"""
import cv2
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import natsort
import numpy as np
import os
from micro_dl.utils.normalize import hist_clipping


def save_predicted_images(input_batch,
                          target_batch,
                          pred_batch,
                          output_dir,
                          batch_idx=None,
                          output_fname=None,
                          ext='jpg',
                          clip_limits=1,
                          font_size=15):
    """
    Saves a batch predicted image to output dir
    Format: rows of [input, target, pred]

    :param np.ndarray input_batch: expected shape [batch_size, n_channels,
     x,y,z]
    :param np.ndarray target_batch: target with the same shape of input_batch
    :param np.ndarray pred_batch: output predicted by the model
    :param str output_dir: dir to store the output images/mosaics
    :param int batch_idx: current batch number/index
    :param str output_fname: fname for saving collage
    :param float clip_limits: top and bottom % of intensity to saturate
    :param int font_size: font size of the image title
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    batch_size = len(input_batch)
    if batch_size == 1:
        assert output_fname is not None, 'need fname for saving image'
        fname = os.path.join(output_dir, '{}.{}'.format(output_fname, ext))


    # 3D images are better saved as movies/gif
    if batch_size != 1:
        assert len(input_batch.shape) == 4, 'saves 2D images only'

    for img_idx in range(batch_size):
        cur_input = input_batch[img_idx]
        cur_target = target_batch[img_idx]
        cur_prediction = pred_batch[img_idx]
        n_channels = cur_input.shape[0]
        fig, ax = plt.subplots(n_channels, 3)
        fig.set_size_inches((15, 5 * n_channels))
        axis_count = 0
        for channel_idx in range(n_channels):
            cur_im = hist_clipping(
                cur_input[channel_idx],
                clip_limits,
                100 - clip_limits,
            )
            ax[axis_count].imshow(cur_im,  cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 0:
                ax[axis_count].set_title('Input', fontsize=font_size)
            axis_count += 1
            cur_im = hist_clipping(
                cur_target[channel_idx],
                clip_limits,
                100 - clip_limits,
            )
            ax[axis_count].imshow(cur_im, cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 1:
                ax[axis_count].set_title('Target', fontsize=font_size)
            axis_count += 1
            cur_im = hist_clipping(
                cur_prediction[channel_idx],
                clip_limits,
                100 - clip_limits,
            )
            ax[axis_count].imshow(cur_im, cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 2:
                ax[axis_count].set_title('Prediction', fontsize=font_size)
            axis_count += 1
        if batch_size != 1:
            fname = os.path.join(
                output_dir,
                '{}.{}'.format(str(batch_idx * batch_size + img_idx), ext)
            )
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_center_slices(image_dir,
                       pos_idx,
                       save_path,
                       mean_std=None,
                       clip_limits=1,
                       margin=20,
                       z_scale=5,
                       z_range=None,
                       channel_str=None,
                       font_size=15,
                       color_map='gray',
                       fig_title=None):
    """
    Given an image directory, loads a z-stack, plots the center cross-sections
    of xy, yz and xz planes with the larger xy section top left, yz top right
    and xz bottom left in the figure.

    :param str image_dir: Directory containing z-stacks
    :param int pos_idx: Which FOV to plot
    :param str save_path: Full path of where to write figure file
    :param tuple mean_std: If None, just assume the image will plot well as is,
        if tuple containing a mean and std (e.g. mean over training data),
        set z-stack mean and std and convert to uint16
    :param float clip_limits: top and bottom % of intensity to saturate
        in histogram clipping
    :param int margin: Number of pixel margin between the three center slices
        xy and xz, yz
    :param int z_scale: How much to upsample in z (to be able to see xz and yz)
    :param list z_range: Min and max z slice from given stack
    :param str channel_str: If there's more than one channel in image_dir
        (e.g. input image dir as opposed to predictions) use this str to select
        which channel to build z-stack from. E.g. '3', 'brightfield'.
    :param int font_size: font size of the image title
    :param str color_map: Matplotlib colormap
    :param str fig_title: Figure title
    """
    search_str = os.path.join(image_dir, "*p{:03d}*".format(pos_idx))
    slice_names = natsort.natsorted(glob.glob(search_str))

    if channel_str is not None:
        slice_names = [s for s in slice_names if channel_str in s]

    # Remove a given nbr of slices from front and back of names
    if z_range is not None:
        assert len(z_range) == 2, 'Z-range must consist of two values'
        slice_names = slice_names[z_range[0]:z_range[1]]
    assert len(slice_names) > 0, \
        "Couldn't find images with given search criteria"

    im_stack = []
    for im_z in slice_names:
        im_stack.append(cv2.imread(im_z, cv2.IMREAD_ANYDEPTH))
    im_stack = np.stack(im_stack, axis=-1)
    # If mean and std tuple exist, scale, otherwise leave as is
    im_norm = im_stack
    if isinstance(mean_std, tuple):
        im_norm = im_stack / im_stack.std() * mean_std[0]
        im_norm = im_norm - im_norm.mean() + mean_std[1]
        # cutoff at 0
        im_norm[im_norm < 0] = 0.
        # Convert to uint16
        im_norm = im_norm.astype(np.uint16)

    # Add xy center slice to plot image (canvas)
    center_slice = hist_clipping(
        im_norm[..., int(len(slice_names) // 2)],
        clip_limits, 100 - clip_limits,
    )
    im_shape = im_stack.shape
    canvas = center_slice.max() * np.ones(
        (im_shape[0] + im_shape[2] * z_scale + margin,
         im_shape[1] + im_shape[2] * z_scale + margin),
        dtype=np.uint16,
    )
    canvas[0:im_shape[0], 0:im_shape[1]] = center_slice
    # add yz center slice
    yz_slice = hist_clipping(
        np.squeeze(im_norm[:, int(im_shape[1] // 2), :]),
        clip_limits, 100 - clip_limits,
    )
    yz_shape = yz_slice.shape
    yz_slice = cv2.resize(yz_slice, (yz_shape[1] * int(z_scale), yz_shape[0]))
    canvas[0:yz_shape[0], im_shape[1] + margin:] = yz_slice
    # add xy center slice
    xy_slice = hist_clipping(
        np.squeeze(im_norm[int(im_shape[1] // 2), :, :]),
        clip_limits, 100 - clip_limits,
    )
    xy_shape = xy_slice.shape
    xy_slice = cv2.resize(xy_slice, (xy_shape[1] * int(z_scale), xy_shape[0]))
    # Need to rotate to fit this slice on the bottom of canvas
    xy_slice = np.rot90(xy_slice)
    canvas[im_shape[0] + margin:, 0:xy_slice.shape[1]] = xy_slice

    plt.imshow(canvas, cmap=color_map)
    plt.axis('off')
    if fig_title is not None:
        plt.title(fig_title, fontsize=font_size)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_mask_overlay(input_image, mask, op_fname, alpha=0.7):
    """
    Plot and save a collage of input, mask, overlay

    :param np.array input_image: 2D input image
    :param np.array mask: 2D mask image
    :param str op_fname: fname will full path for saving the collage as a jpg
    :param int alpha: opacity/transparency for the mask overlay
    """

    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1'
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((15, 5))
    ax[0].imshow(input_image, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis('off')
    # Convert image to uint8 color, scale to 255, and overlay a color contour
    im_rgb = input_image / input_image.max() * 255
    im_rgb = im_rgb.astype(np.uint8)
    im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    try:
        _, contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    except ValueError:
        # Older versions of opencv expects two return values
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    # Draw contours in green with linewidth 2
    im_rgb = cv2.drawContours(im_rgb, contours, -1, (0, 255, 0), 2)
    ax[2].imshow(im_rgb)
    ax[2].axis('off')
    fig.savefig(op_fname, dpi=250)
    plt.close(fig)


def save_plot(x, y, fig_fname, fig_labels=None):
    """
    Plot values y = f(x) and save figure.

    :param list x: x values
    :param list y: y values (same length as x)
    :param str fig_fname: File name including full path
    :param list fig_labels: Labels for x and y axes, and title
    """
    assert len(x) == len(y),\
        "x ({}) and y ({}) must be equal length".format(len(x), len(y))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    if fig_labels is not None:
        assert len(fig_labels) >= 2, "At least x and y labels must be present"
        ax.set_xlabel(fig_labels[0])
        ax.set_ylabel(fig_labels[1])
        if len(fig_labels) == 3:
            ax.set_title(fig_labels[2])
    fig.savefig(fig_fname, dpi=250)
    plt.close(fig)
