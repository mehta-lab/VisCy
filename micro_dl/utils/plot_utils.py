"""Utility functions for plotting"""
import matplotlib.pyplot as plt
import numpy as np
import os


def save_predicted_images(input_batch, target_batch, pred_batch,
                          output_dir, batch_idx):
    """Saves a batch predicted image to output dir as batch_idx.jpg/tiff/png

    Format: rows of [input, target, pred]

    :param np.ndarray input_batch: expected shape [batch_size, n_channels,
     x,y,z]
    :param np.ndarray target_batch: target with the same shape of input_batch
    :param np.ndarray pred_batch: output predicted by the model
    :param str output_dir: dir to store the output images/mosaics
    :param int batch_idx: current batch number/index
    """

    batch_size = input_batch.shape[0]
    # 3D images are better saved as movies/gif
    assert len(batch_size.shape) == 4, 'saves 2D images only'

    for img_idx in range(batch_size):
        cur_input = input_batch[img_idx]
        cur_target = target_batch[img_idx]
        cur_prediction = pred_batch[img_idx]
        n_channels = cur_input.shape[0]
        fig, ax = plt.subplots(n_channels, 3)
        fig.set_size_inches((15, 5 * n_channels))
        axis_count = 0
        for channel_idx in range(n_channels):
            ax[axis_count].imshow(cur_input[channel_idx], cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 0:
                ax[axis_count].title('input')
            ax[axis_count + 1].imshow(cur_target[channel_idx], cmap='gray')
            ax[axis_count + 1].axis('off')
            if axis_count == 1:
                ax[axis_count].title('target')
            ax[axis_count].imshow(cur_prediction[channel_idx], cmap='gray')
            ax[axis_count].axis('off')
            if axis_count == 2:
                ax[axis_count].title('prediction')
            axis_count += 3
        fname = os.path.join(output_dir,
                             '{}.jpg'.format(
                                 str(batch_idx * batch_size + img_idx))
                             )
        fig.savefig(fname, dpi=250)