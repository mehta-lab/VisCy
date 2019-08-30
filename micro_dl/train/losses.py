"""Custom losses"""
from keras import backend as K
from keras.losses import mean_absolute_error
import tensorflow as tf
import numpy as np

import micro_dl.train.metrics as metrics
from micro_dl.utils.aux_utils import get_channel_axis


def mae_loss(y_true, y_pred, mean_loss=True):
    """Mean absolute error

    Keras losses by default calculate metrics along axis=-1, which works with
    image_format='channels_last'. The arrays do not seem to batch flattened,
    change axis if using 'channels_first
    """
    if not mean_loss:
        return K.abs(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.abs(y_pred - y_true), axis=channel_axis)


def mse_loss(y_true, y_pred, mean_loss=True):
    """Mean squared loss

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float: Mean squared error loss
    """
    if not mean_loss:
        return K.square(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.square(y_pred - y_true), axis=channel_axis)


def kl_divergence_loss(y_true, y_pred):
    """KL divergence loss
    D(y||y') = sum(p(y)*log(p(y)/p(y'))

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float: KL divergence loss
    """
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    channel_axis = get_channel_axis(K.image_data_format())
    return K.sum(y_true * K.log(y_true / y_pred), axis=channel_axis)


def dssim_loss(y_true, y_pred):
    """Structural dissimilarity loss + L1 loss
    DSSIM is defined as (1-SSIM)/2
    https://en.wikipedia.org/wiki/Structural_similarity

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :return float: 0.8 * DSSIM + 0.2 * L1
    """
    mae = mean_absolute_error(y_true, y_pred)
    return 0.8 * (1.0 - metrics.ssim(y_true, y_pred) / 2.0) + 0.2 * mae


def ms_ssim_loss(y_true, y_pred):
    """
    Multiscale structural dissimilarity loss + L1 loss
    Uses the same combination weight as the original paper by Wang et al.:
    https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
    Tensorflow doesn't have a 3D version so for stacks the MS-SSIM is the
    mean of individual slices.

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :return float: ms-ssim loss
    """
    mae = mae_loss(y_true, y_pred)
    return 0.84 * (1.0 - metrics.ms_ssim(y_true, y_pred)) + 0.16 * mae


def _split_ytrue_mask(y_true, n_channels):
    """Split the mask concatenated with y_true

    :param keras.tensor y_true: if channels_first, ytrue has shape [batch_size,
     n_channels, y, x]. mask is concatenated as the n_channels+1, shape:
     [[batch_size, n_channels+1, y, x].
    :param int n_channels: number of channels in y_true
    :return:
     keras.tensor ytrue_split - ytrue with the mask removed
     keras.tensor mask_image - bool mask
    """

    try:
        split_axis = get_channel_axis(K.image_data_format())
        y_true_split, mask_image = tf.split(y_true, [n_channels, 1],
                                            axis=split_axis)
        return y_true_split, mask_image
    except Exception as e:
        print('cannot separate mask and y_true' + str(e))


def masked_loss(loss_fn, n_channels):
    """Converts a loss function to mask weighted loss function

    Loss is multiplied by mask. Mask could be binary, discrete or float.
    Provides different weighting of loss according to the mask.
    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!
    Histogram and logical operators are not differentiable, avoid them in loss
    modified_loss = tf.Print(modified_loss, [modified_loss],
                             message='modified_loss', summarize=16)
    :param Function loss_fn: a loss function that returns a loss image to be
     multiplied by mask
    :param int n_channels: number of channels in y_true. The mask is added as
     the last channel in y_true
    :return function masked_loss_fn
    """

    def masked_loss_fn(y_true, y_pred):
        y_true, mask_image = _split_ytrue_mask(y_true, n_channels)
        loss = loss_fn(y_true, y_pred, mean_loss=False)
        total_loss = 0.0
        for ch_idx in range(n_channels):
            cur_loss = loss[:, ch_idx]
            cur_loss = cur_loss * mask_image
            mean_loss = K.mean(cur_loss)
            total_loss += mean_loss
        modified_loss = total_loss / n_channels
        return modified_loss
    return masked_loss_fn


def dice_coef_loss(y_true, y_pred):
    """
    The Dice loss function is defined by 1 - DSC
    since the DSC is in the range [0,1] where 1 is perfect overlap
    and we're looking to minimize the loss.

    :param y_true: true values
    :param y_pred: predicted values
    :return: Dice loss
    """
    return 1. - metrics.dice_coef(y_true, y_pred)


def binary_crossentropy_loss(y_true, y_pred, mean_loss=True):
    """Binary cross entropy loss
    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float: Binary cross entropy loss
    """
    assert len(np.unique(y_true).tolist()) <= 2
    assert len(np.unique(y_pred).tolist()) <= 2

    if not mean_loss:
        return K.binary_crossentropy(y_true, y_pred)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=channel_axis)

