"""Custom losses"""
from keras import backend as K
import tensorflow as tf

import micro_dl.train.metrics as metrics


def mse_binary_wtd(n_channels):
    """Converts a loss function into weighted loss function

    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!

    :mask_image: a binary image (assumes foreground / background classes)
    :return: weighted loss
    """

    def mse_wtd(y_true, y_pred):
        try:
            y_true, mask_image = tf.split(y_true, [n_channels, 1], axis=1)
        except Exception as e:
            print('cannot separate mask and y_true' + str(e))
  
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        weights = K.batch_flatten(mask_image)
        weights = K.cast(weights, 'float32')
        loss = K.square(y_pred - y_true)

        fg_count = K.sum(weights, axis=1)
        total_count = K.cast(K.shape(y_true)[1], 'float32')
        fg_volume_fraction = tf.div(fg_count, total_count)
        bg_volume_fraction = 1-fg_volume_fraction
        # fg_vf is a tensor
        fg_weights = tf.where(fg_volume_fraction >= 0.5,
                              fg_volume_fraction, bg_volume_fraction)
        fg_mask = weights * K.expand_dims(fg_weights, axis=1)
        bg_mask = (1 - weights) * K.expand_dims(1 - fg_weights, axis=1)
        mask = fg_mask + bg_mask
        modified_loss = K.mean(K.sum(loss * mask, axis=1))
        return modified_loss
    return mse_wtd


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