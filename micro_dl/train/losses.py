"""Custom losses"""
from keras import backend as K
import tensorflow as tf


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
            ndims = K.ndim(y_true)
            print('ndims:', ndims)
            y_true, mask_image = tf.split(y_true, [n_channels, 1], axis=1)
        except Exception as e:
            print('cannot separate mask and y_true' + str(e))
  
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        weights = K.batch_flatten(mask_image)
        weights = K.cast(weights, 'float32')
        loss = K.square(y_pred - y_true)

        fg_loss = K.sum(loss * weights, axis=1)
        bg_loss = K.sum(loss * (1 - weights), axis=1)
        fg_count = K.sum(weights, axis=1)
        total_count = K.cast(K.shape(loss)[1], 'float32')
        fg_volume_fraction = tf.div(fg_count, total_count)
        bg_volume_fraction = 1-fg_volume_fraction
        # fg_vf is a tensor
        fg_weight = tf.where(fg_volume_fraction >= 0.5,
                             fg_volume_fraction, bg_volume_fraction) 
        bg_weight = 1 - fg_weight
        fg_loss = K.mean(fg_loss * fg_weight)
        bg_weight = tf.where(fg_volume_fraction >= 0.5,
                             bg_volume_fraction, fg_volume_fraction)
        bg_loss = K.mean(bg_loss * bg_weight)

        wtd_loss = (fg_loss + bg_loss) / 2.0
        print('wtd_loss:', wtd_loss)
        return wtd_loss
    return mse_wtd
