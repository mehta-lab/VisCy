"""Custom losses"""
from keras import backend as K
import tensorflow as tf


def mse_binary_wtd_loss(y_true, y_pred):
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

    print(y_true.get_shape(), 'raw_shape')
    try:
        #[y_true, mask_image] = tf.unstack(y_true, 2, axis=0)
        mask_image = y_true[:, -1, :, :]
        y_true = y_true[:, :-1, :, :]
        print(y_true.get_shape())
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        weights = K.batch_flatten(mask_image)
        weights = K.cast(weights, 'float32')
        loss = K.square(y_pred - y_true)

        if weights is not None:
            fg_loss = K.sum(loss * weights, axis=1)
            bg_loss = K.sum(loss * (1 - weights), axis=1)
            fg_count = K.sum(weights, axis=1)
            total_count = K.cast(K.shape(loss)[1], 'float32')
            fg_weight = 1 - (fg_count / total_count)
            bg_weight = 1 - fg_weight
            wtd_loss = K.mean(fg_loss * fg_weight + bg_loss * bg_weight)
            return wtd_loss

    except Exception as e:
        print('cannot unstack mask and target' + str(e))
