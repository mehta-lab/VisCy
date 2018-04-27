"""Custom losses"""
from keras import backend as K


def weighted_binary_loss(fn):
    """Converts a loss function into weighted loss function

    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    :return: a function with signature fn(y_true, y_pred, weights)
    """

    if fn is None:
        return None

    def weighted(y_true, y_pred, weights):
        """Wrapper function

        :y_true: y_true argument of fn
        :y_pred: y_pred argument of fn
        :mask: a batch_flatten-ed binary image (assumes foreground /
         background classes)
        :return: a scalar tensor
        """

        loss = fn(y_true, y_pred)
        if weights is not None:
            weights = K.cast(weights, K.floatx())
            loss = K.batch_flatten(loss)
            fg_loss = K.sum(loss * weights, axis=1)
            bg_loss = K.sum(loss * (1 - weights), axis=1)
            fg_count = K.sum(weights, axis=1)
            total_count = K.shape(loss)[1]
            fg_weight = 1 - (fg_count / total_count)
            bg_weight = 1 - fg_weight
            wtd_loss = K.mean(fg_loss * fg_weight + bg_loss * bg_weight)
            return wtd_loss
    return weighted

