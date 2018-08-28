"""Custom metrics"""
import keras.backend as K
import tensorflow as tf


def coeff_determination(y_true, y_pred):
    """R^2 Goodness of fit, using as a proxy for accuracy in regression"""

    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )


def mask_coeff_determination(n_channels):
    """split y_true into y_true and mask

    If masked_loss: add a function/method to convert split y_true and pass to
    loss, metrics and callbacks
    """

    def coeff_deter(y_true, y_pred):

        if K.image_data_format() == "channels_last":
            split_axis = -1
        else:
            split_axis = 1
        y_true_split, mask = tf.split(y_true, [n_channels, 1], axis=split_axis)
        r2 = coeff_determination(y_true_split, y_pred)
        return r2
    return coeff_deter


def dice_coef(y_true, y_pred, smooth=1.):
    """
    This is a global non-binary Dice similarity coefficient (DSC)
    with smoothing.
    It computes an approximation of Dice but over the whole batch,
    and it leaves predicted output as continuous. This might help
    alleviate potential discontinuities a binary image level Dice
    might introduce.
    DSC = 2 * |A union B| /(|A| + |B|) = 2 * |ab| / (|a|^2 + |b|^2)
    where a, b are binary vectors
    smoothed DSC = (2 * |ab| + s) / (|a|^2 + |b|^2 + s)
    where s is smoothing constant.
    Although y_pred is not binary, it is assumed to be near binary
    (sigmoid transformed) so |y_pred|^2 is approximated by sum(y_pred).
    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :param float smooth: Constant added for smoothing and to avoid
       divide by zeros
    :return float dice: Smoothed non-binary Dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
