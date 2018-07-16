"""Custom metrics"""
import keras.backend as K


def coeff_determination(y_true, y_pred):
    """R^2 Goodness of fit, using as a proxy for accuracy in regression"""

    SS_res = K.sum(K.square(y_true - y_pred ))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


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
