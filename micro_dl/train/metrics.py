"""Custom metrics"""
import keras.backend as K


def coeff_determination(y_true, y_pred):
    """R^2 Goodness of fit, using as a proxy for accuracy in regression"""

    SS_res = K.sum(K.square(y_true - y_pred ))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )