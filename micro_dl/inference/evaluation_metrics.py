"""Metrics for model evaluation"""
import functools
import numpy as np
import pandas as pd
from skimage.measure import compare_ssim as ssim
import sklearn.metrics
from scipy.stats import pearsonr


def mask_decorator(metric_function):
    """Decorator for masking the metrics

    :param function metric_function: a python function that takes target and
     prediction arrays as input
    :return function wrapper_metric_function: input function that returns
     metrics and masked_metrics if mask was passed as param to input function
    """

    @functools.wraps(metric_function)
    def wrapper_metric_function(**kwargs):
        """Expected inputs cur_target, prediction, mask

        :param dict kwargs: with keys target, prediction and mask all of which
         are np.arrays
        """

        metric = metric_function(target=kwargs['target'],
                                 prediction=kwargs['prediction'])
        if 'mask' in kwargs:
            mask = kwargs['mask']
            cur_target = kwargs['target']
            cur_pred = kwargs['prediction']
            masked_metric = metric_function(target=cur_target[mask],
                                            prediction=cur_pred[mask])
            return [metric, masked_metric]
        return metric
    return wrapper_metric_function


@mask_decorator
def mse_metric(target, prediction):
    """MSE of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float mean squared error
    """
    return np.mean((target - prediction) ** 2)


@mask_decorator
def mae_metric(target, prediction):
    """MAE of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float mean absolute error
    """
    return np.mean(np.abs(target - prediction))


@mask_decorator
def r2_metric(target, prediction):
    """Coefficient of determination of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float coefficient of determination
    """
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    cur_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return cur_r2


@mask_decorator
def corr_metric(target, prediction):
    """Pearson correlation of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float Pearson correlation
    """
    cur_corr = pearsonr(target.flatten(), prediction.flatten())[0]
    return cur_corr


def ssim_metric(target,
                prediction,
                mask=None,
                win_size=21):
    """
    Structural similarity indiex (SSIM) of target and prediction.
    Window size is not passed into function so make sure tiles
    are never smaller than default win_size.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :param np.array/None mask: Mask
    :param int win_size: window size for computing local SSIM
    :return float/list ssim and ssim_masked
    """
    if mask is None:
        cur_ssim = ssim(
            target,
            prediction,
            win_size=win_size,
            data_range=target.max() - target.min(),
        )
        return cur_ssim
    else:
        cur_ssim, cur_ssim_img = ssim(
            target,
            prediction,
            data_range=target.max() - target.min(),
            full=True,
        )
        cur_ssim_masked = np.mean(cur_ssim_img[mask])
        return [cur_ssim, cur_ssim_masked]


def accuracy_metric(target, prediction):
    """Accuracy of binary target and prediction.
    Not using mask decorator for binary data evaluation.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float Accuracy: Accuracy for binarized data
    """
    target_bin = binarize_array(target)
    pred_bin = binarize_array(prediction)
    return sklearn.metrics.accuracy_score(target_bin, pred_bin)


def dice_metric(target, prediction):
    """Dice similarity coefficient (F1 score) of binary target and prediction.
    Reports global metric.
    Not using mask decorator for binary data evaluation.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float dice: Dice for binarized data
    """
    target_bin = binarize_array(target)
    pred_bin = binarize_array(prediction)
    return sklearn.metrics.f1_score(target_bin, pred_bin, average='micro')


def binarize_array(im):
    """Binarize image

    :param np.array im: Prediction or target array
    :return np.array im_bin: Flattened and binarized array
    """
    im_bin = (im.flatten() / im.max()) > .5
    return im_bin.astype(np.uint8)


class MetricsEstimator:
    """Estimate metrics for evaluating a trained model"""

    def __init__(self,
                 metrics_list,
                 masked_metrics=False):
        """
        Init. After instantiating the class you can call metrics estimation
        in xy, xz, yz and xyz orientations assuming images are of shape xyz.
        The first three indices will iterate over planes whereas xyz will
        generate one global 3D metric.

        :param list metrics_list: list of strings with name of metrics
            Currently available metrics:
            'ssim' - Structual similarity index
            'corr' - Correlation
            'r2' -   R squared (coefficient of determination
            'mse' -  Mean squared error
            'mae' -  Mean absolute error
            'acc' -  Accuracy (for binary data, no masks)
            'dice' - Dice similarity coefficient (for binary data, no masks)
        :param bool masked_metrics: get the metrics for the masked region
        """

        available_metrics = {'ssim', 'corr', 'r2', 'mse', 'mae', 'acc', 'dice'}
        assert set(metrics_list).issubset(available_metrics), \
            'only ssim, r2, corr, mse, mae, acc, dice are currently supported'
        self.metrics_list = metrics_list
        self.pd_col_names = metrics_list.copy()
        self.masked_metrics = masked_metrics
        self.metrics_xyz = None
        self.metrics_xy = None
        self.metrics_xz = None
        self.metrics_yz = None
        # No masking for evaluating segmentations (which are masks)
        if 'acc' in metrics_list or 'dice' in metrics_list:
            assert not self.masked_metrics, \
                "Don't use masked metrics if evaluating segmentation"

        if self.masked_metrics:
            self.pd_col_names.append('vol_frac')
            for metric in metrics_list:
                cur_col_name = '{}_masked'.format(metric)
                self.pd_col_names.append(cur_col_name)

        self.pd_col_names.append('pred_name')
        self.fn_mapping = {
            'mae_metric': mae_metric,
            'mse_metric': mse_metric,
            'r2_metric': r2_metric,
            'corr_metric': corr_metric,
            'ssim_metric': ssim_metric,
            'acc_metric': accuracy_metric,
            'dice_metric': dice_metric,
        }

    def get_metrics_xyz(self):
        """Return 3D metrics"""
        return self.metrics_xyz

    def get_metrics_xy(self):
        """Return xy metrics"""
        return self.metrics_xy

    def get_metrics_xz(self):
        """Return xz metrics"""
        return self.metrics_xz

    def get_metrics_yz(self):
        """Return yz metrics"""
        return self.metrics_yz

    @staticmethod
    def mask_to_bool(mask):
        """
        If mask exists and is not boolean, convert.
        Assume mask values == 0 is background

        :param np.array mask: Mask
        :return np.array mask: Mask with boolean dtype
        """
        if mask is not None:
            if mask.dtype != 'bool':
                mask = mask > 0
        return mask

    @staticmethod
    def assert_input(target,
                     prediction,
                     pred_name,
                     mask=None):
        assert isinstance(pred_name, str), \
            'more than one pred_name is passed. Only one target-pred pair ' \
            'is handled per function call'
        assert target.shape == prediction.shape, \
            'The shape of target and prediction are not same: {}, {}'.format(
                target.shape, prediction.shape
            )
        assert target.dtype == prediction.dtype, \
            'The dtype of target and prediction are not same: {}, {}'.format(
                target.dtype, prediction.dtype
            )
        if mask is not None:
            assert target.shape == mask.shape, \
                'The shape of target and mask are not same: {}, {}'.format(
                    target.shape, mask.shape
                )
            assert mask.dtype == 'bool', 'mask is not boolean'

    def compute_metrics_row(self,
                            target,
                            prediction,
                            pred_name,
                            mask):
        """
        Compute one row in metrics dataframe.

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param str pred_name: filename used for saving model prediction
        :param np.array mask: binary mask with foreground / background
        :return: dict metrics_row: a row for a metrics dataframe
        """
        metrics_row = dict.fromkeys(self.pd_col_names)
        metrics_row['pred_name'] = pred_name
        for metric_name in self.metrics_list:
            metric_fn_name = '{}_metric'.format(metric_name)
            metric_fn = self.fn_mapping[metric_fn_name]
            if self.masked_metrics:
                cur_metric_list = metric_fn(
                    target=target,
                    prediction=prediction,
                    mask=mask,
                )
                vol_frac = np.mean(mask)
                metrics_row['vol_frac'] = vol_frac
                metrics_row[metric_name] = cur_metric_list[0]
                metric_name = '{}_masked'.format(metric_name)
                metrics_row[metric_name] = cur_metric_list[1]
            else:
                cur_metric = metric_fn(
                    target=target,
                    prediction=prediction,
                )
                metrics_row[metric_name] = cur_metric
        return metrics_row

    def estimate_xyz_metrics(self,
                             target,
                             prediction,
                             pred_name,
                             mask=None):
        """
        Estimate 3D metrics for the current input, target pair

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param str pred_name: filename used for saving model prediction
        :param np.array mask: binary mask with foreground / background
        """
        mask = self.mask_to_bool(mask)
        self.assert_input(target, prediction, pred_name, mask)
        self.metrics_xyz = pd.DataFrame(columns=self.pd_col_names)
        metrics_row = self.compute_metrics_row(
            target=target,
            prediction=prediction,
            pred_name=pred_name,
            mask=mask,
        )
        # Append to existing dataframe
        self.metrics_xyz = self.metrics_xyz.append(
            metrics_row,
            ignore_index=True,
        )
        print('metrics xyz')
        print(self.metrics_xyz)

    def estimate_xy_metrics(self,
                            target,
                            prediction,
                            pred_name,
                            mask=None):
        """
        Estimate metrics for the current input, target pair
        along each xy slice (in plane)

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param str/list pred_name: filename(s) used for saving model prediction
        :param np.array mask: binary mask with foreground / background
        """
        mask = self.mask_to_bool(mask)
        self.assert_input(target, prediction, pred_name, mask)
        if len(target.shape) == 2:
            target = target[..., np.newaxis]
            prediction = prediction[..., np.newaxis]
        self.metrics_xy = pd.DataFrame(columns=self.pd_col_names)
        # Loop through slices
        for slice_idx in range(target.shape[2]):
            slice_name = "{}_xy{}".format(pred_name, slice_idx)
            cur_mask = mask[..., slice_idx] if mask is not None else None
            metrics_row = self.compute_metrics_row(
                target=target[..., slice_idx],
                prediction=prediction[..., slice_idx],
                pred_name=slice_name,
                mask=cur_mask,
            )
            # Append to existing dataframe
            self.metrics_xy = self.metrics_xy.append(
                metrics_row,
                ignore_index=True,
            )

    def estimate_xz_metrics(self,
                            target,
                            prediction,
                            pred_name,
                            mask=None):
        """
        Estimate metrics for the current input, target pair
        along each xz slice

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param str pred_name: filename used for saving model prediction
        :param np.array mask: binary mask with foreground / background
        """
        mask = self.mask_to_bool(mask)
        self.assert_input(target, prediction, pred_name, mask)
        assert len(target.shape) == 3, 'Dataset is assumed to be 3D'
        self.metrics_xz = pd.DataFrame(columns=self.pd_col_names)
        # Loop through slices
        for slice_idx in range(target.shape[0]):
            slice_name = "{}_xz{}".format(pred_name, slice_idx)
            cur_mask = mask[slice_idx, ...] if mask is not None else None
            metrics_row = self.compute_metrics_row(
                target=target[slice_idx, ...],
                prediction=prediction[slice_idx, ...],
                pred_name=slice_name,
                mask=cur_mask,
            )
            # Append to existing dataframe
            self.metrics_xz = self.metrics_xz.append(
                metrics_row,
                ignore_index=True,
            )

    def estimate_yz_metrics(self,
                            target,
                            prediction,
                            pred_name,
                            mask=None):
        """
        Estimate metrics for the current input, target pair
        along each yz slice

        :param np.array target: ground truth
        :param np.array prediction: model prediction
        :param str pred_name: filename used for saving model prediction
        :param np.array mask: binary mask with foreground / background
        """
        mask = self.mask_to_bool(mask)
        self.assert_input(target, prediction, pred_name, mask)
        assert len(target.shape) == 3, 'Dataset is assumed to be 3D'
        self.metrics_yz = pd.DataFrame(columns=self.pd_col_names)
        # Loop through slices
        for slice_idx in range(target.shape[1]):
            slice_name = "{}_yz{}".format(pred_name, slice_idx)
            cur_mask = mask[:, slice_idx, :] if mask is not None else None
            metrics_row = self.compute_metrics_row(
                target=target[:, slice_idx, :],
                prediction=prediction[:, slice_idx, :],
                pred_name=slice_name,
                mask=cur_mask,
            )
            # Append to existing dataframe
            self.metrics_yz = self.metrics_yz.append(
                metrics_row,
                ignore_index=True,
            )
