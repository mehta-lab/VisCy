import numpy as np
import pytest

import viscy.evaluation.evaluation_metrics as metrics

im_shape = (35, 45, 25)
target_im = np.ones(im_shape)
for i in range(4):
    target_im[..., i + 1] = i + 1
pred_im = target_im.copy() + 2


def test_mse_metric():
    mse = metrics.mse_metric(target=target_im, prediction=pred_im)
    assert mse[0] == 4


def test_mae_metric():
    mae = metrics.mae_metric(target=target_im, prediction=pred_im)
    assert mae[0] == 2


def test_r2_metric():
    r2 = metrics.r2_metric(target=target_im, prediction=pred_im)
    assert r2[0] < -1


def test_corr_metric():
    corr = metrics.corr_metric(target=target_im, prediction=pred_im)
    assert corr > 0.999


def test_ssim_metric():
    ssim = metrics.ssim_metric(target=target_im, prediction=pred_im, win_size=5)
    assert ssim[0] < 1


@pytest.mark.skip(reason="skip segmentation tests")
def test_accuracy_metric():
    target = np.zeros((5, 10, 2))
    target[:, :5, 0] = 255
    pred = np.zeros_like(target)
    pred[:, :, 0] = 255
    acc = metrics.accuracy_metric(target_bin=target, pred_bin=pred)
    assert acc[0] == 0.75


@pytest.mark.skip(reason="skip segmentation tests")
def test_dice_metric():
    target = np.zeros((5, 10, 2)) + 255.0
    pred = np.zeros_like(target)
    pred[:, :, 0] = 255.0
    dice = metrics.dice_metric(target_bin=target, pred_bin=pred)
    assert dice == 0.5


def test_binarize_array():
    im = np.zeros((4, 3, 2))
    im[..., 1] = 15.0
    im_bin = metrics.binarize_array(im)
    assert len(im_bin) == 24
    assert im_bin.max() == 1


def test_mask_to_bool():
    metrics_list = ["acc", "dice"]
    mask = np.zeros(im_shape)
    mask[5:10, 5:10, :] = 1.0
    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
    )
    mask = metrics_inst.mask_to_bool(mask)
    assert mask.dtype == "bool"


def test_xyz_metrics():
    metrics_list = ["ssim", "corr", "r2", "mse", "mae"]
    pred_name = "test_pred"

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xyz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xyz = metrics_inst.get_metrics_xyz()
    assert metrics_xyz.shape == (1, 6)
    metrics_list.append("pred_name")
    assert list(metrics_xyz) == metrics_list


def test_xy_metrics():
    metrics_list = ["ssim", "corr", "r2"]
    pred_name = "test_pred"

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xy_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xy = metrics_inst.get_metrics_xy()
    assert metrics_xy.shape == (im_shape[2], 4)
    metrics_list.append("pred_name")
    assert list(metrics_xy) == metrics_list


@pytest.mark.skip(reason="skip segmentation tests")
def test_xy_metrics_mask():
    metrics_list = ["corr", "r2"]
    pred_name = "test_pred"
    mask = np.zeros_like(target_im)
    mask[5:10, 5:10, :] = 1
    mask = mask > 0

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=True,
    )
    metrics_inst.estimate_xy_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
        mask=mask,
    )
    metrics_xy = metrics_inst.get_metrics_xy()
    assert metrics_xy.shape == (im_shape[2], 6)
    expected_list = [
        "corr",
        "r2",
        "vol_frac",
        "corr_masked",
        "r2_masked",
        "pred_name",
    ]
    assert list(metrics_xy) == expected_list


@pytest.mark.skip(reason="skip segmentation tests")
def test_xy_metrics_mask_and_segmentation():
    metrics_list = ["corr", "r2", "dice"]
    with pytest.raises(AssertionError):
        metrics.MetricsEstimator(
            metrics_list=metrics_list,
            masked_metrics=True,
        )


@pytest.mark.skip(reason="skip segmentation tests")
def test_xy_metrics_segmentation():
    metrics_list = ["acc", "dice"]
    pred_name = "test_pred"
    mask = np.zeros_like(target_im)
    mask[5:10, 5:10, :] = 1.0

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
    )
    metrics_inst.estimate_xy_metrics(
        target=target_im,
        prediction=mask,
        pred_name=pred_name,
    )
    metrics_xy = metrics_inst.get_metrics_xy()
    assert metrics_xy.shape == (im_shape[2], 3)
    expected_list = [
        "acc",
        "dice",
        "pred_name",
    ]
    assert list(metrics_xy) == expected_list


def test_xz_metrics():
    metrics_list = ["ssim", "corr"]
    pred_name = "test_pred"

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xz = metrics_inst.get_metrics_xz()
    assert metrics_xz.shape == (im_shape[0], 3)
    metrics_list.append("pred_name")
    assert list(metrics_xz) == metrics_list


def test_yz_metrics():
    metrics_list = ["ssim", "r2"]
    pred_name = "test_pred"

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_yz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_yz = metrics_inst.get_metrics_yz()
    assert metrics_yz.shape == (im_shape[1], 3)
    metrics_list.append("pred_name")
    assert list(metrics_yz) == metrics_list
