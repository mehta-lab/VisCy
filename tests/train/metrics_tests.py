import keras.backend as K
import nose.tools
import numpy as np
import tensorflow as tf

import micro_dl.train.metrics as metrics


def test_coeff_determination():
    y_true = np.zeros((5, 1, 25, 30), np.float32)
    y_true[..., :10] = 1
    y_pred = np.zeros_like(y_true)
    y_pred[..., :20] = 1
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    r2 = metrics.coeff_determination(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(r2)
        nose.tools.assert_greater(1., res)
        nose.tools.assert_greater(res, -1.)


def test_dice_coef_partial():
    y_true = np.zeros((5, 7, 3), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[1:3, 1:3, 1] = 1.
    y_pred[2:4, 2:4, 1] = 1.
    # y_true and y_pred both have 2x2 ones, with 1x1 ones overlapping
    # so expected DSC is 2*1/(4+4)
    expected_dice = 0.25
    dice = metrics.dice_coef(y_true=y_true, y_pred=y_pred, smooth=0.)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(dice), expected_dice)


def test_dice_coef_complete():
    y_true = np.zeros((5, 7, 3), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[2:4, 2:4, 1] = 1.
    y_pred[2:4, 2:4, 1] = 1.
    # y_true and y_pred both have 2x2 ones, with 2x2 ones overlapping
    # so expected DSC is 2*2*2/(4+4)
    expected_dice = 1.
    dice = metrics.dice_coef(y_true=y_true, y_pred=y_pred, smooth=0.)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(dice), expected_dice)


def test_dice_coef_none():
    y_true = np.zeros((5, 7, 3), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[1:3, 1:3, 1] = 1.
    y_pred[2:4, 5:7, 1] = 1.
    # y_true and y_pred both have 2x2 ones, with 2x2 ones overlapping
    # so expected DSC is 2*0/(4+4)
    expected_dice = 0.
    dice = metrics.dice_coef(y_true=y_true, y_pred=y_pred, smooth=0.)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(dice), expected_dice)


def test_ssim():
    y_true = np.zeros((25, 30, 3), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[10:24, 5:27, :] = 1.
    y_pred[10:24, 5:27, :] = 1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    ssim = metrics.ssim(y_true=y_true, y_pred=y_pred)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(ssim), 1.0)


def test_ms_ssim():
    y_true = np.zeros((175, 230, 3), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[50:150, 5:27, :] = 1.
    y_pred[50:150, 5:27, :] = 1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    ssim = metrics.ms_ssim(y_true=y_true, y_pred=y_pred)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(ssim), 1.)


def test_ms_ssim_negative_cov():
    y_true = np.zeros((175, 230, 3), np.float) - 1.
    y_pred = np.zeros_like(y_true)
    y_true[50:150, 5:27, :] = 1.
    y_pred[50:150, 5:27, :] = -1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    ssim = metrics.ms_ssim(y_true=y_true, y_pred=y_pred)
    with tf.Session() as sess:
        nose.tools.assert_equal(sess.run(ssim), 0.)


def test_ssim_channels_first():
    K.set_image_data_format('channels_first')
    y_true = np.zeros((5, 1, 7, 25, 30), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[2, :, :, 10:24, 5:27] = 1.
    y_pred[2, :, :, 10:24, 5:20] = 1.
    y_true = K.tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = K.tf.convert_to_tensor(y_pred, dtype=tf.float32)
    ssim = metrics.ssim(y_true=y_true, y_pred=y_pred)
    with K.tf.Session() as sess:
        K.set_session(sess)
        nose.tools.assert_greater(1, sess.run(ssim))
        nose.tools.assert_greater(sess.run(ssim), 0.9)


def test_ms_ssim_channels_first():
    K.set_image_data_format('channels_first')
    y_true = np.zeros((5, 1, 7, 170, 180), np.float)
    y_pred = np.zeros_like(y_true)
    y_true[2, :, :, 100:140, 5:127] = 1.
    y_pred[2, :, :, 100:140, 5:120] = 1.
    y_true = K.tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = K.tf.convert_to_tensor(y_pred, dtype=tf.float32)
    ssim = metrics.ms_ssim(y_true=y_true, y_pred=y_pred)
    with K.tf.Session() as sess:
        K.set_session(sess)
        res = sess.run(ssim)
        nose.tools.assert_greater(1, res)
        nose.tools.assert_greater(res, 0.9)


def test_pearson_corr():
    y_true = np.zeros((5, 1, 25, 30), np.float32)
    y_true[:4, ...] = 1
    y_pred = np.zeros_like(y_true)
    y_pred[:2, ...] = 1
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    r = metrics.pearson_corr(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(r)
        # Pearson should be 0.4 <= res <= 0.5
        nose.tools.assert_greater(0.5, res)
        nose.tools.assert_greater(res, 0.4)
