import keras.backend as K
import nose.tools
import numpy as np
import tensorflow as tf

import micro_dl.train.losses as losses


def test_mae_loss():
    y_true = np.zeros((5, 10, 1), np.float32)
    y_true[:, :5, 0] = 2
    y_pred = np.zeros_like(y_true) + 2.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    mae = losses.mae_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(mae)
        mae_expected = np.zeros((5, 10), np.float32)
        mae_expected[:, 5:] = 2
        np.testing.assert_array_equal(res, mae_expected)


def test_mse_loss():
    y_true = np.zeros((5, 10, 2), np.float32)
    y_true[:, :5, :] = 2
    y_pred = np.zeros_like(y_true) + 2.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    mse = losses.mse_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(mse)
        mse_expected = np.zeros((5, 10), np.float32)
        mse_expected[:, 5:] = 4
        np.testing.assert_array_equal(res, mse_expected)


def test_kl_divergence_loss():
    y_true = np.zeros((5, 10, 2), np.float32)
    y_true[:, :5, :] = .5
    y_pred = np.zeros_like(y_true) + .5
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    kl_loss = losses.kl_divergence_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(kl_loss)
        nose.tools.assert_greater(np.min(res), -1.)
        nose.tools.assert_greater(1., np.min(res))


def test_dssim_loss():
    y_true = np.zeros((25, 20, 1), np.float32)
    y_true[:, :10, :] = 1.
    y_pred = np.zeros_like(y_true) + 1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    dssim_loss = losses.dssim_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(dssim_loss)
        nose.tools.assert_greater(np.min(res), .5)
        nose.tools.assert_greater(1., np.min(res))


def test_ms_ssim_loss():
    y_true = np.zeros((180, 170, 2), np.float32)
    y_true[:, :10, :] = 1.
    y_pred = np.zeros_like(y_true) + 1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    dssim_loss = losses.ms_ssim_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(dssim_loss)
        nose.tools.assert_tuple_equal(res.shape, (180, 170))
        nose.tools.assert_greater(np.min(res), .3)
        nose.tools.assert_greater(1., np.min(res))


def test_dice_loss():
    y_true = np.zeros((25, 20, 1), np.float32)
    y_true[:, :10, :] = 1.
    y_pred = np.zeros_like(y_true)
    y_pred[:, :5, :] = 1.
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    dice_loss = losses.dice_coef_loss(y_true, y_pred)
    with tf.Session() as sess:
        res = sess.run(dice_loss)
        nose.tools.assert_greater(res, .3)
        nose.tools.assert_greater(.4, np.min(res))
