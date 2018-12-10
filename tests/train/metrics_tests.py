import nose.tools
import numpy as np
import tensorflow as tf

import micro_dl.train.metrics as metrics


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