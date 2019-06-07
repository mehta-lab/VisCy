import cv2
import glob
import nose.tools
import numpy as np
import os
from testfixtures import TempDirectory

import micro_dl.plotting.plot_utils as plot_utils


def test_save_predicted_images():
    input_batch = np.zeros((1, 1, 15, 25), dtype=np.uint8)
    target_batch = np.ones((1, 1, 15, 25), dtype=np.uint8)
    pred_batch = np.ones((1, 1, 15, 25), dtype=np.uint8)
    with TempDirectory() as tempdir:
        output_dir = tempdir.path
        output_fname = 'test_plot'
        plot_utils.save_predicted_images(
            input_batch=input_batch,
            target_batch=target_batch,
            pred_batch=pred_batch,
            output_dir=output_dir,
            output_fname=output_fname,
        )
        fig_glob = glob.glob(os.path.join(output_dir, '*'))
        nose.tools.assert_equal(len(fig_glob), 1)
        expected_fig = os.path.join(output_dir, 'test_plot.jpg')
        nose.tools.assert_equal(fig_glob[0], expected_fig)


def test_save_center_slices():
    with TempDirectory() as tempdir:
        image_dir = tempdir.path
        save_path = os.path.join(image_dir, 'test_plot.png')
        im = np.zeros((10, 15), dtype=np.uint8)
        for i in range(5):
            im_name = 'im_c005_z00{}_t000_p050.png'.format(i)
            cv2.imwrite(os.path.join(image_dir, im_name), im + i * 10)
        plot_utils.save_center_slices(
            image_dir=image_dir,
            pos_idx=50,
            save_path=save_path,
            mean_std=(100, 10),
            channel_str='c005',
            fig_title='test fig',
        )
        fig_glob = glob.glob(os.path.join(image_dir, '*test_plot*'))
        nose.tools.assert_equal(len(fig_glob), 1)
        nose.tools.assert_equal(fig_glob[0], save_path)


def test_save_mask_overlay():
    with TempDirectory() as tempdir:
        image_dir = tempdir.path
        save_path = os.path.join(image_dir, 'test_plot.png')
        im = 100 * np.ones((10, 15), dtype=np.uint8)
        mask = np.zeros((10, 15), dtype=np.uint8)
        mask[5:7, 5:8] = 1
        plot_utils.save_mask_overlay(im, mask, save_path)
        fig_glob = glob.glob(os.path.join(image_dir, '*'))
        nose.tools.assert_equal(len(fig_glob), 1)
        nose.tools.assert_equal(fig_glob[0], save_path)


def test_save_plot():
    with TempDirectory() as tempdir:
        write_dir = tempdir.path
        save_path = os.path.join(write_dir, 'test_plot.png')
        x = np.arange(10)
        y = [5] * len(x)
        plot_utils.save_plot(x, y, save_path)
        fig_glob = glob.glob(os.path.join(write_dir, '*'))
        nose.tools.assert_equal(len(fig_glob), 1)
        nose.tools.assert_equal(fig_glob[0], save_path)
