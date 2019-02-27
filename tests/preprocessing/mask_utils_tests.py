import nose.tools
import numpy as np
from skimage.filters import gaussian

import micro_dl.utils.masks as mask_utils

uni_thr_tst_image = np.zeros((31, 31))
uni_thr_tst_image[5:10, 8:16] = 127
uni_thr_tst_image[11:21, 2:12] = 97
uni_thr_tst_image[8:12, 3:7] = 31
uni_thr_tst_image[17:29, 17:29] = 61
uni_thr_tst_image[3:14, 17:29] = 47


def test_get_unimodal_threshold():
    input_image = gaussian(uni_thr_tst_image, 1)
    best_thr = mask_utils.get_unimodal_threshold(input_image)
    nose.tools.assert_equal(np.floor(best_thr), 3.0)


def test_unimodal_thresholding():
    input_image = gaussian(uni_thr_tst_image, 1)
    mask = mask_utils.create_unimodal_mask(input_image,
                                           str_elem_size=0)
    np.testing.assert_array_equal(mask, input_image > 3.04)


