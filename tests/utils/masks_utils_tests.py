import nose.tools
import numpy as np
from skimage import draw
from skimage.filters import gaussian

from viscy.utils.masks import (
    create_unimodal_mask,
    get_unet_border_weight_map,
    get_unimodal_threshold,
)

uni_thr_tst_image = np.zeros((31, 31))
uni_thr_tst_image[5:10, 8:16] = 127
uni_thr_tst_image[11:21, 2:12] = 97
uni_thr_tst_image[8:12, 3:7] = 31
uni_thr_tst_image[17:29, 17:29] = 61
uni_thr_tst_image[3:14, 17:29] = 47


def test_get_unimodal_threshold():
    input_image = gaussian(uni_thr_tst_image, 1)
    best_thr = get_unimodal_threshold(input_image)
    nose.tools.assert_equal(np.floor(best_thr), 3.0)


def test_unimodal_thresholding():
    input_image = gaussian(uni_thr_tst_image, 1)
    mask = create_unimodal_mask(input_image, str_elem_size=0)
    nose.tools.assert_equal(input_image.shape, mask.shape)
    nose.tools.assert_true(mask.dtype, bool)
    # Check that mask is somewhat close to simple thresholding
    thresh_im = input_image > 3.04
    nose.tools.assert_true(
        np.abs(np.mean(mask) - np.mean(thresh_im)) < 0.1,
    )


def test_get_unet_border_weight_map():
    # Creating a test image with 3 circles
    # 2 close to each other and one far away
    radius = 10
    params = [(20, 16, radius), (44, 16, radius), (47, 47, radius)]
    mask = np.zeros((64, 64), dtype=np.uint8)
    for i, (cx, cy, radius) in enumerate(params):
        rr, cc = draw.disk((cx, cy), radius)
        mask[rr, cc] = i + 1

    weight_map = get_unet_border_weight_map(mask)

    max_weight_map = np.max(weight_map)
    # weight map between 20, 16 and 44, 16 should be maximum
    # as there is more weight when two objects boundaries overlap
    y_coord = params[0][1]
    for x_coord in range(params[0][0] + radius, params[1][0] - radius):
        distance_near_intersection = weight_map[x_coord, y_coord]
        nose.tools.assert_equal(max_weight_map, distance_near_intersection)
