import nose.tools
import numpy as np
from skimage import draw

import micro_dl.utils.masks as masks_utils


def test_get_unet_border_weight_map():

    # Creating a test image with 3 circles
    # 2 close to each other and one far away
    radius = 10
    params = [(20, 16, radius), (44, 16, radius), (47, 47, radius)]
    mask = np.zeros((64, 64), dtype=np.uint8)
    for i, (cx, cy, radius) in enumerate(params):
        rr, cc = draw.circle(cx, cy, radius)
        mask[rr, cc] = i + 1

    weight_map = masks_utils.get_unet_border_weight_map(mask)

    max_weight_map = np.max(weight_map)
    # weight map between 20, 16 and 44, 16 should be maximum
    # as there is more weight when two objects boundaries overlap
    y_coord = params[0][1]
    for x_coord in range(params[0][0] + radius, params[1][0] - radius):
        distance_near_intersection = weight_map[x_coord, y_coord]
        nose.tools.assert_equal(max_weight_map, distance_near_intersection)
