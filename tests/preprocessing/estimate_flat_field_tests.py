import nose.tools
import numpy as np

import micro_dl.preprocessing.estimate_flat_field as est_flat_field


test_im = np.zeros((10, 15), np.uint8) + 100
test_im[:, 9:] = 200
x, y = np.meshgrid(np.linspace(1, 7, 3), np.linspace(1, 13, 5))
test_coords = np.vstack((x.flatten(), y.flatten())).T
test_values = np.zeros((15,), dtype=np.float64) + 100.
test_values[9:] = 200.


# TODO: Tests broke when flatfield became a class. Fix!
