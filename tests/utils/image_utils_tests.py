import nose.tools
import numpy as np
import numpy.testing

import micro_dl.input.estimate_flat_field
import micro_dl.utils.image_utils as image_utils


# Create a test image and its corresponding coordinates and values
# Create a test image with a bright block to the right
test_im = np.zeros((10, 15), np.uint8) + 100
test_im[:, 9:] = 200
x, y = np.meshgrid(np.linspace(1, 7, 3), np.linspace(1, 13, 5))
test_coords = np.vstack((x.flatten(), y.flatten())).T
test_values = np.zeros((15,), dtype=np.float64) + 100.
test_values[9:] = 200.


def test_sample_block_medians():
    # Test that block sampling is performed correctly
    block_size = 3
    coords, values = micro_dl.input.estimate_flat_field.sample_block_medians(test_im, block_size=block_size)
    # A block size of 3 will create 3*5 block coordinates and values
    np.testing.assert_array_equal(values, test_values)
    np.testing.assert_array_equal(coords, test_coords)


def test_fit_polynomial_surface():
    flatfield = image_utils.fit_polynomial_surface(test_coords,
                                                   test_values,
                                                   im_shape=(10, 15))
    # Since there's a bright block to the right, the left col should be < right col
    nose.tools.assert_true(np.mean(flatfield[:, 0]) < np.mean(flatfield[:, -1]))
    # Since flatfield is normalized, the mean should be close to one
    nose.tools.assert_almost_equal(np.mean(flatfield), 1., places=3)


def test_get_flatfield():
    flatfield = micro_dl.input.estimate_flat_field.get_flatfield(test_im, block_size=3)
    # Since there's a bright block to the right, the left col should be < right col
    nose.tools.assert_true(np.mean(flatfield[:, 0]) < np.mean(flatfield[:, -1]))
    # Since flatfield is normalized, the mean should be close to one
    nose.tools.assert_almost_equal(np.mean(flatfield), 1., places=3)


@nose.tools.raises(ValueError)
def test_negative_flatfield():
    flatfield = micro_dl.input.estimate_flat_field.get_flatfield(
        test_im - 100,
        block_size=3,
    )


def test_preprocess_imstack():
    df_names = ["channel_idx",
                "slice_idx",
                "time_idx",
                "channel_name",
                "file_name",
                "pos_idx"]
    frames_meta = pd.DataFrame(
        columns=df_names,
    )
    im_stack = image_utils.preprocess_imstack(
        time_idx=self.time_idx,
        channel_idx=self.channel_idx,
        slice_idx=16,
        pos_idx=self.pos_idx1,
    )
    self.assertTupleEqual(im_stack.shape, (15, 11, 3))
    im_norm = norm_util.zscore(self.im)
    for z in range(0, 3):
        numpy.testing.assert_array_equal(im_stack[..., z], im_norm)