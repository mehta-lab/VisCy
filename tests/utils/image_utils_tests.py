import nose.tools
import numpy as np

# Create a test image and its corresponding coordinates and values
# Create a test image with a bright block to the right
from micro_dl.utils import image_utils as image_utils
from tests.preprocessing.mask_utils_tests import uni_thr_tst_image

test_im = np.zeros((10, 15), np.uint16) + 100
test_im[:, 9:] = 200
x, y = np.meshgrid(np.linspace(1, 7, 3), np.linspace(1, 13, 5))
test_coords = np.vstack((x.flatten(), y.flatten())).T
test_values = np.zeros((15,), dtype=np.float64) + 100.
test_values[9:] = 200.


def test_upscale_image():
    im_out = image_utils.rescale_image(test_im, 2)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0] * 2)
    nose.tools.assert_equal(im_shape[1], test_shape[1] * 2)
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_downscale_image():
    im_out = image_utils.rescale_image(test_im, 0.5)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], round(test_shape[0] * .5))
    nose.tools.assert_equal(im_shape[1], round(test_shape[1] * .5))
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_samescale_image():
    im_out = image_utils.rescale_image(test_im, 1)
    im_shape = im_out.shape
    test_shape = test_im.shape
    nose.tools.assert_equal(im_shape[0], test_shape[0])
    nose.tools.assert_equal(im_shape[1], test_shape[1])
    nose.tools.assert_equal(im_out[0, 0], test_im[0, 0])
    nose.tools.assert_equal(im_out[-1, -1], test_im[-1, -1])


def test_fit_polynomial_surface():
    flatfield = image_utils.fit_polynomial_surface_2D(
        test_coords,
        test_values,
        im_shape=(10, 15),
    )
    # Since there's a bright block to the right, the left col should be
    # < right col
    nose.tools.assert_true(np.mean(flatfield[:, 0]) <
                           np.mean(flatfield[:, -1]))
    # Since flatfield is normalized, the mean should be close to one
    nose.tools.assert_almost_equal(np.mean(flatfield), 1., places=3)


def test_rescale_volume():
    # shape (5, 31, 31)
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    # upsample isotropically, 0.5 upsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 1.3)
    nose.tools.assert_tuple_equal(res_volume.shape, (6, 40, 40))
    # upsample anisotropically
    res_volume = image_utils.rescale_nd_image(nd_image, [2.1, 1.1, 1.7])
    nose.tools.assert_tuple_equal(res_volume.shape, (10, 34, 53))
    # downsample isotropically, 0.5 downsampling
    res_volume = image_utils.rescale_nd_image(nd_image, 0.7)
    nose.tools.assert_tuple_equal(res_volume.shape, (4, 22, 22))
    # assertion error

@nose.tools.raises(AssertionError)
def test_rescale_volume_vrong_dims():
    nd_image = np.repeat(uni_thr_tst_image[np.newaxis], 5, axis=0)
    image_utils.rescale_nd_image(nd_image, [1.2, 1.8])
