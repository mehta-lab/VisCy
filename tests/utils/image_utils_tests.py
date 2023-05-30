import numpy as np

from viscy.utils.image_utils import grid_sample_pixel_values, preprocess_image


def test_grid_sample_pixel_values():
    im = np.zeros((15, 20))
    row_ids, col_ids, sample_values = grid_sample_pixel_values(
        im,
        grid_spacing=5,
    )
    assert row_ids.tolist() == [5, 5, 5, 10, 10, 10]
    assert col_ids.tolist() == [5, 10, 15, 5, 10, 15]
    assert sample_values.tolist() == [0, 0, 0, 0, 0, 0]


def test_preprocess_image(self):
    im = np.zeros((5, 10, 15, 1))
    im[:, :5, :, :] = 10
    im_proc = preprocess_image(
        im,
        hist_clip_limits=(0, 100),
    )
    self.assertEqual(np.mean(im), np.mean(im_proc))
    self.assertTupleEqual(im_proc.shape, (5, 10, 15))


def test_preprocess_image_norm(self):
    im = np.zeros((5, 10, 15))
    im[:, :5, :] = 10
    im_proc = preprocess_image(
        im,
        normalize_im="dataset",
    )
    self.assertEqual(np.mean(im_proc), 0.0)
    self.assertTupleEqual(im.shape, im_proc.shape)


def test_preprocess_image_mask(self):
    im = np.zeros((5, 10, 15))
    im[:, :5, :] = 10
    im_proc = preprocess_image(
        im,
        is_mask=True,
    )
    self.assertEqual(np.mean(im_proc), 0.5)
    self.assertTupleEqual(im.shape, im_proc.shape)
    self.assertTrue(im_proc.dtype == bool)
