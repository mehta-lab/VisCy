"""Tests for whole-cell watershed segmentation + slice selection."""

import numpy as np
import pytest
import torch

from dynacell.evaluation.segmentation_whole_cell import segment_whole_cell, slice_index

_HAS_CUDA = torch.cuda.is_available()


def test_slice_index_frac():
    """frac selection returns round(fraction * (Z - 1))."""
    vol = np.zeros((11, 4, 4), dtype=np.float32)
    assert slice_index(vol, selection="frac", fraction=0.30) == 3  # round(0.30 * 10)
    assert slice_index(vol, selection="frac", fraction=0.0) == 0
    assert slice_index(vol, selection="frac", fraction=1.0) == 10


def test_slice_index_sharpest():
    """sharpest selection returns the highest-variance plane."""
    vol = np.zeros((5, 8, 8), dtype=np.float32)
    vol[3] = np.tile([0.0, 1.0], (8, 4))  # high-variance checker on plane 3
    assert slice_index(vol, selection="sharpest") == 3


def test_slice_index_invalid():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        slice_index(vol, selection="bogus")


def _two_cell_synthetic_2d(shape=(128, 128)):
    """Two cytoplasmic squares split by a bright membrane wall, two nuclei seeds."""
    h, w = shape
    memb = np.zeros(shape, dtype=np.float32)
    nuc = np.zeros(shape, dtype=np.float32)
    seed = np.zeros(shape, dtype=np.uint16)
    # cytoplasm fill for both cells
    memb[20:108, 16:60] = 0.4
    memb[20:108, 68:112] = 0.4
    # bright walls: outer border + central divider
    memb[18:20, 16:112] = 1.0
    memb[108:110, 16:112] = 1.0
    memb[18:110, 60:68] = 1.0  # divider
    memb[18:110, 14:16] = 1.0
    memb[18:110, 112:114] = 1.0
    # nuclei (fluorescence + seed labels) at the two cell centers
    nuc[56:72, 30:46] = 1.0
    nuc[56:72, 82:98] = 1.0
    seed[56:72, 30:46] = 1
    seed[56:72, 82:98] = 2
    return memb, nuc, seed


@pytest.mark.skipif(not _HAS_CUDA, reason="whole-cell watershed requires a CUDA GPU (cubic)")
def test_segment_whole_cell_2d_one_cell_per_seed():
    """A clean 2-cell synthetic yields one cytoplasm label per seed, walls separating."""
    memb, nuc, seed = _two_cell_synthetic_2d()
    cells = segment_whole_cell(memb, nuc, seed, (0.3, 0.3), cell_voxel_um=0.3, memb_clahe=False, subtract_nuclei=False)
    assert cells.dtype == np.uint16
    assert cells.shape == memb.shape
    n_cells = int(cells.max())
    assert 1 <= n_cells <= 2  # never more cells than seeds
    # the two seed centers fall in distinct cell labels (walls keep them apart)
    left = cells[64, 38]
    right = cells[64, 90]
    assert left != 0 and right != 0 and left != right


@pytest.mark.skipif(not _HAS_CUDA, reason="whole-cell watershed requires a CUDA GPU (cubic)")
def test_segment_whole_cell_subtract_nuclei_carves_seeds():
    """subtract_nuclei zeros the nucleus footprint (cytoplasm-only labels)."""
    memb, nuc, seed = _two_cell_synthetic_2d()
    carved = segment_whole_cell(memb, nuc, seed, (0.3, 0.3), cell_voxel_um=0.3, memb_clahe=False, subtract_nuclei=True)
    assert not carved[seed > 0].any()  # every nucleus voxel is background
    assert int(carved.max()) >= 1  # cytoplasmic shells remain


@pytest.mark.skipif(not _HAS_CUDA, reason="whole-cell watershed requires a CUDA GPU (cubic)")
def test_segment_whole_cell_downscale_roundtrip():
    """cell_voxel_um != native downscales then NN-upscales labels back to native shape."""
    memb, nuc, seed = _two_cell_synthetic_2d()
    cells = segment_whole_cell(memb, nuc, seed, (0.3, 0.3), cell_voxel_um=0.58, memb_clahe=False, subtract_nuclei=False)
    assert cells.dtype == np.uint16
    assert cells.shape == memb.shape  # back at native resolution


@pytest.mark.skipif(not _HAS_CUDA, reason="whole-cell watershed requires a CUDA GPU (cubic)")
def test_segment_whole_cell_3d_runs_full_volume():
    """The 3D path runs on (Z, Y, X) and returns native-shape uint16 labels."""
    memb2d, nuc2d, seed2d = _two_cell_synthetic_2d(shape=(96, 96))
    z = 9
    memb = np.broadcast_to(memb2d, (z, *memb2d.shape)).astype(np.float32).copy()
    nuc = np.broadcast_to(nuc2d, (z, *nuc2d.shape)).astype(np.float32).copy()
    seed = np.broadcast_to(seed2d, (z, *seed2d.shape)).astype(np.uint16).copy()
    cells = segment_whole_cell(
        memb, nuc, seed, (0.3, 0.3, 0.3), cell_voxel_um=0.3, memb_clahe=False, subtract_nuclei=False
    )
    assert cells.dtype == np.uint16
    assert cells.shape == memb.shape
    assert int(cells.max()) >= 1
