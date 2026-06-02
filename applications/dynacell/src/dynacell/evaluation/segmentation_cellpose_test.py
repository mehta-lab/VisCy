"""Tests for the GPU Cellpose-SAM nucleus segmentation backend."""

import numpy as np
import pytest
import torch

from dynacell.evaluation import segmentation
from dynacell.evaluation.cache import cache_paths

_HAS_CUDA = torch.cuda.is_available()
_SPACING = (0.30, 0.11, 0.11)


def test_mask_plate_backend_routing(tmp_path):
    """Default backend keeps the bare path; cellpose gets a ``__cellpose`` infix."""
    paths = cache_paths(tmp_path)
    assert paths.mask_plate("nucleus").name == "nucleus.zarr"
    assert paths.mask_plate("nucleus", "supermodel").name == "nucleus.zarr"
    assert paths.mask_plate("nucleus", "cellpose").name == "nucleus__cellpose.zarr"


def test_instance_mask_plate_routing(tmp_path):
    """Instance plates live under instance_masks/ with a ``__{backend}`` infix."""
    paths = cache_paths(tmp_path)
    assert paths.instance_masks_dir == tmp_path / "instance_masks"
    assert paths.instance_mask_plate("nucleus", "cellpose").name == "nucleus__cellpose.zarr"
    assert (
        paths.instance_mask_plate("membrane", "cellpose_watershed")
        == tmp_path / "instance_masks" / "membrane__cellpose_watershed.zarr"
    )


@pytest.fixture(scope="module")
def cellpose_model():
    if not _HAS_CUDA:
        pytest.skip("Cellpose-SAM segmentation requires a CUDA GPU")
    from dynacell.evaluation.segmentation_cellpose import load_cellpose_model

    return load_cellpose_model(use_gpu=True)


def _synthetic_nucleus(shape=(24, 256, 256)) -> np.ndarray:
    """A single bright ellipsoid on a noisy background."""
    z, y, x = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = ((z - cz) / 6) ** 2 + ((y - cy) / 40) ** 2 + ((x - cx) / 40) ** 2
    img = np.exp(-r2) * 2000.0
    rng = np.random.default_rng(0)
    return (img + rng.normal(50, 10, shape)).astype(np.float32)


def test_segment_nucleus_shape_and_dtype(cellpose_model):
    """Real end-to-end run returns a bool mask at native shape."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus

    img = _synthetic_nucleus()
    mask = segment_nucleus(img, _SPACING, cellpose_model)
    assert mask.dtype == bool
    assert mask.shape == img.shape


def test_segment_nucleus_empty_input(cellpose_model):
    """All-zero input yields an all-False native-shape mask (no cleanup crash)."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus

    img = np.zeros((24, 256, 256), dtype=np.float32)
    mask = segment_nucleus(img, _SPACING, cellpose_model)
    assert mask.shape == img.shape
    assert not mask.any()


def test_segment_dispatch_cellpose_backend(cellpose_model):
    """segment(..., backend='cellpose') routes nucleus to the cellpose pipeline."""
    img = _synthetic_nucleus()
    mask = segmentation.segment(img, "nucleus", seg_model=cellpose_model, backend="cellpose", spacing_zyx=_SPACING)
    assert mask.dtype == bool
    assert mask.shape == img.shape


def test_segment_cellpose_requires_spacing_and_model():
    """cellpose nucleus dispatch raises without seg_model / spacing."""
    img = np.zeros((4, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        segmentation.segment(img, "nucleus", seg_model=None, backend="cellpose", spacing_zyx=_SPACING)
    with pytest.raises(ValueError):
        segmentation.segment(img, "nucleus", seg_model=object(), backend="cellpose", spacing_zyx=None)


def test_segment_cellpose_membrane_not_implemented():
    """cellpose backend is nucleus-only for now."""
    img = np.zeros((4, 32, 32), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        segmentation.segment(img, "membrane", seg_model=object(), backend="cellpose", spacing_zyx=_SPACING)


def test_prepare_segmentation_model_cellpose_requires_cuda():
    """Cellpose backends fail fast (clear error) when run without CUDA.

    ``segment_cpsam`` is GPU-only; building a CPU CellposeModel would only crash
    deeper in the per-FOV loop. ``use_gpu=False`` trips the guard regardless of
    whether the test host has a GPU.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "target_name": "nucleus",
            "use_gpu": False,
            "compute_instance_ap": True,
            "segmentation": {"backend": "cellpose"},
            "io": {"require_complete_cache": False},
        }
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        segmentation.prepare_segmentation_model(cfg)


def test_segment_nucleus_return_labels(cellpose_model):
    """return_labels=True yields native-shape uint16 instances relabeled 1..K."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus

    img = _synthetic_nucleus()
    labels = segment_nucleus(img, _SPACING, cellpose_model, return_labels=True)
    assert labels.dtype == np.uint16
    assert labels.shape == img.shape
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size:  # sequential 1..K with no gaps
        assert ids.tolist() == list(range(1, int(ids.max()) + 1))


def test_segment_nucleus_bool_parity(cellpose_model):
    """The bool footprint equals labels>0 for identical params."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus

    img = _synthetic_nucleus()
    mask = segment_nucleus(img, _SPACING, cellpose_model, return_labels=False)
    labels = segment_nucleus(img, _SPACING, cellpose_model, return_labels=True)
    np.testing.assert_array_equal(mask, labels > 0)


def test_segment_nucleus_2d_slice(cellpose_model):
    """do_3d=False on a (Y, X) slice returns 2D labels at native slice shape."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus_instances

    sl = _synthetic_nucleus()[12]  # mid-z (Y, X)
    labels = segment_nucleus_instances(sl, (0.11, 0.11), cellpose_model, do_3d=False, min_obj_size=10)
    assert labels.dtype == np.uint16
    assert labels.shape == sl.shape  # true 2D, no singleton Z
    assert labels.ndim == 2


def test_segment_nucleus_empty_labels(cellpose_model):
    """All-zero input returns an all-zero uint16 label image."""
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus_instances

    img = np.zeros((24, 256, 256), dtype=np.float32)
    labels = segment_nucleus_instances(img, _SPACING, cellpose_model)
    assert labels.dtype == np.uint16
    assert labels.shape == img.shape
    assert not labels.any()
