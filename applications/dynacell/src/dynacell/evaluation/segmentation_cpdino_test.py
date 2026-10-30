"""Tests for the GPU-resident Cellpose-DINO (cpdino) instance segmentation backend."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dynacell.evaluation import segmentation, segmentation_cpdino
from dynacell.evaluation.segmentation_cpdino import segment_whole_cell_cpdino

_HAS_CUDA = torch.cuda.is_available()
_SPACING_2D = (0.1494, 0.1494)


# ---------------------------------------------------------------------------
# CPU-only logic: the whole-cell carve, dispatch guards, cache identity.
# ---------------------------------------------------------------------------


def _stub_instances(monkeypatch, labels: np.ndarray) -> None:
    """Make segment_cpdino_instances return a fixed label image (no GPU/model needed)."""
    monkeypatch.setattr(segmentation_cpdino, "segment_cpdino_instances", lambda *a, **k: labels)


def test_whole_cell_cpdino_carves_nucleus(monkeypatch):
    """subtract_nuclei=True zeros the nucleus footprint out of the whole-cell labels."""
    cells = np.zeros((8, 8), dtype=np.uint16)
    cells[1:7, 1:7] = 3  # one big whole-cell label
    _stub_instances(monkeypatch, cells)
    seed = np.zeros((8, 8), dtype=np.uint16)
    seed[3:5, 3:5] = 1  # nucleus footprint interior to the cell

    out = segment_whole_cell_cpdino(np.zeros((8, 8), np.float32), seed, _SPACING_2D, object(), subtract_nuclei=True)
    assert out.dtype == np.uint16
    assert (out[3:5, 3:5] == 0).all()  # nucleus carved out
    assert (out[1:3, 1:3] == 3).all()  # cytoplasmic shell preserved
    # the carve must not mutate the (stubbed) input in place
    assert (cells[3:5, 3:5] == 3).all()


def test_whole_cell_cpdino_no_subtract_keeps_cells(monkeypatch):
    """subtract_nuclei=False returns the whole-cell labels untouched."""
    cells = np.full((6, 6), 2, dtype=np.uint16)
    _stub_instances(monkeypatch, cells)
    seed = np.ones((6, 6), dtype=np.uint16)
    out = segment_whole_cell_cpdino(np.zeros((6, 6), np.float32), seed, _SPACING_2D, object(), subtract_nuclei=False)
    np.testing.assert_array_equal(out, cells)


def test_whole_cell_cpdino_none_seed_is_noop(monkeypatch):
    """A None seed (guaranteed cache hit path) leaves the labels intact."""
    cells = np.full((5, 5), 7, dtype=np.uint16)
    _stub_instances(monkeypatch, cells)
    out = segment_whole_cell_cpdino(np.zeros((5, 5), np.float32), None, _SPACING_2D, object(), subtract_nuclei=True)
    np.testing.assert_array_equal(out, cells)


def test_prepare_segmentation_model_cpdino_requires_cuda():
    """cpdino (like the other cellpose backends) fails fast without CUDA."""
    cfg = OmegaConf.create(
        {
            "target_name": "membrane",
            "use_gpu": False,
            "compute_instance_ap": True,
            "segmentation": {"backend": "cpdino", "nuclei_channel_name": "Nuclei", "cpdino": {"model_name": "cpdino"}},
            "io": {"require_complete_cache": False},
        }
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        segmentation.prepare_segmentation_model(cfg)


def _instance_ap_cfg(backend, target_name, *, nuclei=None, compute_ap=True):
    return OmegaConf.create(
        {
            "target_name": target_name,
            "compute_instance_ap": compute_ap,
            "segmentation": {"backend": backend, "nuclei_channel_name": nuclei},
        }
    )


def test_validate_cpdino_valid_for_both_targets():
    """cpdino is valid for nucleus (no nuclei channel) and membrane (with one)."""
    from dynacell.evaluation.pipeline import _validate_instance_ap_config

    _validate_instance_ap_config(_instance_ap_cfg("cpdino", "nucleus"))
    _validate_instance_ap_config(_instance_ap_cfg("cpdino", "membrane", nuclei="Nuclei"))


def test_validate_cpdino_membrane_requires_nuclei_channel():
    """cpdino whole-cell needs a nuclei channel for the carve seeds."""
    from dynacell.evaluation.pipeline import _validate_instance_ap_config

    with pytest.raises(ValueError, match="nuclei_channel_name"):
        _validate_instance_ap_config(_instance_ap_cfg("cpdino", "membrane", nuclei=None))


def test_validate_cpdino_requires_compute_instance_ap():
    """cpdino is an instance backend; it requires compute_instance_ap=true."""
    from dynacell.evaluation.pipeline import _validate_instance_ap_config

    with pytest.raises(ValueError, match="compute_instance_ap"):
        _validate_instance_ap_config(_instance_ap_cfg("cpdino", "nucleus", compute_ap=False))


def _ctx(backend, target_name, **over):
    from dynacell.evaluation.pipeline_cache import _CacheContext

    base = dict(
        paths=None,
        manifest={},
        force={},
        require_complete=False,
        side="gt",
        target_name=target_name,
        spacing=[0.29, 0.108, 0.108],
        patch_size=224,
        backend=backend,
        dimension="2d",
        slice_selection="focus",
        slice_fraction=0.30,
        nuclei_channel_name="Nuclei",
        nuclei_plate_path="/x/H2B.ozx",
        cellpose_params={"min_obj_size": 30, "flow_threshold": 0.4},
        cpdino_params={"model_name": "cpdino", "normalize": True, "min_size": 15, "subtract_nuclei": True},
    )
    base.update(over)
    return _CacheContext(**base)


def test_instance_identity_cpdino_keys_on_cpdino_params():
    """cpdino identity uses cpdino_params, not the cellpose robust-clip params."""
    from dynacell.evaluation.pipeline_cache import _instance_identity

    ident = _instance_identity(_ctx("cpdino", "nucleus"))
    assert ident["model_name"] == "cpdino"
    assert ident["min_size"] == 15
    assert "min_obj_size" not in ident  # cellpose param must not leak in


def test_instance_identity_cpdino_differs_from_cellpose():
    """A cpdino nucleus identity differs from a cellpose nucleus identity."""
    from dynacell.evaluation.pipeline_cache import _instance_identity

    assert _instance_identity(_ctx("cpdino", "nucleus")) != _instance_identity(_ctx("cellpose", "nucleus"))


def test_instance_identity_cpdino_membrane_records_nuclei_source():
    """cpdino whole-cell carve depends on the GT nuclei source, so it is in the identity."""
    from dynacell.evaluation.pipeline_cache import _instance_identity

    ident = _instance_identity(_ctx("cpdino", "membrane"))
    assert ident["nuclei_channel"] == "Nuclei"
    assert ident["nuclei_path"] == "/x/H2B.ozx"
    # nucleus target does not carve, so it does not record the nuclei source
    assert "nuclei_channel" not in _instance_identity(_ctx("cpdino", "nucleus"))


def test_cpdino_infer_kwargs_strips_non_inference_keys():
    """model_name / subtract_nuclei are load/carve controls, not segmenter kwargs."""
    from dynacell.evaluation.pipeline_cache import cpdino_infer_kwargs

    kw = cpdino_infer_kwargs(_ctx("cpdino", "membrane"))
    assert "model_name" not in kw
    assert "subtract_nuclei" not in kw
    assert kw["normalize"] is True
    assert kw["min_size"] == 15


def test_separate_nuclei_store_loaded_for_cpdino_membrane():
    """cpdino whole-cell needs the separate GT-nuclei store on A549 (carve seeds), like watershed.

    Regression: the loader gated the separate-store open on backend=='cellpose_watershed',
    so cpdino membrane fell back to the CAAX plate (no Nuclei channel) and crashed.
    ``target_name`` must be set: ``nuclei_gt_path`` is an inert ``io`` field for every
    non-membrane target, so the helper returns None there.
    """
    from dynacell.evaluation.pipeline import _separate_nuclei_path

    a549 = OmegaConf.create(
        {
            "target_name": "membrane",
            "segmentation": {"backend": "cpdino"},
            "compute_instance_ap": True,
            "io": {"nuclei_gt_path": "/x/H2B_mock.ozx", "gt_path": "/x/CAAX_mock.ozx"},
        }
    )
    assert _separate_nuclei_path(a549) == "/x/H2B_mock.ozx"
    # iPSC single-store: nuclei live in the same cell.zarr -> no separate store.
    ipsc = OmegaConf.create(
        {
            "target_name": "membrane",
            "segmentation": {"backend": "cpdino"},
            "compute_instance_ap": True,
            "io": {"nuclei_gt_path": None, "gt_path": "/x/cell.zarr"},
        }
    )
    assert _separate_nuclei_path(ipsc) is None
    # Non-membrane target: nuclei_gt_path is inert, so no separate store is opened.
    nucleus = OmegaConf.create(
        {
            "target_name": "nucleus",
            "segmentation": {"backend": "cpdino"},
            "compute_instance_ap": True,
            "io": {"nuclei_gt_path": "/x/H2B_mock.ozx", "gt_path": "/x/H2B_mock.ozx"},
        }
    )
    assert _separate_nuclei_path(nucleus) is None


# ---------------------------------------------------------------------------
# GPU end-to-end: real cpdino inference (skipped without CUDA).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpdino_model():
    if not _HAS_CUDA:
        pytest.skip("cpdino segmentation requires a CUDA GPU")
    from dynacell.evaluation.segmentation_cellpose import load_cellpose_model

    return load_cellpose_model(use_gpu=True, model_name="cpdino")


def test_cpdino_backbone_is_dino_vitl(cpdino_model):
    """The cpdino model loads the DINO ViT-L backbone (cubic tiles it at 384)."""
    assert cpdino_model.backbone == "dino_vitl"


def _synthetic_2d(shape=(256, 256)) -> np.ndarray:
    y, x = np.indices(shape)
    cy, cx = (s / 2 for s in shape)
    img = np.exp(-(((y - cy) / 40) ** 2 + ((x - cx) / 40) ** 2)) * 2000.0
    rng = np.random.default_rng(0)
    return (img + rng.normal(50, 10, shape)).astype(np.float32)


def test_segment_cpdino_2d_shape_dtype(cpdino_model):
    """A 2D in-focus slice returns uint16 labels at native shape (no downscale)."""
    from dynacell.evaluation.segmentation_cpdino import segment_cpdino_instances

    sl = _synthetic_2d()
    labels = segment_cpdino_instances(sl, _SPACING_2D, cpdino_model, do_3d=False)
    assert labels.dtype == np.uint16
    assert labels.shape == sl.shape
    assert labels.ndim == 2


def test_segment_cpdino_empty_slice(cpdino_model):
    """An all-zero slice yields all-zero labels (no crash)."""
    from dynacell.evaluation.segmentation_cpdino import segment_cpdino_instances

    img = np.zeros((256, 256), dtype=np.float32)
    labels = segment_cpdino_instances(img, _SPACING_2D, cpdino_model, do_3d=False)
    assert labels.shape == img.shape
    assert not labels.any()
