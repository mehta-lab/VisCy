"""Tests for :func:`dynacell.evaluation.segmentation.segment` input handling."""

from __future__ import annotations

import numpy as np
import pytest
from aicssegmentation.structure_wrapper.seg_sec61b import Workflow_sec61b
from aicssegmentation.structure_wrapper.seg_tomm20 import Workflow_tomm20
from omegaconf import OmegaConf

from dynacell.evaluation import pipeline, precompute_cli, segmentation
from dynacell.evaluation.segmentation import require_cubic_workflows, segment


@pytest.mark.parametrize("target_name", ["er", "mitochondria"])
def test_classical_workflows_leave_the_input_untouched(target_name: str) -> None:
    """The aicssegmentation workflows clip their input in place; segment() must not pass that on.

    The eval segments the GT volume and then scores pixel metrics on the SAME array, so an
    in-place clip silently scores a clipped GT whenever the mask cache is cold (measured:
    SI_SSIM 0.4225 -> 0.3274 on one iPSC ER FOV).
    """
    rng = np.random.default_rng(0)
    img = rng.gamma(2.0, 1.0, size=(8, 64, 64)).astype(np.float32)
    img[4, 32, 32] = 1e3  # an outlier far above mean + 7.5 std, which the workflow clips
    before = img.copy()
    mask = segment(img, target_name, use_gpu=False)
    assert mask.shape == img.shape and mask.dtype == bool
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("target_name,reference", [("er", Workflow_sec61b), ("mitochondria", Workflow_tomm20)])
def test_cubic_cpu_workflows_match_the_reference(target_name, reference):
    rng = np.random.default_rng(17)
    image = rng.gamma(2.0, 1.0, size=(8, 64, 64)).astype(np.float32)
    image[3:5, 16:48, 30:34] += 20.0
    expected = reference(image.copy(), output_type="array").astype(bool)
    actual = segment(image, target_name, use_gpu=False)
    assert expected.any(), "The comparison must include foreground voxels"
    assert not expected.all(), "The comparison must include background voxels"
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target_name,name", [("er", "workflow_sec61b"), ("mitochondria", "workflow_tomm20")])
@pytest.mark.parametrize("use_gpu", [False, True])
def test_cubic_workflow_respects_the_requested_device(monkeypatch, target_name, name, use_gpu):
    image = np.zeros((4, 8, 8), dtype=np.float32)
    calls = []

    def upload(value):
        assert value is image
        calls.append("upload")
        return value

    def workflow(value):
        assert value is image
        calls.append("workflow")
        return np.ones_like(value, dtype=bool)

    monkeypatch.setattr(segmentation, "ascupy", upload)
    monkeypatch.setattr(segmentation._cubic_segmentation, name, workflow)
    result = segment(image, target_name, use_gpu=use_gpu)
    assert calls == (["upload", "workflow"] if use_gpu else ["workflow"])
    assert isinstance(result, np.ndarray) and result.dtype == bool and result.all()


@pytest.mark.parametrize("target_name,name", [("er", "workflow_sec61b"), ("mitochondria", "workflow_tomm20")])
def test_workflow_capability_gate_rejects_only_the_affected_target(monkeypatch, target_name, name):
    monkeypatch.delattr(segmentation._cubic_segmentation, name)
    with pytest.raises(RuntimeError, match=f"requires cubic.segmentation.{name}"):
        require_cubic_workflows(target_name)
    for unaffected in ("nucleus", "membrane", "nucleoli", "lysosomes"):
        require_cubic_workflows(unaffected)


@pytest.mark.parametrize("entry_name", ["evaluate_model", "evaluate_model_grouped"])
def test_eval_checks_workflow_capability_before_loading_data(monkeypatch, entry_name):
    monkeypatch.delattr(segmentation._cubic_segmentation, "workflow_sec61b")
    with pytest.raises(RuntimeError, match="requires cubic.segmentation.workflow_sec61b"):
        getattr(pipeline, entry_name).__wrapped__(OmegaConf.create({"target_name": "er"}))


@pytest.mark.parametrize("build_masks", [False, True])
def test_precompute_requires_workflows_only_when_building_masks(monkeypatch, build_masks):
    monkeypatch.delattr(segmentation._cubic_segmentation, "workflow_sec61b")
    config = OmegaConf.create(
        {
            "target_name": "er",
            "io": {"gt_cache_dir": "/unused"},
            "runtime": {"executor": "serial", "fov_workers": 1, "threads_per_worker": 1},
            "build": dict.fromkeys(("cp", "dinov3", "dynaclr", "celldino", "morphem", "focus"), False)
            | {"masks": build_masks},
        }
    )

    def loading_reached(*args, **kwargs):
        raise ValueError("reached model load")

    monkeypatch.setattr(precompute_cli, "load_eval_models", loading_reached)
    if build_masks:
        with pytest.raises(RuntimeError, match="requires cubic.segmentation.workflow_sec61b"):
            precompute_cli.precompute_gt_artifacts(config)
    else:
        with pytest.raises(ValueError, match="reached model load"):
            precompute_cli.precompute_gt_artifacts(config)
