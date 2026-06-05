"""Smoke tests for ``generate_instance_ap_eval_configs.py``.

Pure-logic + bundled-manifest checks (no dynacell training tree): the scope
filter, per-bucket segmentation overlay, and the A549 cross-store nuclei wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from generate_grouped_eval_configs import ParsedZarr  # noqa: E402
from generate_instance_ap_eval_configs import (  # noqa: E402
    a549_nuclei_store,
    build_leaf,
    in_scope,
    save_dir_for,
)


def _pz(organelle, model, train_set, test_set, condition=None, variant=None) -> ParsedZarr:
    name = f"{organelle}_{model}.zarr"
    return ParsedZarr(
        pred_path=Path(f"/tmp/{test_set}/predictions/{name}"),
        organelle=organelle,
        model=model,
        variant=variant,
        train_set=train_set,
        test_set=test_set,
        condition=condition,
        is_legacy_form=False,
    )


@pytest.mark.parametrize(
    "p, expected",
    [
        (_pz("nucleus", "fnet3d_paper", "ipsc_trained", "ipsc"), True),
        (_pz("membrane", "fcmae_vscyto3d_pretrained", "joint", "ipsc"), True),
        (_pz("nucleus", "unetvit3d", "ipsc_trained", "a549", "mock"), True),
        # pix2pix3d_unetvit: in scope for both instance organelles, all train sets
        (_pz("nucleus", "pix2pix3d_unetvit", "joint", "ipsc"), True),
        (_pz("nucleus", "pix2pix3d_unetvit", "a549_trained", "a549", "zikv"), True),
        (_pz("membrane", "pix2pix3d_unetvit", "ipsc_trained", "a549", "denv"), True),
        # out-of-scope organelles
        (_pz("er", "fnet3d_paper", "ipsc_trained", "ipsc"), False),
        (_pz("mitochondria", "fcmae_vscyto3d_scratch", "a549_trained", "a549", "denv"), False),
        # out-of-scope model (original celldiff dropped; only celldiff_r2 kept)
        (_pz("nucleus", "celldiff", "ipsc_trained", "ipsc", variant="iterative"), False),
        # celldiff_r2 sliding_window/denoise: iPSC only
        (_pz("membrane", "celldiff_r2", "ipsc_trained", "ipsc", variant="sliding_window"), True),
        (_pz("membrane", "celldiff_r2", "a549_trained", "a549", "mock", variant="denoise"), False),
        (_pz("nucleus", "celldiff_r2", "a549_trained", "a549", "zikv", variant="iterative"), True),
    ],
)
def test_in_scope(p: ParsedZarr, expected: bool) -> None:
    """Scope filter keeps nucleus/membrane in-scope models, drops the rest."""
    assert in_scope(p) is expected


def test_a549_nuclei_store_resolves_h2b_per_condition() -> None:
    """The A549 nuclei store is the H2B manifest's test store for that plate."""
    for cond in ("mock", "denv", "zikv"):
        store = a549_nuclei_store(cond)
        assert store.endswith(".ozx") and "H2B" in store


def test_membrane_a549_leaf_wires_cross_store_nuclei() -> None:
    """Membrane × a549 → watershed backend, slice 0.3, per-condition H2B nuclei_gt_path."""
    conds = [
        _pz("membrane", "fnet3d_paper", "a549_trained", "a549", "mock"),
        _pz("membrane", "fcmae_vscyto3d_scratch", "joint", "a549", "zikv"),
    ]
    leaf = build_leaf("membrane", "a549", conds)
    assert leaf["target_name"] == "membrane"
    assert leaf["compute_instance_ap"] is True
    assert leaf["compute_feature_metrics"] is False
    assert leaf["segmentation"]["backend"] == "cellpose_watershed"
    assert leaf["segmentation"]["slice_fraction"] == 0.3
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    for block in leaf["conditions"]:
        assert "H2B" in block["io"]["nuclei_gt_path"]
        assert block["benchmark"]["dataset_ref"]["target"] == "caax"


def test_membrane_ipsc_leaf_has_no_nuclei_gt_path() -> None:
    """Membrane × ipsc reads nuclei from the same cell.zarr → no nuclei_gt_path."""
    leaf = build_leaf("membrane", "ipsc", [_pz("membrane", "fnet3d_paper", "ipsc_trained", "ipsc")])
    assert leaf["segmentation"]["slice_fraction"] == 0.5
    assert leaf["segmentation"]["nuclei_channel_name"] == "Nuclei"
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_nucleus_leaf_is_cellpose_without_nuclei_channel() -> None:
    """Nucleus → backend cellpose, no nuclei_channel_name, no nuclei_gt_path."""
    leaf = build_leaf("nucleus", "ipsc", [_pz("nucleus", "fnet3d_paper", "ipsc_trained", "ipsc")])
    assert leaf["segmentation"]["backend"] == "cellpose"
    assert leaf["segmentation"]["slice_fraction"] == 0.5
    assert "nuclei_channel_name" not in leaf["segmentation"]
    assert "nuclei_gt_path" not in leaf["conditions"][0]["io"]


def test_save_dir_under_instance_ap_parent() -> None:
    """Save dirs land under the dedicated evaluations_instance_ap parent."""
    p = _pz("nucleus", "fnet3d_paper", "a549_trained", "a549", "mock")
    sd = save_dir_for(p)
    assert "evaluations_instance_ap" in sd.parts
    assert sd.name == "eval_fnet3d_a549trained_nucleus_mock"
