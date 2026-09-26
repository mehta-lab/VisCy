"""Integration tests for ``lite_subset_sim.py``.

Each test writes a tiny synthetic eval dir (the CSVs and per-cell embedding ``.npz``
files a real ``dynacell evaluate`` run leaves behind), loads it through the real
:class:`lite_subset_sim.System`, and compares the block-sum estimators against the
pipeline's own ``feature_metrics._kid`` run on the explicit cell arrays. Every cohort
stays under 1000 cells, where the pipeline's ``100 x min(1000, n)`` subset mean is a
single full-set estimate, so the two must agree to float32 precision.

Run::

    uv run --no-sync pytest applications/dynacell/tools/lite_subset_sim_test.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lite_subset_sim import System, load_systems, mmd2_from_features, poly3_features, poly_kernel

from dynacell.evaluation.cp_reference import DatasetFit, fit_cp_reference, write_cp_reference
from dynacell.evaluation.feature_metrics import _kid, _median_cosine_similarity

_BLOCKS = [("A/1/0", 0), ("A/1/0", 1), ("A/1/1", 0), ("A/1/1", 1), ("A/1/2", 0), ("A/1/2", 1)]
_CELLS_PER_BLOCK = [7, 12, 9, 15, 5, 11]
_D_DEEP = 6
_D_CP = 5
_DATASET = "synthetic-set"


def _write_reference(path: Path) -> dict:
    """Fit a real CP reference whose mask drops column 1 and whose scaler is far from the eval's cells.

    The reference GT is offset and rescaled relative to the eval dir's cells, so the
    reference transform and the retired per-side z-score give clearly different KIDs.
    """
    rng = np.random.default_rng(99)
    gt = rng.normal(size=(200, _D_CP)) * 2.0 + 3.0
    gt[:, 1] = 0.0  # constant -> dropped by the GT-only variance filter
    fit = DatasetFit(
        dataset=_DATASET,
        cells=gt,
        record={"positions": sorted({f for f, _ in _BLOCKS}), "gt_cache_dir": None, "cp_cache_built_at": None},
        in_mask_fit=True,
    )
    payload = fit_cp_reference(
        [fit], target_name="nucleus", feature_names=tuple(f"f{i}" for i in range(_D_CP)), cp_identity={}, lite={}
    )
    write_cp_reference(payload, path)
    return payload


def _reference_transform(payload: dict, x: np.ndarray) -> np.ndarray:
    """Apply the pipeline's CP transform spelled out from the payload: mask, then this dataset's scaler."""
    scaler = payload["scalers"][_DATASET]
    return (x[:, np.array(payload["keep_mask"])] - np.array(scaler["mean"])) / np.array(scaler["std"])


def _pipeline_kid(pred: np.ndarray, gt: np.ndarray) -> float:
    """KID exactly as the eval pipeline scores it (100 subsets of min(1000, n), seed 2020)."""
    return _kid(pred, gt, 100, 1000, 2020)[0]


def _write_eval_dir(path: Path, seed: int, reference: Path) -> dict[str, dict[str, np.ndarray]]:
    """Write a synthetic eval dir scored in ``reference`` and return its per-cell arrays by extractor token.

    Pred embeddings are GT shifted and rescaled, so every KID is well away from zero.
    """
    rng = np.random.default_rng(seed)
    (path / "embeddings").mkdir(parents=True)
    fov = np.array([f for (f, _), n in zip(_BLOCKS, _CELLS_PER_BLOCK) for _ in range(n)])
    tp = np.array([t for (_, t), n in zip(_BLOCKS, _CELLS_PER_BLOCK) for _ in range(n)])
    pd.DataFrame(
        {"FOV": [f for f, _ in _BLOCKS], "Timepoint": [t for _, t in _BLOCKS], "SI_SSIM": rng.random(len(_BLOCKS))}
    ).to_csv(path / "pixel_metrics.csv", index=False)
    pd.DataFrame(
        {"FOV": [f for f, _ in _BLOCKS], "Timepoint": [t for _, t in _BLOCKS], "Dice": rng.random(len(_BLOCKS))}
    ).to_csv(path / "mask_metrics.csv", index=False)
    pd.DataFrame({"Dataset_DINOv3_KID": [0.0]}).to_csv(path / "feature_metrics.csv", index=False)
    cells = {}
    for tok, d in (("dinov3", _D_DEEP), ("cp", _D_CP)):
        gt = rng.normal(size=(len(fov), d))
        pred = 1.5 * gt + 0.8 + 0.3 * rng.normal(size=gt.shape)
        for side, arr in (("gt", gt), ("pred", pred)):
            np.savez(
                path / "embeddings" / f"{side}_{tok}_single_cell_embeddings.npz", embeddings=arr, fov=fov, timepoint=tp
            )
        cells[tok] = {"pred": pred, "gt": gt, "fov": fov, "timepoint": tp}
    sha256 = json.loads(reference.read_text())["sha256"]
    (path / "cp_selected_feature_mask.json").write_text(
        json.dumps({"reference_path": str(reference), "reference_sha256": sha256, "dataset": _DATASET})
    )
    return cells


def _cells_of(cells: dict[str, np.ndarray], sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute force: stack every cell of each block in ``sel``, once per occurrence."""
    rows = [
        np.flatnonzero((cells["fov"] == _BLOCKS[b][0]) & (cells["timepoint"] == _BLOCKS[b][1])) for b in sel.tolist()
    ]
    idx = np.concatenate(rows)
    return cells["pred"][idx], cells["gt"][idx]


def _per_side_zscore(x: np.ndarray) -> np.ndarray:
    """Apply the retired GLCM+ transform (each side by its own stats); used only to show discrimination."""
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


@pytest.fixture
def system(tmp_path: Path) -> tuple[System, dict[str, dict[str, np.ndarray]], dict]:
    """One synthetic system over :data:`_BLOCKS`, its explicit cell arrays, and its CP reference payload."""
    reference = tmp_path / "nucleus.json"
    payload = _write_reference(reference)
    path = tmp_path / "nucleus" / "fnet3d_paper" / "ipsc" / "ipsc"
    cells = _write_eval_dir(path, seed=0, reference=reference)
    return System(path, _BLOCKS), cells, payload


def test_poly3_feature_map_reproduces_the_kernel() -> None:
    """``Phi(X) Phi(Y)^T`` is the degree-3 poly kernel, and the feature MMD is the kernel MMD."""
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(40, 4)), rng.normal(size=(40, 4)) + 0.5
    np.testing.assert_allclose(poly3_features(X) @ poly3_features(Y).T, poly_kernel(X, Y), rtol=1e-10, atol=1e-12)
    kxx, kyy, kxy = poly_kernel(X, X), poly_kernel(Y, Y), poly_kernel(X, Y)
    m = len(X)
    direct = (kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.sum() / (m * m)
    assert mmd2_from_features(poly3_features(X), poly3_features(Y)) == pytest.approx(direct, rel=1e-10)
    assert direct > 0.05


@pytest.mark.parametrize("sel", [np.arange(len(_BLOCKS)), np.array([1, 2, 5])], ids=["all", "subset"])
def test_block_sum_kid_matches_pipeline_kid(system, sel: np.ndarray) -> None:
    """Block-sum deep KID and direct GLCM+ KID equal ``feature_metrics._kid`` on the same cells."""
    sysobj, cells, payload = system
    out = sysobj.metrics(sel)
    pred, gt = _cells_of(cells["dinov3"], sel)
    assert out["DINOv3_KID"] == pytest.approx(_pipeline_kid(pred, gt), rel=1e-4)
    assert out["DINOv3_MedCos"] == pytest.approx(_median_cosine_similarity(pred, gt), rel=1e-10)
    assert out["n_cells"] == len(pred)
    raw_pred, raw_gt = _cells_of(cells["cp"], sel)
    expected = _pipeline_kid(_reference_transform(payload, raw_pred), _reference_transform(payload, raw_gt))
    assert out["CP_KID"] == pytest.approx(expected, rel=1e-4)
    # Guard against a vacuous pass: both estimates are far from zero, and the retired per-side
    # z-score on the same cells gives a clearly different CP KID.
    assert out["DINOv3_KID"] > 0.1 and abs(out["CP_KID"]) > 0.01
    mask = np.array(payload["keep_mask"])
    old = _pipeline_kid(_per_side_zscore(raw_pred[:, mask]), _per_side_zscore(raw_gt[:, mask]))
    assert out["CP_KID"] != pytest.approx(old, rel=0.1)


def test_multiset_weights_equal_brute_force_duplication(system) -> None:
    """A block drawn twice scores exactly as if its cells were physically duplicated."""
    sysobj, cells, payload = system
    sel = np.array([0, 0, 1, 3, 3, 3, 5])
    out = sysobj.metrics(sel)
    pred, gt = _cells_of(cells["dinov3"], sel)
    assert out["DINOv3_KID"] == pytest.approx(_pipeline_kid(pred, gt), rel=1e-4)
    cos = np.einsum("ij,ij->i", pred, gt) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1))
    assert out["DINOv3_MedCos"] == pytest.approx(float(np.median(cos)), rel=1e-12)
    cp_pred, cp_gt = _cells_of(cells["cp"], sel)
    assert out["CP_KID"] == pytest.approx(
        _pipeline_kid(_reference_transform(payload, cp_pred), _reference_transform(payload, cp_gt)), rel=1e-4
    )
    assert out["SI_SSIM"] == pytest.approx(float(np.mean(sysobj.row_vals["SI_SSIM"][sel])), rel=1e-12)
    # Guard against a vacuous pass: the multiset differs from the plain set of its blocks.
    assert out["DINOv3_KID"] != pytest.approx(sysobj.metrics(np.unique(sel))["DINOv3_KID"], rel=1e-3)


def test_kid_is_nan_below_the_pipeline_minimum(system) -> None:
    """Under 16 cells both the block-sum KID and the pipeline KID are NaN; at 16+ neither is."""
    sysobj, cells, _ = system
    small = np.array([4])  # 5 cells
    out = sysobj.metrics(small)
    assert np.isnan(out["DINOv3_KID"]) and np.isnan(out["CP_KID"])
    assert np.isnan(_pipeline_kid(*_cells_of(cells["dinov3"], small)))
    enough = np.array([0, 2])  # 16 cells
    assert np.isfinite(sysobj.metrics(enough)["DINOv3_KID"])
    assert np.isfinite(_pipeline_kid(*_cells_of(cells["dinov3"], enough)))


def test_load_systems_keeps_only_the_common_rows(tmp_path: Path) -> None:
    """Systems whose row sets differ are scored on the intersection."""
    reference = tmp_path / "nucleus.json"
    _write_reference(reference)
    for model in ("fnet3d_paper", "unetvit3d"):
        _write_eval_dir(tmp_path / "nucleus" / model / "ipsc" / "ipsc", seed=2, reference=reference)
    px = tmp_path / "nucleus" / "unetvit3d" / "ipsc" / "ipsc" / "pixel_metrics.csv"
    pd.read_csv(px).iloc[:-1].to_csv(px, index=False)
    systems, blocks = load_systems("nucleus", "ipsc", ["fnet3d_paper", "unetvit3d"], data_root=tmp_path)
    assert [s.name for s in systems] == ["fnet3d_paper/ipsc", "unetvit3d/ipsc"]
    assert blocks == sorted(_BLOCKS[:-1])


def test_misaligned_embeddings_raise(tmp_path: Path) -> None:
    """Pred and GT embeddings from different cells are refused, not skipped."""
    path = tmp_path / "nucleus" / "fnet3d_paper" / "ipsc" / "ipsc"
    reference = tmp_path / "nucleus.json"
    _write_reference(reference)
    _write_eval_dir(path, seed=3, reference=reference)
    npz = path / "embeddings" / "gt_dinov3_single_cell_embeddings.npz"
    with np.load(npz) as z:
        arrays = dict(z)
    arrays["fov"] = arrays["fov"][::-1]
    np.savez(npz, **arrays)
    with pytest.raises(ValueError, match="not aligned"):
        System(path, _BLOCKS)
