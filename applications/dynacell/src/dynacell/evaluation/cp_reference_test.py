"""Tests for the per-target CP reference and the pipeline's use of it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dynacell.evaluation import cp_reference
from dynacell.evaluation.cache import StaleCacheError
from dynacell.evaluation.cp_reference import (
    STD_FLOOR_FRACTION,
    DatasetFit,
    fit_cp_reference,
    load_cp_reference,
    payload_sha256,
    resolve_cp_reference_path,
    sidecar_cp_space,
    write_cp_reference,
)
from dynacell.evaluation.feature_metrics import compute_feature_similarity_pairwise
from dynacell.evaluation.feature_select import select_gt_features
from dynacell.evaluation.paths import DATA_ROOT, LITE_DATA_ROOT
from dynacell.evaluation.pipeline import (
    _BackboneLists,
    _cp_row_features,
    _extend_backbone,
    _stage_cp_dataset_inputs,
)

_N_FEATURES = 8
_NAMES = tuple(f"f{i}" for i in range(_N_FEATURES))
_POSITIONS = ["A/1/0", "A/1/1"]
_LITE_ROWS = 150  # the lite set's GT cells: set-a's first position


def _blocks(gt: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    """GT cells as the eval stages them: the two halves as timepoints 0 and 1 of A/1/0."""
    half = gt.shape[0] // 2
    return {("A/1/0", 0): gt[:half], ("A/1/0", 1): gt[half:]}


def _gt(n: int = 300, seed: int = 0, offset: float = 10.0, scale: float = 1.0) -> np.ndarray:
    """Synthetic GT CP matrix: 6 independent columns, one near-duplicate of f0, one constant."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, _N_FEATURES)) * np.arange(1, _N_FEATURES + 1)) * scale + offset
    x[:, 6] = x[:, 0] * 2.0 + 1e-6 * rng.standard_normal(n)  # |r| ~ 1 with f0
    x[:, 7] = 3.0  # constant
    return x


def _fit(
    dataset: str,
    cells: np.ndarray,
    *,
    in_mask_fit: bool = True,
    built_at: str | None = None,
    parent: str | None = None,
    positions: list[str] = _POSITIONS,
) -> DatasetFit:
    return DatasetFit(
        dataset=dataset,
        cells=cells,
        record={"positions": positions, "cp_cache_built_at": built_at},
        in_mask_fit=in_mask_fit,
        parent=parent,
    )


def _reference(tmp_path: Path, fits: list[DatasetFit]):
    """Fit, write and load a reference; returns ``(CPReference, payload)``."""
    payload = fit_cp_reference(fits, target_name="er", feature_names=_NAMES, cp_identity={"cp_feature_version": "test"})
    path = tmp_path / "er.json"
    write_cp_reference(payload, path, force=True)
    return load_cp_reference(path, target_name="er"), payload


@pytest.fixture
def two_sets(tmp_path: Path):
    """A reference over two test sets whose GT differ by a large offset and scale, plus HEK and one lite."""
    a, b, hek = _gt(seed=0), _gt(seed=1, offset=40.0, scale=3.0), _gt(seed=2, offset=-5.0)
    lite = _fit("set-a-lite", a[:_LITE_ROWS], in_mask_fit=False, parent="set-a", positions=_POSITIONS[:1])
    ref, payload = _reference(tmp_path, [_fit("set-a", a), _fit("set-b", b), _fit("hek", hek, in_mask_fit=False), lite])
    return ref, payload, {"set-a": a, "set-b": b, "hek": hek}


def _lists(pred: np.ndarray, gt: np.ndarray) -> _BackboneLists:
    """One-FOV ``_BackboneLists`` holding ``(pred, gt)`` as two timepoints."""
    bb = _BackboneLists()
    half = pred.shape[0] // 2
    _extend_backbone(bb, pred[:half], gt[:half], "A/1/0", 0)
    _extend_backbone(bb, pred[half:], gt[half:], "A/1/0", 1)
    return bb


def test_mask_is_pooled_over_the_mask_fit_sets_only(two_sets) -> None:
    """The mask is the GT-only selection over the mask-fit sets; HEK does not enter it."""
    ref, payload, cells = two_sets
    expected = select_gt_features(np.vstack([cells["set-a"], cells["set-b"]]))
    np.testing.assert_array_equal(ref.keep_mask, expected)
    assert not ref.keep_mask[7]  # constant in every set
    # HEK entering the fit would change the mask: its offset reshapes the pooled correlations.
    with_hek = select_gt_features(np.vstack([cells["set-a"], cells["set-b"], cells["hek"]]))
    assert not np.array_equal(with_hek, expected)
    assert payload["fit"]["mask_fit"]["datasets"] == ["set-a", "set-b"]
    assert payload["fit"]["mask_fit"]["n_cells"] == 600
    assert payload["fit"]["datasets"]["hek"]["in_mask_fit"] is False


def test_each_set_gets_its_own_gt_scaler(two_sets) -> None:
    """A set's scaler standardizes exactly that set's GT; a lite set reuses its parent's."""
    ref, payload, cells = two_sets
    for name in ("set-a", "set-b", "hek"):
        space = ref.for_dataset(name)
        z = space.transform(cells[name])
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-9)
        assert payload["fit"]["datasets"][name]["n_cells"] == 300
    # set-b's GT in set-a's space is far from standardized: the scalers really differ.
    assert np.abs(ref.for_dataset("set-a").transform(cells["set-b"]).mean(axis=0)).max() > 5
    lite = ref.for_dataset("set-a-lite")
    assert lite.is_lite and lite.scaler_dataset == "set-a"
    np.testing.assert_array_equal(lite.mean, ref.for_dataset("set-a").mean)
    with pytest.raises(KeyError, match="no scaler for dataset 'unknown'"):
        ref.for_dataset("unknown")


def test_std_floor_fires_on_a_feature_constant_within_one_set(tmp_path: Path) -> None:
    """A pooled-kept feature that is constant within one set gets floored std, and is recorded."""
    a, b = _gt(seed=0), _gt(seed=1)
    # Varies in the pool (via set-a) but is ~constant within set-b. Distinct values, so the
    # pooled variance filter keeps it; its set-b std (~1e-9) is far below the floor.
    b[:, 2] = 5.0 + 1e-9 * np.random.default_rng(3).standard_normal(b.shape[0])
    ref, payload = _reference(tmp_path, [_fit("set-a", a), _fit("set-b", b)])
    kept = payload["kept_feature_names"]
    assert "f2" in kept
    assert payload["scalers"]["set-b"]["floored_features"] == ["f2"]
    assert payload["scalers"]["set-a"]["floored_features"] == []
    j = kept.index("f2")
    pooled_std = np.vstack([a, b])[:, ref.keep_mask].std(axis=0)[j]
    assert payload["scalers"]["set-b"]["std"][j] == pytest.approx(STD_FLOOR_FRACTION * pooled_std)
    assert np.isfinite(ref.for_dataset("set-b").transform(b)).all()


def test_pooled_zero_variance_is_refused(monkeypatch) -> None:
    """A kept feature with zero POOLED variance is refused: the floor would have nothing to scale by.

    The variance filter normally drops such a column first, so the selector is
    replaced by keep-everything to reach the guard.
    """
    # Patch the module object this file imported: other tests re-create dynacell.evaluation.*
    # in sys.modules, so a dotted-path patch could land on a different module.
    monkeypatch.setattr(cp_reference, "select_gt_features", lambda gt, **kw: np.ones(gt.shape[1], dtype=bool))
    with pytest.raises(ValueError, match="zero-variance"):
        fit_cp_reference([_fit("set-a", _gt())], target_name="er", feature_names=_NAMES, cp_identity={})


def test_empty_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="no GT cells"):
        fit_cp_reference(
            [_fit("set-a", np.empty((0, _N_FEATURES)))], target_name="er", feature_names=_NAMES, cp_identity={}
        )


def test_hash_ignores_build_time_and_detects_edits(tmp_path: Path) -> None:
    """Identical inputs hash identically; a hand edit fails the load."""
    _, first = _reference(tmp_path, [_fit("set-a", _gt())])
    assert first["sha256"] == payload_sha256({**first, "created_at": "another time"})
    path = tmp_path / "er.json"
    edited = json.loads(path.read_text())
    edited["scalers"]["set-a"]["std"][0] *= 2.0
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        load_cp_reference(path, target_name="er")


def test_harmless_recache_keeps_the_hash(tmp_path: Path) -> None:
    """A rebuild whose scalers are identical but whose fit provenance moved keeps the hash.

    The new provenance (e.g. a GT cache re-stamped with the same values) is written
    without ``force``, so the eval checks see the new ``built_at`` while no
    final-metrics cache stamped with the old hash is invalidated.
    """
    cells = _gt()
    first = fit_cp_reference(
        [_fit("set-a", cells, built_at="t0")], target_name="er", feature_names=_NAMES, cp_identity={}
    )
    fit_b = _fit("set-a", cells, built_at="t1")
    fit_b.record["positions"] = [*_POSITIONS]
    second = fit_cp_reference([fit_b], target_name="er", feature_names=_NAMES, cp_identity={})
    assert first["sha256"] == second["sha256"]
    assert first["fit"] != second["fit"]

    path = tmp_path / "er.json"
    write_cp_reference(first, path)
    assert write_cp_reference(second, path) is True  # same hash, new provenance: no --force needed
    assert json.loads(path.read_text())["fit"]["datasets"]["set-a"]["cp_cache_built_at"] == "t1"
    assert write_cp_reference(second, path) is False


def test_changed_scaler_changes_the_hash() -> None:
    """Anything that moves a CP number -- here one dataset's mean -- changes the hash."""
    lite = _fit("set-a-lite", _gt()[:10], in_mask_fit=False, parent="set-a")
    payload = fit_cp_reference(
        [_fit("set-a", _gt()), _fit("set-b", _gt(seed=1)), lite], target_name="er", feature_names=_NAMES, cp_identity={}
    )
    moved = json.loads(json.dumps(payload))
    moved["scalers"]["set-b"]["mean"][0] += 1e-9
    assert payload_sha256(moved) != payload["sha256"]
    relinked = json.loads(json.dumps(payload))
    relinked["lite"]["set-a-lite"]["parent"] = "set-b"
    assert payload_sha256(relinked) != payload["sha256"]
    gate_field = json.loads(json.dumps(payload))
    gate_field["fit"]["datasets"]["set-b"]["n_cells"] = 1  # a content-gate field is integrity-protected
    assert payload_sha256(gate_field) != payload["sha256"]
    audit_only = json.loads(json.dumps(payload))
    audit_only["fit"]["datasets"]["set-b"]["cp_cache_built_at"] = "2099-01-01T00:00:00+00:00"
    assert payload_sha256(audit_only) == payload["sha256"]


def test_hand_edited_gate_field_fails_the_load(tmp_path: Path) -> None:
    """Editing a recorded gate moment without re-hashing is refused at load; the exact GT sha is audit only."""
    payload = fit_cp_reference([_fit("set-a", _gt())], target_name="er", feature_names=_NAMES, cp_identity={})
    audit = json.loads(json.dumps(payload))
    audit["fit"]["datasets"]["set-a"]["gt_matrix_sha256"] = "0" * 64
    (tmp_path / "er.json").write_text(json.dumps(audit))
    load_cp_reference(tmp_path / "er.json", target_name="er")  # audit-only field: not hashed
    payload["fit"]["datasets"]["set-a"]["gt_moments"]["mean"][0] += 1.0
    (tmp_path / "er.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        load_cp_reference(tmp_path / "er.json", target_name="er")


def test_write_is_atomic_and_refuses_a_different_reference(tmp_path: Path) -> None:
    """Identical hash is a no-op; a different one needs force; no tmp file is left behind."""
    first = fit_cp_reference([_fit("set-a", _gt())], target_name="er", feature_names=_NAMES, cp_identity={})
    other = fit_cp_reference([_fit("set-a", _gt(seed=5))], target_name="er", feature_names=_NAMES, cp_identity={})
    path = tmp_path / "er.json"
    assert write_cp_reference(first, path) is True
    assert write_cp_reference({**first, "created_at": "later"}, path) is False
    with pytest.raises(FileExistsError, match="--force"):
        write_cp_reference(other, path)
    assert json.loads(path.read_text())["sha256"] == first["sha256"]
    assert write_cp_reference(other, path, force=True) is True
    assert json.loads(path.read_text())["sha256"] == other["sha256"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["er.json"]


def test_load_refuses_missing_and_other_target(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_cp_reference.py --target er"):
        load_cp_reference(tmp_path / "absent.json", target_name="er")
    _reference(tmp_path, [_fit("set-a", _gt())])
    with pytest.raises(ValueError, match="was fit for"):
        load_cp_reference(tmp_path / "er.json", target_name="nucleus")


def test_registry_path_is_shared_by_lite_and_full() -> None:
    """``reference_path: null`` resolves under DATA_ROOT, never LITE_DATA_ROOT."""
    config = OmegaConf.create({"target_name": "mitochondria", "feature_metrics": {"cp": {"reference_path": None}}})
    path = resolve_cp_reference_path(config)
    assert path == DATA_ROOT / "cp_reference" / "mitochondria.json"
    assert LITE_DATA_ROOT not in path.parents
    config.feature_metrics.cp.reference_path = "/x/ref.json"
    assert resolve_cp_reference_path(config) == Path("/x/ref.json")
    config.target_name = "golgi"
    config.feature_metrics.cp.reference_path = None
    with pytest.raises(ValueError, match="no CP reference for target"):
        resolve_cp_reference_path(config)


def test_relative_reference_override_resolves_to_an_absolute_path(tmp_path: Path, monkeypatch) -> None:
    """A relative override is made absolute, so the sidecar path survives a later cwd change."""
    monkeypatch.chdir(tmp_path)
    config = OmegaConf.create({"target_name": "er", "feature_metrics": {"cp": {"reference_path": "refs/er.json"}}})
    path = resolve_cp_reference_path(config)
    assert path.is_absolute()
    assert path == tmp_path.resolve() / "refs" / "er.json"


def test_positions_must_match_the_fit(two_sets) -> None:
    """Every dataset (lite included) must score exactly the positions its cells were recorded over."""
    ref, _, _ = two_sets
    ref.for_dataset("set-a").check_positions(_POSITIONS)
    with pytest.raises(ValueError, match="compute_feature_metrics=false for smoke runs"):
        ref.for_dataset("set-a").check_positions(_POSITIONS[:1])
    with pytest.raises(ValueError, match="adds \\['B/1/0'\\]"):
        ref.for_dataset("set-a").check_positions([*_POSITIONS, "B/1/0"])
    ref.for_dataset("set-a-lite").check_positions(_POSITIONS[:1])
    with pytest.raises(ValueError, match="skips 1"):
        ref.for_dataset("set-a-lite").check_positions(["B/1/0"])


def test_content_gate_passes_identical_cells_and_refuses_a_value_change(two_sets) -> None:
    """Identical cells, or cells jittered at GPU level, pass; a same-count value change or a count change raises."""
    ref, _, cells = two_sets
    space = ref.for_dataset("set-a")
    gt = cells["set-a"]
    space.check_gt_cells(_blocks(gt.copy()))  # e.g. force_recompute.gt_cp reproducing the same cells
    space.check_gt_cells(dict(reversed(list(_blocks(gt).items()))))  # staging order does not matter
    space.check_gt_cells(_blocks(gt * (1 + 1e-12)))  # far above GPU jitter (~1e-15), still passes
    changed = gt.copy()
    changed[5, 2] += 1.0  # same count, one cell moved by ~1/3 sd
    with pytest.raises(StaleCacheError, match="(?s)differ from the CP reference fit .*--target er --force"):
        space.check_gt_cells(_blocks(changed))
    with pytest.raises(StaleCacheError, match="299 GT cells vs 300"):
        space.check_gt_cells(_blocks(gt[:-1]))
    lite = ref.for_dataset("set-a-lite")
    lite.check_gt_cells(_blocks(gt[:_LITE_ROWS]))
    with pytest.raises(StaleCacheError):
        lite.check_gt_cells(_blocks(gt))  # the parent's cells are not the lite set's


def test_staged_dataset_arrays_are_the_dataset_scaler_transform(two_sets, tmp_path: Path) -> None:
    """Dataset-level CP metric arrays are exactly ``transform`` with the eval dataset's scaler.

    Catches a per-side z-score (or any other scaler) sneaking back in: set-b's
    scaler is far from set-a's, and the prediction is off-scale on purpose.
    """
    ref, _, cells = two_sets
    gt = cells["set-b"]
    pred = 0.5 * gt + 7.0
    space = ref.for_dataset("set-b")
    staged = _stage_cp_dataset_inputs(_lists(pred, gt), space, _blocks(gt), tmp_path)
    np.testing.assert_array_equal(staged[1], space.transform(pred))
    np.testing.assert_array_equal(staged[2], space.transform(gt))
    np.testing.assert_array_equal(staged[3], pred[:, ref.keep_mask])  # probe: masked, unscaled
    np.testing.assert_array_equal(staged[4], gt[:, ref.keep_mask])
    assert not np.allclose(staged[1], ref.for_dataset("set-a").transform(pred))
    per_side = (pred[:, ref.keep_mask] - pred[:, ref.keep_mask].mean(0)) / pred[:, ref.keep_mask].std(0)
    assert not np.allclose(staged[1], per_side)
    sidecar = json.loads((tmp_path / "cp_selected_feature_mask.json").read_text())
    assert (sidecar["dataset"], sidecar["scaler_dataset"], sidecar["reference_sha256"]) == (
        "set-b",
        "set-b",
        ref.sha256,
    )


def test_staging_refuses_gt_cells_that_differ_from_the_fit(two_sets, tmp_path: Path) -> None:
    """The dataset-level stage runs the content gate before writing its sidecar."""
    ref, _, cells = two_sets
    gt = cells["set-a"]
    with pytest.raises(StaleCacheError, match="differ from the CP reference fit"):
        _stage_cp_dataset_inputs(_lists(gt[:-1], gt[:-1]), ref.for_dataset("set-a"), _blocks(gt[:-1]), tmp_path)
    assert not (tmp_path / "cp_selected_feature_mask.json").exists()


def test_row_features_use_the_dataset_scaler(two_sets) -> None:
    """Per-row CP inputs are the same dataset-scaler transform as the dataset-level stage."""
    ref, _, cells = two_sets
    gt = cells["set-b"][:20]
    pred = gt * 0.5
    space = ref.for_dataset("set-b")
    p, g = _cp_row_features(pred, gt, space)
    np.testing.assert_array_equal(p, space.transform(pred))
    np.testing.assert_array_equal(g, space.transform(gt))
    empty = np.empty((0, _N_FEATURES))
    assert _cp_row_features(empty, empty, space)[0] is empty
    with pytest.raises(StaleCacheError, match="CP feature dimension mismatch"):
        _cp_row_features(np.ones((4, 58)), np.ones((4, 58)), space)


def test_two_models_share_one_mask(two_sets, tmp_path: Path) -> None:
    """Two different predictions against one GT are scored on the identical feature subset.

    The retired selection, which pooled GT with each model's prediction, gives these
    two models different masks.
    """
    ref, _, cells = two_sets
    gt = cells["set-a"]
    rng = np.random.default_rng(1)
    pred_a = gt + 0.1 * rng.standard_normal(gt.shape)
    pred_b = gt.copy()
    pred_b[:, 6] = rng.standard_normal(gt.shape[0]) * 50.0  # decorrelates f6 from f0 in the pool
    pred_b[:, 7] = rng.standard_normal(gt.shape[0])  # a constant GT column that varies in pred

    masks = []
    for i, pred in enumerate((pred_a, pred_b)):
        save_dir = tmp_path / f"model_{i}"
        save_dir.mkdir()
        _stage_cp_dataset_inputs(_lists(pred, gt), ref.for_dataset("set-a"), _blocks(gt), save_dir)
        masks.append(json.loads((save_dir / "cp_selected_feature_mask.json").read_text())["keep_mask"])
    assert masks[0] == masks[1] == [bool(b) for b in ref.keep_mask]
    old_a = select_gt_features(np.vstack([gt, pred_a]))
    old_b = select_gt_features(np.vstack([gt, pred_b]))
    assert not np.array_equal(old_a, old_b)


def test_over_smoothing_registers_in_the_shared_space(two_sets) -> None:
    """``pred = 0.5 * GT + c`` (compressed range + offset) now scores clearly off GT.

    The retired per-side z-score maps any per-feature affine distortion of GT back
    onto GT exactly, so it reported a perfect match. The dataset's GT scaler keeps
    the distortion, so the same prediction is now penalized.
    """
    ref, _, cells = two_sets
    gt = cells["set-a"]
    pred = 0.5 * gt + 7.0
    space = ref.for_dataset("set-a")

    sel_p, sel_t = space.select(pred), space.select(gt)
    old_p = (sel_p - sel_p.mean(0)) / (sel_p.std(0) + 1e-8)
    old_t = (sel_t - sel_t.mean(0)) / (sel_t.std(0) + 1e-8)
    old = compute_feature_similarity_pairwise(old_p, old_t, "CP", compute_fid=False)
    new = compute_feature_similarity_pairwise(space.transform(pred), space.transform(gt), "CP", compute_fid=False)
    # KID of GT against itself: the estimator's floor for "indistinguishable" (not exactly 0).
    floor = compute_feature_similarity_pairwise(space.transform(gt), space.transform(gt), "CP", compute_fid=False)

    assert old["CP_Median_Cosine_Similarity"] == pytest.approx(1.0, abs=1e-6)
    assert old["CP_KID"] == pytest.approx(floor["CP_KID"], abs=1e-4)
    assert new["CP_Median_Cosine_Similarity"] < 0.9
    assert new["CP_KID"] > 1.0
    assert new["CP_KID"] > 20 * abs(floor["CP_KID"])


def test_sidecar_binds_the_recorded_reference_and_dataset(two_sets, tmp_path: Path) -> None:
    """An eval dir's sidecar resolves to its reference + dataset; a changed reference is refused."""
    ref, _, cells = two_sets
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    lite_gt = cells["set-a"][:_LITE_ROWS]
    _stage_cp_dataset_inputs(_lists(lite_gt, lite_gt), ref.for_dataset("set-a-lite"), _blocks(lite_gt), eval_dir)
    space = sidecar_cp_space(eval_dir)
    assert (space.dataset, space.scaler_dataset, space.reference_sha256) == ("set-a-lite", "set-a", ref.sha256)
    np.testing.assert_array_equal(space.mean, ref.for_dataset("set-a").mean)

    rebuilt = [_fit("set-a", _gt(seed=8)), _fit("set-a-lite", lite_gt, in_mask_fit=False, parent="set-a")]
    _reference(tmp_path, rebuilt)  # rebuilt in place
    with pytest.raises(ValueError, match="changed since the eval"):
        sidecar_cp_space(eval_dir)


def test_binding_separates_datasets_and_tracks_the_gt_matrix(tmp_path: Path) -> None:
    """The binding differs per dataset (lite vs parent too) and moves with that dataset's GT matrix, not built_at."""
    a = _gt()
    fits = [
        _fit("set-a", a, built_at="t0"),
        _fit("set-b", _gt(seed=1), built_at="t0"),
        _fit("set-a-lite", a[:_LITE_ROWS], in_mask_fit=False, parent="set-a", positions=_POSITIONS[:1]),
    ]
    ref, payload = _reference(tmp_path, fits)
    bindings = {d: ref.for_dataset(d).binding_sha256 for d in ("set-a", "set-b", "set-a-lite")}
    assert len(set(bindings.values())) == 3

    def rebound(edit) -> dict[str, str]:
        moved = json.loads(json.dumps(payload))
        edit(moved)
        moved["sha256"] = payload_sha256(moved)  # a legitimate rebuild re-hashes
        (tmp_path / "er.json").write_text(json.dumps(moved))
        again = load_cp_reference(tmp_path / "er.json", target_name="er")
        return {d: again.for_dataset(d).binding_sha256 for d in bindings}

    assert rebound(lambda p: p["fit"]["datasets"]["set-a"].update(cp_cache_built_at="t1")) == bindings
    assert rebound(lambda p: p["fit"]["datasets"]["set-a"].update(gt_matrix_sha256="0" * 64)) == bindings
    gt = rebound(lambda p: p["fit"]["datasets"]["set-a"]["gt_moments"]["mean"].__setitem__(0, 1.0))
    assert gt["set-a"] != bindings["set-a"]
    assert gt["set-b"] == bindings["set-b"] and gt["set-a-lite"] == bindings["set-a-lite"]
    lite_gt = rebound(lambda p: p["fit"]["lite"]["set-a-lite"]["gt_moments"]["std"].__setitem__(0, 1.0))
    assert lite_gt["set-a-lite"] != bindings["set-a-lite"] and lite_gt["set-a"] == bindings["set-a"]


def test_lite_staging_uses_the_parent_scaler(two_sets, tmp_path: Path) -> None:
    """A lite eval's staged CP arrays are the PARENT scaler's transform of pred and GT, never a lite refit.

    pred != GT and the lite cells' own statistics differ from the parent's, so an
    eval-time refit on the lite cells (or on either side) gives different arrays.
    """
    ref, _, cells = two_sets
    lite_gt = cells["set-a"][:_LITE_ROWS]
    pred = 0.5 * lite_gt + 1.0
    lite, parent = ref.for_dataset("set-a-lite"), ref.for_dataset("set-a")
    staged = _stage_cp_dataset_inputs(_lists(pred, lite_gt), lite, _blocks(lite_gt), tmp_path)
    np.testing.assert_array_equal(staged[1], parent.transform(pred))
    np.testing.assert_array_equal(staged[2], parent.transform(lite_gt))
    kept = lite_gt[:, ref.keep_mask]
    refit = (kept - kept.mean(axis=0)) / kept.std(axis=0)
    assert not np.allclose(staged[2], refit)
    assert not np.allclose(staged[1], staged[2])
    sidecar = json.loads((tmp_path / "cp_selected_feature_mask.json").read_text())
    assert (sidecar["dataset"], sidecar["scaler_dataset"]) == ("set-a-lite", "set-a")


def test_binding_covers_only_this_datasets_space(tmp_path: Path) -> None:
    """Adding an unrelated dataset keeps an existing dataset's binding; changing its own scaler moves it."""
    a, b = _gt(), _gt(seed=1)
    base = [_fit("set-a", a), _fit("set-b", b)]
    ref, payload = _reference(tmp_path, base)
    before = ref.for_dataset("set-a").binding_sha256

    (tmp_path / "added").mkdir()
    added, _ = _reference(tmp_path / "added", [*base, _fit("hek", _gt(seed=2, offset=-5.0), in_mask_fit=False)])
    assert added.sha256 != ref.sha256  # the whole reference did change
    assert added.for_dataset("set-a").binding_sha256 == before

    moved = json.loads(json.dumps(payload))
    moved["scalers"]["set-a"]["std"][0] *= 1.5
    moved["sha256"] = payload_sha256(moved)
    (tmp_path / "er.json").write_text(json.dumps(moved))
    assert load_cp_reference(tmp_path / "er.json", target_name="er").for_dataset("set-a").binding_sha256 != before
    floored = json.loads(json.dumps(payload))
    floored["scalers"]["set-a"]["floored_features"] = ["f0"]
    floored["sha256"] = payload_sha256(floored)
    (tmp_path / "er.json").write_text(json.dumps(floored))
    assert load_cp_reference(tmp_path / "er.json", target_name="er").for_dataset("set-a").binding_sha256 != before


def test_cosine_is_blind_to_contraction_toward_the_gt_mean_but_kid_is_not(two_sets) -> None:
    """``pred = mean + 0.5 * (GT - mean)`` has median cosine exactly 1 after the scaler; KID still flags it.

    Centering on the dataset's GT mean makes each pred/GT cell pair parallel, and
    cosine ignores magnitude, so GLCM+ cosine measures direction only and
    over-smoothing toward the mean shows up in KID (documented in the evaluation
    README; the reviewer measured KID 0.194 vs a floor of -0.041 on real cells).
    """
    ref, _, cells = two_sets
    gt = cells["set-a"]
    mean = gt.mean(axis=0)
    pred = mean + 0.5 * (gt - mean)
    space = ref.for_dataset("set-a")
    got = compute_feature_similarity_pairwise(space.transform(pred), space.transform(gt), "CP", compute_fid=False)
    floor = compute_feature_similarity_pairwise(space.transform(gt), space.transform(gt), "CP", compute_fid=False)
    assert got["CP_Median_Cosine_Similarity"] == pytest.approx(1.0, abs=1e-6)
    assert got["CP_KID"] > floor["CP_KID"] + 0.1


def test_old_sidecar_survives_a_rebuild_that_leaves_its_dataset_unchanged(tmp_path: Path) -> None:
    """A sidecar pins its dataset's CP space, not the whole reference.

    Adding an unrelated dataset (a new HEK scaler, outside the mask fit) rebuilds the
    reference with a new sha256 but leaves set-a's binding, so the old sidecar still
    loads; refitting set-a's own scaler makes it raise.
    """
    a, b = _gt(), _gt(seed=1)
    base = [_fit("set-a", a), _fit("set-b", b)]
    ref, _ = _reference(tmp_path, base)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _stage_cp_dataset_inputs(_lists(a, a), ref.for_dataset("set-a"), _blocks(a), eval_dir)
    old_sha = ref.sha256

    added, _ = _reference(tmp_path, [*base, _fit("hek", _gt(seed=2, offset=-5.0), in_mask_fit=False)])
    assert added.sha256 != old_sha
    assert sidecar_cp_space(eval_dir).reference_sha256 == added.sha256  # bound to the current reference

    _reference(tmp_path, [_fit("set-a", a * 1.1), _fit("set-b", b)])  # set-a's scaler moves
    with pytest.raises(ValueError, match="CP space of set-a .* changed since the eval"):
        sidecar_cp_space(eval_dir)
