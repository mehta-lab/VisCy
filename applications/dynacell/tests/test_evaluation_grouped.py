"""Parity test for the grouped multi-condition eval driver.

Builds two independent fixtures (each its own pred + GT plates + mask
caches), runs them once via ``evaluate_predictions_grouped`` and once
via two back-to-back ``evaluate_predictions`` calls, and asserts the
outputs are byte-equal per condition.

Cache-only design (same shape as
``test_evaluation_pipeline_parallel_cpu.py``): ``target_name=er`` +
``io.require_complete_cache=true`` + ``compute_feature_metrics=false``
means no segmenter / extractors / cubic are loaded — the test
validates the **loop structure** of the grouped driver, not the
model-sharing speedup (which is hermetic-untestable).

Fixture-building helpers live in ``_eval_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dynacell.evaluation.cross_condition_probe import GROUP_PROBE_FILENAME

from ._eval_fixtures import (
    N_POSITIONS,
    T,
    build_eval_config,
    build_fixture,
    live_pipeline_module,
    read_position_arrays,
)


def _build_grouped_config(
    cond_a_root: Path,
    cond_b_root: Path,
    save_root: Path,
):
    """Construct the grouped config with two condition overlays."""
    pred_a, gt_a, gt_cache_a, pred_cache_a = build_fixture(cond_a_root)
    pred_b, gt_b, gt_cache_b, pred_cache_b = build_fixture(cond_b_root)

    base = build_eval_config(
        pred_a,
        gt_a,
        gt_cache_a,
        pred_cache_a,
        save_root / "_unused",
        executor="serial",
        fov_workers=1,
    )
    # Strip the placeholder io/save — conditions own those entirely.
    base["io"]["pred_path"] = None
    base["io"]["gt_path"] = None
    base["io"]["gt_cache_dir"] = None
    base["io"]["pred_cache_dir"] = None
    base["save"]["save_dir"] = None

    base["conditions"] = [
        {
            "name": "cond_a",
            "io": {
                "pred_path": str(pred_a),
                "gt_path": str(gt_a),
                "gt_cache_dir": str(gt_cache_a),
                "pred_cache_dir": str(pred_cache_a),
            },
            "save": {"save_dir": str(save_root / "cond_a_grouped")},
        },
        {
            "name": "cond_b",
            "io": {
                "pred_path": str(pred_b),
                "gt_path": str(gt_b),
                "gt_cache_dir": str(gt_cache_b),
                "pred_cache_dir": str(pred_cache_b),
            },
            "save": {"save_dir": str(save_root / "cond_b_grouped")},
        },
    ]
    return base


def test_grouped_matches_sequential_per_condition(tmp_path: Path):
    """Grouped eval produces byte-equal outputs to sequential per-condition runs."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    pred_a, gt_a, gt_cache_a, pred_cache_a = build_fixture(cond_a_root)
    pred_b, gt_b, gt_cache_b, pred_cache_b = build_fixture(cond_b_root)
    save_a_seq = save_root / "cond_a_seq"
    save_b_seq = save_root / "cond_b_seq"
    save_a_seq.mkdir()
    save_b_seq.mkdir()

    pipeline = live_pipeline_module()
    cfg_a_seq = build_eval_config(pred_a, gt_a, gt_cache_a, pred_cache_a, save_a_seq, executor="serial", fov_workers=1)
    cfg_b_seq = build_eval_config(pred_b, gt_b, gt_cache_b, pred_cache_b, save_b_seq, executor="serial", fov_workers=1)
    pixel_a_seq, mask_a_seq, _ = pipeline.evaluate_predictions(cfg_a_seq)
    pixel_b_seq, mask_b_seq, _ = pipeline.evaluate_predictions(cfg_b_seq)

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    pipeline = live_pipeline_module()
    results = pipeline.evaluate_predictions_grouped(grouped_cfg)

    assert [name for name, _ in results] == ["cond_a", "cond_b"]
    (_, (pixel_a_grp, mask_a_grp, _)), (_, (pixel_b_grp, mask_b_grp, _)) = results

    def _sort_key(row: dict):
        return (row["FOV"], row["Timepoint"])

    assert sorted(pixel_a_grp, key=_sort_key) == sorted(pixel_a_seq, key=_sort_key)
    assert sorted(mask_a_grp, key=_sort_key) == sorted(mask_a_seq, key=_sort_key)
    assert sorted(pixel_b_grp, key=_sort_key) == sorted(pixel_b_seq, key=_sort_key)
    assert sorted(mask_b_grp, key=_sort_key) == sorted(mask_b_seq, key=_sort_key)

    for cond_name, save_seq, save_grp in [
        ("cond_a", save_a_seq, save_root / "cond_a_grouped"),
        ("cond_b", save_b_seq, save_root / "cond_b_grouped"),
    ]:
        arrs_seq = read_position_arrays(save_seq / "segmentation_results.zarr")
        arrs_grp = read_position_arrays(save_grp / "segmentation_results.zarr")
        assert set(arrs_seq.keys()) == set(arrs_grp.keys()), f"{cond_name} position set mismatch"
        for pos_name, arr_seq in arrs_seq.items():
            np.testing.assert_array_equal(arr_seq, arrs_grp[pos_name], err_msg=f"{cond_name}/{pos_name}")

    assert len(pixel_a_grp) == N_POSITIONS * T
    assert len(pixel_b_grp) == N_POSITIONS * T


def test_grouped_rejects_model_loading_field_overrides(tmp_path: Path):
    """Per-condition overrides on model-loading fields must raise — including condition 0."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    # Sabotage condition B's target_name.
    grouped_cfg["conditions"][1]["target_name"] = "membrane"

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="model-loading field"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_rejects_condition0_model_field_override(tmp_path: Path):
    """A model-loading override smuggled into condition 0 must raise, not silently re-baseline.

    Earlier wiring used ``conditions[0]``-merged config as the invariant
    baseline, which would have made condition 0 implicitly trusted to
    redefine ``target_name`` / ``feature_extractor.*`` etc. Guards
    against regression of that asymmetry.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"][0]["target_name"] = "membrane"

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="model-loading field"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_rejects_instance_ap_field_override(tmp_path: Path):
    """The instance-AP toggles are grouped invariants — a per-condition override raises.

    ``compute_instance_ap`` (and the ``segmentation.*`` / ``instance_metrics.*``
    instance fields) gate the seg backend and the shared instance-label cache
    identity, so a condition that flips one would diverge from the bundle.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"][1]["compute_instance_ap"] = True

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="model-loading field"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


@pytest.mark.parametrize("condition", [0, 1])
def test_grouped_rejects_foreground_override(tmp_path: Path, condition: int):
    """A per-condition ``pixel_metrics.foreground`` raises: one bucket must not mix FG/no-FG or recipes."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"][condition]["pixel_metrics"] = {
        "foreground": {"enabled": True, "smooth_sigma_um": 0.5, "feather_sigma_um": 0.5}
    }

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="pixel_metrics.foreground"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_warns_loudly_on_process_plus_cache_miss(tmp_path: Path, capsys):
    """``executor=process`` + multiple conditions + cache-miss prints a loud WARNING.

    This combination is the worst case for amortization: each
    condition's worker pool independently re-loads SuperModel + the
    three deep extractors, so cold-start cost multiplies by
    ``N_workers × N_conditions``. The driver should warn the user
    upfront and recommend ``executor=serial``.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["runtime"]["executor"] = "process"
    grouped_cfg["runtime"]["fov_workers"] = 2
    grouped_cfg["io"]["require_complete_cache"] = False

    # The grouped driver prints the warning before iterating conditions.
    # The eval will then fail/succeed depending on cache state; we don't
    # care — we just want the warning text in stdout.
    pipeline = live_pipeline_module()
    try:
        pipeline.evaluate_predictions_grouped(grouped_cfg)
    except Exception:
        pass

    out = capsys.readouterr().out
    assert "WARNING" in out, f"expected loud warning, got: {out!r}"
    assert "executor=process" in out
    assert "require_complete_cache=false" in out
    assert "executor=serial" in out


def test_grouped_mild_note_on_process_plus_cache_only(tmp_path: Path, capsys):
    """``executor=process`` + ``require_complete_cache=true`` gets a mild note, not the loud WARNING.

    Under the cache-only path the segmenter is usually skipped (the
    deep extractors still load when ``compute_feature_metrics=true``,
    so the cost isn't strictly zero — but it's bounded). The driver
    should inform but not alarm.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["runtime"]["executor"] = "process"
    grouped_cfg["runtime"]["fov_workers"] = 2
    # require_complete_cache stays True from the cache-only fixture.

    pipeline = live_pipeline_module()
    pipeline.evaluate_predictions_grouped(grouped_cfg)
    out = capsys.readouterr().out
    assert "[grouped] note:" in out
    assert "segmenter usually skipped" in out
    assert "WARNING" not in out, f"unexpected loud warning under cache-only path: {out!r}"


def test_grouped_rejects_require_complete_cache_override(tmp_path: Path):
    """``io.require_complete_cache`` is a grouped invariant — overrides must raise.

    The base config's ``io.require_complete_cache`` determines whether
    ``load_eval_models`` instantiates a real ``SuperModel`` or returns
    ``None``. Letting a condition flip this would mean the shared models
    bundle disagrees with the condition's actual needs (None when
    cache-miss expects a real segmenter, or loaded but never used). Guard
    by treating it as a model-loading invariant.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"][1]["io"]["require_complete_cache"] = False

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="require_complete_cache"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_rejects_pred_cache_dir_flip_for_nucleus(tmp_path: Path):
    """A per-condition ``io.pred_cache_dir=None`` flip must raise for nucleus/membrane.

    Under ``target_name ∈ {nucleus, membrane}`` + ``require_complete_cache=true``,
    ``prepare_segmentation_model`` returns ``None`` when ``io.pred_cache_dir``
    is set (pred masks served from cache) but loads ``SuperModel`` when
    ``io.pred_cache_dir is None`` (per-T loop falls back to
    ``segment(predict[t], seg_model=...)``). If the base config has
    ``pred_cache_dir`` set so the shared bundle skips SuperModel, but a
    condition overrides ``pred_cache_dir=None``, the per-T loop in that
    condition would call ``segment(...)`` with ``seg_model=None`` and crash.
    The grouped driver must catch this dangerous-direction flip pre-eval.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["target_name"] = "nucleus"
    # Force the dangerous direction: base has pred_cache_dir set (so the
    # baseline doesn't load SuperModel), then cond_a flips it to None
    # (which would crash at runtime as ``segment(predict[t], seg_model=None)``).
    grouped_cfg["io"]["pred_cache_dir"] = str(cond_a_root / "_unused_base_pred_cache")
    grouped_cfg["conditions"][0]["io"]["pred_cache_dir"] = None

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="pred_cache_dir"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_works_on_struct_mode_config(tmp_path: Path):
    """Grouped driver must work on Hydra-composed (struct-mode) configs.

    Real ``dynacell evaluate-grouped`` invocations go through
    ``@hydra.main`` which always produces a struct-mode ``DictConfig``.
    ``OmegaConf.merge`` propagates struct mode, so merging an overlay
    carrying a ``name`` label (outside the schema) raises
    ``ConfigKeyError``, and stripping ``conditions`` / ``name`` with
    ``del`` raises ``ConfigTypeError``. The driver must escape struct
    mode internally; this test guards against regression.
    """
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()

    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    OmegaConf.set_struct(grouped_cfg, True)

    pipeline = live_pipeline_module()
    results = pipeline.evaluate_predictions_grouped(grouped_cfg)
    assert [name for name, _ in results] == ["cond_a", "cond_b"]


def test_grouped_rejects_empty_conditions(tmp_path: Path):
    """Missing or empty 'conditions' list must raise an informative error."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()
    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["conditions"] = []

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)


def test_grouped_only_conditions_runs_the_named_subset(tmp_path: Path):
    """``only_conditions`` scores the named conditions and leaves the others untouched."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()
    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["only_conditions"] = ["cond_b"]
    OmegaConf.set_struct(grouped_cfg, True)

    pipeline = live_pipeline_module()
    results = pipeline.evaluate_predictions_grouped(grouped_cfg)

    assert [name for name, _ in results] == ["cond_b"]
    ((_, (pixel_b, _, _)),) = results
    assert len(pixel_b) == N_POSITIONS * T
    assert (save_root / "cond_b_grouped" / "pixel_metrics.csv").is_file()
    assert not (save_root / "cond_a_grouped").exists()


def test_grouped_only_conditions_keeps_index_names():
    """An unnamed condition is selected by its leaf index, and keeps that label."""
    pipeline = live_pipeline_module()
    conditions = OmegaConf.create([{"name": "cond_a"}, {}])
    assert [name for name, _ in pipeline._select_conditions(conditions, ["1"])] == ["1"]


@pytest.mark.parametrize("only", ["01", 1])
def test_grouped_only_conditions_rejects_a_scalar(only):
    """A bracketless override is a scalar, not a list; it must not select by its characters."""
    pipeline = live_pipeline_module()
    conditions = OmegaConf.create([{}, {}])
    with pytest.raises(ValueError, match="must be a list"):
        pipeline._select_conditions(conditions, only)


@pytest.mark.parametrize("only", [["cond_c"], ["cond_a", "cond_c"], []])
def test_grouped_only_conditions_rejects_unknown_or_empty(tmp_path: Path, only: list[str]):
    """An unknown name or an empty selection raises before any condition is scored."""
    cond_a_root = tmp_path / "fixture_a"
    cond_b_root = tmp_path / "fixture_b"
    cond_a_root.mkdir()
    cond_b_root.mkdir()
    save_root = tmp_path / "saves"
    save_root.mkdir()
    grouped_cfg = _build_grouped_config(cond_a_root, cond_b_root, save_root)
    grouped_cfg["only_conditions"] = only

    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="only_conditions"):
        pipeline.evaluate_predictions_grouped(grouped_cfg)
    assert not (save_root / "cond_a_grouped").exists()
    assert not (save_root / "cond_b_grouped").exists()


def _probe_rerun_config(tmp_path: Path, only: list[str], *, force: bool = False, extra: tuple[dict, ...] = ()):
    """A mock + denv grouped config restricted to ``only``; returns ``(config, save_dirs)``.

    ``extra`` appends conditions as given, after the mock and denv ones.
    """
    base = build_eval_config(
        tmp_path / "pred.zarr",
        tmp_path / "gt.zarr",
        tmp_path / "g",
        tmp_path / "p",
        tmp_path,
        executor="serial",
        fov_workers=1,
    )
    (tmp_path / "pred.zarr").mkdir()
    save_dirs = {cond: tmp_path / "er" / "model" / "ipsc" / f"a549__{cond}" for cond in ("mock", "denv")}
    for d in save_dirs.values():
        d.mkdir(parents=True)
    conditions = [{"name": cond, "save": {"save_dir": str(d)}} for cond, d in save_dirs.items()]
    config = OmegaConf.merge(
        base,
        {
            "conditions": [*conditions, *extra],
            "only_conditions": only,
            "force_recompute": {"final_metrics": force},
            "cross_condition_probe": {"enabled": True},
        },
    )
    return config, save_dirs


def _stub_grouped_scoring(pipeline, monkeypatch, cached: set[Path]) -> tuple[list[list[Path]], list[Path]]:
    """Stub scoring; ``cached`` dirs hold metrics that a non-forced run would reuse.

    The cache check keeps the real one's order: a force flag rejects first, and a
    missing prediction store raises. The probe stub writes a fresh CSV into every
    infected dir it can pair with a mock. Returns the dirs each probe call received
    and the dirs whose cache was checked.
    """
    calls: list[list[Path]] = []
    checked: list[Path] = []

    def cache_valid(cfg):
        checked.append(Path(cfg.save.save_dir))
        if cfg.force_recompute.all or cfg.force_recompute.final_metrics:
            return False
        if not Path(cfg.io.pred_path).exists():
            raise FileNotFoundError(cfg.io.pred_path)
        return Path(cfg.save.save_dir) in cached

    def probe(dirs, n_splits, rng_seed):
        calls.append(list(dirs))
        if not any(d.name.endswith("__mock") for d in dirs):
            return []
        written = [d / GROUP_PROBE_FILENAME for d in dirs if d.name.endswith("__denv")]
        for path in written:
            path.write_text("fresh")
        return written

    monkeypatch.setattr(pipeline, "apply_dataset_ref", lambda cfg: None)
    monkeypatch.setattr(pipeline, "load_eval_models", lambda cfg: object())
    monkeypatch.setattr(pipeline, "prediction_sources", lambda *a: {})
    monkeypatch.setattr(
        pipeline, "evaluate_predictions", lambda cfg, *, models, cp_space, prediction_snapshot: ([], [], [])
    )
    monkeypatch.setattr(pipeline, "save_metrics", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_final_metrics_cache_valid", cache_valid)
    monkeypatch.setattr(pipeline, "_cross_condition_run_for_group", probe)
    return calls, checked


def test_mock_only_rerun_reprobes_cached_infected(tmp_path: Path, monkeypatch):
    """Rescoring mock alone re-probes each infected condition whose cache is current."""
    pipeline = live_pipeline_module()
    config, dirs = _probe_rerun_config(tmp_path, ["mock"])
    (dirs["denv"] / GROUP_PROBE_FILENAME).write_text("stale")
    calls, _ = _stub_grouped_scoring(pipeline, monkeypatch, cached={dirs["denv"]})

    pipeline.evaluate_predictions_grouped(config)

    assert calls == [[dirs["mock"], dirs["denv"]]]
    assert (dirs["denv"] / GROUP_PROBE_FILENAME).read_text() == "fresh"


@pytest.mark.parametrize("mock_cached", [True, False])
def test_infected_only_rerun_never_keeps_a_stale_probe(tmp_path: Path, monkeypatch, mock_cached: bool):
    """Rescoring an infected condition re-probes it against a current mock, else drops its old CSV."""
    pipeline = live_pipeline_module()
    config, dirs = _probe_rerun_config(tmp_path, ["denv"])
    probe_csv = dirs["denv"] / GROUP_PROBE_FILENAME
    probe_csv.write_text("stale")
    calls, _ = _stub_grouped_scoring(pipeline, monkeypatch, cached={dirs["mock"]} if mock_cached else set())

    pipeline.evaluate_predictions_grouped(config)

    assert calls == [[dirs["denv"], dirs["mock"]] if mock_cached else [dirs["denv"]]]
    if mock_cached:
        assert probe_csv.read_text() == "fresh"
    else:
        assert not probe_csv.exists()


def test_forced_mock_rerun_pairs_no_unverified_counterpart(tmp_path: Path, monkeypatch):
    """A forced rerun trusts no counterpart cache; the infected probe on the old mock is dropped.

    ``force_recompute`` is how a recipe change the cache check cannot see (e.g.
    ``feature_metrics.focus_slab``) is applied, so pairing the rescored mock with an
    infected cache from before the change would mix recipes.
    """
    pipeline = live_pipeline_module()
    config, dirs = _probe_rerun_config(tmp_path, ["mock"], force=True)
    probe_csv = dirs["denv"] / GROUP_PROBE_FILENAME
    probe_csv.write_text("stale")
    calls, _ = _stub_grouped_scoring(pipeline, monkeypatch, cached={dirs["denv"]})

    pipeline.evaluate_predictions_grouped(config)

    assert calls == [[dirs["mock"]]]
    assert not probe_csv.exists()


def test_subset_rerun_skips_conditions_outside_its_probe_groups(tmp_path: Path, monkeypatch):
    """Only real counterparts are checked; another model's or a missing-store sibling cannot fail the run."""
    pipeline = live_pipeline_module()
    missing = str(tmp_path / "missing.zarr")
    other_model = tmp_path / "er" / "other" / "ipsc" / "a549__denv"
    zikv = tmp_path / "er" / "model" / "ipsc" / "a549__zikv"
    extra = (
        {"name": "other", "io": {"pred_path": missing}, "save": {"save_dir": str(other_model)}},
        {"name": "zikv", "io": {"pred_path": missing}, "save": {"save_dir": str(zikv)}},
    )
    config, dirs = _probe_rerun_config(tmp_path, ["mock"], extra=extra)
    calls, checked = _stub_grouped_scoring(pipeline, monkeypatch, cached={dirs["denv"]})

    pipeline.evaluate_predictions_grouped(config)

    assert other_model not in checked
    assert calls == [[dirs["mock"], dirs["denv"]]]
