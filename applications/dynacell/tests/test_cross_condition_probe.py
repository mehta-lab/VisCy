"""Integration tests for the cross-condition (infected-vs-mock) linear probe.

Exercises the real ``cross_condition_probe`` code (no stubs) on small synthetic
per-cell embeddings written in the same NPZ layout the eval pipeline emits.
"""

import csv
from pathlib import Path

import numpy as np

from dynacell.evaluation.cross_condition_probe import (
    _FEATURE_TYPES,
    GROUP_PROBE_FILENAME,
    run,
    run_for_group,
)


def _write_group_embeddings(eval_dir: Path, *, seed: int, n_fovs: int = 4, n_per_fov: int = 25, dim: int = 16) -> None:
    """Write gt+pred NPZ for every feature space into ``eval_dir/embeddings``.

    Cells are spread over ``n_fovs`` FOVs so the FOV-stratified GroupKFold has
    groups to split on. Two conditions written with different ``seed`` draw from
    the same distribution, so the probe should read near chance (AUROC ~0.5) —
    note a pure location shift would be cancelled by the per-side MAD scaler,
    so we test the contract, not a synthesized separation.
    """
    rng = np.random.default_rng(seed)
    emb = eval_dir / "embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    n = n_fovs * n_per_fov
    fov = np.array([f"0/0/fov{(i // n_per_fov):04d}" for i in range(n)])
    tp = np.zeros(n, dtype=np.int32)
    for source in ("gt", "pred"):
        for feat in _FEATURE_TYPES:
            x = rng.normal(size=(n, dim)).astype(np.float32)
            np.savez(emb / f"{source}_{feat}_single_cell_embeddings.npz", embeddings=x, fov=fov, timepoint=tp)


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def test_run_for_group_writes_probe_csv(tmp_path):
    """mock vs denv -> a CSV in the denv dir only, with valid per-feature AUROC."""
    mock = tmp_path / "eval_demo_membrane_mock"
    denv = tmp_path / "eval_demo_membrane_denv"
    _write_group_embeddings(mock, seed=1)
    _write_group_embeddings(denv, seed=2)

    written = run_for_group([mock, denv])

    assert written == [denv / GROUP_PROBE_FILENAME]
    assert not (mock / GROUP_PROBE_FILENAME).exists()  # mock is the reference, no CSV
    rows = _read_rows(denv / GROUP_PROBE_FILENAME)
    assert len(rows) == len(_FEATURE_TYPES) * 2  # each feature x {pred, gt}
    assert {r["pair"] for r in rows} == {"mock_vs_denv"}
    assert {r["source"] for r in rows} == {"pred", "gt"}
    # Every probe ran and produced a valid AUROC; same-distribution cohorts read
    # near chance (the per-side MAD scaler nulls any pure location difference).
    for r in rows:
        a = float(r["auroc_mean"])
        assert 0.0 <= a <= 1.0
        assert abs(a - 0.5) < 0.2, r


def test_run_for_group_requires_mock_reference(tmp_path):
    """No mock dir -> nothing computed or written."""
    denv = tmp_path / "eval_demo_membrane_denv"
    _write_group_embeddings(denv, seed=3)
    assert run_for_group([denv]) == []
    assert not (denv / GROUP_PROBE_FILENAME).exists()


def test_run_for_group_handles_zikv(tmp_path):
    """mock + zikv -> CSV with the mock_vs_zikv pair in the zikv dir."""
    mock = tmp_path / "eval_demo_nucleus_mock"
    zikv = tmp_path / "eval_demo_nucleus_zikv"
    _write_group_embeddings(mock, seed=4)
    _write_group_embeddings(zikv, seed=5)
    written = run_for_group([mock, zikv])
    assert written == [zikv / GROUP_PROBE_FILENAME]
    rows = _read_rows(zikv / GROUP_PROBE_FILENAME)
    assert {r["pair"] for r in rows} == {"mock_vs_zikv"}


def test_run_for_group_partitions_multiple_models(tmp_path):
    """Conditions from several models in one call are probed per model (no collision).

    The grouped-eval driver passes every condition save dir of a bucket, which
    folds many models, each with its own mock/denv dirs. Two dirs map to
    ``denv`` (one per model) — historically this raised ``duplicate condition``
    and skipped the whole bucket's probe. They must instead be partitioned by
    model prefix and each probed independently.
    """
    dirs = []
    for model in ("modelA", "modelB"):
        for cond, seed in (("mock", 1), ("denv", 2)):
            d = tmp_path / f"eval_{model}_membrane_{cond}"
            _write_group_embeddings(d, seed=seed)
            dirs.append(d)

    written = run_for_group(dirs)

    # One CSV per model's denv dir (mock is the reference) — both models covered.
    assert set(written) == {
        tmp_path / "eval_modelA_membrane_denv" / GROUP_PROBE_FILENAME,
        tmp_path / "eval_modelB_membrane_denv" / GROUP_PROBE_FILENAME,
    }
    for p in written:
        rows = _read_rows(p)
        assert len(rows) == len(_FEATURE_TYPES) * 2
        assert "morphem" in {r["feature_type"] for r in rows}


def test_run_longform_covers_all_pairs_and_sources(tmp_path):
    """The CLI ``run`` writes one long-form CSV over both default pairs x sources."""
    mock = tmp_path / "eval_demo_er_mock"
    denv = tmp_path / "eval_demo_er_denv"
    zikv = tmp_path / "eval_demo_er_zikv"
    _write_group_embeddings(mock, seed=6)
    _write_group_embeddings(denv, seed=7)
    _write_group_embeddings(zikv, seed=8)
    out = tmp_path / "probe.csv"
    run([mock, denv, zikv], out)
    rows = _read_rows(out)
    # 4 features x 2 pairs x 2 sources.
    assert len(rows) == len(_FEATURE_TYPES) * 2 * 2
    assert {r["pair"] for r in rows} == {"mock_vs_denv", "mock_vs_zikv"}
