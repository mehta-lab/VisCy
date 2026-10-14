import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.orchestration import matrix_eval
from dynaclr.evaluation.representation import representation_as_x


def _unit(tmp_path: Path, *, input_paths: tuple[Path, ...] = ()) -> matrix_eval.EvaluationUnit:
    plan_path = tmp_path / "plan.yml"
    plan_path.write_text("matrix: matrix.yml\noutput_root: out\nevaluations: []\n")
    evaluation = matrix_eval.EvaluationSpec(name="smooth", type="smoothness")
    plan = matrix_eval.MatrixEvaluationPlan(
        matrix="matrix.yml",
        output_root=str(tmp_path / "out"),
        evaluations=[evaluation],
    )
    return matrix_eval.EvaluationUnit(
        plan_path=plan_path,
        plan=plan,
        model={
            "family": "family",
            "run": "run",
            "datasets_root": str(tmp_path),
        },
        checkpoint_name="last",
        evaluation=evaluation,
        embedding_key="X_normalized_pca80",
        input_paths=input_paths,
        output_dir=tmp_path / "out" / "family" / "run" / "last" / "smooth",
    )


def _write_embedding(path: Path, *, normalized: bool = True) -> None:
    obs = pd.DataFrame(
        {
            "experiment": ["dataset"] * 3,
            "marker": ["SEC61B"] * 3,
            "fov_name": ["0"] * 3,
            "track_id": [1, 1, 1],
            "t": [0, 1, 2],
        },
        index=["a", "b", "c"],
    )
    adata = ad.AnnData(X=np.ones((3, 4), dtype=np.float32), obs=obs)
    if normalized:
        adata.obsm["X_normalized_pca80"] = np.full((3, 2), 2.0, dtype=np.float32)
    adata.write_zarr(path)


def test_plan_defaults_to_normalized_representation():
    plan = matrix_eval.MatrixEvaluationPlan(
        matrix="matrix.yml",
        output_root="out",
        evaluations=[{"name": "smooth", "type": "smoothness"}],
    )
    assert plan.embedding_key == "X_normalized_pca80"


def test_success_manifest_skips_only_matching_fingerprint(tmp_path):
    first = _unit(tmp_path, input_paths=(tmp_path / "a.zarr",))
    first.output_dir.mkdir(parents=True)
    first.success_path.write_text(
        json.dumps({"status": "complete", "fingerprint": matrix_eval.unit_fingerprint(first)})
    )
    assert matrix_eval.unit_is_complete(first)

    expanded = _unit(tmp_path, input_paths=(tmp_path / "a.zarr", tmp_path / "b.zarr"))
    assert not matrix_eval.unit_is_complete(expanded)


def test_submit_command_carries_representation_and_overwrite(tmp_path):
    unit = _unit(tmp_path)
    command = matrix_eval.build_submit_cmd(unit, overwrite=True)
    assert command[0] == "sbatch"
    assert "X_normalized_pca80" in command
    assert command[-1] == "1"


def test_validate_unit_inputs_requires_selected_representation(tmp_path):
    path = tmp_path / "SEC61B.zarr"
    _write_embedding(path, normalized=False)
    unit = _unit(tmp_path, input_paths=(path,))
    with pytest.raises(KeyError, match="X_normalized_pca80"):
        matrix_eval.validate_unit_inputs(unit)

    raw_unit = matrix_eval.EvaluationUnit(**{**unit.__dict__, "embedding_key": "X"})
    matrix_eval.validate_unit_inputs(raw_unit)


def test_run_unit_writes_success_and_then_skips(tmp_path, monkeypatch):
    path = tmp_path / "SEC61B.zarr"
    _write_embedding(path)
    unit = _unit(tmp_path, input_paths=(path,))
    calls = []
    monkeypatch.setitem(matrix_eval._RUNNERS, "smoothness", lambda resolved: calls.append(resolved.label))

    assert matrix_eval.run_unit(unit)
    assert len(calls) == 1
    assert matrix_eval.unit_is_complete(unit)
    assert not matrix_eval.run_unit(unit)
    assert len(calls) == 1


def test_representation_as_x_supports_marker_specific_width():
    adata = ad.AnnData(X=np.zeros((4, 8), dtype=np.float32))
    adata.obsm["X_normalized_pca80"] = np.ones((4, 3), dtype=np.float32)
    selected = representation_as_x(adata, "X_normalized_pca80")
    assert selected.shape == (4, 3)
    np.testing.assert_array_equal(selected.X, 1.0)
