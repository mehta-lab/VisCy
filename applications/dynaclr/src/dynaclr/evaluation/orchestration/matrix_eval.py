"""Prepare analysis representations, then run evaluations over a model matrix.

This is the development-friendly counterpart to :mod:`matrix`: it submits one
plain SLURM job per model/checkpoint/evaluation and does not invoke Nextflow.
For normalized evaluations it first schedules one shared Step-0 job per model
checkpoint (raw X -> X_control_mad -> X_normalized_pca80). Completion is
recorded by a fingerprinted ``_SUCCESS.json`` manifest. The default behavior
skips current preparation/evaluation units; ``--overwrite`` reruns them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click
import yaml
from pydantic import BaseModel, Field, model_validator

from dynaclr.evaluation.orchestration.matrix import load_matrix, resolve_checkpoints
from dynaclr.evaluation.predict_triplet import plan_predict_runs
from viscy_data.collection import load_collection
from viscy_utils.cli_utils import load_config
from viscy_utils.compose import load_composed_config

_TASK_SBATCH = Path("applications/dynaclr/tools/matrix_eval.sbatch")
_PREPARE_SBATCH = Path("applications/dynaclr/tools/normalize.sbatch")
_SUCCESS_FILE = "_SUCCESS.json"
_RAW_EMBEDDING_KEY = "X"
_DEFAULT_PREPARED_KEY = "X_normalized_pca80"
_DEFAULT_RECIPE = "applications/dynaclr/configs/evaluation/recipes/witness_gmm_pooled_joint_pca80.yaml"


class SlurmResources(BaseModel):
    """Resources for one direct evaluation job."""

    partition: str = "cpu"
    cpus: int = Field(default=8, ge=1)
    memory: str = "128G"
    time: str = "1-00:00:00"
    gpus: int = Field(default=0, ge=0)


class EvaluationSpec(BaseModel):
    """One named metric requested by a matrix-evaluation plan."""

    name: str
    type: Literal["embedding_consistency", "smoothness", "linear_classifiers", "temporal_forecast"]
    config: str | None = None
    embedding_key: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    resources: SlurmResources | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def validate_config(self) -> "EvaluationSpec":
        if self.type in {"embedding_consistency", "temporal_forecast"} and not self.config:
            raise ValueError(f"evaluation {self.name!r} requires config")
        return self


class RepresentationPreparation(BaseModel):
    """Evaluation Step 0: persist control-MAD, then fit pooled marker PCA."""

    enabled: bool = True
    recipe: str = _DEFAULT_RECIPE
    resources: SlurmResources = Field(default_factory=lambda: SlurmResources(cpus=8, memory="128G", time="1-00:00:00"))


class MatrixEvaluationPlan(BaseModel):
    """Top-level direct evaluation plan."""

    matrix: str
    output_root: str
    embedding_key: str = "X_normalized_pca80"
    prepare_representation: RepresentationPreparation = Field(default_factory=RepresentationPreparation)
    resources: SlurmResources = Field(default_factory=SlurmResources)
    evaluations: list[EvaluationSpec]

    @model_validator(mode="after")
    def validate_evaluations(self) -> "MatrixEvaluationPlan":
        names = [item.name for item in self.evaluations]
        if not names:
            raise ValueError("evaluations must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("evaluation names must be unique")
        return self


@dataclass(frozen=True)
class EvaluationUnit:
    """Resolved model/checkpoint/evaluation work unit."""

    plan_path: Path
    plan: MatrixEvaluationPlan
    model: dict[str, Any]
    checkpoint_name: str
    evaluation: EvaluationSpec
    embedding_key: str
    input_paths: tuple[Path, ...]
    output_dir: Path

    @property
    def label(self) -> str:
        return (
            f"{self.model['family']}/{self.model['run']}/{self.checkpoint_name}"
            f" [{self.evaluation.name}; {self.embedding_key}]"
        )

    @property
    def success_path(self) -> Path:
        return self.output_dir / _SUCCESS_FILE


def _resolve_path(value: str | Path, plan_path: Path) -> Path:
    """Resolve repo-relative paths first, then paths relative to the plan."""
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = plan_path.parent / path
    return candidate if candidate.exists() else path


def load_evaluation_plan(path: Path) -> MatrixEvaluationPlan:
    """Load and validate a matrix-evaluation YAML."""
    with Path(path).open() as stream:
        return MatrixEvaluationPlan(**(yaml.safe_load(stream) or {}))


def _expected_paths(model: dict[str, Any], checkpoint_name: str) -> tuple[Path, ...]:
    collection = load_collection(Path(model["collection"]))
    runs = plan_predict_runs(
        collection,
        model_family=model["family"],
        run=model["run"],
        ckpt_name=checkpoint_name,
        datasets_root=model["datasets_root"],
        markers=model.get("markers"),
    )
    return tuple(dict.fromkeys(item.output_path for item in runs))


def build_units(
    plan_path: Path,
    *,
    evaluation_names: set[str] | None = None,
    model_families: set[str] | None = None,
    embedding_key_override: str | None = None,
) -> list[EvaluationUnit]:
    """Expand a plan into model x checkpoint x evaluation units."""
    plan_path = Path(plan_path)
    plan = load_evaluation_plan(plan_path)
    matrix_path = _resolve_path(plan.matrix, plan_path)
    output_root = _resolve_path(plan.output_root, plan_path)
    models = load_matrix(matrix_path)
    units: list[EvaluationUnit] = []
    for model in models:
        if model_families and model["family"] not in model_families:
            continue
        for checkpoint_name, _ in resolve_checkpoints(model):
            paths = _expected_paths(model, checkpoint_name)
            for evaluation in plan.evaluations:
                if not evaluation.enabled:
                    continue
                if evaluation_names and evaluation.name not in evaluation_names:
                    continue
                embedding_key = embedding_key_override or evaluation.embedding_key or plan.embedding_key
                output_dir = output_root / model["family"] / model["run"] / checkpoint_name / evaluation.name
                units.append(
                    EvaluationUnit(
                        plan_path=plan_path,
                        plan=plan,
                        model=model,
                        checkpoint_name=checkpoint_name,
                        evaluation=evaluation,
                        embedding_key=embedding_key,
                        input_paths=paths,
                        output_dir=output_dir,
                    )
                )
    return units


def _evaluation_config_path(unit: EvaluationUnit) -> Path | None:
    value = unit.evaluation.config
    if unit.evaluation.type == "linear_classifiers" and value in (None, "model.eval_config"):
        value = unit.model.get("eval_config")
    if value is None:
        return None
    return _resolve_path(value, unit.plan_path)


def _resolved_evaluation_config(unit: EvaluationUnit) -> dict[str, Any] | None:
    path = _evaluation_config_path(unit)
    if path is None:
        return None
    if unit.evaluation.type == "embedding_consistency":
        return load_composed_config(path)
    return load_config(path)


def _representation_fingerprints(unit: EvaluationUnit) -> list[str | None]:
    """Return the saved Step-0 fit identity for each input store."""
    if unit.embedding_key == _RAW_EMBEDDING_KEY:
        return [None] * len(unit.input_paths)
    import zarr
    from anndata.io import read_elem

    fingerprints: list[str | None] = []
    for path in unit.input_paths:
        try:
            root = zarr.open_group(str(path), mode="r", use_consolidated=False)
            metadata = read_elem(root[f"uns/{unit.embedding_key}"])
            fingerprints.append(str(metadata.get("pca_fit_id") or metadata.get("pca_model_checksums")))
        except (KeyError, OSError, TypeError):
            fingerprints.append(None)
    return fingerprints


def unit_fingerprint(unit: EvaluationUnit) -> str:
    """Fingerprint the requested computation and exact expected store list."""
    payload = {
        "schema_version": 1,
        "model_family": unit.model["family"],
        "run": unit.model["run"],
        "checkpoint_name": unit.checkpoint_name,
        "evaluation": unit.evaluation.model_dump(mode="json", exclude={"resources"}),
        "resolved_config": _resolved_evaluation_config(unit),
        "embedding_key": unit.embedding_key,
        "input_paths": [str(path) for path in unit.input_paths],
        "representation_fingerprints": _representation_fingerprints(unit),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def unit_is_complete(unit: EvaluationUnit) -> bool:
    """Whether the success manifest matches the current unit fingerprint."""
    try:
        manifest = json.loads(unit.success_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return manifest.get("status") == "complete" and manifest.get("fingerprint") == unit_fingerprint(unit)


def _zarr_complete(path: Path) -> bool:
    return path.is_dir() and ((path / "zarr.json").exists() or (path / ".zgroup").exists())


def _has_representation(path: Path, embedding_key: str) -> bool:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if embedding_key == _RAW_EMBEDDING_KEY:
        return "X" in root
    return "obsm" in root and embedding_key in root["obsm"]


def validate_unit_inputs(unit: EvaluationUnit) -> None:
    """Fail before evaluation when a store or requested representation is absent."""
    incomplete = [str(path) for path in unit.input_paths if not _zarr_complete(path)]
    if incomplete:
        preview = "\n".join(f"  - {path}" for path in incomplete[:20])
        raise FileNotFoundError(f"{unit.label}: {len(incomplete)} embedding store(s) missing/incomplete:\n{preview}")
    missing_key = [str(path) for path in unit.input_paths if not _has_representation(path, unit.embedding_key)]
    if missing_key:
        preview = "\n".join(f"  - {path}" for path in missing_key[:20])
        raise KeyError(
            f"{unit.label}: representation {unit.embedding_key!r} missing from "
            f"{len(missing_key)} store(s):\n{preview}\n"
            "Run `dynaclr run-matrix --stages normalize` or select `--embedding-key X`."
        )


def _validate_raw_inputs(unit: EvaluationUnit) -> None:
    incomplete = [str(path) for path in unit.input_paths if not _zarr_complete(path)]
    if incomplete:
        preview = "\n".join(f"  - {path}" for path in incomplete[:20])
        raise FileNotFoundError(
            f"{unit.label}: {len(incomplete)} raw embedding store(s) missing/incomplete:\n{preview}"
        )


def _preparation_recipe(unit: EvaluationUnit) -> Path:
    return _resolve_path(unit.plan.prepare_representation.recipe, unit.plan_path)


def _plain_metadata(value: Any) -> Any:
    """Convert AnnData/Zarr array scalars to ordinary JSON-comparable values."""
    if isinstance(value, dict):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_metadata(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def representation_is_current(unit: EvaluationUnit) -> bool:
    """Whether Step 0 matches the exact current collection and recipe."""
    if unit.embedding_key == _RAW_EMBEDDING_KEY:
        return True
    if unit.embedding_key != _DEFAULT_PREPARED_KEY:
        return _has_representation(unit.input_paths[0], unit.embedding_key) if unit.input_paths else False
    if not unit.plan.prepare_representation.enabled:
        return all(_has_representation(path, unit.embedding_key) for path in unit.input_paths)

    import zarr
    from anndata.io import read_elem

    from dynaclr.evaluation.mmd.export_representation import load_pooled_representation_config

    expected_paths = [str(path) for path in unit.input_paths]
    expected_config = load_pooled_representation_config(_preparation_recipe(unit))
    expected_normalization = expected_config.representation.model_dump(mode="json")
    expected_aliases = expected_config.condition_aliases
    fit_ids: set[str] = set()
    for path in unit.input_paths:
        try:
            root = zarr.open_group(str(path), mode="r", use_consolidated=False)
            if "obsm/X_control_mad" not in root or "uns/X_control_mad" not in root:
                return False
            metadata = read_elem(root[f"uns/{unit.embedding_key}"])
        except (KeyError, OSError, TypeError):
            return False
        if (
            int(metadata.get("schema_version", 0)) < 2
            or list(metadata.get("fit_input_paths", ())) != expected_paths
            or _plain_metadata(metadata.get("normalization")) != _plain_metadata(expected_normalization)
            or _plain_metadata(metadata.get("condition_aliases")) != _plain_metadata(expected_aliases)
        ):
            return False
        fit_ids.add(str(metadata.get("pca_fit_id", "")))
    return len(fit_ids) == 1 and bool(next(iter(fit_ids), ""))


def build_prepare_cmd(unit: EvaluationUnit, *, overwrite: bool = False) -> list[str]:
    """Build evaluation Step-0 submission for one model/checkpoint."""
    resources = unit.plan.prepare_representation.resources
    markers = ",".join(unit.model.get("markers") or ())
    cmd = [
        "sbatch",
        f"--partition={resources.partition}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.memory}",
        f"--time={resources.time}",
    ]
    if resources.gpus:
        cmd.append(f"--gpus={resources.gpus}")
    cmd.extend(
        [
            str(_PREPARE_SBATCH),
            str(unit.model["collection"]),
            unit.model["family"],
            unit.model["run"],
            unit.checkpoint_name,
            str(unit.model["datasets_root"]),
            markers,
            str(_preparation_recipe(unit)),
        ]
    )
    return cmd


def prepare_representation(unit: EvaluationUnit, *, overwrite: bool = False) -> bool:
    """Run Step 0 locally, returning False when the pooled fit is current."""
    if unit.embedding_key == _RAW_EMBEDDING_KEY:
        return False
    if unit.embedding_key != _DEFAULT_PREPARED_KEY:
        raise ValueError(f"automatic Step 0 only prepares {_DEFAULT_PREPARED_KEY!r}, got {unit.embedding_key!r}")
    if not unit.plan.prepare_representation.enabled:
        validate_unit_inputs(unit)
        return False
    if not overwrite and representation_is_current(unit):
        return False

    from dynaclr.evaluation.mmd.checkpoint_representation import export_checkpoint_representation

    export_checkpoint_representation(
        Path(unit.model["collection"]),
        model_family=unit.model["family"],
        run=unit.model["run"],
        ckpt_name=unit.checkpoint_name,
        datasets_root=Path(unit.model["datasets_root"]),
        markers=unit.model.get("markers"),
        recipe=_preparation_recipe(unit),
        overwrite=True,
    )
    return True


def _api_embedding_key(value: str) -> str | None:
    return None if value == _RAW_EMBEDDING_KEY else value


def _run_embedding_consistency(unit: EvaluationUnit) -> None:
    from dynaclr.evaluation.mmd.config import EmbeddingConsistencyConfig
    from dynaclr.evaluation.mmd.consistency import run_consistency_qc

    raw = dict(_resolved_evaluation_config(unit) or {})
    raw.update(
        {
            "model_family": unit.model["family"],
            "run": unit.model["run"],
            "ckpt_name": unit.checkpoint_name,
            "datasets_root": str(unit.model["datasets_root"]),
            "output_dir": str(unit.output_dir),
            "embedding_key": _api_embedding_key(unit.embedding_key),
        }
    )
    config = EmbeddingConsistencyConfig(**raw)
    run_consistency_qc(config, input_paths=[str(path) for path in unit.input_paths])


def _run_smoothness(unit: EvaluationUnit) -> None:
    from dynaclr.evaluation.benchmarking.smoothness.evaluate_smoothness import main as smoothness_main

    datasets_root = Path(unit.model["datasets_root"])
    models = []
    for path in unit.input_paths:
        try:
            dataset = path.relative_to(datasets_root).parts[0]
        except ValueError:
            dataset = path.parent.name
        models.append({"path": str(path), "label": f"{dataset}__{path.stem}"})
    evaluation = {
        "output_dir": str(unit.output_dir),
        "embedding_key": _api_embedding_key(unit.embedding_key),
        "fail_fast": True,
        **unit.evaluation.parameters,
    }
    unit.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = unit.output_dir / "resolved_smoothness.yml"
    resolved_path.write_text(yaml.safe_dump({"models": models, "evaluation": evaluation}, sort_keys=False))
    smoothness_main.callback(config=resolved_path)
    summary = unit.output_dir / "combined_smoothness_stats.csv"
    if not summary.exists():
        raise RuntimeError(f"smoothness did not write {summary}")


def _run_linear_classifiers(unit: EvaluationUnit) -> None:
    from dynaclr.evaluation.evaluate_config import EvaluationConfig
    from dynaclr.evaluation.linear_classifiers.orchestrated import run_linear_classifiers

    resolved = _resolved_evaluation_config(unit)
    if resolved is None:
        raise ValueError(f"{unit.label}: no linear-classifier config")
    evaluation_config = EvaluationConfig(**resolved)
    if evaluation_config.linear_classifiers is None:
        raise ValueError(f"{unit.label}: config has no linear_classifiers section")
    classifier_config = evaluation_config.linear_classifiers.model_copy(
        update={
            "embedding_key": _api_embedding_key(unit.embedding_key),
            # Direct matrix evaluation is a benchmark, not a registry release.
            # Publishing normalized-space pipelines requires an equally
            # representation-aware inference path and must be explicit.
            "publish_dir": (
                evaluation_config.linear_classifiers.publish_dir
                if unit.evaluation.parameters.get("publish", False)
                else None
            ),
        }
    )
    metrics = run_linear_classifiers(list(unit.input_paths), classifier_config, unit.output_dir)
    if metrics.empty:
        raise RuntimeError(f"{unit.label}: no classifier metrics were produced")


def _run_temporal_forecast(unit: EvaluationUnit) -> None:
    from dynaclr.evaluation.temporal.config_forecast import (
        TemporalForecastConfig,
    )
    from dynaclr.evaluation.temporal.forecast import run_temporal_forecast

    resolved = dict(_resolved_evaluation_config(unit) or {})
    if "temporal_forecast" in resolved:
        resolved = resolved["temporal_forecast"]
    config = TemporalForecastConfig(**resolved)
    run_temporal_forecast(
        unit.input_paths,
        config,
        unit.output_dir,
        embedding_key=unit.embedding_key,
    )


_RUNNERS = {
    "embedding_consistency": _run_embedding_consistency,
    "smoothness": _run_smoothness,
    "linear_classifiers": _run_linear_classifiers,
    "temporal_forecast": _run_temporal_forecast,
}


def _write_success(unit: EvaluationUnit) -> None:
    unit.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "fingerprint": unit_fingerprint(unit),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_family": unit.model["family"],
        "run": unit.model["run"],
        "checkpoint_name": unit.checkpoint_name,
        "evaluation": unit.evaluation.name,
        "embedding_key": unit.embedding_key,
        "input_paths": [str(path) for path in unit.input_paths],
    }
    fd, temporary = tempfile.mkstemp(prefix="._SUCCESS.", suffix=".json", dir=unit.output_dir)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, unit.success_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_unit(unit: EvaluationUnit, *, overwrite: bool = False) -> bool:
    """Run one unit, returning False when a current unit was skipped."""
    if not overwrite and unit_is_complete(unit):
        click.echo(f"SKIP {unit.label}: current success manifest")
        return False
    validate_unit_inputs(unit)
    unit.output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"RUN  {unit.label} -> {unit.output_dir}")
    _RUNNERS[unit.evaluation.type](unit)
    _write_success(unit)
    return True


def _resources(unit: EvaluationUnit) -> SlurmResources:
    return unit.evaluation.resources or unit.plan.resources


def build_submit_cmd(
    unit: EvaluationUnit,
    *,
    overwrite: bool = False,
    dependency: str | None = None,
) -> list[str]:
    """Build one sbatch command for a direct evaluation unit."""
    resources = _resources(unit)
    cmd = [
        "sbatch",
        f"--partition={resources.partition}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.memory}",
        f"--time={resources.time}",
    ]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    if resources.gpus:
        cmd.append(f"--gpus={resources.gpus}")
    cmd.extend(
        [
            str(_TASK_SBATCH),
            str(unit.plan_path),
            unit.model["family"],
            unit.model["run"],
            unit.checkpoint_name,
            unit.evaluation.name,
            unit.embedding_key,
            "1" if overwrite else "0",
        ]
    )
    return cmd


def _submit(cmd: list[str]) -> str:
    output = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return output.strip().split()[-1]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-c", "--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--evaluation", "evaluation_names", multiple=True, help="Run only these named evaluations.")
@click.option("--model-family", "model_families", multiple=True, help="Run only these matrix model families.")
@click.option(
    "--embedding-key",
    default=None,
    help="Override the plan representation for every selected evaluation; use X for raw adata.X.",
)
@click.option("--overwrite/--no-overwrite", default=False, help="Rerun current units; default skips them.")
@click.option("--dry-run", is_flag=True, help="Print pending sbatch commands without submitting.")
@click.option("--local", is_flag=True, help="Run pending units sequentially in this process instead of SLURM.")
def main(
    config: Path,
    evaluation_names: tuple[str, ...],
    model_families: tuple[str, ...],
    embedding_key: str | None,
    overwrite: bool,
    dry_run: bool,
    local: bool,
) -> None:
    """Prepare requested representations, then evaluate without Nextflow."""
    units = build_units(
        config,
        evaluation_names=set(evaluation_names) or None,
        model_families=set(model_families) or None,
        embedding_key_override=embedding_key,
    )
    if not units:
        raise click.ClickException("selection matched no evaluation units")
    unknown = set(evaluation_names) - {unit.evaluation.name for unit in units}
    if unknown:
        raise click.ClickException(f"unknown evaluation name(s): {sorted(unknown)}")

    pending = units if overwrite else [unit for unit in units if not unit_is_complete(unit)]
    click.echo(f"{len(units)} unit(s): {len(units) - len(pending)} current, {len(pending)} pending")
    if not pending:
        return

    grouped: dict[tuple[str, str, str], list[EvaluationUnit]] = {}
    for unit in pending:
        grouped.setdefault((unit.model["family"], unit.model["run"], unit.checkpoint_name), []).append(unit)
    if local:
        for group in grouped.values():
            for unit in group:
                _validate_raw_inputs(unit)
            prepared = next(
                (unit for unit in group if unit.embedding_key == _DEFAULT_PREPARED_KEY),
                None,
            )
            if prepared is not None:
                changed = prepare_representation(prepared, overwrite=overwrite)
                click.echo(
                    f"{'RUN ' if changed else 'SKIP'} evaluation Step 0 for "
                    f"{prepared.model['family']}/{prepared.model['run']}/{prepared.checkpoint_name}"
                )
            for unit in group:
                run_unit(unit, overwrite=overwrite)
        return
    if not dry_run:
        # Fail before any submission if raw model outputs are incomplete. A
        # missing derived representation is expected: Step 0 creates it.
        for unit in pending:
            _validate_raw_inputs(unit)
    for group in grouped.values():
        prepared = next(
            (unit for unit in group if unit.embedding_key == _DEFAULT_PREPARED_KEY),
            None,
        )
        dependency: str | None = None
        if prepared is not None and (overwrite or not representation_is_current(prepared)):
            prepare_cmd = build_prepare_cmd(prepared, overwrite=overwrite)
            label = (
                f"{prepared.model['family']}/{prepared.model['run']}/"
                f"{prepared.checkpoint_name} [prepare-representation]"
            )
            if dry_run:
                click.echo(f"# {label}\n{' '.join(prepare_cmd)}")
                dependency = "<prepare-representation>"
            else:
                dependency = _submit(prepare_cmd)
                click.echo(f"{label} -> job {dependency}")
        for unit in group:
            if dependency is None and not dry_run:
                validate_unit_inputs(unit)
            cmd = build_submit_cmd(
                unit,
                overwrite=overwrite,
                dependency=dependency if unit.embedding_key == _DEFAULT_PREPARED_KEY else None,
            )
            if dry_run:
                click.echo(f"# {unit.label}\n{' '.join(cmd)}")
            else:
                job_id = _submit(cmd)
                click.echo(f"{unit.label} -> job {job_id}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-c", "--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-family", required=True)
@click.option("--run", required=True)
@click.option("--ckpt-name", required=True)
@click.option("--evaluation", "evaluation_name", required=True)
@click.option("--embedding-key", required=True)
@click.option("--overwrite/--no-overwrite", default=False)
def task_main(
    config: Path,
    model_family: str,
    run: str,
    ckpt_name: str,
    evaluation_name: str,
    embedding_key: str,
    overwrite: bool,
) -> None:
    """Execute one resolved matrix-evaluation unit (SLURM worker entry point)."""
    matches = [
        unit
        for unit in build_units(
            config,
            evaluation_names={evaluation_name},
            model_families={model_family},
            embedding_key_override=embedding_key,
        )
        if unit.model["run"] == run and unit.checkpoint_name == ckpt_name
    ]
    if len(matches) != 1:
        raise click.ClickException(f"expected one evaluation unit, found {len(matches)}")
    run_unit(matches[0], overwrite=overwrite)
