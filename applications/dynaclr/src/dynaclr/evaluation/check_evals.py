"""Check completion status of eval runs defined in an eval registry YAML.

Derives status from filesystem sentinels rather than stored state, so it
is always ground-truth.

Usage
-----
dynaclr check-evals -r eval_registry.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
import yaml

from dynaclr.evaluation.evaluate_config import EvaluationConfig
from viscy_utils.cli_utils import load_config

_STEP_SENTINELS: dict[str, str] = {
    "predict": "embeddings/embeddings.zarr",
    "split": "configs/viewer.yaml",
    "reduce_dimensionality": "configs/reduce.yaml",
    "reduce_combined": "configs/reduce_combined.yaml",
    "smoothness": "smoothness/combined_smoothness_stats.csv",
    "plot": "plots",
    "linear_classifiers": "linear_classifiers/metrics_summary.csv",
}

Status = Literal["done", "partial", "pending"]


def _check_mmd_step(output_dir: Path, eval_cfg: EvaluationConfig) -> bool:
    """Return True if all MMD blocks have at least one result CSV."""
    if not eval_cfg.mmd:
        return True  # no MMD configured — not a blocking step
    for i, block in enumerate(eval_cfg.mmd):
        block_name = block.name if block.name else f"mmd_{i}"
        block_dir = output_dir / "mmd" / block_name
        if not any(block_dir.glob("*.csv")):
            return False
    return True


def _check_plot_step(output_dir: Path) -> bool:
    """Return True if the plots directory has at least one PDF."""
    plots_dir = output_dir / "plots"
    return any(plots_dir.rglob("*.pdf"))


def _missing_steps(eval_cfg: EvaluationConfig) -> list[str]:
    """Return steps from eval_cfg.steps that have not yet produced their sentinel output."""
    output_dir = Path(eval_cfg.output_dir)
    missing = []
    for step in eval_cfg.steps:
        if step == "mmd":
            if not _check_mmd_step(output_dir, eval_cfg):
                missing.append(step)
        elif step == "plot":
            if not _check_plot_step(output_dir):
                missing.append(step)
        elif step in _STEP_SENTINELS:
            sentinel = output_dir / _STEP_SENTINELS[step]
            if not sentinel.exists():
                missing.append(step)
        # unknown steps: skip silently
    return missing


def _model_status(eval_cfg: EvaluationConfig, force_rerun: bool) -> tuple[Status, list[str]]:
    """Return (status, missing_steps) for one model entry."""
    if force_rerun:
        return "pending", ["(force_rerun=true)"]
    missing = _missing_steps(eval_cfg)
    if not missing:
        return "done", []
    if len(missing) < len(eval_cfg.steps):
        return "partial", missing
    return "pending", missing


def _load_registry(registry_path: Path) -> list[dict]:
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data["models"]


def check_evals(registry: Path, workspace_dir: Path | None) -> None:
    """Print a markdown table showing completion status for each registered model."""
    models = _load_registry(registry)

    rows = []
    for entry in models:
        name = entry["name"]
        force_rerun = entry.get("force_rerun", False)
        eval_config_path = Path(entry["eval_config"])

        # Resolve relative paths against workspace_dir (if provided) or registry location
        if not eval_config_path.is_absolute():
            base = workspace_dir if workspace_dir else registry.parent.parent.parent.parent
            eval_config_path = base / eval_config_path

        try:
            raw = load_config(eval_config_path)
            eval_cfg = EvaluationConfig(**raw)
            status, missing = _model_status(eval_cfg, force_rerun)
            missing_str = ", ".join(missing) if missing else "—"
        except FileNotFoundError as e:
            status = "pending"
            missing_str = f"config not found: {e}"
        except Exception as e:  # noqa: BLE001
            status = "pending"
            missing_str = f"error: {e}"

        rows.append((name, status, missing_str))

    # Print markdown table
    col_name = max(len(r[0]) for r in rows)
    col_status = max(len(r[1]) for r in rows)
    col_missing = max(len(r[2]) for r in rows)

    col_name = max(col_name, len("Model"))
    col_status = max(col_status, len("Status"))
    col_missing = max(col_missing, len("Missing Steps"))

    header = f"| {'Model':<{col_name}} | {'Status':<{col_status}} | {'Missing Steps':<{col_missing}} |"
    sep = f"| {'-' * col_name} | {'-' * col_status} | {'-' * col_missing} |"
    click.echo(header)
    click.echo(sep)
    for name, status, missing_str in rows:
        click.echo(f"| {name:<{col_name}} | {status:<{col_status}} | {missing_str:<{col_missing}} |")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-r",
    "--registry",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to eval_registry.yaml",
)
@click.option(
    "--workspace-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace root for resolving relative eval_config paths. Defaults to four levels above the registry file.",
)
def main(registry: Path, workspace_dir: Path | None) -> None:
    """Print a markdown table showing eval completion status for each registered model.

    Status is derived from filesystem sentinels — never stored manually.
    Set force_rerun: true in the registry to mark a model for re-execution
    regardless of existing outputs.
    """
    check_evals(registry, workspace_dir)


if __name__ == "__main__":
    main()
