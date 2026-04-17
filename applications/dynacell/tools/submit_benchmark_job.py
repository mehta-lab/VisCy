r"""Submit a dynacell benchmark leaf via sbatch.

Composes the leaf via :func:`viscy_utils.compose.load_composed_config`,
extracts the top-level ``launcher:`` block, strips reserved keys from the
resolved config, renders an sbatch script from
``tools/sbatch_template.sbatch``, writes both to ``{run_root}/resolved/``
and ``{run_root}/slurm/``, and submits via ``sbatch`` (unless
``--dry-run``).

Usage::

    uv run python applications/dynacell/tools/submit_benchmark_job.py \
        applications/dynacell/configs/benchmarks/virtual_staining/train/er/ipsc_confocal/celldiff.yml \
        --dry-run
"""

from __future__ import annotations

import argparse
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from viscy_utils.compose import load_composed_config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* with dict-deep, list-replace semantics.

    Mirrors viscy_utils.compose._deep_merge so we don't import a private helper
    across package boundaries.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# SBATCH directive order — matches Dihan's run_celldiff.slurm byte-for-byte.
_SBATCH_DIRECTIVE_ORDER = (
    ("job_name", "--job-name"),
    ("time", "--time"),
    ("nodes", "--nodes"),
    ("ntasks", "--ntasks"),
    ("partition", "--partition"),
    ("cpus_per_task", "--cpus-per-task"),
    ("gpus", "--gpus"),
    ("mem", "--mem"),
    ("constraint", "--constraint"),
    # output and error are derived from run_root below.
)


class SbatchTemplate(string.Template):
    """Template using ``@@`` as delimiter to pass shell ``$VAR`` through verbatim."""

    delimiter = "@@"


def _parse_override(token: str) -> tuple[list[str], Any]:
    """Parse ``key.path=value`` into (path-segments, parsed-value).

    ``${...}`` interpolation is rejected outright (load_composed_config is
    pure stdlib — allowing OmegaConf-style interpolation here would create
    a semantic gap between the compose path and the override path).
    """
    if "=" not in token:
        raise SystemExit(f"--override {token!r}: missing '=' (expected key.path=value)")
    key, value = token.split("=", 1)
    if value.startswith("${"):
        raise SystemExit(f"--override {token!r}: ${{...}} interpolation is not supported")
    parsed = yaml.safe_load(value)
    return key.split("."), parsed


def _apply_override(composed: dict, path: list[str], value: Any) -> None:
    """Deep-merge a single dotlist override into *composed*."""
    nested: Any = value
    for seg in reversed(path):
        nested = {seg: nested}
    merged = _deep_merge(composed, nested)
    composed.clear()
    composed.update(merged)


def _render_sbatch_directives(job_name: str, run_root: str, sbatch: dict) -> str:
    """Render ordered ``#SBATCH`` lines matching Dihan's exact layout."""
    values = dict(sbatch)
    values.setdefault("job_name", job_name)
    lines = []
    for key, flag in _SBATCH_DIRECTIVE_ORDER:
        if key not in values:
            raise SystemExit(f"hardware profile missing sbatch.{key}")
        raw = values[key]
        rendered = f'"{raw}"' if flag == "--constraint" else str(raw)
        lines.append(f"#SBATCH {flag}={rendered}")
    lines.append(f"#SBATCH --output={run_root}/slurm/%j.out")
    lines.append(f"#SBATCH --error={run_root}/slurm/%j.err")
    return "\n".join(lines)


def _render_env_block(env: dict | None) -> str:
    if not env:
        return ""
    return "\n".join(f"export {k}={v}" for k, v in env.items())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("leaf", type=Path, help="path to a benchmark leaf YAML")
    ap.add_argument("--dry-run", action="store_true", help="render both files but skip sbatch")
    ap.add_argument("--print-script", action="store_true", help="print rendered sbatch to stdout")
    ap.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="print resolved YAML (launcher+benchmark stripped) to stdout",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="dotlist override, deep-merged after compose (repeatable)",
    )
    return ap.parse_args(argv)


def submit(argv: list[str] | None = None) -> int:
    """Render and submit the leaf; return process exit code."""
    args = _parse_args(argv)

    composed = load_composed_config(args.leaf)
    for token in args.override:
        path, value = _parse_override(token)
        _apply_override(composed, path, value)

    if "launcher" not in composed:
        raise SystemExit("leaf is missing required 'launcher:' block")
    launcher = composed.pop("launcher")
    composed.pop("benchmark", None)

    mode = launcher.get("mode")
    job_name = launcher.get("job_name")
    run_root = launcher.get("run_root")
    sbatch = launcher.get("sbatch", {})
    env = launcher.get("env", {})
    if mode not in ("fit", "predict"):
        raise SystemExit(f"launcher.mode must be 'fit' or 'predict' (got {mode!r})")
    if not job_name:
        raise SystemExit("launcher.job_name must be non-empty")
    if not run_root or not str(run_root).startswith("/"):
        raise SystemExit(f"launcher.run_root must be an absolute path (got {run_root!r})")

    # Consistency: hardware profile's gpu count must match trainer.devices.
    trainer_devices = composed.get("trainer", {}).get("devices")
    sbatch_gpus = sbatch.get("gpus")
    if trainer_devices != sbatch_gpus:
        raise SystemExit(
            f"trainer.devices={trainer_devices!r} does not match "
            f"launcher.sbatch.gpus={sbatch_gpus!r}. "
            f"Check --override values or hardware profile."
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    run_root_path = Path(run_root)
    resolved_dir = run_root_path / "resolved"
    slurm_dir = run_root_path / "slurm"
    resolved_path = resolved_dir / f"{mode}_{job_name}_{timestamp}.yml"
    sbatch_path = slurm_dir / f"{timestamp}_{job_name}.sbatch"

    template_text = (Path(__file__).parent / "sbatch_template.sbatch").read_text()
    override_suffix = "".join(f" --override {t}" for t in args.override)
    rendered = SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, str(run_root), sbatch),
        run_root=str(run_root),
        env_block=_render_env_block(env),
        mode=mode,
        resolved_config=str(resolved_path),
        overrides=override_suffix,
    )

    if args.print_resolved_config:
        sys.stdout.write(yaml.safe_dump(composed, default_flow_style=False))
    if args.print_script:
        sys.stdout.write(rendered)
    if args.dry_run and not (args.print_script or args.print_resolved_config):
        sys.stdout.write(rendered)

    if not args.dry_run:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(yaml.safe_dump(composed, default_flow_style=False))
        sbatch_path.write_text(rendered)
        subprocess.run(["sbatch", str(sbatch_path)], check=True)

    return 0


if __name__ == "__main__":
    sys.exit(submit())
