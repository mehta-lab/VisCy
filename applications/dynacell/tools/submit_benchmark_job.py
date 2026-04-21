r"""Submit a dynacell benchmark leaf via sbatch.

Composes the leaf via :func:`viscy_utils.compose.load_composed_config`,
extracts the top-level ``launcher:`` block, strips reserved keys from the
resolved config, renders an sbatch script from
``tools/sbatch_template.sbatch``, writes both to ``{run_root}/resolved/``
and ``{run_root}/slurm/``, and submits via ``sbatch`` (unless
``--dry-run``).

Usage::

    uv run python applications/dynacell/tools/submit_benchmark_job.py \
        applications/dynacell/configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/train.yml \
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dynacell._compose_hook import _dynacell_ref_resolver
from viscy_utils.compose import deep_merge, load_composed_config

_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SBATCH_DIRECTIVE_ORDER = (
    ("job_name", "--job-name"),
    ("time", "--time"),
    ("nodes", "--nodes"),
    ("ntasks_per_node", "--ntasks-per-node"),
    ("partition", "--partition"),
    ("cpus_per_task", "--cpus-per-task"),
    ("gpus", "--gpus"),
    ("mem", "--mem"),
    ("constraint", "--constraint"),
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


def _apply_override(composed: dict, path: list[str], value: Any) -> dict:
    """Deep-merge a single dotlist override and return the new config."""
    nested: Any = value
    for seg in reversed(path):
        nested = {seg: nested}
    return deep_merge(composed, nested)


_OPTIONAL_SBATCH_DIRECTIVES = frozenset({"constraint"})


def _render_sbatch_directives(job_name: str, run_root: str, sbatch: dict) -> str:
    """Render ordered ``#SBATCH`` lines. Order is pinned; output/error appended last.

    Optional directives (currently ``constraint``) are skipped when the
    value is missing or null — profiles can set ``constraint: null`` to
    express "run on any GPU."
    """
    values = dict(sbatch)
    values.setdefault("job_name", job_name)
    lines = []
    for key, flag in _SBATCH_DIRECTIVE_ORDER:
        if key not in values:
            if key in _OPTIONAL_SBATCH_DIRECTIVES:
                continue
            raise SystemExit(f"hardware profile missing sbatch.{key}")
        raw = values[key]
        if raw is None and key in _OPTIONAL_SBATCH_DIRECTIVES:
            continue
        rendered = f'"{raw}"' if flag == "--constraint" else str(raw)
        lines.append(f"#SBATCH {flag}={rendered}")
    lines.append(f"#SBATCH --output={run_root}/slurm/%j.out")
    lines.append(f"#SBATCH --error={run_root}/slurm/%j.err")
    return "\n".join(lines)


def _render_env_block(env: dict | None) -> str:
    """Render ``export KEY=VALUE`` lines, shlex-quoting values and validating keys."""
    if not env:
        return ""
    lines = []
    for k, v in env.items():
        if not _VALID_ENV_NAME.match(str(k)):
            raise SystemExit(f"launcher.env key {k!r} is not a valid shell identifier")
        lines.append(f"export {k}={shlex.quote(str(v))}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("leaf", type=Path, help="path to a benchmark leaf YAML")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write resolved config + sbatch to launcher.run_root but skip submission "
        "(requires write permission). Combine with --print-* to suppress writes.",
    )
    ap.add_argument(
        "--print-script",
        action="store_true",
        help="preview rendered sbatch to stdout. No disk writes, no submission, "
        "safe on any run_root (overrides --dry-run's disk write).",
    )
    ap.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="preview resolved YAML (launcher+benchmark stripped) to stdout. "
        "No disk writes, no submission (overrides --dry-run's disk write).",
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
    # Shared-group writes: resolved/ and slurm/ artifacts land on a shared
    # project path (`launcher.run_root`), so guarantee g+w regardless of the
    # caller's login umask. The sbatch template re-asserts umask 0002 on the
    # compute node for wandb/checkpoint/prediction outputs.
    os.umask(0o002)
    args = _parse_args(argv)

    composed = load_composed_config(args.leaf, resolver=_dynacell_ref_resolver)
    for token in args.override:
        path, value = _parse_override(token)
        composed = _apply_override(composed, path, value)

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

    # Consistency: under SLURM, Lightning's SLURMEnvironment derives
    # world_size from SLURM_NTASKS — not from trainer.devices — and
    # rejects bare `--ntasks`, demanding `--ntasks-per-node` (see
    # SLURMEnvironment._validate_srun_variables). If ntasks_per_node
    # ≠ devices, DDP silently runs with the wrong world_size and only
    # some GPUs train. Invariant: trainer.devices == sbatch.ntasks_per_node,
    # and sbatch.gpus == sbatch.nodes × trainer.devices.
    trainer = composed.get("trainer", {})
    devices = trainer.get("devices")
    nodes = sbatch.get("nodes", 1)
    ntasks_per_node = sbatch.get("ntasks_per_node")
    gpus = sbatch.get("gpus")
    if not isinstance(devices, int) or ntasks_per_node != devices or gpus != nodes * devices:
        raise SystemExit(
            f"topology mismatch: trainer.devices={devices!r}, sbatch.nodes={nodes!r}, "
            f"sbatch.ntasks_per_node={ntasks_per_node!r}, sbatch.gpus={gpus!r}. "
            f"Must satisfy devices == ntasks_per_node and gpus == nodes × devices. "
            f"Check --override values or hardware profile."
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    run_root_path = Path(run_root)
    resolved_dir = run_root_path / "resolved"
    slurm_dir = run_root_path / "slurm"
    resolved_path = resolved_dir / f"{mode}_{job_name}_{timestamp}.yml"
    sbatch_path = slurm_dir / f"{timestamp}_{job_name}.sbatch"

    template_text = (Path(__file__).parent / "sbatch_template.sbatch").read_text()
    rendered = SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, str(run_root), sbatch),
        run_root=str(run_root),
        env_block=_render_env_block(env),
        mode=mode,
        resolved_config=str(resolved_path),
    )

    if args.print_resolved_config:
        sys.stdout.write(yaml.safe_dump(composed, default_flow_style=False))
    if args.print_script:
        sys.stdout.write(rendered)

    # Preview contract:
    # - --print-* (either) = pure preview: no disk writes, no submission.
    #   Safe against run_roots the caller can't write to.
    # - --dry-run alone = write artifacts to run_root but don't submit.
    #   Requires write permission on launcher.run_root. Use --print-script
    #   to also see the rendered sbatch on stdout.
    # - --dry-run combined with --print-* = --print-* wins (preview).
    # - Bare invocation = write + submit.
    preview_only = args.print_script or args.print_resolved_config
    skip_submit = preview_only or args.dry_run
    if not preview_only:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(yaml.safe_dump(composed, default_flow_style=False))
        sbatch_path.write_text(rendered)
    if not skip_submit:
        subprocess.run(["sbatch", str(sbatch_path)], check=True)

    return 0


if __name__ == "__main__":
    sys.exit(submit())
