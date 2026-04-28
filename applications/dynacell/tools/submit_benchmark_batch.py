r"""Submit a sequence of dynacell predict leaves as ONE sbatch job.

Composes each leaf via :func:`viscy_utils.compose.load_composed_config`,
strips reserved keys, writes one resolved config per leaf to
``{run_root}/resolved/`` with stable plate-scoped filenames, then renders
ONE sbatch script (from ``sbatch_template_batch.sbatch``) that invokes
``python -m dynacell predict --config <resolved>`` for each leaf in
order. Useful for chaining per-plate predict leaves into a single sbatch
to amortize queue submission and GPU allocation.

Constraints (predict-only by design):
  * All leaves must share ``launcher.mode == 'predict'`` and the same
    ``launcher.run_root``. Mixing modes or run_roots raises.
  * Leaves' SBATCH directives are merged from the FIRST leaf's
    ``launcher.sbatch`` block; subsequent leaves' sbatch blocks must
    match (same hardware profile). The composite job_name is taken
    from ``--job-name`` if provided, else derived from the first
    leaf with a ``_batch`` suffix.
  * Wall time defaults to ``--time`` if provided; else uses the first
    leaf's ``sbatch.time`` multiplied by len(leaves) (rounded up to
    the next hour).

Usage::

    LEAVES=applications/dynacell/configs/benchmarks/virtual_staining/er/fnet3d_paper/ipsc_confocal
    uv run python applications/dynacell/tools/submit_benchmark_batch.py \
        $LEAVES/predict__a549_mantis_2024_11_07.yml \
        $LEAVES/predict__a549_mantis_2024_10_31.yml \
        $LEAVES/predict__a549_mantis_2025_07_24.yml \
        $LEAVES/predict__a549_mantis_2025_08_26.yml \
        --job-name FNet3DPaper_PRED_SEC61B_ON_A549_ALL \
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

import yaml

from dynacell._compose_hook import _dynacell_ref_resolver
from viscy_utils.compose import load_composed_config

_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REPO_ROOT = Path(__file__).resolve().parents[3]

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
    ("exclude", "--exclude"),
)
_OPTIONAL_SBATCH_DIRECTIVES = frozenset({"constraint", "exclude"})


class SbatchTemplate(string.Template):
    """``@@`` delimiter to pass shell ``$VAR`` through verbatim."""

    delimiter = "@@"


def _render_sbatch_directives(job_name: str, run_root: str, sbatch: dict) -> str:
    values = dict(sbatch)
    values["job_name"] = job_name
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
    ap.add_argument("leaves", nargs="+", type=Path, help="paths to predict leaf YAMLs (>=1)")
    ap.add_argument("--job-name", default=None, help="composite SLURM job name (default: derived)")
    ap.add_argument(
        "--time",
        default=None,
        help="composite SLURM walltime override (default keeps head leaf's sbatch.time)",
    )
    ap.add_argument("--dry-run", action="store_true", help="render artifacts but don't sbatch")
    ap.add_argument("--print-script", action="store_true", help="print rendered sbatch to stdout, no writes")
    ap.add_argument("--parsable", action="store_true", help="invoke sbatch with --parsable")
    return ap.parse_args(argv)


def _stable_resolved_name(composed: dict) -> str:
    """Build a stable, plate-scoped filename from benchmark identifiers.

    Falls back to ``benchmark.experiment_id`` if present, else the
    leaf's job_name + plate suffix from ``predict_set``.
    """
    bench = composed.get("benchmark", {}) or {}
    exp_id = bench.get("experiment_id")
    if exp_id:
        return f"{exp_id}.yml"
    raise SystemExit("leaf is missing benchmark.experiment_id; cannot derive stable resolved filename")


def submit(argv: list[str] | None = None) -> int:
    """Compose, render, and submit a chained predict sbatch from N leaves."""
    os.umask(0o002)
    args = _parse_args(argv)

    if len(args.leaves) < 1:
        raise SystemExit("at least one leaf is required")

    composed_list: list[dict] = []
    launcher_list: list[dict] = []
    for leaf in args.leaves:
        composed = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
        if "launcher" not in composed:
            raise SystemExit(f"{leaf}: missing required 'launcher:' block")
        launcher = composed.pop("launcher")
        composed.pop("benchmark", None)
        composed_list.append(composed)
        launcher_list.append(launcher)

    # All leaves must agree on mode + run_root.
    modes = {ln.get("mode") for ln in launcher_list}
    if modes != {"predict"}:
        raise SystemExit(f"all leaves must be mode=predict (got {modes!r})")
    run_roots = {ln.get("run_root") for ln in launcher_list}
    if len(run_roots) != 1:
        raise SystemExit(f"all leaves must share launcher.run_root (got {run_roots!r})")
    run_root = next(iter(run_roots))
    if not run_root or not str(run_root).startswith("/"):
        raise SystemExit(f"launcher.run_root must be an absolute path (got {run_root!r})")

    # All leaves must agree on sbatch hardware profile (we render one set of
    # directives). Differences in `time` are tolerated because we override
    # the composite wall.
    head_sbatch = dict(launcher_list[0].get("sbatch", {}))
    for i, ln in enumerate(launcher_list[1:], 1):
        sb = ln.get("sbatch", {})
        for key in ("nodes", "ntasks_per_node", "partition", "cpus_per_task", "gpus", "mem"):
            if sb.get(key) != head_sbatch.get(key):
                raise SystemExit(
                    f"leaf {args.leaves[i]}: sbatch.{key}={sb.get(key)!r} "
                    f"differs from head sbatch.{key}={head_sbatch.get(key)!r}"
                )
    head_env = launcher_list[0].get("env", {})

    # Compose-time sanity: predict leaves must declare batch-relevant fields
    # consistently. `trainer.devices` should equal sbatch.ntasks_per_node
    # (predict typically runs single-GPU; if a leaf opts in to multi-GPU,
    # the head's directive will apply).
    for i, c in enumerate(composed_list):
        devices = c.get("trainer", {}).get("devices")
        if not isinstance(devices, int) or devices != head_sbatch.get("ntasks_per_node"):
            raise SystemExit(
                f"leaf {args.leaves[i]}: trainer.devices={devices!r} must equal "
                f"head sbatch.ntasks_per_node={head_sbatch.get('ntasks_per_node')!r}"
            )

    # Composite walltime: --time overrides; else keep head's sbatch.time
    # (existing hardware profiles set this to 4-00:00:00, which is plenty
    # for sequential predict on a handful of plates).
    if args.time:
        head_sbatch["time"] = args.time

    # Composite job name.
    job_name = args.job_name or (launcher_list[0].get("job_name", "predict") + "_batch")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    run_root_path = Path(run_root)
    resolved_dir = run_root_path / "resolved"
    slurm_dir = run_root_path / "slurm"

    resolved_paths: list[Path] = []
    for composed in composed_list:
        # Reattach benchmark for the stable-name derivation, then drop again.
        bench_stub = {"experiment_id": composed.get("_drop_marker")}  # placeholder
        # We popped benchmark earlier; restore experiment_id by re-reading the
        # source leaf's benchmark.experiment_id field. Read leaf YAML again
        # without compose to extract the stable identifier.
        del bench_stub  # unused
        leaf_path = args.leaves[composed_list.index(composed)]
        with leaf_path.open() as f:
            raw_leaf = yaml.safe_load(f) or {}
        exp_id = (raw_leaf.get("benchmark", {}) or {}).get("experiment_id")
        if not exp_id:
            raise SystemExit(f"{leaf_path}: missing benchmark.experiment_id")
        resolved_paths.append(resolved_dir / f"{exp_id}__{timestamp}.yml")

    # Render the predict invocation block: one srun per leaf, in order.
    invocations = "\n\n".join(
        f"echo '[batch] step {i + 1}/{len(args.leaves)}: {p.name}'\nsrun uv run python -m dynacell predict --config {p}"
        for i, p in enumerate(resolved_paths)
    )

    template_text = (Path(__file__).parent / "sbatch_template_batch.sbatch").read_text()
    rendered = SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, str(run_root), head_sbatch),
        run_root=str(run_root),
        env_block=_render_env_block(head_env),
        repo_root=str(_REPO_ROOT),
        predict_invocations=invocations,
    )

    sbatch_path = slurm_dir / f"{timestamp}_{job_name}.sbatch"

    if args.print_script:
        sys.stdout.write(rendered)
        return 0

    resolved_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for composed, p in zip(composed_list, resolved_paths):
        p.write_text(yaml.safe_dump(composed, default_flow_style=False))
    sbatch_path.write_text(rendered)

    if args.dry_run:
        print(f"[dry-run] sbatch script: {sbatch_path}")
        for p in resolved_paths:
            print(f"[dry-run] resolved:    {p}")
        return 0

    sbatch_cmd = ["sbatch"]
    if args.parsable:
        sbatch_cmd.append("--parsable")
    sbatch_cmd.append(str(sbatch_path))
    if args.parsable:
        result = subprocess.run(sbatch_cmd, check=True, stdout=subprocess.PIPE, text=True)
        print(result.stdout.strip())
    else:
        subprocess.run(sbatch_cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(submit())
