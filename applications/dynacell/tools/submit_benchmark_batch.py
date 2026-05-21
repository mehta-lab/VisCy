r"""Submit a sequence of dynacell predict leaves as ONE sbatch job.

Two submission shapes are supported:

* **Serial (default)** — Composes each leaf, writes one resolved
  config per leaf to ``{run_root}/resolved/``, and renders ONE sbatch
  script (from ``sbatch_template_batch.sbatch``) that invokes
  ``python -m dynacell predict --config <resolved>`` for each leaf in
  order. One GPU runs all leaves sequentially in one allocation.
* **Array (``--array``)** — Renders ONE sbatch array (from
  ``sbatch_template_array.sbatch``) with ``--array=0-(N-1)`` (optionally
  capped via ``--max-array-concurrency K`` → ``%K``). Each array task
  picks its resolved config from the bash ``CONFIGS=(...)`` block by
  ``$SLURM_ARRAY_TASK_ID``. Up to K tasks run concurrently, each on its
  own GPU allocation.

Constraints (predict-only by design):
  * All leaves must share ``launcher.mode == 'predict'`` and the same
    ``launcher.run_root``. Mixing modes or run_roots raises.
  * Leaves' SBATCH directives are taken from the FIRST leaf's
    ``launcher.sbatch`` block; subsequent leaves' sbatch blocks must
    match (same hardware profile). ``--allow-mixed-directives``
    (array mode only) relaxes this by grouping leaves into directive
    buckets and submitting one array per bucket.
  * Wall time defaults to ``--time`` if provided; else uses the first
    leaf's ``sbatch.time``.

Usage::

    LEAVES=applications/dynacell/configs/benchmarks/virtual_staining/er/fnet3d_paper/ipsc_confocal

    # Serial: 4 plates back-to-back in one allocation.
    uv run python applications/dynacell/tools/submit_benchmark_batch.py \
        $LEAVES/predict__a549_mantis_*.yml \
        --job-name FNet3DPaper_PRED_SEC61B_ON_A549_ALL \
        --dry-run

    # Array: 4 plates in parallel (capped at 2 concurrent).
    uv run python applications/dynacell/tools/submit_benchmark_batch.py \
        $LEAVES/predict__a549_mantis_*.yml \
        --array --max-array-concurrency 2 \
        --job-name FNet3DPaper_PRED_SEC61B_ON_A549_ALL \
        --dry-run

Re-running over plates whose output stores already contain the prediction
channel requires ``--overwrite`` (a dedicated alias that walks
``trainer.callbacks`` and sets ``HCSPredictionWriter.init_args.overwrite=True``
on every leaf). Plain ``--override`` deep-merges by dict key only and cannot
reach list elements.
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


_PRED_WRITER_CLASS_SUFFIX = "HCSPredictionWriter"


def _parse_override(token: str) -> tuple[list[str], Any]:
    """Parse ``key.path=value`` into (path-segments, parsed-value).

    Mirrors :func:`submit_benchmark_job._parse_override`. List indexing
    syntax (``foo[0].bar``) is *not* supported — :func:`deep_merge`
    operates on dict keys only. Use ``--overwrite`` for the prediction
    writer's overwrite flag and accept that other list-element overrides
    require editing the leaf YAML.
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


def _apply_overwrite_alias(composed: dict, leaf_path: Path) -> None:
    """Set ``init_args.overwrite=True`` on every ``HCSPredictionWriter`` callback.

    Mutates ``composed`` in place. Walks ``trainer.callbacks`` and
    matches by ``class_path`` ending in ``HCSPredictionWriter`` — robust
    against re-ordered callback lists and additional callbacks
    (``LearningRateMonitor``, etc.). Raises if no writer is found, since
    that means the alias cannot do what the user asked.
    """
    callbacks = composed.get("trainer", {}).get("callbacks", [])
    if not isinstance(callbacks, list):
        raise SystemExit(
            f"{leaf_path}: trainer.callbacks must be a list to use --overwrite (got {type(callbacks).__name__})"
        )
    matched = 0
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        if str(cb.get("class_path", "")).endswith(_PRED_WRITER_CLASS_SUFFIX):
            cb.setdefault("init_args", {})["overwrite"] = True
            matched += 1
    if matched == 0:
        class_paths = [cb.get("class_path") for cb in callbacks if isinstance(cb, dict)]
        raise SystemExit(
            f"{leaf_path}: --overwrite requested but no HCSPredictionWriter callback "
            f"found under trainer.callbacks (got class_paths={class_paths!r})"
        )


def _render_sbatch_directives(
    job_name: str,
    run_root: str,
    sbatch: dict,
    array_spec: str | None = None,
) -> str:
    """Render ``#SBATCH`` lines.

    Parameters
    ----------
    job_name
        Value for ``--job-name``.
    run_root
        Path used for the default ``%j``/``%A_%a`` output/error logs.
    sbatch
        Hardware-profile dict (nodes, gpus, mem, ...).
    array_spec
        When set (e.g. ``"0-15"`` or ``"0-15%4"``), emit ``#SBATCH
        --array=<spec>`` and route logs through ``%A_%a`` instead of
        ``%j`` so each array task writes its own file.
    """
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
    if array_spec is not None:
        lines.append(f"#SBATCH --array={array_spec}")
        lines.append(f"#SBATCH --output={run_root}/slurm/%A_%a.out")
        lines.append(f"#SBATCH --error={run_root}/slurm/%A_%a.err")
    else:
        lines.append(f"#SBATCH --output={run_root}/slurm/%j.out")
        lines.append(f"#SBATCH --error={run_root}/slurm/%j.err")
    return "\n".join(lines)


# SBATCH directive fields whose equality across leaves is required so that a
# single ``#SBATCH`` block can stand in for all of them. ``time`` is not
# included because :func:`submit` recomputes the composite walltime; the
# remaining keys (constraint, exclude) come along with hardware identity.
_DIRECTIVE_BUCKET_KEYS: tuple[str, ...] = (
    "nodes",
    "ntasks_per_node",
    "partition",
    "cpus_per_task",
    "gpus",
    "mem",
    "constraint",
    "exclude",
)


def _directive_bucket_key(sbatch: dict) -> tuple:
    """Stable, hashable identity for a leaf's hardware-profile directives.

    Used by ``--allow-mixed-directives`` to group leaves with matching
    SBATCH blocks into separate array submissions.
    """
    return tuple((k, sbatch.get(k)) for k in _DIRECTIVE_BUCKET_KEYS)


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
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="dotlist override applied to every leaf after compose, deep-merged. "
        "Repeatable. ${...} interpolation not supported. Note: list-index syntax "
        "(callbacks[0]) is NOT honored — deep_merge operates on dict keys. Use "
        "--overwrite for the common case of forcing prediction-writer overwrite.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="set init_args.overwrite=True on every HCSPredictionWriter callback "
        "in every leaf, after compose. Required to re-run a leaf whose output "
        "store already contains the prediction channel. Off by default.",
    )
    ap.add_argument(
        "--array",
        action="store_true",
        help="emit ONE sbatch array (--array=0-(N-1)) instead of N serial "
        "invocations within one sbatch. Each array task runs one leaf on its "
        "own GPU allocation; up to --max-array-concurrency run concurrently.",
    )
    ap.add_argument(
        "--max-array-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="when --array is set, append ``%%N`` to the array spec to cap "
        "concurrent tasks (e.g. --array --max-array-concurrency 4 emits "
        "``--array=0-(N-1)%%4``). Default: no cap. Ignored without --array.",
    )
    ap.add_argument(
        "--allow-mixed-directives",
        action="store_true",
        help="when --array is set, allow leaves with different SBATCH hardware "
        "profiles. Leaves are grouped by directive identity and one array job "
        "is submitted per group; a list of job IDs is returned. Without this "
        "flag, --array requires every leaf to share identical SBATCH directives.",
    )
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


def _compose_leaves(
    leaf_paths: list[Path],
    parsed_overrides: list[tuple[list[str], Any]],
    overwrite: bool,
) -> tuple[list[dict], list[dict], list[str]]:
    """Compose every leaf and return (composed_configs, launchers, exp_ids).

    ``composed_configs[i]`` has the ``launcher`` and ``benchmark`` keys
    stripped (LightningCLI rejects unknown roots). ``launchers[i]`` is
    the popped launcher block. ``exp_ids[i]`` is the leaf's
    ``benchmark.experiment_id`` (used to derive stable resolved-config
    filenames).
    """
    composed_list: list[dict] = []
    launcher_list: list[dict] = []
    exp_ids: list[str] = []
    for leaf in leaf_paths:
        composed = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
        for path, value in parsed_overrides:
            composed = _apply_override(composed, path, value)
        if overwrite:
            _apply_overwrite_alias(composed, leaf)
        if "launcher" not in composed:
            raise SystemExit(f"{leaf}: missing required 'launcher:' block")
        launcher = composed.pop("launcher")
        bench = composed.pop("benchmark", None) or {}
        exp_id = bench.get("experiment_id")
        if not exp_id:
            raise SystemExit(f"{leaf}: missing benchmark.experiment_id")
        composed_list.append(composed)
        launcher_list.append(launcher)
        exp_ids.append(exp_id)
    return composed_list, launcher_list, exp_ids


def _validate_uniform_sbatch(
    leaf_paths: list[Path],
    launcher_list: list[dict],
) -> dict:
    """Validate that every leaf shares mode + run_root + SBATCH hardware.

    Returns the validated head ``sbatch`` dict. Used by both serial mode
    and ``--array`` without ``--allow-mixed-directives``.
    """
    modes = {ln.get("mode") for ln in launcher_list}
    if modes != {"predict"}:
        raise SystemExit(f"all leaves must be mode=predict (got {modes!r})")
    run_roots = {ln.get("run_root") for ln in launcher_list}
    if len(run_roots) != 1:
        raise SystemExit(f"all leaves must share launcher.run_root (got {run_roots!r})")
    run_root = next(iter(run_roots))
    if not run_root or not str(run_root).startswith("/"):
        raise SystemExit(f"launcher.run_root must be an absolute path (got {run_root!r})")
    head_sbatch = dict(launcher_list[0].get("sbatch", {}))
    for i, ln in enumerate(launcher_list[1:], 1):
        sb = ln.get("sbatch", {})
        for key in ("nodes", "ntasks_per_node", "partition", "cpus_per_task", "gpus", "mem"):
            if sb.get(key) != head_sbatch.get(key):
                raise SystemExit(
                    f"leaf {leaf_paths[i]}: sbatch.{key}={sb.get(key)!r} "
                    f"differs from head sbatch.{key}={head_sbatch.get(key)!r} "
                    f"(use --array --allow-mixed-directives to bucket by hardware)"
                )
    return head_sbatch


def _validate_devices(
    leaf_paths: list[Path],
    composed_list: list[dict],
    ntasks_per_node: int | None,
) -> None:
    """Per-leaf topology check: trainer.devices must equal ntasks_per_node."""
    for i, c in enumerate(composed_list):
        devices = c.get("trainer", {}).get("devices")
        if not isinstance(devices, int) or devices != ntasks_per_node:
            raise SystemExit(
                f"leaf {leaf_paths[i]}: trainer.devices={devices!r} must equal "
                f"head sbatch.ntasks_per_node={ntasks_per_node!r}"
            )


def _write_resolved_configs(
    composed_list: list[dict],
    exp_ids: list[str],
    resolved_dir: Path,
    timestamp: str,
) -> list[Path]:
    """Write each composed config to ``{resolved_dir}/{exp_id}__{timestamp}.yml``."""
    resolved_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for composed, exp_id in zip(composed_list, exp_ids):
        p = resolved_dir / f"{exp_id}__{timestamp}.yml"
        p.write_text(yaml.safe_dump(composed, default_flow_style=False))
        paths.append(p)
    return paths


def _render_serial_sbatch(
    job_name: str,
    run_root: str,
    head_sbatch: dict,
    head_env: dict,
    resolved_paths: list[Path],
) -> str:
    """Render ``sbatch_template_batch.sbatch`` with N srun invocations in series."""
    invocations = "\n\n".join(
        (
            f"echo '[batch] step {i + 1}/{len(resolved_paths)}: {p.name}'\n"
            f"srun uv run python -m dynacell predict --config {p}"
        )
        for i, p in enumerate(resolved_paths)
    )
    template_text = (Path(__file__).parent / "sbatch_template_batch.sbatch").read_text()
    return SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, run_root, head_sbatch),
        run_root=run_root,
        env_block=_render_env_block(head_env),
        repo_root=str(_REPO_ROOT),
        predict_invocations=invocations,
    )


def _render_array_sbatch(
    job_name: str,
    run_root: str,
    head_sbatch: dict,
    head_env: dict,
    resolved_paths: list[Path],
    max_concurrency: int | None,
) -> str:
    """Render ``sbatch_template_array.sbatch`` with ``--array=0-(N-1)[%K]``."""
    n = len(resolved_paths)
    if n < 1:
        raise SystemExit("array submission requires at least one leaf")
    array_spec = f"0-{n - 1}"
    if max_concurrency is not None:
        if max_concurrency < 1:
            raise SystemExit(f"--max-array-concurrency must be >=1 (got {max_concurrency!r})")
        array_spec += f"%{max_concurrency}"
    configs_list = "\n".join(f'  "{p}"' for p in resolved_paths)
    template_text = (Path(__file__).parent / "sbatch_template_array.sbatch").read_text()
    return SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, run_root, head_sbatch, array_spec=array_spec),
        run_root=run_root,
        env_block=_render_env_block(head_env),
        repo_root=str(_REPO_ROOT),
        configs_list=configs_list,
    )


def _sbatch_submit(
    sbatch_path: Path,
    parsable: bool,
) -> str | None:
    """Invoke ``sbatch`` on a rendered script. Returns parsable stdout when requested."""
    sbatch_cmd = ["sbatch"]
    if parsable:
        sbatch_cmd.append("--parsable")
    sbatch_cmd.append(str(sbatch_path))
    if parsable:
        result = subprocess.run(sbatch_cmd, check=True, stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    subprocess.run(sbatch_cmd, check=True)
    return None


def submit(argv: list[str] | None = None) -> int:
    """Compose, render, and submit a chained predict sbatch (serial or array)."""
    os.umask(0o002)
    args = _parse_args(argv)

    if len(args.leaves) < 1:
        raise SystemExit("at least one leaf is required")
    if args.max_array_concurrency is not None and not args.array:
        raise SystemExit("--max-array-concurrency requires --array")
    if args.allow_mixed_directives and not args.array:
        raise SystemExit("--allow-mixed-directives requires --array")

    parsed_overrides = [_parse_override(t) for t in args.override]
    composed_list, launcher_list, exp_ids = _compose_leaves(args.leaves, parsed_overrides, args.overwrite)

    # mode invariant: every leaf must be predict mode.
    modes = {ln.get("mode") for ln in launcher_list}
    if modes != {"predict"}:
        raise SystemExit(f"all leaves must be mode=predict (got {modes!r})")

    # run_root invariant: uniform unless --allow-mixed-directives bumps it
    # into bucket grouping. Validate each leaf's run_root is an absolute path.
    for i, ln in enumerate(launcher_list):
        rr = ln.get("run_root")
        if not rr or not str(rr).startswith("/"):
            raise SystemExit(f"leaf {args.leaves[i]}: launcher.run_root must be an absolute path (got {rr!r})")
    run_roots = {ln.get("run_root") for ln in launcher_list}
    if len(run_roots) != 1 and not (args.array and args.allow_mixed_directives):
        raise SystemExit(
            f"all leaves must share launcher.run_root (got {sorted(run_roots)!r}); "
            f"use --array --allow-mixed-directives to bucket by run_root"
        )
    run_root = next(iter(run_roots)) if len(run_roots) == 1 else None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")

    # Group leaves by (SBATCH directive identity, run_root). In strict
    # mode every leaf must share both; --allow-mixed-directives buckets
    # them and submits one sbatch per bucket (each bucket writes its
    # artifacts into its own run_root).
    if args.array and args.allow_mixed_directives:
        buckets: dict[tuple, list[int]] = {}
        for i, ln in enumerate(launcher_list):
            key = (_directive_bucket_key(ln.get("sbatch", {})), ln.get("run_root"))
            buckets.setdefault(key, []).append(i)
    else:
        head_sbatch_validated = _validate_uniform_sbatch(args.leaves, launcher_list)
        buckets = {(_directive_bucket_key(head_sbatch_validated), run_root): list(range(len(args.leaves)))}

    rendered_outputs: list[tuple[Path, list[Path]]] = []  # (sbatch_path, resolved_paths)
    for bucket_idx, (bucket_key, leaf_idxs) in enumerate(buckets.items()):
        bucket_leaf_paths = [args.leaves[i] for i in leaf_idxs]
        bucket_composed = [composed_list[i] for i in leaf_idxs]
        bucket_launchers = [launcher_list[i] for i in leaf_idxs]
        bucket_exp_ids = [exp_ids[i] for i in leaf_idxs]

        # bucket_key = (directive_tuple, run_root); use the bucket's run_root
        # so each bucket's artifacts go into the right tree under mixed mode.
        _, bucket_run_root = bucket_key
        bucket_run_root_path = Path(bucket_run_root)
        bucket_resolved_dir = bucket_run_root_path / "resolved"
        bucket_slurm_dir = bucket_run_root_path / "slurm"

        head_sbatch = dict(bucket_launchers[0].get("sbatch", {}))
        head_env = bucket_launchers[0].get("env", {})

        _validate_devices(bucket_leaf_paths, bucket_composed, head_sbatch.get("ntasks_per_node"))

        if args.time:
            head_sbatch["time"] = args.time

        # Suffix mixed-directive bucket index into job_name so multiple
        # submissions in one invocation get distinguishable names + sbatch
        # filenames.
        base_job_name = args.job_name or (bucket_launchers[0].get("job_name", "predict") + "_batch")
        job_name = f"{base_job_name}_g{bucket_idx}" if len(buckets) > 1 else base_job_name

        resolved_paths = _write_resolved_configs(bucket_composed, bucket_exp_ids, bucket_resolved_dir, timestamp)

        if args.array:
            rendered = _render_array_sbatch(
                job_name,
                str(bucket_run_root),
                head_sbatch,
                head_env,
                resolved_paths,
                args.max_array_concurrency,
            )
        else:
            rendered = _render_serial_sbatch(
                job_name,
                str(bucket_run_root),
                head_sbatch,
                head_env,
                resolved_paths,
            )

        bucket_slurm_dir.mkdir(parents=True, exist_ok=True)
        sbatch_path = bucket_slurm_dir / f"{timestamp}_{job_name}.sbatch"
        rendered_outputs.append((sbatch_path, resolved_paths))

        if args.print_script:
            sys.stdout.write(rendered)
            continue

        sbatch_path.write_text(rendered)

    if args.print_script:
        return 0
    if args.dry_run:
        for sbatch_path, resolved_paths in rendered_outputs:
            print(f"[dry-run] sbatch script: {sbatch_path}")
            for p in resolved_paths:
                print(f"[dry-run] resolved:    {p}")
        return 0

    job_ids: list[str] = []
    for sbatch_path, _ in rendered_outputs:
        result = _sbatch_submit(sbatch_path, args.parsable)
        if result is not None:
            job_ids.append(result)
    if args.parsable and job_ids:
        print("\n".join(job_ids))
    return 0


if __name__ == "__main__":
    sys.exit(submit())
