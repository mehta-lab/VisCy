r"""Submit dynacell predict leaves as one or more chunked sbatch jobs.

Composes each leaf via :func:`viscy_utils.compose.load_composed_config`,
strips reserved keys, writes one resolved config per leaf to
``{run_root}/resolved/`` with stable plate-scoped filenames, then renders
sbatch script(s) (from ``sbatch_template_batch.sbatch``) that invoke
``python -m dynacell predict --config <resolved>``.

Two layouts, selected by ``--parallel P`` (default 1):
  * ``P=1``: one sbatch job runs all N leaves sequentially via ``srun``.
  * ``P>1``: leaves are split into ``ceil(N/P)`` independent sbatch jobs,
    each running up to P leaves concurrently as bare-background processes
    on the shared GPU allocation (no per-process srun — that would try to
    subdivide GRES). ``cpus_per_task`` scales with chunk size and
    ``OMP_NUM_THREADS`` is pinned per process so threads don't
    oversubscribe.

Constraints (predict-only by design):
  * All leaves must share ``launcher.mode == 'predict'`` and the same
    ``launcher.run_root``. Mixing modes or run_roots raises.
  * Leaves' SBATCH directives are merged from the FIRST leaf's
    ``launcher.sbatch`` block; subsequent leaves' sbatch blocks must
    match (same hardware profile). The composite job_name is taken
    from ``--job-name`` if provided, else derived from the first
    leaf with a ``_batch`` suffix; chunked jobs append ``_{i:02d}of{n:02d}``.
  * Wall time defaults to ``--time`` if provided; else keeps the head
    leaf's ``sbatch.time`` verbatim (a chunked job runs at most P leaves
    concurrently, so per-leaf wall time is usually enough).

Failure handling:
  * For ``P>1``, the rendered bash captures each background PID and
    propagates non-zero exit codes via ``wait $pid`` per child, so a
    single crashed predict fails the whole sbatch (no silent partial
    success masked by ``wait`` returning only the last child's status).
  * The submit loop catches ``sbatch`` failures per chunk, reports which
    chunks made it into the queue and which were skipped, and exits 1.

Usage::

    LEAVES=applications/dynacell/configs/benchmarks/virtual_staining/er/fnet3d_paper/ipsc_confocal
    uv run python applications/dynacell/tools/submit_benchmark_batch.py \
        $LEAVES/predict__a549_mantis_2024_11_07.yml \
        $LEAVES/predict__a549_mantis_2024_10_31.yml \
        $LEAVES/predict__a549_mantis_2025_07_24.yml \
        $LEAVES/predict__a549_mantis_2025_08_26.yml \
        --job-name FNet3DPaper_PRED_SEC61B_ON_A549_ALL \
        --parallel 2 \
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

# Soft cap for the rendered ``cpus_per_task`` under --parallel > 1. Most cluster
# nodes top out at 64–128 cores; requesting more makes the chunk pend forever or
# get rejected. Emit a warning above this threshold and let the user decide.
_CPUS_SOFT_CAP = 128


def _positive_int(value: str) -> int:
    """Argparse ``type=`` validator that rejects non-positive ints."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an int, got {value!r}") from exc
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


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
        "--parallel",
        type=_positive_int,
        default=1,
        metavar="P",
        help="run P predict invocations concurrently per sbatch job, and split "
        "the N leaves across ceil(N/P) sbatch jobs (default 1 = one sbatch, "
        "N leaves sequential). With P>1 each chunk's sbatch runs its <=P leaves "
        "as bare-background processes on the shared GPU allocation (no per-"
        "process srun — that would subdivide GRES), each writing its own log "
        "under run_root/slurm. cpus_per_task scales with the chunk size and "
        "OMP_NUM_THREADS is pinned per process so threads don't oversubscribe. "
        "Mem is left at the head leaf's value (predict at batch_size=1 has a "
        "tiny footprint for the current dynacell configs; recipes with larger "
        "models or num_workers>0 may need a head-leaf bump). 2-4 fit comfortably "
        "on a single H200 or A40.",
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


def submit(argv: list[str] | None = None) -> int:
    """Compose, render, and submit a chained predict sbatch from N leaves."""
    os.umask(0o002)
    args = _parse_args(argv)

    if len(args.leaves) < 1:
        raise SystemExit("at least one leaf is required")

    parsed_overrides = [_parse_override(t) for t in args.override]

    composed_list: list[dict] = []
    launcher_list: list[dict] = []
    for leaf in args.leaves:
        composed = load_composed_config(leaf, resolver=_dynacell_ref_resolver)
        for path, value in parsed_overrides:
            composed = _apply_override(composed, path, value)
        if args.overwrite:
            _apply_overwrite_alias(composed, leaf)
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

    # P=1 (default) keeps the original single-sbatch chained-srun layout:
    # ALL leaves run sequentially inside one allocation. P>1 splits the N
    # leaves into ceil(N/P) sbatches, each running up to P concurrent leaves
    # as bare-background processes on the shared GPU (no per-process srun,
    # which would try to subdivide the GRES). Each concurrent process writes
    # its own log under run_root/slurm so output doesn't interleave.
    if args.parallel > len(resolved_paths):
        sys.stderr.write(
            f"[submit] --parallel {args.parallel} capped to {len(resolved_paths)} "
            f"(only {len(resolved_paths)} leaf(s))\n"
        )
    parallel = min(args.parallel, len(resolved_paths))
    n_chunks = 1 if parallel == 1 else (len(resolved_paths) + parallel - 1) // parallel

    # Pre-write all resolved YAMLs once; chunks reference the same files.
    if not args.print_script:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        for composed, p in zip(composed_list, resolved_paths):
            p.write_text(yaml.safe_dump(composed, default_flow_style=False))

    template_text = (Path(__file__).parent / "sbatch_template_batch.sbatch").read_text()
    base_cpus = int(head_sbatch.get("cpus_per_task", 1))

    def _render_chunk(chunk_idx: int, chunk_paths: list[Path]) -> tuple[Path, str, str]:
        """Render the sbatch script for one chunk; return (script_path, name, body)."""
        suffix = f"_{chunk_idx + 1:02d}of{n_chunks:02d}"
        chunk_job_name = job_name if n_chunks == 1 else f"{job_name}{suffix}"
        # Scale cpus_per_task by chunk size so each concurrent predict gets the
        # head leaf's per-process CPU budget. Mem is NOT scaled — predict at
        # batch_size=1 has a tiny footprint for the current dynacell configs
        # (model ~128 MB, no backward activations, num_workers=0). Keep
        # ntasks_per_node/gpus unchanged — concurrent leaves share the single GPU.
        chunk_sbatch = dict(head_sbatch)
        if parallel > 1:
            scaled_cpus = base_cpus * len(chunk_paths)
            chunk_sbatch["cpus_per_task"] = scaled_cpus
            if scaled_cpus > _CPUS_SOFT_CAP:
                sys.stderr.write(
                    f"[submit] WARNING: chunk {chunk_idx + 1}/{n_chunks} requests "
                    f"cpus_per_task={scaled_cpus} (> soft cap {_CPUS_SOFT_CAP}); "
                    f"may pend forever or be rejected. Lower --parallel or pick a "
                    f"head profile with a smaller cpus_per_task.\n"
                )

        if parallel == 1:
            # P=1: srun-based sequential layout. srun handles signal propagation
            # and exit codes natively; no PID gymnastics needed.
            lines = []
            for i, p in enumerate(chunk_paths):
                p_q = shlex.quote(str(p))
                lines.append(f"echo '[batch] step {i + 1}/{len(chunk_paths)}: {p.name}'")
                lines.append(f"srun uv run python -m dynacell predict --config {p_q}")
                lines.append("")
            invocations = "\n".join(lines).rstrip()
        else:
            # P>1: bare-background processes share the GPU. Pin OMP_NUM_THREADS
            # per process so the P × base_cpus allocation isn't oversubscribed
            # by every child reading SLURM_CPUS_PER_TASK. Capture PIDs and
            # propagate non-zero exit codes — bare `wait` (no args) returns only
            # the LAST child's status, masking earlier crashes.
            lines = [
                f"echo '[batch] chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_paths)} concurrent'",
                "pids=()",
                "fail=0",
            ]
            # Build the log redirection path by concatenating shell-quoted
            # fragments with the unquoted ``${SLURM_JOB_ID}`` so bash expands the
            # job-id at runtime while keeping the directory and stem safe from
            # metachars leaking from ``experiment_id``.
            slurm_dir_q = shlex.quote(str(slurm_dir))
            for p in chunk_paths:
                stem_q = shlex.quote(p.stem)
                p_q = shlex.quote(str(p))
                log = f"{slurm_dir_q}/${{SLURM_JOB_ID}}_{stem_q}.log"
                lines.append(
                    f"OMP_NUM_THREADS={base_cpus} MKL_NUM_THREADS={base_cpus} "
                    f"uv run python -m dynacell predict --config {p_q} > {log} 2>&1 &"
                )
                lines.append("pids+=($!)")
            lines.extend(
                [
                    'for pid in "${pids[@]}"; do',
                    '  wait "$pid" || fail=$?',
                    "done",
                    'if [ "$fail" -ne 0 ]; then',
                    '  echo "[batch] one or more concurrent predicts failed (exit=$fail); see per-leaf logs" >&2',
                    '  exit "$fail"',
                    "fi",
                ]
            )
            invocations = "\n".join(lines)

        body = SbatchTemplate(template_text).substitute(
            sbatch_directives=_render_sbatch_directives(chunk_job_name, str(run_root), chunk_sbatch),
            run_root=str(run_root),
            env_block=_render_env_block(head_env),
            repo_root=str(_REPO_ROOT),
            predict_invocations=invocations,
        )
        script_path = slurm_dir / f"{timestamp}_{chunk_job_name}.sbatch"
        return script_path, chunk_job_name, body

    # P=1 packs ALL leaves into a single chunk (sequential sbatch). P>1 slices
    # into chunks of P; the final chunk may be shorter when N is not divisible.
    chunk_size = len(resolved_paths) if parallel == 1 else parallel
    chunk_artifacts: list[tuple[Path, str, str, list[Path]]] = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        chunk_paths = resolved_paths[start : start + chunk_size]
        script_path, chunk_name, body = _render_chunk(chunk_idx, chunk_paths)
        chunk_artifacts.append((script_path, chunk_name, body, chunk_paths))

    if args.print_script:
        # Single chunk → emit the body verbatim (pipeable to sbatch). Multi-chunk
        # output is for inspection only; refuse to print since concatenated
        # #!/bin/bash + #SBATCH blocks would not parse as one sbatch.
        if n_chunks > 1:
            raise SystemExit(
                "--print-script is single-chunk only; with --parallel > 1 over "
                f"{len(resolved_paths)} leaf(s), {n_chunks} sbatch scripts would be "
                "rendered. Use --dry-run to write them all to disk and inspect."
            )
        sys.stdout.write(chunk_artifacts[0][2])
        return 0

    for script_path, _name, body, _paths in chunk_artifacts:
        script_path.write_text(body)

    if args.dry_run:
        for script_path, _name, _body, paths in chunk_artifacts:
            print(f"[dry-run] sbatch script: {script_path}")
            for p in paths:
                print(f"[dry-run] resolved:    {p}")
        return 0

    # Submit each chunk; on failure, report submitted-vs-skipped so the user
    # can scancel orphans manually rather than chasing an opaque traceback.
    job_ids: list[str] = []
    failures: list[tuple[str, str]] = []
    for script_path, name, _body, _paths in chunk_artifacts:
        sbatch_cmd = ["sbatch"]
        if args.parsable:
            sbatch_cmd.append("--parsable")
        sbatch_cmd.append(str(script_path))
        try:
            if args.parsable:
                result = subprocess.run(sbatch_cmd, check=True, stdout=subprocess.PIPE, text=True)
                jid = result.stdout.strip()
                job_ids.append(jid)
                # One job ID per line on stdout (back-compat with single-chunk
                # callers parsing $(... --parsable)); name annotation on stderr.
                print(jid)
                sys.stderr.write(f"[submit] {jid}\t{name}\n")
            else:
                subprocess.run(sbatch_cmd, check=True)
                sys.stderr.write(f"[submit] queued {name}\n")
        except subprocess.CalledProcessError as exc:
            failures.append((name, str(exc)))
            sys.stderr.write(f"[submit] FAILED to queue {name}: {exc}\n")

    if failures:
        sys.stderr.write(
            f"[submit] {len(failures)}/{len(chunk_artifacts)} chunk(s) failed to submit; "
            f"{len(job_ids)} already queued: {','.join(job_ids) if job_ids else '<none>'}\n"
            f"[submit] scancel the queued chunk(s) above if you want to abort the whole batch.\n"
        )
        return 1
    if job_ids:
        sys.stderr.write(f"[submit] queued {len(job_ids)} job(s): {','.join(job_ids)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(submit())
