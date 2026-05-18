#!/usr/bin/env python3
r"""Submit a sequence of dynacell evaluate calls (one per predict leaf) as ONE sbatch job.

For each leaf, the eval invocation is rendered the same way
``submit_evaluation_job.py`` would render it (same predict_set / target /
io.pred_path / save.save_dir Hydra overrides). All invocations run in
series on a single GPU within one sbatch allocation.

Eval doesn't need Lightning's DDP profile — it's a single-GPU Hydra job —
so a fixed single-GPU sbatch profile is baked into this script (no separate
``.sbatch`` template).

Usage::

    LEAF=applications/dynacell/configs/benchmarks/virtual_staining/membrane/fnet3d_paper/a549_mantis
    uv run python applications/dynacell/tools/submit_evaluation_batch.py \
        $LEAF/predict__a549_mantis_mock.yml \
        $LEAF/predict__a549_mantis_denv.yml \
        $LEAF/predict__a549_mantis_zikv.yml \
        --job-name EVAL_MEMBRANE_FNET3D_A549_ON_A549 \
        --dry-run

Re-running plates whose save_dir already contains metrics requires
``--overwrite`` (which adds ``force_recompute.all=true`` to every per-leaf
eval invocation).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dynacell.evaluation.save_paths import (
    DEFAULT_EVAL_RUN_ROOT as _DEFAULT_RUN_ROOT,
)
from dynacell.evaluation.save_paths import (
    ORGANELLE_EVAL_TARGET as _ORGANELLE_EVAL_TARGET,
)
from dynacell.evaluation.save_paths import (
    eval_predict_set_group as _eval_predict_set_group,
)
from dynacell.evaluation.save_paths import (
    eval_save_dir,
)
from dynacell.evaluation.save_paths import (
    extract_predict_output_store as _extract_output_store,
)
from dynacell.evaluation.save_paths import (
    paper_key as _paper_key,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("leaves", nargs="+", type=Path, help="paths to predict leaf YAMLs (>=1)")
    ap.add_argument(
        "--job-name",
        default=None,
        help="composite SLURM job name. Default: EVAL_<ORG>_<MODEL>_<TRAIN>_ON_<TEST>.",
    )
    ap.add_argument(
        "--time",
        default="4:00:00",
        help="SLURM walltime (default 4:00:00). One completed regen-metrics single-"
        "leaf iPSC eval (33009553) took 40 min, so a serial 3-plate A549 chain "
        "runs ~2h; 4h gives ~2× headroom for jitter / larger cohorts. With "
        "--parallel N, total elapsed drops by N× — the cap is then trivially "
        "comfortable. Walltime is billed at actual elapsed (not allocated), so "
        "bumping this is free insurance.",
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="directory under which `resolved/` and `slurm/` land. "
        "Default: first leaf's save_dir grandparent if it exists, else "
        f"{_DEFAULT_RUN_ROOT}.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="add force_recompute.all=true to every per-leaf eval invocation.",
    )
    ap.add_argument(
        "--regen-metrics",
        action="store_true",
        help="add force_recompute.final_metrics=true to every per-leaf eval invocation. "
        "Re-runs the metric loop and rewrites embeddings while reusing cached GT "
        "masks / CP / deep features. Mutually exclusive with --overwrite.",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of eval processes to run concurrently on the single allocated "
        "GPU (default 1 = serial, current behavior). Eval is GPU-light (~2 GB of "
        "~80 GB observed, 0–17%% utilization), so 2–4 parallel processes typically "
        "fit on one GPU without contention. CPU and memory allocations scale "
        "linearly with this value (4 × parallel CPUs, 16G × parallel mem). "
        "Auto-capped at len(leaves) — has no effect on single-leaf iPSC tuples.",
    )
    ap.add_argument("--dry-run", action="store_true", help="render sbatch script but do not submit")
    ap.add_argument("--print-script", action="store_true", help="print rendered sbatch to stdout, no writes")
    ap.add_argument("--parsable", action="store_true", help="invoke sbatch with --parsable")
    return ap.parse_args(argv)


def _resolve_one_leaf(
    leaf_path: Path,
) -> tuple[dict[str, object], str, str, str, Path, str]:
    """Compose a predict leaf and derive its eval-side overrides + identifiers.

    Returns
    -------
    overrides : dict
        Flat dict of Hydra overrides (dotted keys preserved).
    organelle : str
        Config-side organelle key.
    code_model : str
        Config-side model name.
    trained_on : str
        Config-side train-set key (e.g. ``ipsc_confocal``).
    save_dir : Path
        Canonical eval save_dir.
    test_plate : str
        ``ipsc`` | ``mock`` | ``denv`` | ``zikv``.
    """
    from dynacell._compose_hook import _dynacell_ref_resolver  # noqa: PLC0415
    from viscy_utils.compose import load_composed_config  # noqa: PLC0415

    if not leaf_path.is_file():
        raise SystemExit(f"leaf does not exist: {leaf_path}")
    composed = load_composed_config(leaf_path, resolver=_dynacell_ref_resolver)
    benchmark = composed.get("benchmark") or {}
    organelle = benchmark.get("organelle")
    code_model = benchmark.get("model_name")
    trained_on = benchmark.get("trained_on")
    dataset_ref = benchmark.get("dataset_ref") or {}
    dataset_name = dataset_ref.get("dataset")
    if not organelle or not code_model or not trained_on:
        raise SystemExit(f"{leaf_path}: missing benchmark.{{organelle, model_name, trained_on}}")
    if not dataset_name:
        raise SystemExit(f"{leaf_path}: missing benchmark.dataset_ref.dataset (needed to pick predict_set group)")
    output_store = _extract_output_store(composed, leaf_path)
    if not output_store.exists():
        raise SystemExit(
            f"predict output zarr does not exist: {output_store}\n  referenced by:                    {leaf_path}"
        )

    target_group = _ORGANELLE_EVAL_TARGET[organelle]
    predict_set_group = _eval_predict_set_group(organelle, dataset_name)

    target_yaml = (
        _REPO_ROOT
        / "applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/eval/target"
        / f"{target_group}.yaml"
    )
    if not target_yaml.is_file():
        raise SystemExit(f"eval target group YAML missing: {target_yaml}")

    leaf_stem = leaf_path.stem
    # CellDiff iPSC variants (`predict__ipsc_confocal__<variant>.yml`) all map
    # to test_plate="ipsc" — save_paths collapses celldiff variants to one
    # paper key, so the eval save_dir is shared across variants.
    if leaf_stem == "predict__ipsc_confocal" or leaf_stem.startswith("predict__ipsc_confocal__"):
        test_plate = "ipsc"
    elif leaf_stem.endswith("_mock"):
        test_plate = "mock"
    elif leaf_stem.endswith("_denv"):
        test_plate = "denv"
    elif leaf_stem.endswith("_zikv"):
        test_plate = "zikv"
    else:
        raise SystemExit(f"cannot infer test plate from leaf filename: {leaf_path.name}")

    save_dir = eval_save_dir(
        organelle=organelle,
        code_model=code_model,
        train_set=trained_on,
        test_plate=test_plate,
    )
    overrides: dict[str, object] = {
        "predict_set": predict_set_group,
        "target": target_group,
        "io.pred_path": str(output_store),
        "save.save_dir": str(save_dir),
        "compute_feature_metrics": True,
    }
    return overrides, organelle, code_model, trained_on, save_dir, test_plate


def _composite_job_name(organelle: str, code_model: str, trained_on: str, test_plates: list[str]) -> str:
    train_key_short = {
        "ipsc_confocal": "IPSC",
        "a549_mantis": "A549",
        "joint_ipsc_confocal_a549_mantis": "JOINT",
    }[trained_on]
    # Test scope: all-iPSC, all-A549, or mixed (the local runner enforces single test_set,
    # but support graceful naming if a user batches arbitrary leaves).
    if test_plates == ["ipsc"]:
        test_scope = "IPSC"
    elif set(test_plates) <= {"mock", "denv", "zikv"}:
        test_scope = "A549"
    else:
        test_scope = "MIXED"
    return f"EVAL_{organelle.upper()}_{_paper_key(code_model).upper()}_{train_key_short}_ON_{test_scope}"


# Single-GPU sbatch profile. Eval is single-process Hydra; no DDP.
# CPU/mem right-sized from in-flight sample (33009706 / 33009742 / 33009751 /
# 33009818 / 33009859 / 33009873 / 33009924, all on the regen-metrics path):
# observed peak RSS 2.7–4.0 GB and AveCPU ≈ 1.4 cores. 4 CPUs / 16 GB per
# concurrent eval gives 2.5× / 4× headroom. With --parallel N, CPU and mem
# scale linearly while a single GPU stays shared across N processes (GPU is
# the slack resource at ~2 GB / 0–17% per eval).
_CPUS_PER_EVAL = 4
_MEM_GB_PER_EVAL = 16


def _sbatch_directives(parallel: int) -> tuple[tuple[str, str], ...]:
    """Build SBATCH directives, scaling CPU/mem with the parallel multiplier."""
    return (
        ("--partition", "gpu"),
        ("--nodes", "1"),
        ("--ntasks-per-node", "1"),
        ("--gpus", "1"),
        ("--cpus-per-task", str(_CPUS_PER_EVAL * parallel)),
        ("--mem", f"{_MEM_GB_PER_EVAL * parallel}G"),
    )


def _render_sbatch(
    job_name: str,
    walltime: str,
    run_root: Path,
    cmds: list[list[str]],
    test_plates: list[str],
    parallel: int,
) -> str:
    """Render the sbatch script.

    With ``parallel == 1`` (default), leaves run serially via ``srun``. With
    ``parallel > 1``, leaves are grouped into waves of up to ``parallel``
    bare-background processes that share the single allocated GPU; each wave
    waits for all of its processes before the next wave starts. Each parallel
    process gets its own per-leaf stdout/stderr log file under
    ``run_root/slurm`` so output doesn't interleave.
    """
    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")
    if len(test_plates) != len(cmds):
        raise ValueError(f"test_plates / cmds length mismatch: {len(test_plates)} vs {len(cmds)}")

    lines = ["#!/bin/bash", ""]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --time={walltime}")
    for flag, val in _sbatch_directives(parallel):
        lines.append(f"#SBATCH {flag}={val}")
    lines.append(f"#SBATCH --output={run_root}/slurm/%j.out")
    lines.append(f"#SBATCH --error={run_root}/slurm/%j.err")
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("umask 0002")
    lines.append(f"mkdir -p -m 775 {run_root}/slurm")
    lines.append("")
    lines.append("ml uv 2>/dev/null || true")
    lines.append(f"cd {_REPO_ROOT}")
    lines.append("")
    lines.append("scontrol show job $SLURM_JOB_ID || true")
    lines.append("nvidia-smi || true")
    lines.append("")

    if parallel == 1:
        for i, cmd in enumerate(cmds, 1):
            joined = " ".join(cmd)
            lines.append(f"echo '[batch] step {i}/{len(cmds)}'")
            lines.append(f"srun {joined}")
            lines.append("")
    else:
        # Wave the leaves into chunks of `parallel`. Each chunk runs concurrently
        # as bare background processes on the shared allocation (no per-process
        # `srun` step — that would try to subdivide the GRES). Each process gets
        # its own log file keyed by test_plate so output is debuggable.
        n_waves = (len(cmds) + parallel - 1) // parallel
        for wave_idx in range(n_waves):
            start = wave_idx * parallel
            end = min(start + parallel, len(cmds))
            wave_cmds = cmds[start:end]
            wave_plates = test_plates[start:end]
            lines.append(f"echo '[batch] wave {wave_idx + 1}/{n_waves} ({len(wave_cmds)} concurrent)'")
            for plate, cmd in zip(wave_plates, wave_cmds, strict=True):
                joined = " ".join(cmd)
                log = f"{run_root}/slurm/${{SLURM_JOB_ID}}_{plate}.log"
                lines.append(f"{joined} > {log} 2>&1 &")
            lines.append("wait")
            lines.append("")
    return "\n".join(lines) + "\n"


def submit(argv: list[str] | None = None) -> int:
    """Render and submit a chained-eval sbatch from N predict leaves."""
    os.umask(0o002)
    args = _parse_args(argv)

    if args.overwrite and args.regen_metrics:
        raise SystemExit("--overwrite and --regen-metrics are mutually exclusive")

    overrides_list: list[dict[str, object]] = []
    organelles: list[str] = []
    code_models: list[str] = []
    trained_ons: list[str] = []
    save_dirs: list[Path] = []
    test_plates: list[str] = []
    for leaf in args.leaves:
        overrides, organelle, code_model, trained_on, save_dir, test_plate = _resolve_one_leaf(leaf)
        if args.overwrite:
            overrides["force_recompute.all"] = True
        elif args.regen_metrics:
            overrides["force_recompute.final_metrics"] = True
        overrides_list.append(overrides)
        organelles.append(organelle)
        code_models.append(code_model)
        trained_ons.append(trained_on)
        save_dirs.append(save_dir)
        test_plates.append(test_plate)

    if len(set(organelles)) != 1:
        raise SystemExit(f"all leaves must share benchmark.organelle (got {sorted(set(organelles))})")
    if len(set(code_models)) != 1:
        raise SystemExit(f"all leaves must share benchmark.model_name (got {sorted(set(code_models))})")
    if len(set(trained_ons)) != 1:
        raise SystemExit(f"all leaves must share benchmark.trained_on (got {sorted(set(trained_ons))})")
    organelle = organelles[0]
    code_model = code_models[0]
    trained_on = trained_ons[0]

    job_name = args.job_name or _composite_job_name(organelle, code_model, trained_on, test_plates)

    if args.run_root is not None:
        run_root = args.run_root
    else:
        gp = save_dirs[0].parent.parent
        run_root = gp if gp.exists() else _DEFAULT_RUN_ROOT

    # Build per-leaf command lines.
    cmds: list[list[str]] = []
    for overrides in overrides_list:
        cmd = ["uv", "run", "dynacell", "evaluate"]
        for k, v in overrides.items():
            token = "true" if v is True else "false" if v is False else str(v)
            cmd.append(f"{k}={token}")
        cmds.append(cmd)

    # Cap parallelism at the number of leaves — no point allocating CPU/mem for
    # idle slots on a single-leaf iPSC tuple.
    parallel = max(1, min(args.parallel, len(cmds)))
    if args.parallel > parallel:
        print(
            f"[submit] --parallel {args.parallel} capped to {parallel} (only {len(cmds)} leaf(s))",
            file=sys.stderr,
        )

    rendered = _render_sbatch(job_name, args.time, run_root, cmds, test_plates, parallel)

    if args.print_script:
        sys.stdout.write(rendered)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    resolved_dir = run_root / "resolved"
    slurm_dir = run_root / "slurm"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = slurm_dir / f"{timestamp}_{job_name}.sbatch"
    sbatch_path.write_text(rendered)

    # Also drop a per-leaf overrides YAML for provenance, matching the
    # single-leaf submitter's --dry-run output shape.
    import yaml  # noqa: PLC0415

    for i, overrides in enumerate(overrides_list):
        plate = test_plates[i]
        yml_path = resolved_dir / f"evaluate_{job_name}_{plate}_{timestamp}.yml"
        yml_path.write_text(yaml.safe_dump(overrides, default_flow_style=False))

    if args.dry_run:
        print(f"[dry-run] sbatch:    {sbatch_path}")
        for i in range(len(overrides_list)):
            plate = test_plates[i]
            print(f"[dry-run] overrides: {resolved_dir / f'evaluate_{job_name}_{plate}_{timestamp}.yml'}")
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
