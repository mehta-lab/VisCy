r"""Run many DynaCLR models in parallel through train → predict → eval.

Reads a matrix YAML and, per model, submits three SLURM jobs chained by
``--dependency=afterok`` (train → predict → eval). Models run fully in parallel;
within a model the stages chain. The eval side reuses ``dynaclr eval`` /
the Nextflow ``eval_from_embeddings`` entry; predict reuses ``dynaclr predict-batch``.

Matrix YAML — ``defaults:`` block + per-model ``train_sbatch`` (the ``.sh``):

    defaults:
      ckpt_name: last
      collection: applications/dynaclr/configs/collections/<...>.yml
      eval_config: applications/dynaclr/configs/evaluation/<...>.yaml
      datasets_root: /hpc/projects/intracellular_dashboard/organelle_dynamics
      predict_flags: {z_range: [15, 45], z_reduction: mip, reference_pixel_size: 0.1494}
    models:
      - train_sbatch: applications/dynaclr/configs/training/DynaCLR-2D/<...>.sh
      - train_sbatch: applications/dynaclr/configs/training/DynaCLR-3D/<...>.sh

``family`` / ``run`` / ``train_configs`` are parsed from each ``train_sbatch``'s
``export PROJECT=`` / ``RUN_NAME=`` / ``CONFIGS=`` lines (override by giving the
fields explicitly on the model entry). Shared fields come from ``defaults``.

Output:
  * ``--print-cmd``: print the chained sbatch commands (no submission).
  * ``--dry-run``: alias for ``--print-cmd``.
  * default: submit; capture each job id and wire ``afterok`` dependencies.

Usage::

    dynaclr run-matrix \
        -c applications/dynaclr/configs/matrix/<name>.yml \
        [--stages train,predict,eval] [--dry-run]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import click
import yaml

from dynaclr.evaluation.orchestration.predict_batch import check_ai_ready

# Wrapper sbatch scripts the launcher submits for the predict / eval links.
_PREDICT_SBATCH = Path("applications/dynaclr/tools/predict.sbatch")
_EVAL_SBATCH = Path("applications/dynaclr/tools/eval.sbatch")

_ALL_STAGES = ("train", "predict", "eval")


def matrix_preflight(models: list[dict]) -> None:
    """Report AI-readiness for every model's collection before submitting anything.

    Runs once upfront (only relevant when the predict stage is included) so all
    non-AI-ready datasets are surfaced before any job queues. Raises if any
    dataset is missing ``focus_slice`` (needs manual QC) or ``normalization``.
    """
    seen: set[str] = set()
    missing_focus: list[str] = []
    missing_norm: list[str] = []
    for model in models:
        collection = model.get("collection")
        if not collection or str(collection) in seen:
            continue
        seen.add(str(collection))
        for r in check_ai_ready(Path(collection)):
            if r["missing_focus_slice"]:
                missing_focus.append(f"{r['name']} ({r['data_path']})")
            if r["missing_normalization"]:
                missing_norm.append(f"{r['name']} ({r['data_path']})")

    if missing_norm:
        print("[preflight] datasets missing normalization (run `viscy preprocess` — safe/auto):", file=sys.stderr)
        for m in missing_norm:
            print(f"    - {m}", file=sys.stderr)
    if missing_focus:
        lines = "\n".join(f"    - {m}" for m in missing_focus)
        raise ValueError(
            "[preflight] datasets missing focus_slice — run QC manually before the matrix "
            "(z-focus needs per-dataset physics params):\n"
            f"{lines}\n"
            "Run `qc run -c <dataset>/qc_config.yml` for each, then retry."
        )
    if missing_norm:
        raise ValueError(
            "[preflight] datasets missing normalization (see above). Run "
            "`viscy preprocess --data_path <zarr> --channel_names=-1` for each, then retry."
        )
    print("[preflight] all datasets AI-ready.", file=sys.stderr)


# `export NAME="value"` or `export NAME=value` (value may be quoted).
_EXPORT_RE = re.compile(r'^\s*export\s+(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', re.MULTILINE)


def parse_train_sbatch(sbatch_path: Path) -> dict:
    """Parse PROJECT / RUN_NAME / CONFIGS out of a training ``.sh``.

    Parameters
    ----------
    sbatch_path : Path
        A DynaCLR training script that exports ``PROJECT``, ``RUN_NAME``, and
        ``CONFIGS`` (space-separated config paths).

    Returns
    -------
    dict
        ``{"family": PROJECT, "run": RUN_NAME, "train_configs": [..]}``.

    Raises
    ------
    ValueError
        If any of the three exports is missing (fail loud — a malformed script
        must not silently mis-resolve).
    """
    text = Path(sbatch_path).read_text()
    exports: dict[str, str] = {}
    for m in _EXPORT_RE.finditer(text):
        name = m.group(1)
        value = next(g for g in m.groups()[1:] if g is not None)
        exports[name] = value
    missing = [k for k in ("PROJECT", "RUN_NAME", "CONFIGS") if k not in exports]
    if missing:
        raise ValueError(f"{sbatch_path} is missing required export(s): {', '.join(missing)}")
    return {
        "family": exports["PROJECT"],
        "run": exports["RUN_NAME"],
        "train_configs": exports["CONFIGS"].split(),
    }


def resolve_model(model: dict, defaults: dict) -> dict:
    """Merge defaults into a model entry and resolve identity from its ``.sh``.

    ``{**defaults, **model}`` (model wins); ``predict_flags`` is shallow-merged.
    ``family`` / ``run`` / ``train_configs`` are parsed from ``train_sbatch``
    unless the entry provides them explicitly (override).
    """
    merged = {**defaults, **model}
    merged["predict_flags"] = {**defaults.get("predict_flags", {}), **model.get("predict_flags", {})}

    if "train_sbatch" not in merged:
        raise ValueError(f"model entry has no 'train_sbatch' (and no explicit identity): {model}")
    parsed = parse_train_sbatch(Path(merged["train_sbatch"]))
    # Explicit fields on the entry override the parsed values.
    for key in ("family", "run", "train_configs"):
        merged.setdefault(key, parsed[key])

    for required in ("family", "run"):
        if required not in merged:
            raise ValueError(f"model missing required field '{required}' after resolve: {merged.get('train_sbatch')}")
    if "ckpt_name" not in merged and "ckpt_names" not in merged:
        raise ValueError(f"model needs 'ckpt_name' or 'ckpt_names' after resolve: {merged.get('train_sbatch')}")
    return merged


def load_matrix(path: Path) -> list[dict]:
    """Load the matrix YAML and return the list of fully-resolved model dicts."""
    doc = yaml.safe_load(Path(path).read_text())
    defaults = doc.get("defaults", {})
    models = doc.get("models", [])
    if not models:
        raise ValueError(f"matrix {path} has no 'models'.")
    return [resolve_model(m, defaults) for m in models]


def resolve_checkpoints(model: dict) -> list[tuple[str, str]]:
    """Return the (ckpt_name, checkpoint_path) units for a model's sweep.

    Supports a checkpoint sweep via ``ckpt_names: [..]`` (labels) and/or
    ``checkpoints: [..]`` (explicit paths), falling back to the scalar
    ``ckpt_name`` / ``checkpoint``. When both lists are given they are zipped
    (must be equal length); a list of labels with no explicit paths yields empty
    paths (derived downstream). One train job feeds every unit.
    """
    names = model.get("ckpt_names") or [model.get("ckpt_name", "last")]
    paths = model.get("checkpoints")
    if paths is None:
        scalar = model.get("checkpoint")
        paths = [scalar] * len(names) if scalar else [""] * len(names)
    if len(paths) != len(names):
        raise ValueError(
            f"ckpt_names ({len(names)}) and checkpoints ({len(paths)}) length mismatch for {model.get('train_sbatch')}"
        )
    return [(str(n), str(p or "")) for n, p in zip(names, paths)]


def build_train_cmd(model: dict) -> list[str]:
    """The train sbatch command (the model's own .sh; env baked in)."""
    return ["sbatch", str(model["train_sbatch"])]


def build_predict_cmd(model: dict, ckpt_name: str, checkpoint: str) -> list[str]:
    """The predict sbatch command for one checkpoint of a model.

    ``markers`` and ``predict_flags`` are forwarded as positional values to
    the prediction wrapper.
    """
    markers = ",".join(model["markers"]) if model.get("markers") else ""
    predict_flags = model.get("predict_flags", {})
    z_range = predict_flags.get("z_range")
    if z_range is not None and len(z_range) != 2:
        raise ValueError("predict_flags.z_range must contain exactly two values.")
    z_start, z_end = ("", "") if z_range is None else map(str, z_range)
    return [
        "sbatch",
        str(_PREDICT_SBATCH),
        model["collection"],
        model["family"],
        model["run"],
        ckpt_name,
        str(model["datasets_root"]),
        checkpoint,  # empty = derive from run dir
        markers,  # empty = all channels
        z_start,
        z_end,
        str(predict_flags.get("z_reduction") or ""),
        str(predict_flags.get("reference_pixel_size") or ""),
        str(predict_flags.get("batch_size") or ""),
    ]


def build_eval_cmd(model: dict, ckpt_name: str) -> list[str]:
    """The eval sbatch command for one checkpoint of a model."""
    return [
        "sbatch",
        str(_EVAL_SBATCH),
        model["eval_config"],
        model["family"],
        model["run"],
        ckpt_name,
        str(model["datasets_root"]),
    ]


def build_stage_cmds(model: dict, stages: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Flat (stage, command) list for the FIRST checkpoint — used by tests / simple linear view.

    The full sweep fan-out (train once → predict→eval per checkpoint) is built in
    :func:`run_model`. This helper keeps the single-checkpoint shape that unit
    tests assert against.
    """
    ckpt_name, checkpoint = resolve_checkpoints(model)[0]
    m = {**model, "ckpt_name": ckpt_name, "checkpoint": checkpoint}
    cmds: list[tuple[str, list[str]]] = []
    if "train" in stages:
        cmds.append(("train", build_train_cmd(m)))
    if "predict" in stages:
        cmds.append(("predict", build_predict_cmd(m, ckpt_name, checkpoint)))
    if "eval" in stages:
        cmds.append(("eval", build_eval_cmd(m, ckpt_name)))
    return cmds


def _submit(cmd: list[str]) -> str:
    """Submit an sbatch command and return its job id (parsed from stdout)."""
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    # sbatch prints "Submitted batch job 12345"
    return out.strip().split()[-1]


def _emit(stage: str, label: str, cmd: list[str], dep_jid: str | None, print_only: bool) -> str | None:
    """Print or submit one stage command with an optional afterok dependency.

    Returns the submitted job id (or None when printing).
    """
    dep = f"--dependency=afterok:{dep_jid if not print_only else '<prev>'}" if dep_jid else None
    full = cmd[:1] + ([dep] if dep else []) + cmd[1:]
    if print_only:
        print(f"# {label} [{stage}]")
        print(" ".join(full))
        return None
    jid = _submit(full)
    print(f"{label} [{stage}] -> job {jid}", file=sys.stderr)
    return jid


def run_model(model: dict, stages: tuple[str, ...], print_only: bool) -> None:
    """Submit/print a model's jobs: train ONCE, then predict→eval per checkpoint.

    A model may sweep multiple checkpoints (``ckpt_names`` / ``checkpoints``).
    Train runs once; each checkpoint gets its own predict→eval chain, both
    ``--dependency=afterok`` on the shared train job (models and per-checkpoint
    chains otherwise run in parallel via normal SLURM scheduling).
    """
    base = f"{model['family']}/{model['run']}"
    train_jid: str | None = None
    if "train" in stages:
        train_jid = _emit("train", base, build_train_cmd(model), None, print_only)

    for ckpt_name, checkpoint in resolve_checkpoints(model):
        label = f"{base}/{ckpt_name}"
        prev = train_jid  # predict depends on the shared train (if any)
        if "predict" in stages:
            prev = _emit("predict", label, build_predict_cmd(model, ckpt_name, checkpoint), prev, print_only)
        if "eval" in stages:
            _emit("eval", label, build_eval_cmd(model, ckpt_name), prev, print_only)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--matrix",
    "matrix",
    type=click.Path(path_type=Path),
    required=True,
    help="Matrix YAML (defaults + models).",
)
@click.option(
    "--stages",
    default=",".join(_ALL_STAGES),
    help=f"Comma-separated subset of {_ALL_STAGES} to run (default: all).",
)
@click.option("--dry-run", is_flag=True, help="Print the chained commands, submit nothing.")
@click.option("--print-cmd", is_flag=True, help="Alias for --dry-run.")
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip the upfront AI-ready (normalization/focus_slice) check across all datasets.",
)
def main(matrix: Path, stages: str, dry_run: bool, print_cmd: bool, skip_preflight: bool) -> None:
    """Run many models through train→predict→eval in parallel (SLURM afterok chain)."""
    stage_tuple = tuple(s.strip() for s in stages.split(",") if s.strip())
    bad = [s for s in stage_tuple if s not in _ALL_STAGES]
    if bad:
        raise click.BadParameter(f"unknown stage(s): {bad}; valid: {_ALL_STAGES}", param_hint="--stages")

    models = load_matrix(matrix)
    print_only = dry_run or print_cmd

    # Upfront preflight: only meaningful when we will predict, and only on a real
    # submission (dry-run must stay side-effect-free / offline).
    if "predict" in stage_tuple and not print_only and not skip_preflight:
        matrix_preflight(models)

    print(f"{len(models)} model(s); stages={stage_tuple}; {'DRY-RUN' if print_only else 'SUBMITTING'}", file=sys.stderr)
    for model in models:
        run_model(model, stage_tuple, print_only)


if __name__ == "__main__":
    main()
