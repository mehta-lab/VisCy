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
from dynaclr.evaluation.paths import DATASETS_ROOT

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


def _embedding_complete(output_path: Path) -> bool:
    """Whether a per-marker embedding zarr exists and looks fully written.

    Metadata-only (no array load): the dir must exist and carry a top-level
    ``zarr.json`` (v3) or ``.zgroup`` (v2). A bare directory left by a
    crash mid-write (no group metadata) counts as NOT complete, so it re-runs.
    """
    return output_path.exists() and ((output_path / "zarr.json").exists() or (output_path / ".zgroup").exists())


def resolve_datasets_to_run(models: list[dict], overwrite: bool) -> list[dict]:
    """Prune already-embedded datasets so predict runs only what's missing.

    Progressive-collection pre-step (login node, before any submit). For each
    model row and each of its checkpoints, computes the expected per-reporter
    output zarrs via :func:`plan_predict_runs` and marks an experiment DONE
    when all of its expected zarrs are complete. Fully-done rows are dropped;
    partially-done rows get a pruned temp collection (only the missing
    experiments) and their ``collection`` repointed to it. ``overwrite`` skips
    the check entirely (whole collection runs).

    Read-only w.r.t. the datasets (only checks existence); the only writes are
    pruned collection YAMLs next to the original collection.

    Parameters
    ----------
    models : list[dict]
        Resolved model rows (from :func:`load_matrix`).
    overwrite : bool
        If ``True``, return ``models`` unchanged (run everything).

    Returns
    -------
    list[dict]
        Filtered rows: fully-done rows removed; survivors may have their
        ``collection`` set to a pruned temp YAML.
    """
    from dynaclr.evaluation.predict_triplet import plan_predict_runs
    from viscy_data.collection import load_collection, save_collection

    if overwrite:
        print("[skip-existing] --overwrite: running the full collection for every row.", file=sys.stderr)
        return models

    survivors: list[dict] = []
    for model in models:
        family, run = model["family"], model["run"]
        collection = load_collection(Path(model["collection"]))
        markers = model.get("markers")
        datasets_root = model.get("datasets_root", DATASETS_ROOT)

        # An experiment is done only when done for EVERY checkpoint this row sweeps.
        done_names: set[str] = None  # type: ignore[assignment]
        for ckpt_name, _ in resolve_checkpoints(model):
            runs = plan_predict_runs(
                collection,
                model_family=family,
                run=run,
                ckpt_name=ckpt_name,
                datasets_root=datasets_root,
                markers=markers,
            )
            paths_by_exp: dict[str, list[Path]] = {}
            for r in runs:
                paths_by_exp.setdefault(r.experiment, []).append(r.output_path)
            ckpt_done = {exp for exp, paths in paths_by_exp.items() if all(_embedding_complete(p) for p in paths)}
            done_names = ckpt_done if done_names is None else (done_names & ckpt_done)

        done_names = done_names or set()
        all_names = [exp.name for exp in collection.experiments]
        missing_names = [n for n in all_names if n not in done_names]
        label = f"{family}/{run}"

        if not missing_names:
            print(f"[skip-existing] {label}: all {len(all_names)} dataset(s) done → skipping row.", file=sys.stderr)
            continue
        if not done_names:
            print(f"[skip-existing] {label}: {len(missing_names)} dataset(s) to run (none done).", file=sys.stderr)
            survivors.append(model)
            continue

        # Partial: write a pruned collection with only the missing experiments.
        pruned = collection.model_copy(
            update={"experiments": [e for e in collection.experiments if e.name in missing_names]}
        )
        src = Path(model["collection"])
        pruned_path = src.with_name(f"{src.stem}.pending-{family}-{run}{src.suffix}")
        save_collection(pruned, pruned_path)
        model = {**model, "collection": str(pruned_path)}
        print(
            f"[skip-existing] {label}: {len(done_names)} done (skipped: {sorted(done_names)}), "
            f"{len(missing_names)} to run: {missing_names} → pruned collection {pruned_path}",
            file=sys.stderr,
        )
        survivors.append(model)

    return survivors


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
    merged.setdefault("model_type", "dynaclr")

    if merged["model_type"] == "foundation":
        for required in ("family", "run", "ckpt_name", "training_config"):
            if required not in merged:
                raise ValueError(
                    f"foundation model entry needs explicit '{required}' (no train_sbatch parsing): {model}"
                )
        return merged

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
    the prediction wrapper. ``model_type`` (dynaclr | foundation) and
    ``training_config`` (foundation only) are the last two positionals; for
    dynaclr rows ``training_config`` is empty and ``checkpoint`` is forwarded.
    """
    markers = ",".join(model["markers"]) if model.get("markers") else ""
    predict_flags = model.get("predict_flags", {})
    z_range = predict_flags.get("z_range")
    z_window = predict_flags.get("z_window")
    if z_range is not None and z_window is not None:
        raise ValueError("predict_flags: z_range and z_window are mutually exclusive; set only one.")
    if predict_flags.get("reference_pixel_size_z_um") is not None and z_window is None:
        raise ValueError("predict_flags.reference_pixel_size_z_um requires predict_flags.z_window.")
    if z_range is not None and len(z_range) != 2:
        raise ValueError("predict_flags.z_range must contain exactly two values.")
    z_start, z_end = ("", "") if z_range is None else map(str, z_range)
    model_type = model.get("model_type", "dynaclr")
    training_config = str(model["training_config"]) if model_type == "foundation" else ""
    args = [
        "sbatch",
        str(_PREDICT_SBATCH),
        model["collection"],
        model["family"],
        model["run"],
        ckpt_name,
        str(model["datasets_root"]),
        checkpoint,  # empty = derive from run dir (foundation: unused)
        markers,  # empty = all channels
        z_start,
        z_end,
        str(predict_flags.get("z_reduction") or ""),
        str(predict_flags.get("reference_pixel_size") or ""),
        str(predict_flags.get("batch_size") or ""),
        model_type,  # dynaclr | foundation
        training_config,  # foundation only; empty for dynaclr
        str(z_window or ""),  # focus-centered Z window WIDTH (excl. with z_range)
        str(predict_flags.get("focus_channel") or ""),
        str(predict_flags.get("z_focus_offset") or ""),
    ]
    if predict_flags.get("reference_pixel_size_z_um") is not None:
        args.append(str(predict_flags["reference_pixel_size_z_um"]))
    return args


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
    is_foundation = model.get("model_type", "dynaclr") == "foundation"
    cmds: list[tuple[str, list[str]]] = []
    if "train" in stages and not is_foundation:
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
    is_foundation = model.get("model_type", "dynaclr") == "foundation"
    train_jid: str | None = None
    if "train" in stages and not is_foundation:
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
    default="predict,eval",
    help=f"Comma-separated subset of {_ALL_STAGES} to run (default: predict,eval). "
    "Pass 'train,predict,eval' to also submit the DynaCLR training job "
    "(foundation rows never train).",
)
@click.option("--dry-run", is_flag=True, help="Print the chained commands, submit nothing.")
@click.option("--print-cmd", is_flag=True, help="Alias for --dry-run.")
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip the upfront AI-ready (normalization/focus_slice) check across all datasets.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Re-embed the whole collection. Default (--no-overwrite): skip datasets whose embeddings "
    "already exist and predict only the newly-added ones (progressive collections).",
)
def main(matrix: Path, stages: str, dry_run: bool, print_cmd: bool, skip_preflight: bool, overwrite: bool) -> None:
    """Run many models through train→predict→eval in parallel (SLURM afterok chain)."""
    stage_tuple = tuple(s.strip() for s in stages.split(",") if s.strip())
    bad = [s for s in stage_tuple if s not in _ALL_STAGES]
    if bad:
        raise click.BadParameter(f"unknown stage(s): {bad}; valid: {_ALL_STAGES}", param_hint="--stages")

    models = load_matrix(matrix)
    print_only = dry_run or print_cmd

    # Skip-existing pre-step: prune already-embedded datasets so predict runs only the
    # missing ones (login node, read-only w.r.t. datasets — safe in dry-run too). Only
    # meaningful when predicting.
    if "predict" in stage_tuple:
        models = resolve_datasets_to_run(models, overwrite)

    # Upfront preflight: only meaningful when we will predict, and only on a real
    # submission (dry-run must stay side-effect-free / offline).
    if "predict" in stage_tuple and not print_only and not skip_preflight:
        matrix_preflight(models)

    print(f"{len(models)} model(s); stages={stage_tuple}; {'DRY-RUN' if print_only else 'SUBMITTING'}", file=sys.stderr)
    for model in models:
        run_model(model, stage_tuple, print_only)


if __name__ == "__main__":
    main()
