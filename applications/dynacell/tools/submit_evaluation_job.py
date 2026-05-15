#!/usr/bin/env python3
r"""Stage a ``dynacell evaluate`` invocation from a predict leaf.

Reads a per-plate predict leaf (the same YAML consumed by
``submit_benchmark_job.py``), composes it via
:func:`viscy_utils.compose.load_composed_config`, extracts the predicted
zarr path (``output_store``) and the eval-side ``predict_set`` group name
(via ``benchmark.dataset_ref.dataset``), maps the leaf's organelle to the
eval ``target`` group, and builds the literal ``dynacell evaluate``
override token list.

Output:
  * ``--print-cmd``: print the command (one token per line) to stdout.
  * ``--dry-run``: write the command + a flat YAML of overrides to
    ``<run_root>/resolved/evaluate_<job_name>_<timestamp>.{cmd,yml}``.
  * default: execute the command via ``subprocess.run``.

Usage::

    LEAF=applications/dynacell/configs/benchmarks/virtual_staining/membrane
    uv run python applications/dynacell/tools/submit_evaluation_job.py \
        $LEAF/fnet3d_paper/a549_mantis/predict__a549_mantis_mock.yml \
        --dry-run --print-cmd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

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
    ap.add_argument("leaf", type=Path, help="path to a predict leaf YAML")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="add ``force_recompute.all=true`` to the override list. Required to "
        "regenerate metrics already present in save_dir.",
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "directory under which `resolved/` provenance lands. "
            f"Default: derived from save_dir's grandparent, fallback {_DEFAULT_RUN_ROOT}."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="render command + overrides yaml to <run_root>/resolved/ but do not exec.",
    )
    ap.add_argument(
        "--print-cmd",
        action="store_true",
        help="print the literal `dynacell evaluate` command to stdout (no exec). "
        "Combine with --dry-run to also write provenance files.",
    )
    return ap.parse_args(argv)


def submit(argv: list[str] | None = None) -> int:
    """Stage and (optionally) submit a `dynacell evaluate` call."""
    os.umask(0o002)
    args = _parse_args(argv)

    # Compose lazily so --help works without an editable dynacell install.
    from dynacell._compose_hook import _dynacell_ref_resolver  # noqa: PLC0415
    from viscy_utils.compose import load_composed_config  # noqa: PLC0415

    leaf_path = args.leaf.resolve()
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
        raise SystemExit(
            f"{leaf_path}: missing benchmark.{{organelle, model_name, trained_on}} "
            f"(got organelle={organelle!r}, model_name={code_model!r}, trained_on={trained_on!r})"
        )
    if not dataset_name:
        raise SystemExit(f"{leaf_path}: missing benchmark.dataset_ref.dataset (needed to pick eval predict_set group)")

    # Resolve the predict output zarr (must exist on disk — predict already ran).
    output_store = _extract_output_store(composed, leaf_path)
    if not output_store.exists():
        raise SystemExit(
            f"predict output zarr does not exist: {output_store}\n"
            f"  referenced by:                    {leaf_path}\n"
            f"  (run `dynacell predict -c <resolved>` first)"
        )

    # Map predict-side identifiers to eval-side Hydra groups.
    target_group = _ORGANELLE_EVAL_TARGET[organelle]
    predict_set_group = _eval_predict_set_group(organelle, dataset_name)

    # Eval `target` group must exist on disk under the external searchpath.
    target_yaml = (
        _REPO_ROOT
        / "applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/eval/target"
        / f"{target_group}.yaml"
    )
    if not target_yaml.is_file():
        raise SystemExit(f"eval target group YAML missing: {target_yaml}")

    # Plate label for save_dir: ipsc | mock | denv | zikv.
    # Predict leaves are named predict__<test_set>[_<cond>].yml. Extract the
    # trailing condition for a549; the bare ipsc_confocal predict leaf is iPSC.
    leaf_stem = leaf_path.stem  # predict__a549_mantis_mock
    if leaf_stem == "predict__ipsc_confocal":
        test_plate = "ipsc"
    elif leaf_stem.endswith("_mock"):
        test_plate = "mock"
    elif leaf_stem.endswith("_denv"):
        test_plate = "denv"
    elif leaf_stem.endswith("_zikv"):
        test_plate = "zikv"
    else:
        raise SystemExit(
            f"cannot infer test plate from leaf filename: {leaf_path.name} "
            f"(expected predict__ipsc_confocal.yml or predict__*_{{mock,denv,zikv}}.yml)"
        )

    save_dir = eval_save_dir(
        organelle=organelle,
        code_model=code_model,
        train_set=trained_on,
        test_plate=test_plate,
    )

    # job_name: EVAL_<ORG>_<MODEL_PAPER>_<TRAIN>_<PLATE>.
    train_key_short = {
        "ipsc_confocal": "IPSC",
        "a549_mantis": "A549",
        "joint_ipsc_confocal_a549_mantis": "JOINT",
    }[trained_on]
    plate_upper = "IPSC" if test_plate == "ipsc" else test_plate.upper()
    job_name = f"EVAL_{organelle.upper()}_{_paper_key(code_model).upper()}_{train_key_short}_{plate_upper}"

    # run_root: prefer save_dir's grandparent if it exists; else flag; else default.
    if args.run_root is not None:
        run_root = args.run_root
    else:
        grandparent = save_dir.parent.parent
        run_root = grandparent if grandparent.exists() else _DEFAULT_RUN_ROOT

    overrides: dict[str, object] = {
        "predict_set": predict_set_group,
        "target": target_group,
        "io.pred_path": str(output_store),
        "save.save_dir": str(save_dir),
        "compute_feature_metrics": True,
    }
    if args.overwrite:
        overrides["force_recompute.all"] = True

    cmd_tokens = ["uv", "run", "dynacell", "evaluate"]
    for k, v in overrides.items():
        # Hydra wants lowercase booleans; everything else is straight repr.
        if isinstance(v, bool):
            token_val = "true" if v else "false"
        else:
            token_val = str(v)
        cmd_tokens.append(f"{k}={token_val}")

    if args.print_cmd:
        sys.stdout.write("\n".join(cmd_tokens) + "\n")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    resolved_dir = run_root / "resolved"
    cmd_path = resolved_dir / f"evaluate_{job_name}_{timestamp}.cmd"
    yml_path = resolved_dir / f"evaluate_{job_name}_{timestamp}.yml"

    if args.dry_run:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        cmd_path.write_text("\n".join(cmd_tokens) + "\n")
        yml_path.write_text(yaml.safe_dump(overrides, default_flow_style=False))
        if not args.print_cmd:
            print(f"[dry-run] cmd:       {cmd_path}")
            print(f"[dry-run] overrides: {yml_path}")
            print(f"[dry-run] job_name:  {job_name}")
            print(f"[dry-run] save_dir:  {save_dir}")
        return 0

    # Execute. Surface stdout/stderr live; raise on non-zero.
    subprocess.run(cmd_tokens, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(submit())
