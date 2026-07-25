#!/usr/bin/env python3
"""Generate the HEK third-cell-type predict leaves from their A549 siblings.

Emits one predict leaf per ``(organelle, model, train_set)`` in the probe roster,
pointing an already-trained checkpoint at the HEK ``a549xy`` store. Nothing is
trained: this is the predict half of NeurIPS response item O.

**The sibling is the A549 leaf, not the iPSC one.** For each tuple the generator
reads ``<organelle>/<model_dir>/<train_dir>/predict__a549_mantis_mock.yml``
because that leaf is already a *foreign test set* leaf — it carries the
source-only predict normalizations and the gene-keyed ``dataset_ref.target``
override that a cross-cell-type predict needs. The iPSC leaf would not, and for
CELL-Diff there is no bare ``predict__ipsc_confocal.yml`` to read at all (only
the ``__iterative`` / ``__sliding_window`` / ``__denoise`` variants).

**Identity comes from inverting the sibling's own output_store**, via
``paths.key_from_prediction_store``, rather than from re-deriving it. That keeps
two conventions from being restated (and drifting): the CELL-Diff path token
depends on the training pool, not the checkpoint (iPSC-trained is
``celldiff_r2_iterative``, A549/joint are ``celldiff_r2``), and the config
directory name is not the path token (config dir ``celldiff`` -> token
``celldiff_r2*``). The leaf is written under the CONFIG dir spelling and its
``output_store`` uses the PATH token.

Usage::

    uv run python applications/dynacell/tools/generate_hek_predict_configs.py --dry-run
    uv run python applications/dynacell/tools/generate_hek_predict_configs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from dynacell.evaluation import paths

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/benchmarks/virtual_staining"

# Roster: the paper's focus trio (main.tex:279 / :1118) x 3 training pools x the
# two QC-passing HEK organelles. Excludes UNeXt2, UNetViT3D and pix2pix3d — the
# GAN in particular is absent from the submitted baseline suite, and three
# response drafts rest on "one generative family".
_MODEL_DIRS: tuple[str, ...] = ("fnet3d_paper", "fcmae_vscyto3d_pretrained", "celldiff")
_TRAIN_DIRS: tuple[str, ...] = ("ipsc_confocal", "a549_mantis", "joint_ipsc_confocal_a549_mantis")

# organelle config dir -> (manifest target key, eval-side target fragment)
_ORGANELLES: dict[str, tuple[str, str]] = {
    "membrane": ("kras", "membrane"),
    "mito": ("tomm70a", "mito_tomm20"),
}
_ARM = "a549xy"
# Job-name stems, kept short enough to stay readable in squeue.
_JOB_STEM: dict[str, str] = {
    "fnet3d_paper": "FNET3D",
    "fcmae_vscyto3d_pretrained": "VSCYTO3D",
    "celldiff": "CELLDIFF",
}
_TRAIN_STEM: dict[str, str] = {"ipsc": "IPSCTR", "a549": "A549TR", "joint": "JOINTTR"}


def _train_stem(train_set: str) -> str:
    """Job-name stem for a canonical train_set token, including deconv provenance.

    ER/mito A549- and joint-trained checkpoints carry their GT provenance in the
    token (``a549__deconv``, ``joint__legacy_deconvgt``), which the HEK path
    inherits: the token describes what the MODEL was trained against, and that
    does not change because it now predicts on HEK. Keeping it also keeps the HEK
    row comparable to the A549 row for the same checkpoint.
    """
    base = train_set.split("__", 1)[0]
    if base not in _TRAIN_STEM:
        raise ValueError(f"unknown train_set base {base!r} from token {train_set!r}")
    suffix = "_DECONVGT" if "deconv" in train_set else ""
    return _TRAIN_STEM[base] + suffix


# FCMAE predicts full_image at fp32: ~7 GB on A549 640x960x15, ~13 GB on the HEK
# 960x1184x15. hardware_predict_any_gpu leaves constraint null, which admits
# 24 GB L4s; pin >=48 GB for headroom. submit_benchmark_batch does not compare
# `constraint` across leaves, so this does not force --allow-mixed-directives.
_CONSTRAINT = "a40|l40s|a6000|a100|h100|h200"


def _sibling(organelle: str, model_dir: str, train_dir: str) -> Path:
    return _CONFIG_ROOT / organelle / model_dir / train_dir / "predict__a549_mantis_mock.yml"


def _model_overlay(base: list[str]) -> str:
    """Pull the model overlay fragment name out of a sibling leaf's ``base`` list."""
    for entry in base:
        if "model_overlays/" in entry:
            return entry.rsplit("/", 1)[-1]
    raise ValueError(f"no model_overlays/ entry in base list {base!r}")


def build_leaf(organelle: str, model_dir: str, train_dir: str) -> tuple[Path, str]:
    """Return (output leaf path, YAML text) for one roster tuple."""
    sibling_path = _sibling(organelle, model_dir, train_dir)
    if not sibling_path.is_file():
        raise FileNotFoundError(f"no A549 sibling leaf to derive from: {sibling_path}")
    sibling = yaml.safe_load(sibling_path.read_text())

    ckpt = sibling["model"]["init_args"]["ckpt_path"]
    if "REPLACE_ME" in str(ckpt):
        raise ValueError(f"{sibling_path} has a placeholder ckpt_path ({ckpt}); exclude this tuple")

    # Invert the sibling's canonical output_store to recover (organelle, model
    # token, train_set) without restating the CELL-Diff / config-dir conventions.
    key = paths.key_from_prediction_store(paths.extract_predict_output_store(sibling, sibling_path))
    target_key, target_fragment = _ORGANELLES[organelle]
    out_store = paths.prediction_store(key.organelle, key.model, key.train_set, "hek", _ARM)
    run_root = out_store.parent
    predict_set = f"hek_mantis_{target_key}_{_ARM}"
    job = f"{_JOB_STEM[model_dir]}_PRED_{organelle.upper()}_{_train_stem(key.train_set)}_HEK"

    body = {
        "base": [
            f"../../../_internal/shared/model/predict_sets/{predict_set}.yml",
            f"../../../_internal/shared/model/targets/{target_fragment}.yml",
            f"../../../_internal/shared/model/model_overlays/{_model_overlay(sibling['base'])}",
            "../../../_internal/shared/model/launcher_profiles/mode_predict.yml",
            "../../../_internal/shared/model/launcher_profiles/hardware_predict_any_gpu.yml",
            "../../../_internal/shared/model/launcher_profiles/runtime_shared.yml",
        ],
        "benchmark": {
            "task": "virtual_staining",
            "organelle": organelle,
            "trained_on": train_dir,
            "predict_set": predict_set,
            "model_name": sibling["benchmark"]["model_name"],
            "experiment_id": f"{organelle}__{train_dir}__{model_dir}__{predict_set}",
            "dataset_ref": {"target": target_key},
        },
        "model": {"init_args": {"ckpt_path": ckpt}},
        "data": {
            "init_args": {
                "normalizations": [
                    {
                        "class_path": "viscy_transforms.NormalizeSampled",
                        "init_args": {
                            "keys": ["Phase3D"],
                            "level": "fov_statistics",
                            "subtrahend": "mean",
                            "divisor": "std",
                        },
                    }
                ],
                "augmentations": [],
            }
        },
        "trainer": {
            "callbacks": [
                {
                    "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                    "init_args": {"output_store": str(out_store)},
                }
            ]
        },
        "launcher": {
            "job_name": job,
            "run_root": str(run_root),
            "sbatch": {"constraint": _CONSTRAINT},
        },
    }

    header = (
        f"# {sibling['benchmark']['model_name']} predict: {organelle} trained on {train_dir},\n"
        f"# predicting against the HEK293T third-cell-type probe ({target_key.upper()}, {_ARM} geometry).\n"
        f"# Auto-generated by tools/generate_hek_predict_configs.py from\n"
        f"# {sibling_path.relative_to(_CONFIG_ROOT)} — checkpoint lifted verbatim.\n"
        f"# Evaluation-only: no HEK training. Path token is {key.model!r} (config dir {model_dir!r}).\n"
    )
    leaf_path = _CONFIG_ROOT / organelle / model_dir / train_dir / f"predict__{predict_set}.yml"
    return leaf_path, header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)


def main(argv: list[str] | None = None) -> int:
    """Generate every HEK predict leaf in the roster."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = ap.parse_args(argv)

    written = 0
    errors: list[str] = []
    for organelle in _ORGANELLES:
        for model_dir in _MODEL_DIRS:
            for train_dir in _TRAIN_DIRS:
                try:
                    leaf_path, text = build_leaf(organelle, model_dir, train_dir)
                except (FileNotFoundError, ValueError) as exc:
                    errors.append(f"{organelle}/{model_dir}/{train_dir}: {exc}")
                    continue
                rel = leaf_path.relative_to(_CONFIG_ROOT)
                if args.dry_run:
                    print(f"[dry-run] {rel}")
                else:
                    leaf_path.write_text(text)
                    print(f"[gen] {rel}")
                written += 1

    if errors:
        print(f"\n[FAIL] {len(errors)} tuple(s) could not be generated:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"\n[ok] {written} leaves {'planned' if args.dry_run else 'written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
