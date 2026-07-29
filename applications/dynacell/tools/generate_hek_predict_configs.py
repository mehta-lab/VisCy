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
import copy
import sys
from pathlib import Path

import yaml

from dynacell.evaluation import paths

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/benchmarks/virtual_staining"

# Roster: the paper's focus trio (main.tex:279 / :1118) plus pix2pix3d, x 3
# training pools x the two QC-passing HEK organelles. Still excludes UNeXt2 and
# bare UNetViT3D. pix2pix3d was added on request: it is the second *generative*
# family in the suite (CELL-Diff being the first), so the cross-cell-type result
# no longer rests on a single generative model.
_MODEL_DIRS: tuple[str, ...] = (
    "fnet3d_paper",
    "fcmae_vscyto3d_pretrained",
    "celldiff",
    "pix2pix3d_unetvit",
)
_TRAIN_DIRS: tuple[str, ...] = ("ipsc_confocal", "a549_mantis", "joint_ipsc_confocal_a549_mantis")

# Roster holes: tuples with no trained model to point at *yet*. Recorded explicitly so
# a short run is a stated 5-of-6 rather than a generic failure — the placeholder-ckpt
# guard in build_leaf would otherwise report this as an error every single run.
#
# TEMPORARY. This is a missed training submission, not a structural absence, and the
# backfill fit is in flight (job 35053575, authored as
# mito/pix2pix3d_unetvit/ipsc_confocal/train_4gpu_modernized.yml). Delete this entry
# once that fit produces a checkpoint and the A549-mock predict sibling this generator
# reads has a real ckpt_path; the tuple then generates like any other.
_NO_TRAINED_MODEL: dict[tuple[str, str, str], str] = {
    ("mito", "pix2pix3d_unetvit", "ipsc_confocal"): (
        "iPSC-trained mito GAN not trained yet — backfill fit in flight (job 35053575). "
        "Its train_4gpu_modernized.yml was the only gap in the 12-cell pix2pix grid, so the "
        "fit was never submitted; wandb has A549_TOMM20 and JOINT_TOMM20 but no iPSC_TOMM20 "
        "run in any state. Remove this entry once the checkpoint exists"
    ),
}

# organelle config dir -> (manifest target key, eval-side target fragment)
_ORGANELLES: dict[str, tuple[str, str]] = {
    "membrane": ("kras", "membrane"),
    "mito": ("tomm70a", "mito_tomm20"),
}
_ARM = "a549xy"
# Marker of a deconvolved GT channel in a fit config (e.g. Structure_deconvolved).
DECONV_SUFFIX = "_deconvolved"

# Z depth of the A549 assembly and of the HEK a549xy stores. CELL-Diff is the only
# model in the roster that pins ``z_window_size`` in its leaf, and it pins it to the
# store's FULL depth (48 on A549, and the celldiff_predict overlay default of 40 is
# likewise the full iPSC depth) so each FOV is predicted in a single window. Carrying
# 48 (or falling through to the overlay's 40) onto the 64-plane HEK stack instead
# slides the window: 64-40+1 = 25 windows per FOV, measured at ~1.6 h/window on an
# H200, i.e. ~160 h/leaf versus ~10 h for four full-depth windows. FNet3D and
# VSCyto3D set no z_window_size and genuinely slide their 15-plane window, which
# already yields the full Z=64 output; they must not be rescaled.
#
# pix2pix3d also pins z_window_size, but in its OVERLAY (=8) rather than its leaf,
# and 8 is a genuinely sliding window like FNet3D's 15, not a full-depth one. The
# rescale is keyed on the leaf precisely so that distinction survives: the leaf
# carries no window, _hek_z_window returns None, and the overlay's 8 composes
# through untouched. Rescaling it would be wrong twice over — the ViT generator is
# fixed at 512x512xZ and cannot take a 64-plane input at all.
_A549_Z = 48
_HEK_Z = 64
# Job-name stems, kept short enough to stay readable in squeue.
_JOB_STEM: dict[str, str] = {
    "fnet3d_paper": "FNET3D",
    "fcmae_vscyto3d_pretrained": "VSCYTO3D",
    "celldiff": "CELLDIFF",
    "pix2pix3d_unetvit": "PIX2PIX3D",
}
_TRAIN_STEM: dict[str, str] = {"ipsc": "IPSCTR", "a549": "A549TR", "joint": "JOINTTR"}


def _train_stem(train_set: str) -> str:
    """Job-name stem for a canonical train_set token.

    No deconv-provenance suffix: ``_assert_raw_trained`` rejects any
    deconvolution-trained fit before this is reached, and ``train_set`` comes from
    ``paths._norm_train_set`` over the three roster training dirs, none of which
    normalize to a deconv token. Every stem reaching this function is therefore
    plain ``ipsc`` / ``a549`` / ``joint``.
    """
    base = train_set.split("__", 1)[0]
    if base not in _TRAIN_STEM:
        raise ValueError(f"unknown train_set base {base!r} from token {train_set!r}")
    return _TRAIN_STEM[base]


# FCMAE predicts full_image at fp32: ~7 GB on A549 640x960x15, ~13 GB on the HEK
# 960x1184x15. hardware_predict_any_gpu leaves constraint null, which admits
# 24 GB L4s; pin >=48 GB for headroom. submit_benchmark_batch does not compare
# `constraint` across leaves, so this does not force --allow-mixed-directives.
_CONSTRAINT = "a40|l40s|a6000|a100|h100|h200"


def _sibling(organelle: str, model_dir: str, train_dir: str) -> Path:
    return _CONFIG_ROOT / organelle / model_dir / train_dir / "predict__a549_mantis_mock.yml"


def _assert_raw_trained(organelle: str, model_dir: str, train_dir: str) -> str:
    """Refuse any tuple whose fit trained against a deconvolved GT channel.

    HEK is scored against a RAW fluorescence target, so a deconvolution-trained
    model would be compared across a GT-processing difference on top of the
    cell-type and marker shifts. The authority is the fit config's
    ``target_channel``, not the path token: the A549 ER/mito predict leaves still
    carry stale ``a549__deconv`` / ``joint__legacy_deconvgt`` output_store tokens
    from before the raw-flip retrain, while the live predictions and evals sit
    under the plain ``a549`` / ``joint`` tokens and every current mito fit uses
    ``target_channel: Structure``. Trusting the token would both mislabel the
    provenance and write the HEK results to a path no table reads.
    """
    train_yaml = _CONFIG_ROOT / organelle / model_dir / train_dir / "train.yml"
    if not train_yaml.is_file():
        raise FileNotFoundError(f"no fit config to check GT provenance: {train_yaml}")
    text = train_yaml.read_text()
    if DECONV_SUFFIX in text:
        raise ValueError(
            f"{train_yaml.relative_to(_CONFIG_ROOT)} references a deconvolved GT channel; "
            "HEK is scored against raw fluorescence, so deconvolution-trained models are "
            "out of scope for this probe"
        )
    return text


def _model_overlay(base: list[str]) -> str:
    """Pull the model overlay fragment name out of a sibling leaf's ``base`` list."""
    for entry in base:
        if "model_overlays/" in entry:
            return entry.rsplit("/", 1)[-1]
    raise ValueError(f"no model_overlays/ entry in base list {base!r}")


def _hek_z_window(sibling: dict, sibling_path: Path) -> int | None:
    """Rescale a full-depth ``z_window_size`` from the A549 depth to the HEK depth.

    Returns ``None`` when the sibling pins no window (FNet3D, VSCyto3D — they
    slide and need no override). Raises when the sibling pins a window that is
    not the A549 full depth, since then the full-depth assumption behind the
    rescale no longer holds and the intended HEK window is not derivable.
    """
    window = sibling.get("data", {}).get("init_args", {}).get("z_window_size")
    if window is None:
        return None
    if window != _A549_Z:
        raise ValueError(
            f"{sibling_path.name} pins z_window_size={window}, which is not the A549 full depth "
            f"{_A549_Z}; the full-depth rescale to the HEK depth {_HEK_Z} does not apply — set the "
            "HEK window explicitly"
        )
    return _HEK_Z


def _writer_init_args(sibling: dict, sibling_path: Path) -> dict:
    """Return the sibling's ``HCSPredictionWriter`` init_args, minus ``output_store``.

    Everything else the sibling tells the writer to do is a property of how that MODEL
    tiles, not of which test set it runs on, so it carries over to HEK verbatim. This is
    the same inherit-don't-restate rule ``build_leaf`` applies to the data/model blocks,
    and for the same reason: pix2pix3d predicts set ``z_reduction: center`` (each plane
    taken from the depth window centred on it rather than the mean of all 8 covering
    windows, which halves mid-band power). Restating only ``output_store`` here silently
    dropped that on every regeneration, reverting it to the mean blend.
    """
    for cb in sibling.get("trainer", {}).get("callbacks", []):
        if cb.get("class_path", "").endswith("HCSPredictionWriter"):
            init = copy.deepcopy(cb.get("init_args", {}))
            init.pop("output_store", None)
            return init
    raise ValueError(f"{sibling_path.name} has no HCSPredictionWriter callback to inherit writer args from")


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
    _assert_raw_trained(organelle, model_dir, train_dir)
    # The MODEL token comes from the sibling (it encodes the CELL-Diff pool
    # convention), but the TRAIN_SET token comes from the training config dir. The
    # sibling's own token is stale for ER/mito (see _assert_raw_trained), and using
    # it would put the HEK row under a train_set no table reads.
    train_set = paths._norm_train_set(train_dir)
    target_key, target_fragment = _ORGANELLES[organelle]
    out_store = paths.prediction_store(key.organelle, key.model, train_set, "hek", _ARM)
    run_root = out_store.parent
    predict_set = f"hek_mantis_{target_key}_{_ARM}"
    job = f"{_JOB_STEM[model_dir]}_PRED_{organelle.upper()}_{_train_stem(train_set)}_HEK"

    # Inherit the sibling's data/model init_args wholesale and override only what
    # HEK's geometry forces. Hardcoding a normalization block here instead was a
    # real bug: it is right for FNet3D and VSCyto3D (NormalizeSampled at
    # fov_statistics) but wrong for CELL-Diff, whose leaves all use MinMaxSampled
    # at timepoint_statistics, so the CELL-Diff HEK predicts were feeding z-scored
    # input to a min-max-trained model. Nothing about a change of TEST SET should
    # touch how the model's input is prepared, so the generator no longer has an
    # opinion about it.
    data_init: dict = copy.deepcopy(sibling.get("data", {}).get("init_args", {}))
    model_init: dict = copy.deepcopy(sibling["model"]["init_args"])
    z_window = _hek_z_window(sibling, sibling_path)
    if z_window is not None:
        data_init["z_window_size"] = z_window

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
        "model": {"init_args": model_init},
        "data": {"init_args": data_init},
        "trainer": {
            "callbacks": [
                {
                    "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                    "init_args": {"output_store": str(out_store), **_writer_init_args(sibling, sibling_path)},
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
        f"# {sibling_path.relative_to(_CONFIG_ROOT)} — checkpoint and the sibling's\n"
        f"# data/model init_args (normalizations, predict hparams) and prediction-writer\n"
        f"# args lifted verbatim.\n"
        f"# Fit verified to train against a raw (non-deconvolved) GT channel.\n"
        f"# Evaluation-only: no HEK training. Path token is {key.model!r} (config dir {model_dir!r}).\n"
    )
    if z_window is not None:
        header += (
            f"# z_window_size rescaled {_A549_Z} -> {_HEK_Z} (the HEK stack depth): this model\n"
            f"# predicts one full-depth window per FOV, not a sliding window.\n"
        )
    leaf_path = _CONFIG_ROOT / organelle / model_dir / train_dir / f"predict__{predict_set}.yml"
    return leaf_path, header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)


def main(argv: list[str] | None = None) -> int:
    """Generate every HEK predict leaf in the roster."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = ap.parse_args(argv)

    written = 0
    skipped = 0
    errors: list[str] = []
    for organelle in _ORGANELLES:
        for model_dir in _MODEL_DIRS:
            for train_dir in _TRAIN_DIRS:
                hole = _NO_TRAINED_MODEL.get((organelle, model_dir, train_dir))
                if hole is not None:
                    print(f"[skip] {organelle}/{model_dir}/{train_dir}: {hole}")
                    skipped += 1
                    continue
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
    tail = f" ({skipped} skipped: no trained model)" if skipped else ""
    print(f"\n[ok] {written} leaves {'planned' if args.dry_run else 'written'}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
