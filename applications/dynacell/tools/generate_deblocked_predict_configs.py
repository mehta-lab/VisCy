#!/usr/bin/env python3
"""Generate lattice-free pix2pix3d predict leaves for the qualitative overview figure.

``UNetViT3D`` predictions carry two grid-locked artifacts that are visible as block
structure in figure panels: the ViT bottleneck's 32 px token lattice (``unpatchify``
lays independently projected token blocks side by side) and the decoder's
ConvTranspose checkerboard at 2/4/8 px. Both are present *in* distribution on every
organelle. ``engine._phase_shift_average`` cancels them by averaging tiled
predictions over shifts of the analysis grid, and ``HCSPredictionWriter``'s
``z_reduction='center'`` keeps that from costing sharpness by taking each plane from
one forward pass instead of the mean of ``z_window_size``.

Each generated leaf inherits its canonical predict sibling wholesale -- same
checkpoint, same normalizations, same test store -- and overrides exactly four
things::

    model.init_args.predict_phase_shifts   [0, 5, 11, 22]
    trainer.callbacks[0].z_reduction       center
    trainer.callbacks[0].output_store      <bucket>/prediction_deblocked.zarr
    data.init_args.include_fov_names       [<the FOV the figure displays>]

**The output is a sibling store, not the canonical one.** The eval generators
discover predictions with ``glob("*/*/*/prediction.zarr")``, so
``prediction_deblocked.zarr`` is invisible to them and every metric table keeps
reading the canonical store. That is deliberate: shift averaging is test-time
augmentation, and applying it to pix2pix3d alone while the other five model
families run one pass would tilt the comparison. It is a rendering fix for the
qualitative figure until the same treatment is measured across all models.

**The FOV roster is the figure's own selection**, not an independent choice --
``fig_prediction_overview_focus.pick_crop`` per (organelle, test set), which is
``_pred_overview_common.PANEL_OVERRIDE`` where set and otherwise the top
foreground-ranked crop of the VSCyto3D (iPSC) eval. Pinning it cuts each job from
all 12 positions to the one that is rendered. Re-derive after any change to that
selection; a stale FOV here silently leaves the figure reading the canonical store's
blocky panel for a position the deblocked store never covered.

Usage::

    uv run python applications/dynacell/tools/generate_deblocked_predict_configs.py --dry-run
    uv run python applications/dynacell/tools/generate_deblocked_predict_configs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from dynacell.evaluation import paths

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/benchmarks/virtual_staining"
_MODEL_DIR = "pix2pix3d_unetvit"

# Offsets, in pixels, averaged over as the outer product on both YX axes: 16 tiled
# passes. Zeroes the period-2 and period-4 phasors exactly and holds 8/16/32 at
# ~0.21-0.27, which measured best of every four-offset set tried -- see
# engine._phase_shift_average for why the sets that cancel period 32 exactly come
# out two to three times worse.
_PHASE_SHIFTS = [0, 5, 11, 22]

# (organelle config dir, predict-set stem, canonical test token, condition, FOV)
# from the figure's own pick_crop; timepoint and focus plane are recorded for
# provenance only, since a predict covers every timepoint of the pinned FOV.
_PANELS: tuple[tuple[str, str, str, str | None, str, int, int], ...] = (
    ("nucleus", "ipsc_confocal", "ipsc", None, "41/68804/8999", 0, 22),
    ("nucleus", "a549_mantis_mock", "a549", "mock", "0/0/fov0000", 8, 16),
    ("membrane", "ipsc_confocal", "ipsc", None, "41/73112/8476", 0, 20),
    ("membrane", "a549_mantis_mock", "a549", "mock", "0/0/fov0002", 9, 20),
    ("membrane", "hek_mantis_kras_a549xy", "hek", "a549xy", "0/KRAS/000001", 0, 43),
    ("er", "ipsc_confocal", "ipsc", None, "4/40085/5421", 0, 27),
    ("er", "a549_mantis_mock", "a549", "mock", "0/0/fov0009", 6, 16),
    ("mito", "ipsc_confocal", "ipsc", None, "4/11476/16725", 0, 24),
    ("mito", "a549_mantis_mock", "a549", "mock", "0/0/fov0005", 7, 16),
    ("mito", "hek_mantis_tomm70a_a549xy", "hek", "a549xy", "0/TOMM70A/001000", 0, 28),
)

_TRAIN_DIRS: tuple[str, ...] = ("ipsc_confocal", "a549_mantis", "joint_ipsc_confocal_a549_mantis")

# There is no iPSC-trained mito pix2pix3d checkpoint (the fit leaf exists, the run was
# never done), so those three panels are grey in the figure and have nothing to
# de-block. Recorded explicitly so a 27-of-30 run reads as expected, not as breakage.
_ROSTER_HOLES: frozenset[tuple[str, str]] = frozenset({("mito", "ipsc_confocal")})

_TEST_STEM = {"ipsc": "IPSC", "a549": "A549", "hek": "HEK"}


def _train_stem(train_set: str) -> str:
    return {"ipsc": "IPSCTR", "a549": "A549TR", "joint": "JOINTTR"}[train_set]


def build_leaf(organelle: str, train_dir: str, panel: tuple) -> tuple[Path, str]:
    """Return (output leaf path, YAML text) for one (organelle, pool, panel) tuple."""
    _, predict_set, test_set, condition, fov, timepoint, focus_plane = panel
    sibling_path = _CONFIG_ROOT / organelle / _MODEL_DIR / train_dir / f"predict__{predict_set}.yml"
    if not sibling_path.is_file():
        raise FileNotFoundError(f"no canonical predict leaf to derive from: {sibling_path}")
    sibling = yaml.safe_load(sibling_path.read_text())
    # The leaf inherits the sibling's checkpoint, so a roster hole that slipped
    # through _ROSTER_HOLES should fail here rather than at submission time.
    ckpt = Path(sibling["model"]["init_args"]["ckpt_path"])
    if not ckpt.is_file():
        raise FileNotFoundError(f"{sibling_path.name} points at a checkpoint that does not exist: {ckpt}")

    train_set = paths._norm_train_set(train_dir)
    # Build the store from the canonical grammar rather than from the sibling's own
    # output_store: the ER/mito A549/joint leaves still carry the pre-retokenize
    # deconv-provenance train_set token (`a549__deconv`, `joint__legacy_deconvgt`),
    # which no table reads any more.
    canonical = paths.prediction_store(organelle, _MODEL_DIR, train_set, test_set, condition)
    out_store = canonical.with_name("prediction_deblocked.zarr")
    experiment_id = f"{organelle}__{train_dir}__{_MODEL_DIR}__{predict_set}__deblocked"
    job = f"PIX2PIX3D_DEBLOCK_{organelle.upper()}_{_train_stem(train_set)}_{_TEST_STEM[test_set]}"

    body = {
        "base": [f"./predict__{predict_set}.yml"],
        "benchmark": {"experiment_id": experiment_id},
        "model": {"init_args": {"predict_phase_shifts": _PHASE_SHIFTS}},
        "data": {"init_args": {"include_fov_names": [fov]}},
        "trainer": {
            "callbacks": [
                {
                    "class_path": "viscy_utils.callbacks.prediction_writer.HCSPredictionWriter",
                    "init_args": {"output_store": str(out_store), "z_reduction": "center"},
                }
            ]
        },
        "launcher": {"job_name": job, "run_root": str(out_store.parent / "deblocked")},
    }

    header = (
        f"# pix2pix3d_unetvit DEBLOCKED predict: {organelle} trained on {train_dir},\n"
        f"# predicting {predict_set} -- the tiling/lattice fix applied to the one FOV the\n"
        f"# qualitative overview figure renders for this row ({fov}, t={timepoint},\n"
        f"# focus plane {focus_plane}).\n"
        f"# Auto-generated by tools/generate_deblocked_predict_configs.py from\n"
        f"# {sibling_path.relative_to(_CONFIG_ROOT)} -- edit the generator, not this file.\n"
        f"# Writes prediction_deblocked.zarr beside the canonical prediction.zarr, which is\n"
        f"# left untouched so every metric table keeps its one-pass predictions.\n"
    )
    return sibling_path.with_name(f"predict_deblocked__{predict_set}.yml"), header + yaml.safe_dump(
        body, sort_keys=False, default_flow_style=False
    )


def main() -> int:
    """Emit one deblocked predict leaf per roster tuple."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print what would be written and exit")
    args = ap.parse_args()

    written = 0
    for panel in _PANELS:
        organelle = panel[0]
        for train_dir in _TRAIN_DIRS:
            if (organelle, train_dir) in _ROSTER_HOLES:
                print(f"[skip] {organelle}/{train_dir}: no trained checkpoint")
                continue
            out_path, text = build_leaf(organelle, train_dir, panel)
            rel = out_path.relative_to(_CONFIG_ROOT)
            if args.dry_run:
                print(f"[dry-run] would write {rel}")
                continue
            out_path.write_text(text)
            print(f"[write] {rel}")
            written += 1
    print(f"\n{'planned' if args.dry_run else 'wrote'} {len(_PANELS) * len(_TRAIN_DIRS) - 3} leaves")
    return 0 if args.dry_run or written else 1


if __name__ == "__main__":
    sys.exit(main())
