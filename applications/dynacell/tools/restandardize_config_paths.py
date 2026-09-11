#!/usr/bin/env python
"""Codemod: rewrite dynacell predict/train config-leaf paths to the canonical grammar.

Part of the Phase-7 canonicalization. After the artifact migration
(``run_migration.py``) relocates checkpoints + predictions onto the
``dynacell.evaluation.paths`` grammar, the config leaves that drive re-training /
re-prediction must point at the new locations, or Phase-9 re-predict reads stale
checkpoints and writes to legacy prediction dirs.

Scope (this tool): the hand-authored ``train.yml`` and ``predict__*.yml`` leaves
under ``applications/dynacell/configs/benchmarks/virtual_staining/<organelle>/
<model>/<train_term>/``. Generated eval leaves (grouped / instance_ap) are handled
by fixing their GENERATORS + regeneration, not here.

Rewrite rules (per the verified field inventory) — a value is rewritten IFF it
resolves to a MIGRATED artifact; everything else is preserved:

- **checkpoint paths** (value under ``models/dynacell/`` or ``models/cell_diff_vs_viscy/``):
  ``model.init_args.ckpt_path`` (predict: the trained ckpt to LOAD), ``launcher.run_root``
  / ``ModelCheckpoint.dirpath`` / ``logger.save_dir`` (train: the model OUTPUT dir).
  Rewritten via the SAME old-model-dir -> canonical-dest map the manifest builder
  computes (``collect_checkpoints``), preserving any ``/checkpoints/<file>`` tail.
  This correctly sends ER/mito A549 ckpts to the raw ``a549`` home (not the
  ``a549__deconv`` prediction home). A checkpoint NOT in the map (iPSC-trained,
  external init, published baseline, ``REPLACE_ME`` placeholder) is left untouched.
- **prediction output** (``HCSPredictionWriter.output_store``, a data-tree ``.zarr``):
  ``normalize_legacy(old) -> prediction_store(...)`` — byte-identical to the
  manifest's prediction dest.
- **predict ``launcher.run_root``** (a data-tree ``predictions/`` dir): set to the
  canonical prediction leaf dir (``output_store``'s parent), co-locating logs with
  the prediction.

PRESERVED (never rewritten): training-input ``data*.data_path``, eval GT
``nuclei_gt_path``, ``scratch_dir``, and any checkpoint value outside the two
migrated model roots.

Dry-run by default (prints a unified diff); ``--apply`` writes in place. Config
leaves are git-tracked, so ``--apply`` is fully reversible via git.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

from build_migration_manifest import (
    CONFIG_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    _referenced_model_dirs,
    collect_checkpoints,
)

from dynacell.evaluation import paths

CONFIG_ROOT_REL = "applications/dynacell/configs/benchmarks/virtual_staining"

# Value roots.
_MODELS_MIGRATED_ROOTS = (
    str(MODELS_ROOT) + "/",  # .../models/dynacell/
    str(MODELS_ROOT.parent / "cell_diff_vs_viscy") + "/",
)

# A ``key: /hpc/projects/...`` line (captures indent, key, value; tolerates inline
# trailing whitespace). Values are unquoted in these leaves.
_KV = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<val>/hpc/projects/\S+)\s*$")

# Keys that may carry a rewritable path (others with /hpc values are inputs we preserve).
_CKPT_OR_DIR_KEYS = frozenset({"ckpt_path", "run_root", "dirpath", "save_dir"})
_PRED_OUT_KEY = "output_store"


def build_ckpt_map(models_root: Path) -> dict[str, str]:
    """Return {old model-dir -> canonical dest} for every migrated checkpoint run.

    Includes both the dedup ``move`` winner and its ``dedup_legacy`` siblings, so a
    config pointing at EITHER cross-root copy resolves to the same canonical dest.

    Re-enumerates the LIVE models tree — valid only BEFORE ``run_migration`` has
    relocated the checkpoints (else the legacy dirs are gone and the map is empty).
    After migrating, pass the frozen manifest to ``load_ckpt_map_from_manifest``.
    """
    rows, gaps = collect_checkpoints(
        models_root, full_hardlink_check=False, referenced_dirs=_referenced_model_dirs(CONFIG_ROOT)
    )
    if gaps:
        raise RuntimeError("checkpoint enumeration has gaps; refusing to codemod:\n" + "\n".join(gaps))
    return {r.src: r.dest for r in rows if r.status in ("move", "dedup_legacy")}


def load_ckpt_map_from_manifest(manifest: Path) -> dict[str, str]:
    """Return {old model-dir -> canonical dest} from a frozen manifest CSV.

    The codemod runs AFTER ``run_migration`` relocates the checkpoints, so the live
    models tree can no longer be re-enumerated for the legacy dirs. The manifest is
    the frozen record of what moved (checkpoint ``move`` + ``dedup_legacy`` rows) —
    the same {src -> dest} pairs ``build_ckpt_map`` would have produced pre-migration.
    """
    out: dict[str, str] = {}
    with manifest.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["kind"] == "checkpoint" and row["status"] in ("move", "dedup_legacy"):
                out[row["src"]] = row["dest"]
    if not out:
        raise RuntimeError(f"no checkpoint move rows in manifest {manifest}; refusing to codemod")
    return out


def _rewrite_ckpt_value(value: str, ckpt_map: dict[str, str]) -> str | None:
    """Rewrite a checkpoint-tree value via the map, preserving any file/subdir tail.

    Handles every on-disk shape a checkpoint-tree value takes in these leaves:

    - a bare model dir (``save_dir`` / train ``run_root``) — an exact map key;
    - ``<model_dir>/checkpoints`` or ``<model_dir>/checkpoints/<file>`` (``dirpath`` /
      ``ckpt_path``), split on ``/checkpoints``;
    - a bare best-epoch hardlink alias at ``<model_dir>/<file>`` with no ``checkpoints/``
      subdir (several legacy iPSC runs store the best ckpt this way), or a sibling subdir
      like ``<model_dir>/smoke`` — split on the final path separator.

    Without the last branch ``find`` returned ``-1``, ``model_dir`` swallowed the whole
    value, the map lookup missed, and the stale (migrated-away) path was silently kept.

    Returns the canonical value, or None if the value is not a migrated checkpoint
    (outside the two model roots, or a model dir absent from the map) -> preserve.
    """
    if not value.startswith(_MODELS_MIGRATED_ROOTS):
        return None
    if value in ckpt_map:  # bare model dir (save_dir / train run_root), no tail
        return ckpt_map[value]
    idx = value.find("/checkpoints")
    if idx != -1:
        model_dir, tail = value[:idx], value[idx:]
    else:
        parent, _, filename = value.rpartition("/")
        model_dir, tail = parent, "/" + filename
    dest = ckpt_map.get(model_dir)
    if dest is None:
        return None
    return dest + tail


def _rewrite_prediction_value(value: str) -> str | None:
    """Rewrite a data-tree prediction ``.zarr`` value via normalize_legacy. None -> preserve."""
    if not value.endswith(".zarr"):
        return None
    key = paths.normalize_legacy(value)
    if key is None:
        return None
    return str(paths.prediction_store(key.organelle, key.model, key.train_set, key.test_set, key.condition))


def rewrite_leaf(text: str, ckpt_map: dict[str, str]) -> tuple[str, list[str]]:
    """Return (new_text, changes) for one predict/train leaf. changes is human-readable."""
    lines = text.splitlines(keepends=True)
    changes: list[str] = []

    # Pass 1: canonical prediction output (drives predict run_root).
    canonical_pred: str | None = None
    for line in lines:
        m = _KV.match(line)
        if m and m.group("key") == _PRED_OUT_KEY:
            canonical_pred = _rewrite_prediction_value(m.group("val"))
            break

    out: list[str] = []
    for line in lines:
        m = _KV.match(line)
        if not m:
            out.append(line)
            continue
        key, val, indent = m.group("key"), m.group("val"), m.group("indent")
        new_val: str | None = None
        if key == _PRED_OUT_KEY:
            new_val = _rewrite_prediction_value(val)
        elif key == "run_root" and val.startswith(str(DATA_ROOT) + "/"):
            # predict working dir -> canonical prediction leaf dir (output_store parent)
            new_val = str(Path(canonical_pred).parent) if canonical_pred else None
        elif key in _CKPT_OR_DIR_KEYS:
            new_val = _rewrite_ckpt_value(val, ckpt_map)
        if new_val is not None and new_val != val:
            out.append(f"{indent}{key}: {new_val}\n" if line.endswith("\n") else f"{indent}{key}: {new_val}")
            changes.append(f"{key}: {val}\n         -> {new_val}")
        else:
            out.append(line)
    return "".join(out), changes


def iter_leaves(config_root: Path) -> list[Path]:
    """Every hand-authored predict/train leaf under the organelle config tree."""
    leaves = sorted(config_root.rglob("predict__*.yml"))
    leaves += sorted(config_root.rglob("train.yml"))
    leaves += sorted(config_root.rglob("train_*.yml"))  # train_4gpu / train_smoke variants
    return leaves


def main(argv: list[str] | None = None) -> int:
    """CLI: dry-run (diff) or --apply the predict/train leaf path codemod."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / CONFIG_ROOT_REL,
        help="root of the benchmark config tree (default: repo config root)",
    )
    ap.add_argument("--apply", action="store_true", help="write changes in place (default: dry-run diff)")
    ap.add_argument("--models-root", type=Path, default=MODELS_ROOT, help="checkpoint models root (for the ckpt map)")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="frozen manifest CSV; build the ckpt map from it instead of re-scanning the "
        "(already-migrated) models tree. Required when run AFTER run_migration --apply.",
    )
    ap.add_argument("--show-diff", action="store_true", help="print a unified diff per changed file")
    args = ap.parse_args(argv)

    ckpt_map = load_ckpt_map_from_manifest(args.manifest) if args.manifest else build_ckpt_map(args.models_root)
    leaves = iter_leaves(args.config_root)
    n_changed = 0
    n_field_changes = 0
    for leaf in leaves:
        text = leaf.read_text()
        new_text, changes = rewrite_leaf(text, ckpt_map)
        if not changes:
            continue
        n_changed += 1
        n_field_changes += len(changes)
        rel = leaf.relative_to(args.config_root)
        print(f"{'APPLY' if args.apply else 'DRY'} {rel}  ({len(changes)} field(s))")
        if args.show_diff:
            diff = difflib.unified_diff(
                text.splitlines(), new_text.splitlines(), fromfile=str(rel), tofile=str(rel), lineterm=""
            )
            print("\n".join(diff))
        else:
            for c in changes:
                print(f"    {c}")
        if args.apply:
            leaf.write_text(new_text)

    print(f"\n{'applied' if args.apply else 'would change'}: {n_changed} leaf file(s), {n_field_changes} field(s)")
    print(f"checkpoint map: {len(ckpt_map)} old-dir -> canonical-dest entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
