#!/usr/bin/env python
"""Codemod: bake each predict leaf's best-by-monitor checkpoint into ``ckpt_path``.

After the canonical-path migration + ``restandardize_config_paths`` codemod, predict
leaves point at a checkpoint DIRECTORY that is canonical but may carry a stale FILENAME
(the path codemod preserves the tail): the retrained ER/mito A549/joint leaves still
name a pre-flip ``epoch=NNN`` file that no longer exists. This pass rewrites
``model.init_args.ckpt_path`` in every predict leaf to the actual best checkpoint
resolved from disk, so the final evals run straight from the leaf with no ``--ckpt
best`` override.

Resolution per leaf (only for a ``ckpt_path`` value under ``MODELS_ROOT``):

- normal trained dir (has ``last*.ckpt``): best = newest ``last*.ckpt``'s
  ``best_model_path`` (rebased onto the dir), else the highest ``epoch=*.ckpt``.
  Uses ``submit_benchmark_job.resolve_best_ckpt`` — the same resolver ``--ckpt best``
  uses at submit time — so a baked leaf and ``--ckpt best`` agree by construction.
- pure-alias dir (a curated ``best_ep*.ckpt`` with no ``last*.ckpt`` / ``epoch=*.ckpt``
  to scan, as several legacy iPSC runs store it): the current value IS the curated
  best — kept, not clobbered.
- unresolvable / external / published baseline / ``REPLACE_ME`` placeholder: preserved
  and reported (never trained -> no best to list).

Only ``predict__*.yml`` leaves are touched; a train leaf's ``ckpt_path`` is an init /
resume source, not a prediction target. Dry-run by default (prints a per-leaf report);
``--apply`` writes in place. Leaves are git-tracked, so ``--apply`` is reversible.
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
from collections.abc import Callable
from pathlib import Path

from submit_benchmark_job import resolve_best_ckpt

from dynacell.evaluation import paths

CONFIG_ROOT_REL = "applications/dynacell/configs/benchmarks/virtual_staining"

_MODELS_ROOT_PREFIX = str(paths.MODELS_ROOT) + "/"
_CKPT_LINE = re.compile(r"^(?P<indent>\s*)ckpt_path:\s*(?P<val>\S+)\s*$")
_REPLACE_ME = "REPLACE_ME"

Resolver = Callable[[Path], "Path | None"]


def _ckpt_dir_for(value: str) -> Path:
    """Return the directory ``resolve_best_ckpt`` scans for a ``ckpt_path`` value.

    ``<model_dir>/checkpoints/<file>`` -> ``<model_dir>/checkpoints``; a bare alias at
    ``<model_dir>/<file>`` -> ``<model_dir>`` (the alias's own dir).
    """
    idx = value.find("/checkpoints")
    if idx != -1:
        return Path(value[: idx + len("/checkpoints")])
    return Path(value).parent


def canonical_ckpt_dir_for_leaf(rel: Path) -> Path | None:
    """Grammar-derived canonical checkpoint dir from a leaf's config-tree position.

    ``<organelle>/<model>/<train_term>/predict__<test>.yml`` ->
    ``MODELS_ROOT/<train_set>/<organelle>/<model>/checkpoints``. Used only as a fallback
    when a leaf's baked ``ckpt_path`` directory is stale/empty (a path-codemod miss).
    Returns None when the organelle/train_term don't normalize to a canonical tuple
    (e.g. the ``_no_train_*`` ablation trees). ``REPLACE_ME`` / external leaves are
    short-circuited before this fallback is consulted, so a dir computed for them is
    never used.
    """
    parts = rel.parts
    if len(parts) < 4:
        return None
    organelle_tok, model, train_tok = parts[0], parts[1], parts[2]
    try:
        return paths.checkpoint_dir(paths._norm_organelle(organelle_tok), model, paths._norm_train_set(train_tok))
    except ValueError:
        return None


def resolve_leaf_ckpt(value: str, resolver: Resolver, canonical_dir: Path | None = None) -> tuple[str | None, str]:
    """Return (new_ckpt_path_or_None, category) for one ``ckpt_path`` value.

    ``new`` is None when the value is preserved. ``category`` is one of: ``rebaked``,
    ``repointed`` (baked dir stale -> resolved from the grammar-canonical dir),
    ``already_best``, ``kept_current``, ``unresolvable``, ``external``, ``replace_me``.
    """
    if value.startswith(_REPLACE_ME):
        return None, "replace_me"
    if not value.startswith(_MODELS_ROOT_PREFIX):
        return None, "external"
    current_dir = _ckpt_dir_for(value)
    best = resolver(current_dir)
    if best is None and canonical_dir is not None and canonical_dir != current_dir:
        # baked dir is stale/empty (a path-codemod miss); resolve from the canonical home
        best = resolver(canonical_dir)
        if best is not None:
            return str(best), "repointed"
    if best is None:
        # No last*.ckpt / epoch=*.ckpt to scan. Keep a curated alias that exists;
        # otherwise the value is genuinely unresolvable (report, do not blank).
        return None, "kept_current" if Path(value).is_file() else "unresolvable"
    new = str(best)
    if new == value:
        return None, "already_best"
    return new, "rebaked"


def bake_leaf(
    text: str, resolver: Resolver, canonical_dir: Path | None = None
) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (new_text, changes) for one predict leaf. changes: (old, new, category)."""
    out: list[str] = []
    changes: list[tuple[str, str, str]] = []
    for line in text.splitlines(keepends=True):
        m = _CKPT_LINE.match(line)
        if not m:
            out.append(line)
            continue
        val = m.group("val")
        new, category = resolve_leaf_ckpt(val, resolver, canonical_dir)
        if new is not None:
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{m.group('indent')}ckpt_path: {new}{nl}")
            changes.append((val, new, category))
        else:
            out.append(line)
            changes.append((val, val, category))
    return "".join(out), changes


def iter_predict_leaves(config_root: Path) -> list[Path]:
    """Every hand-authored predict leaf under the organelle config tree."""
    return sorted(config_root.rglob("predict__*.yml"))


def main(argv: list[str] | None = None) -> int:
    """CLI: dry-run (report) or --apply the best-ckpt baking over predict leaves."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / CONFIG_ROOT_REL,
        help="root of the benchmark config tree (default: repo config root)",
    )
    ap.add_argument("--apply", action="store_true", help="write changes in place (default: dry-run report)")
    args = ap.parse_args(argv)

    # Many leaves (per-condition, per-test) share one checkpoint dir; memoize so each
    # dir's last*.ckpt is torch.load-ed once, not once per leaf.
    resolver = functools.lru_cache(maxsize=None)(resolve_best_ckpt)

    tally: dict[str, int] = {}
    n_files_changed = 0
    for leaf in iter_predict_leaves(args.config_root):
        rel = leaf.relative_to(args.config_root)
        text = leaf.read_text()
        new_text, changes = bake_leaf(text, resolver, canonical_ckpt_dir_for_leaf(rel))
        for old, new, category in changes:
            tally[category] = tally.get(category, 0) + 1
            if category in ("rebaked", "repointed"):
                print(f"{category.upper():11s} {rel}\n    {old}\n -> {new}")
            elif category in ("unresolvable", "replace_me"):
                print(f"{category.upper():11s} {rel}\n    {old}")
        if new_text != text:
            n_files_changed += 1
            if args.apply:
                leaf.write_text(new_text)

    print(f"\n{'applied' if args.apply else 'would change'}: {n_files_changed} predict leaf file(s)")
    print("categories: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
