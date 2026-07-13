#!/usr/bin/env python
"""Build the Phase-7 canonical-path migration manifest (READ-ONLY, no ``mv``).

Enumerates the on-disk dynacell campaign artifacts and computes their canonical
destinations with :mod:`dynacell.evaluation.paths` — the single source of truth.
The tool deliberately carries **no independent skip policy**: whatever
:func:`paths.normalize_legacy` / :func:`paths.checkpoint_dir` map is migrated;
whatever they refuse (``None`` / ``ValueError``) is reported as a skip or a gap.
The only tool-side exception is a noise-run filter for checkpoint run dirs
(``*_smoke`` / ``*_debug`` / ``*_sanity``), which would otherwise canonicalize
onto — and collide with — their real sibling run.

Three artifact families:

1. **checkpoints** — ``<root>/<term>/<org>/<run_dir>/`` model dirs (both the
   ``dynacell`` and ``cell_diff_vs_viscy`` roots) whose ``run_dir`` carries a
   ``checkpoints/`` subdir. The WHOLE ``run_dir`` moves (carrying ``resolved/`` +
   ``wandb/`` + ``checkpoints/preflip_deconv/`` …) to ``checkpoint_dir(...).parent``,
   renaming the run-dir leaf to its canonical model code key.
2. **predictions** — ``<domain>/{predictions,joint_predictions[_v2]}/*.zarr``.
   Legacy dual-home writes the same joint zarr into both ``predictions/`` and
   ``joint_predictions/`` (``cp -al`` hardlinks); both normalize to one canonical
   ``prediction.zarr``. The colliding group is deduped to a single ``move`` winner
   (verified hardlink-identical) plus ``dedup_legacy`` rows left in place.
3. **evals** — every ``<domain>/evaluations*|joint_evaluations/<leaf>`` dir.

Output: a CSV manifest (``--out``) consumed by ``run_migration.py`` and a
human-readable ``.report.txt`` beside it. Columns::

    kind, status, src, dest, reason

``status`` ∈ ``move`` | ``dedup_legacy`` | ``skip``. The apply tool consumes only
``move`` rows. Exits non-zero when any artifact is UNMAPPED (a real hole in
``normalize_legacy``) or a canonical dest collides (two distinct sources) —
never guesses, never silently drops.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import astuple, dataclass
from pathlib import Path

from dynacell.evaluation import paths

MODELS_ROOT: Path = paths.MODELS_ROOT
DATA_ROOT: Path = paths.DATA_ROOT
# Checkpoints live under two source roots; the canonical tree consolidates both
# under the models root (dynacell), so cell_diff_vs_viscy entries are cross-root
# moves (still a same-filesystem rename — both under models/). The sibling root is
# derived from ``models_root`` inside collect_checkpoints so a test can point it
# at a fixture.
# Campaign train terms (the raw on-disk source dir names). iPSC-trained-only
# checkpoints (their own top-level term) are out of this campaign's scope.
MODELS_TERMS: tuple[str, ...] = ("a549_mantis", "joint_ipsc_confocal_a549_mantis")
PRED_SUBDIRS: tuple[str, ...] = ("predictions", "joint_predictions", "joint_predictions_v2")
# Run-dir markers that identify smoke/debug runs (excluded — they canonicalize
# onto their real sibling's dest and would collide).
NOISE_RUN_MARKERS: tuple[str, ...] = ("smoke", "debug", "sanity")

# Non-artifact eval leaf names / suffixes to skip (not eval outputs).
_EVAL_SKIP_EXACT: frozenset[str] = frozenset({"slurm"})
_EVAL_SKIP_SUFFIX: tuple[str, ...] = ("_CPU", "_local_gpu")

# Eval parents whose leaves are in the coherent canonical scope: the focus-2D
# ``*_with_embeddings`` triad (default track) + the instance-AP subtrack. Other
# normalize_legacy-mapped parents (ablation ``evaluations_{randinit,cytoland,
# infectionft}`` and the FT ``*_cytolandft*`` / ``*_infectionft_dynacellft*``
# families) are OWN-TRACK: paths.py maps their eval dirs but SKIPS their prediction
# zarrs (the ``_cytoland`` / ``_infectionft`` prediction skip), so their configs
# cannot be fully canonicalized. Excluded by default; ``--include-own-track-evals``
# folds them anyway.
_CANONICAL_EVAL_PARENTS: frozenset[str] = frozenset(
    {
        "evaluations_with_embeddings",
        "evaluations_a549trained_with_embeddings",
        "evaluations_jointtrained_with_embeddings",
        "evaluations_instance_ap",
    }
)


CSV_HEADER: tuple[str, ...] = ("kind", "status", "src", "dest", "reason")


@dataclass(frozen=True)
class Row:
    """One manifest row: an artifact and its resolved migration disposition."""

    kind: str  # checkpoint | prediction | eval
    status: str  # move | dedup_legacy | skip
    src: str
    dest: str  # "" for skip
    reason: str


def _is_noise_run(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in NOISE_RUN_MARKERS)


def _pred_skip_reason(name: str) -> str | None:
    """Deliberate normalize_legacy prediction skips (own track / stale), not gaps."""
    if name in paths._SKIP_ZARR_FILENAMES:
        return "alias-dup (stale short-name)"
    if any(tok in name for tok in ("_randinit", "_cytoland", "_infectionft")):
        return "ablation-track (own eval track)"
    return None


# ---------------------------------------------------------------------------
# Hardlink-equivalence check for deduping prediction dual-homes
# ---------------------------------------------------------------------------


def _stat_key(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino)


def _group_hardlinked(srcs: list[str], full: bool, sentinel: str) -> tuple[bool, str]:
    """Return (equivalent, detail) for a group of dirs mapping to one dest.

    Fast mode (default): identical ``sentinel`` inode ``(dev, ino)`` across the
    group — two stats per member, no tree walk. A shared sentinel inode
    (``zarr.json`` for predictions, ``checkpoints/last.ckpt`` for model dirs) is a
    strong ``cp -al`` signature. Full mode (``--full-hardlink-check``): the entire
    ``(relpath -> (dev, ino))`` map is identical (a true hardlink tree). A group
    that fails is NOT safe to dedup by selection — surfaced as a collision, never
    silently dropped.
    """
    for s in srcs:
        if not (Path(s) / sentinel).is_file():
            return False, f"no {sentinel} in {s}"
    key0 = _stat_key(Path(srcs[0]) / sentinel)
    for s in srcs[1:]:
        if _stat_key(Path(s) / sentinel) != key0:
            return False, f"{sentinel} inode differs ({s})"
    if not full:
        return True, f"{sentinel} inode match (fast)"

    def relino(root_str: str) -> dict[str, tuple[int, int]]:
        root = Path(root_str)
        return {str(f.relative_to(root)): _stat_key(f) for f in root.rglob("*") if f.is_file()}

    base = relino(srcs[0])
    for s in srcs[1:]:
        if relino(s) != base:
            return False, f"tree inode mismatch ({s})"
    return True, f"{len(base)} files fully hardlinked"


def _dedup_group(
    kind: str,
    dest: str,
    srcs: list[str],
    full: bool,
    sentinel: str,
    prefer_root: Path | None = None,
) -> tuple[list[Row], str | None]:
    """Resolve a group of sources mapping to one canonical dest.

    A single source is a plain ``move``. Multiple sources must be hardlink-identical
    (a legit ``cp -al`` dual-home) — one becomes the ``move`` winner, the rest
    ``dedup_legacy`` (left in place). ``prefer_root`` wins the tie (keep the copy
    already under the canonical root; leave the other tree, e.g. a cross-user root,
    untouched). Returns (rows, gap): a non-hardlinked group yields a gap, no rows.
    """
    if len(srcs) == 1:
        return [Row(kind, "move", srcs[0], dest, "")], None

    def sort_key(s: str) -> tuple[bool, str]:
        under_pref = prefer_root is not None and Path(s).is_relative_to(prefer_root)
        return (not under_pref, s)  # prefer_root first, then lexicographic

    ordered = sorted(srcs, key=sort_key)
    equivalent, detail = _group_hardlinked(ordered, full=full, sentinel=sentinel)
    if not equivalent:
        return [], f"COLLISION (non-hardlinked, unsafe to dedup) {dest}: {detail}; sources={ordered}"
    winner = ordered[0]
    rows = [Row(kind, "move", winner, dest, f"dedup winner of {len(ordered)} ({detail})")]
    rows += [Row(kind, "dedup_legacy", o, dest, f"hardlink dup of {winner}") for o in ordered[1:]]
    return rows, None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def collect_checkpoints(models_root: Path, full_hardlink_check: bool) -> tuple[list[Row], list[str]]:
    """Whole model-dir moves for every checkpointed run in scope. Returns (rows, gaps).

    A model dir hardlink-duplicated across the two source roots (``cp -al``
    consolidation) canonicalizes to one dest; the group is deduped to a single
    ``move`` winner (preferring the copy already under the canonical dynacell root)
    plus ``dedup_legacy`` rows for the others.
    """
    rows: list[Row] = []
    gaps: list[str] = []
    dest_to_srcs: dict[str, list[str]] = defaultdict(list)
    source_roots = (models_root, models_root.parent / "cell_diff_vs_viscy")
    for src_root in source_roots:
        for term in MODELS_TERMS:
            term_dir = src_root / term
            if not term_dir.is_dir():
                continue
            for org_dir in sorted(p for p in term_dir.iterdir() if p.is_dir()):
                for model_dir in sorted(p for p in org_dir.iterdir() if p.is_dir()):
                    if not (model_dir / "checkpoints").is_dir():
                        continue
                    if _is_noise_run(model_dir.name):
                        rows.append(Row("checkpoint", "skip", str(model_dir), "", f"noise run dir ({model_dir.name})"))
                        continue
                    try:
                        model = paths.canonical_model_name(model_dir.name)
                        # Move the whole model dir -> canonical <train_set>/<org>/<model>.
                        dest = paths.checkpoint_dir(org_dir.name, model, term, models_root=models_root).parent
                    except ValueError as exc:
                        gaps.append(f"{model_dir}  -> ERR {exc}")
                        continue
                    dest_to_srcs[str(dest)].append(str(model_dir))

    for dest, srcs in sorted(dest_to_srcs.items()):
        group_rows, gap = _dedup_group(
            "checkpoint",
            dest,
            sorted(srcs),
            full=full_hardlink_check,
            sentinel="checkpoints/last.ckpt",
            prefer_root=models_root,
        )
        rows.extend(group_rows)
        if gap:
            gaps.append(gap)
    return rows, gaps


def collect_predictions(data_root: Path, full_hardlink_check: bool) -> tuple[list[Row], list[str], list[str]]:
    """Prediction-zarr moves with hardlinked dual-home dedup. Returns (rows, skips, gaps)."""
    rows: list[Row] = []
    skips: list[str] = []
    gaps: list[str] = []
    dest_to_srcs: dict[str, list[str]] = defaultdict(list)
    for domain in ("a549", "ipsc"):
        for sub in PRED_SUBDIRS:
            d = data_root / domain / sub
            if not d.is_dir():
                continue
            for z in sorted(d.glob("*.zarr")):
                key = paths.normalize_legacy(z, data_root=data_root)
                if key is None:
                    reason = _pred_skip_reason(z.name)
                    if reason:
                        skips.append(f"{z}  [{reason}]")
                    else:
                        gaps.append(f"{z}  -> UNMAPPED (normalize_legacy None)")
                    continue
                try:
                    dest = paths.prediction_store(
                        key.organelle, key.model, key.train_set, key.test_set, key.condition, data_root=data_root
                    )
                except ValueError as exc:
                    gaps.append(f"{z}  -> ERR {exc}")
                    continue
                dest_to_srcs[str(dest)].append(str(z))

    for dest, srcs in sorted(dest_to_srcs.items()):
        group_rows, gap = _dedup_group("prediction", dest, sorted(srcs), full=full_hardlink_check, sentinel="zarr.json")
        rows.extend(group_rows)
        if gap:
            gaps.append(gap)
    return rows, skips, gaps


def collect_evals(data_root: Path, include_own_track: bool) -> tuple[list[Row], list[str], list[str]]:
    """Eval-dir moves for every normalize_legacy-mapped leaf. Returns (rows, skips, gaps)."""
    rows: list[Row] = []
    skips: list[str] = []
    gaps: list[str] = []
    for domain in ("a549", "ipsc"):
        base = data_root / domain
        if not base.is_dir():
            continue
        for evroot in sorted(
            p
            for p in base.iterdir()
            if p.is_dir() and (p.name.startswith("evaluations") or p.name == "joint_evaluations")
        ):
            own_track = evroot.name not in _CANONICAL_EVAL_PARENTS
            for leafdir in sorted(p for p in evroot.iterdir() if p.is_dir()):
                if leafdir.name in _EVAL_SKIP_EXACT or leafdir.name.endswith(_EVAL_SKIP_SUFFIX):
                    skips.append(f"{leafdir}  [non-artifact dir]")
                    continue
                key = paths.normalize_legacy(leafdir, data_root=data_root)
                if key is None:
                    skips.append(f"{leafdir}  [normalize->None (stale pre-2D / dropped deconv-GT / own track)]")
                    continue
                if own_track and not include_own_track:
                    skips.append(f"{leafdir}  [own-track eval ({evroot.name}); predictions not canonicalized]")
                    continue
                try:
                    dest = paths.eval_leaf(
                        key.organelle,
                        key.model,
                        key.train_set,
                        key.test_set,
                        key.condition,
                        component=key.component,
                        track=key.track,
                        data_root=data_root,
                    )
                except ValueError as exc:
                    gaps.append(f"{leafdir}  -> ERR {exc}")
                    continue
                rows.append(Row("eval", "move", str(leafdir), str(dest), f"track={key.track}"))
    return rows, skips, gaps


# ---------------------------------------------------------------------------
# Collision guard + report
# ---------------------------------------------------------------------------


def _dest_collisions(rows: list[Row]) -> list[str]:
    """Return errors for any canonical ``move`` dest reached by >1 distinct source."""
    by_dest: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.status == "move":
            by_dest[r.dest].append(r.src)
    errs = []
    for dest, srcs in sorted(by_dest.items()):
        if len(srcs) > 1:
            errs.append(f"DEST COLLISION {dest} <- {srcs}")
    return errs


def _write_manifest(rows: list[Row], out: Path) -> None:
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(astuple(r))


def _summarize(rows: list[Row], kind: str) -> str:
    move = sum(1 for r in rows if r.kind == kind and r.status == "move")
    dedup = sum(1 for r in rows if r.kind == kind and r.status == "dedup_legacy")
    skip = sum(1 for r in rows if r.kind == kind and r.status == "skip")
    extra = f", {dedup} dedup_legacy" if dedup else ""
    extra += f", {skip} skip" if skip else ""
    return f"{move} move{extra}"


def _write_report(
    report: Path,
    rows: list[Row],
    skips: list[str],
    gaps: list[str],
    collisions: list[str],
) -> None:
    lines: list[str] = []
    lines.append("Phase-7 canonical-path migration manifest — dry-run report")
    lines.append("=" * 64)
    for kind in ("checkpoint", "prediction", "eval"):
        lines.append(f"{kind:12s}: {_summarize(rows, kind)}")
    n_move = sum(1 for r in rows if r.status == "move")
    lines.append(f"{'TOTAL move':12s}: {n_move}")
    lines.append("")
    lines.append(f"deliberate skips (not migrated, left in place): {len(skips)}")
    lines.append(f"GAPS (UNMAPPED — must be 0): {len(gaps)}")
    lines.append(f"DEST COLLISIONS (must be 0): {len(collisions)}")
    lines.append("")
    if gaps:
        lines.append("---- GAPS ----")
        lines.extend(f"  {g}" for g in gaps)
        lines.append("")
    if collisions:
        lines.append("---- DEST COLLISIONS ----")
        lines.extend(f"  {c}" for c in collisions)
        lines.append("")
    lines.append("---- checkpoint moves ----")
    for r in rows:
        if r.kind == "checkpoint" and r.status == "move":
            tag = f"  [{r.reason}]" if r.reason else ""
            lines.append(f"  {r.src}\n    -> {r.dest}{tag}")
    lines.append("")
    lines.append("---- deliberate skips ----")
    lines.extend(f"  {s}" for s in skips)
    report.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI: enumerate artifacts, write the manifest CSV + report, exit 1 on gaps/collisions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="manifest CSV output path")
    ap.add_argument(
        "--full-hardlink-check",
        action="store_true",
        help="verify prediction dedup groups by the full per-file (relpath -> inode) "
        "tree comparison (slow; walks every zarr). Default: zarr.json inode match only "
        "(use --full-hardlink-check for the pre-apply run).",
    )
    ap.add_argument(
        "--include-own-track-evals",
        action="store_true",
        help="also migrate own-track eval dirs (ablation randinit/cytoland/infectionft + "
        "FT families) whose prediction zarrs are NOT canonicalized by paths.py. Default: "
        "migrate only the coherent canonical set (*_with_embeddings triad + instance_ap).",
    )
    ap.add_argument(
        "--allow-gaps",
        action="store_true",
        help="write the manifest and exit 0 even if UNMAPPED artifacts / collisions exist "
        "(for inspection; default is to exit non-zero)",
    )
    args = ap.parse_args(argv)

    ck_rows, ck_gaps = collect_checkpoints(MODELS_ROOT, full_hardlink_check=args.full_hardlink_check)
    pr_rows, pr_skips, pr_gaps = collect_predictions(DATA_ROOT, full_hardlink_check=args.full_hardlink_check)
    ev_rows, ev_skips, ev_gaps = collect_evals(DATA_ROOT, include_own_track=args.include_own_track_evals)

    rows = ck_rows + pr_rows + ev_rows
    gaps = ck_gaps + pr_gaps + ev_gaps
    collisions = _dest_collisions(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, args.out)
    report = args.out.with_suffix(".report.txt")
    _write_report(report, rows, pr_skips + ev_skips, gaps, collisions)

    for kind in ("checkpoint", "prediction", "eval"):
        print(f"{kind:12s}: {_summarize(rows, kind)}")
    print(f"{'skip':12s}: {len(pr_skips) + len(ev_skips) + sum(1 for r in rows if r.status == 'skip')}")
    print(f"{'GAPS':12s}: {len(gaps)}")
    print(f"{'COLLISIONS':12s}: {len(collisions)}")
    print(f"manifest -> {args.out}")
    print(f"report   -> {report}")
    for g in gaps:
        print(f"  GAP {g}")
    for c in collisions:
        print(f"  {c}")

    if (gaps or collisions) and not args.allow_gaps:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
