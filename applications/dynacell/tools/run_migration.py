#!/usr/bin/env python
"""Apply (or dry-run / roll back) the Phase-7 canonical-path migration.

Consumes the CSV manifest from ``build_migration_manifest.py`` and executes the
``move`` rows as whole-directory renames (``Path.rename``) so every sibling of a
moved dir travels with it (``checkpoints/`` + ``resolved/`` + ``wandb/`` for a
model dir; the whole ``prediction.zarr`` / eval-leaf tree otherwise).

Safety model
------------
* **Dry-run by default.** ``--apply`` is required to touch disk.
* **Journaled.** Every executed rename is appended to a journal CSV
  (``<manifest>.journal.csv`` by default) as ``ts,kind,src,dest,result,detail``.
  ``--rollback`` replays the journal in reverse (dest -> src).
* **Idempotent.** A row whose ``src`` is already gone and whose ``dest`` already
  exists is treated as previously-applied and skipped, so a re-run after an
  interruption resumes cleanly.
* **Pre-flight, always (even dry-run):** every ``move`` src exists; no ``dest``
  pre-exists (except the idempotent already-applied case); src and dest are on the
  same filesystem (rename is not cross-device); src/dest namespaces are disjoint
  (no dest nested under any src or vice-versa). Any violation aborts before a
  single rename — never a partial, un-journaled mutation.

``dedup_legacy`` rows are never moved (they are hardlinked duplicates of a ``move``
winner, left in place as legacy). ``skip`` rows are ignored.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import errno
import sys
from dataclasses import dataclass
from pathlib import Path

JOURNAL_HEADER: tuple[str, ...] = ("ts", "kind", "src", "dest", "result", "detail")


@dataclass(frozen=True)
class Move:
    """One planned rename: whole-dir ``src`` -> canonical ``dest``."""

    kind: str
    src: Path
    dest: Path
    reason: str


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_moves(manifest: Path) -> list[Move]:
    """Read the ``move`` rows from the manifest CSV (skips dedup_legacy / skip)."""
    moves: list[Move] = []
    with manifest.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["status"] != "move":
                continue
            moves.append(Move(row["kind"], Path(row["src"]), Path(row["dest"]), row["reason"]))
    return moves


def _existing_ancestor(path: Path) -> Path:
    """Return the nearest existing ancestor of ``path`` (for a same-device check)."""
    p = path
    while not p.exists():
        if p.parent == p:
            return p
        p = p.parent
    return p


def preflight(moves: list[Move]) -> tuple[list[Move], list[Move], list[Move], list[str]]:
    """Classify moves and collect blocking errors.

    Returns (pending, merges, already_applied, errors). ``pending`` are
    src-present / dest-absent renames to run; ``merges`` are eval leaves whose
    canonical dest dir already exists (the prediction move creates the shared
    leaf and drops ``prediction.zarr`` into it first) — the eval's children are
    merged into that leaf rather than whole-dir renamed onto it; ``already`` are
    idempotent no-ops.
    """
    pending: list[Move] = []
    merges: list[Move] = []
    already: list[Move] = []
    errors: list[str] = []

    srcs = {m.src for m in moves}
    dests = {m.dest for m in moves}

    # Namespace disjointness: no dest may be nested under any src (or vice-versa),
    # else moving the outer dir first would drag/lose the inner one.
    for d in dests:
        for s in srcs:
            if d == s or s in d.parents or d in s.parents:
                errors.append(f"NESTED src/dest: src={s} <-> dest={d}")

    # Classification has to reason about the tree as it will be when each move
    # RUNS, not as it is now. apply_moves walks `pending` in manifest order
    # (ck_rows + pr_rows + ev_rows) and calls dest.parent.mkdir(parents=True),
    # and prediction_store() returns `<leaf>/prediction.zarr` -- a child of the
    # eval move's dest. So the prediction move materializes the eval's dest
    # before the eval move is reached: a dest that looks absent here exists, and
    # is non-empty, by then. Judging on the current tree classified those evals
    # as whole-dir renames, and `src.rename(dest)` then raised ENOTEMPTY mid-run
    # with checkpoint and prediction renames already committed -- exactly the
    # partial mutation this module's docstring promises cannot happen.
    will_exist: set[Path] = set()
    for m in moves:
        will_exist.update(m.dest.parents)

    seen_dest: dict[Path, Path] = {}
    for m in moves:
        if m.dest in seen_dest and seen_dest[m.dest] != m.src:
            errors.append(f"DEST COLLISION {m.dest} <- {seen_dest[m.dest]} and {m.src}")
        seen_dest[m.dest] = m.src

        src_exists = m.src.exists()
        dest_exists = m.dest.exists() or m.dest in will_exist
        if src_exists and not dest_exists:
            # Same-filesystem (rename is not cross-device)?
            src_dev = m.src.stat().st_dev
            dest_dev = _existing_ancestor(m.dest).stat().st_dev
            if src_dev != dest_dev:
                errors.append(
                    f"CROSS-DEVICE (rename unsafe) src={m.src} (dev {src_dev}) dest={m.dest} (dev {dest_dev})"
                )
            else:
                pending.append(m)
        elif not src_exists and dest_exists:
            already.append(m)
        elif src_exists and dest_exists:
            # Canonical eval leaf == the prediction leaf: prediction.zarr is
            # already inside the dest dir. A whole-dir rename would collide, so
            # merge the eval's children into the leaf iff none of them clash.
            if m.kind == "eval" and m.src.is_dir() and (m.dest.is_dir() or m.dest in will_exist):
                # A dest that only `will_exist` has no children on disk yet, so
                # nothing can clash with it; the prediction move that creates it
                # writes prediction.zarr, which an eval src never owns.
                clashes = (
                    sorted(c.name for c in m.src.iterdir() if (m.dest / c.name).exists()) if m.dest.is_dir() else []
                )
                if clashes:
                    errors.append(f"MERGE CLASH {m.dest} already has {clashes} (src {m.src})")
                elif m.src.stat().st_dev != _existing_ancestor(m.dest).stat().st_dev:
                    errors.append(f"CROSS-DEVICE (merge unsafe) src={m.src} dest={m.dest}")
                else:
                    merges.append(m)
            else:
                errors.append(f"DEST ALREADY EXISTS (unexpected) src={m.src} dest={m.dest}")
        else:  # neither exists
            errors.append(f"SRC MISSING and DEST ABSENT (lost?) src={m.src} dest={m.dest}")
    return pending, merges, already, errors


def _append_journal(journal: Path, kind: str, src: Path, dest: Path, result: str, detail: str) -> None:
    new = not journal.exists()
    with journal.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(JOURNAL_HEADER)
        w.writerow((_now(), kind, str(src), str(dest), result, detail))


def apply_moves(pending: list[Move], merges: list[Move], journal: Path, dry_run: bool) -> int:
    """Execute (or preview) the pending renames and eval-leaf merges. Returns count applied."""
    applied = 0
    for m in pending:
        if dry_run:
            print(f"  [dry-run] {m.kind}: {m.src}\n            -> {m.dest}")
            continue
        m.dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            m.src.rename(m.dest)
        except OSError as exc:
            _append_journal(journal, m.kind, m.src, m.dest, "error", f"{type(exc).__name__}: {exc}")
            if exc.errno == errno.EXDEV:
                raise RuntimeError(f"cross-device rename despite pre-flight; src={m.src} dest={m.dest}") from exc
            raise
        _append_journal(journal, m.kind, m.src, m.dest, "applied", m.reason)
        applied += 1
        print(f"  moved {m.kind}: {m.src.name} -> {m.dest}")

    # Eval-leaf merges: rename each child of the eval src into the shared
    # (prediction-occupied) dest leaf. Journal one row PER CHILD so --rollback
    # reverses exactly, never touching prediction.zarr (which the eval never owns).
    for m in merges:
        children = sorted(m.src.iterdir())
        if dry_run:
            print(f"  [dry-run] MERGE {m.kind}: {m.src} ({len(children)} entries)\n            -> {m.dest}/")
            continue
        m.dest.mkdir(parents=True, exist_ok=True)
        for child in children:
            target = m.dest / child.name
            if target.exists():
                _append_journal(journal, m.kind, child, target, "error", "merge target exists")
                raise FileExistsError(f"merge target exists: {target}")
            try:
                child.rename(target)
            except OSError as exc:
                _append_journal(journal, m.kind, child, target, "error", f"{type(exc).__name__}: {exc}")
                raise
            _append_journal(journal, m.kind, child, target, "applied", f"merge {m.reason}")
        m.src.rmdir()  # src is now empty; fails loudly if not
        applied += 1
        print(f"  merged {m.kind}: {m.src.name} ({len(children)} entries) -> {m.dest}/")
    return applied


def rollback(journal: Path, dry_run: bool) -> int:
    """Reverse applied journal rows (dest -> src), newest first."""
    if not journal.exists():
        raise FileNotFoundError(f"no journal at {journal}")
    with journal.open(newline="") as fh:
        applied_rows = [r for r in csv.DictReader(fh) if r["result"] == "applied"]
    reversed_count = 0
    for r in reversed(applied_rows):
        src, dest = Path(r["src"]), Path(r["dest"])
        if not dest.exists():
            print(f"  [skip] dest gone (already reversed?): {dest}")
            continue
        if src.exists():
            raise RuntimeError(f"cannot reverse: src already exists {src}")
        if dry_run:
            print(f"  [dry-run] reverse {r['kind']}: {dest}\n            -> {src}")
            reversed_count += 1
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        dest.rename(src)
        _append_journal(journal, r["kind"], dest, src, "reversed", "rollback")
        reversed_count += 1
        print(f"  reversed {r['kind']}: {dest.name} -> {src}")
    return reversed_count


def main(argv: list[str] | None = None) -> int:
    """CLI: dry-run / apply / rollback the manifest's ``move`` rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest CSV from build_migration_manifest.py")
    ap.add_argument("--journal", type=Path, default=None, help="journal CSV (default: <manifest>.journal.csv)")
    ap.add_argument("--apply", action="store_true", help="execute renames (default: dry-run preview)")
    ap.add_argument("--rollback", action="store_true", help="reverse applied journal rows (dest -> src)")
    ap.add_argument("--kind", choices=("checkpoint", "prediction", "eval"), help="restrict to one artifact family")
    ap.add_argument("--limit", type=int, default=None, help="apply at most N moves (canary)")
    args = ap.parse_args(argv)

    journal = args.journal or args.manifest.with_suffix(".journal.csv")
    dry_run = not args.apply

    if args.rollback:
        n = rollback(journal, dry_run)
        print(f"{'[dry-run] would reverse' if dry_run else 'reversed'} {n} move(s); journal {journal}")
        return 0

    moves = load_moves(args.manifest)
    if args.kind:
        moves = [m for m in moves if m.kind == args.kind]
    pending, merges, already, errors = preflight(moves)

    print(
        f"manifest moves: {len(moves)}  (pending {len(pending)}, merges {len(merges)}, "
        f"already-applied {len(already)}, errors {len(errors)})"
    )
    for e in errors:
        print(f"  ERROR {e}")
    if errors:
        print("ABORT: pre-flight errors — no renames performed.")
        return 1

    if args.limit is not None:
        pending = pending[: args.limit]
        merges = merges[: max(0, args.limit - len(pending))]
        print(f"--limit {args.limit}: applying first {len(pending)} rename(s) + {len(merges)} merge(s)")

    print(f"{'DRY-RUN (no changes)' if dry_run else 'APPLYING'} — {len(pending)} rename(s), {len(merges)} merge(s):")
    applied = apply_moves(pending, merges, journal, dry_run)
    if dry_run:
        print(
            f"[dry-run] would move {len(pending)} + merge {len(merges)}; "
            f"{len(already)} already applied. journal -> {journal}"
        )
    else:
        print(f"applied {applied}; {len(already)} already applied. journal -> {journal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
