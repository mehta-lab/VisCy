#!/usr/bin/env python3
"""Generate the Spotlight-v2 grouped eval buckets from a roster.

Spotlight v2 scores every iPSC-trained baseline and Spotlight arm under ONE eval
pin, into a root that the paper never reads. Isolation is by root, as for
DynaCell-lite: ``save.save_dir`` and ``io.pred_cache_dir`` use the unchanged
canonical grammar (:func:`paths.eval_leaf`, :func:`paths.pred_cache_dir`) with
``data_root=V2_ROOT``::

    V2_ROOT/<organelle>/<model>/ipsc/<test>[__<cond>]/            save_dir
    V2_ROOT/<test>/eval_cache_pred/<organelle>/<model>/ipsc/<leaf>/  pred cache

The GT feature caches are SHARED with the canonical buckets (``io.gt_cache_dir``
comes from the dataset manifest, untouched here). They depend only on the GT and
are keyed by ``(gene_cond, focus_slab.halfwidth)``; these leaves inherit the
``eval.yaml`` halfwidth exactly like every canonical nucleus/membrane bucket, so
they read the same key rather than flipping it.

Every pred-side artifact is force-recomputed (masks, instances, CP and all deep
features) plus ``final_metrics``: the eval caches carry no prediction provenance,
and later waves add stores (post-hoc clipped predictions) whose pixels could
otherwise be served from a stale cache. ``gt_*`` stays cached.

Buckets are keyed by ``(wave, organelle, test leaf)`` and named
``spotlight_v2_<organelle>_<leaf>__<wave>``. A wave is a set of models scored
together; add new arms (``_segaux``, seed replicates, a retrained baseline,
clipped stores) as a NEW wave so re-running it never re-scores, and overwrites,
the conditions of an earlier wave. A ``(organelle, model, leaf)`` may appear in
only one wave.

Model tokens must be ``paths.PAPER_KEY`` keys (condition names use the paper
key). By default each model's store is the canonical
``paths.prediction_store(organelle, model, "ipsc", ...)``; a roster entry may give
``pred_paths: {<leaf>: <path>}`` for stores outside that grammar.

Every store is gated before anything is written: its position set must equal
the GT test store's, each position's T/Z/Y/X must match the GT, and every array
must hold ``prod(ceil(shape / chunks))`` chunk files. ``pred_path.is_dir()`` (the
grouped generator's only check) passes on a half-written store.

Usage::

    uv run --no-sync python applications/dynacell/tools/generate_spotlight_v2_eval_configs.py --dry-run
    uv run --no-sync python applications/dynacell/tools/generate_spotlight_v2_eval_configs.py
    uv run --no-sync python applications/dynacell/tools/generate_spotlight_v2_eval_configs.py --wave wave1a
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from generate_grouped_eval_configs import (
    _HYDRA_HEADER,
    _LEAF_OUT_ROOT,
    _MANIFEST_ROOT,
    ParsedZarr,
    benchmark_dataset_ref,
    build_leaf_yaml,
    condition_name,
)

from dynacell.evaluation import paths

V2_ROOT = paths.DATA_ROOT / "_spotlight_v2"
DEFAULT_ROSTER = Path(__file__).resolve().parent / "spotlight_v2_eval_roster.yaml"
BUCKET_PREFIX = "spotlight_v2_"
TRAIN_SET = "ipsc"
_TRAIN_BUCKET = "ipsc_trained"
ORGANELLES: frozenset[str] = frozenset({"nucleus", "membrane"})
LEAVES: tuple[str, ...] = ("ipsc", "a549__mock", "a549__denv", "a549__zikv")
_GATE_WORKERS = 16

# Every pred-side artifact the eval caches (see _resolve_force in
# evaluation/pipeline_cache.py). NOT ``all``: that also forces gt_*.
PRED_FORCE: dict[str, bool] = {
    "final_metrics": True,
    "pred_masks": True,
    "pred_cp": True,
    "pred_dinov3": True,
    "pred_dynaclr": True,
    "pred_celldino": True,
    "pred_morphem": True,
    "pred_instances": True,
}


def split_leaf(leaf: str) -> tuple[str, str | None]:
    """Return ``(test_set, condition)`` for a ``<test>[__<cond>]`` leaf token."""
    if leaf not in LEAVES:
        raise ValueError(f"unknown test leaf {leaf!r}; expected one of {LEAVES}")
    test_set, _, condition = leaf.partition("__")
    return test_set, condition or None


def bucket_name(wave: str, organelle: str, leaf: str) -> str:
    """Return the bucket directory name for ``(wave, organelle, leaf)``."""
    return f"{BUCKET_PREFIX}{organelle}_{leaf.replace('__', '_')}__{wave}"


def load_roster(path: Path) -> list[tuple[str, str, str, str, Path]]:
    """Expand a roster YAML into ``(wave, organelle, model, leaf, pred_path)`` rows.

    Raises
    ------
    ValueError
        On an unknown organelle/leaf/model or a duplicate ``(organelle, model, leaf)``.
    """
    with path.open() as f:
        roster = yaml.safe_load(f)
    default_leaves = roster["leaves"]
    rows: list[tuple[str, str, str, str, Path]] = []
    seen: dict[tuple[str, str, str], str] = {}
    for wave, by_organelle in roster["waves"].items():
        for organelle, entries in by_organelle.items():
            if organelle not in ORGANELLES:
                raise ValueError(f"wave {wave!r}: organelle {organelle!r} not in {sorted(ORGANELLES)}")
            for entry in entries:
                if isinstance(entry, str):
                    entry = {"model": entry}
                model = entry["model"]
                if model not in paths.PAPER_KEY:
                    raise ValueError(f"wave {wave!r}: model {model!r} is not a paths.PAPER_KEY key")
                overrides = entry.get("pred_paths", {})
                for leaf in entry.get("leaves", default_leaves):
                    test_set, condition = split_leaf(leaf)
                    key = (organelle, model, leaf)
                    if key in seen:
                        raise ValueError(f"{key} is in wave {seen[key]!r} and wave {wave!r}")
                    seen[key] = wave
                    pred = overrides.get(leaf) or paths.prediction_store(
                        organelle, model, TRAIN_SET, test_set, condition
                    )
                    rows.append((wave, organelle, model, leaf, Path(pred)))
    return rows


def _array_layout(array_dir: Path) -> tuple[list[int], list[int], Path] | None:
    """Return ``(shape, chunks, chunk_root)`` for a zarr v2/v3 array dir, else None."""
    v3 = array_dir / "zarr.json"
    if v3.is_file():
        meta = json.loads(v3.read_text())
        if meta.get("node_type") != "array":
            return None
        encoding = meta.get("chunk_key_encoding", {"name": "default"})["name"]
        chunk_root = array_dir / "c" if encoding == "default" else array_dir
        return meta["shape"], meta["chunk_grid"]["configuration"]["chunk_shape"], chunk_root
    v2 = array_dir / ".zarray"
    if v2.is_file():
        meta = json.loads(v2.read_text())
        return meta["shape"], meta["chunks"], array_dir
    return None


def _count_chunk_files(root: Path) -> int:
    """Count files below ``root``, excluding zarr metadata."""
    n = 0
    stack = [root]
    while stack:
        current = stack.pop()
        if not current.is_dir():
            continue
        with os.scandir(current) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.name not in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}:
                    n += 1
    return n


def _positions(store: Path) -> dict[str, Path]:
    """Map ``row/col/fov`` to its group dir for an HCS plate store."""
    return {
        f"{r.name}/{c.name}/{fov.name}": fov
        for r in sorted(p for p in store.iterdir() if p.is_dir())
        for c in sorted(p for p in r.iterdir() if p.is_dir())
        for fov in sorted(p for p in c.iterdir() if p.is_dir())
    }


def store_problems(pred_path: Path, gt_store: Path) -> list[str]:
    """Return why ``pred_path`` is not a complete prediction of ``gt_store`` (empty = complete).

    Checks the position set against the GT test store, each position's full-res
    ``(T, Z, Y, X)`` against the GT's (channels differ by design), and every
    array's chunk-file count against ``prod(ceil(shape / chunks))``.
    """
    if not pred_path.is_dir():
        return [f"missing store {pred_path}"]
    pred_pos = _positions(pred_path)
    gt_pos = _positions(gt_store)
    problems: list[str] = []
    if set(pred_pos) != set(gt_pos):
        missing = sorted(set(gt_pos) - set(pred_pos))
        extra = sorted(set(pred_pos) - set(gt_pos))
        problems.append(f"position set differs from {gt_store}: missing {missing[:3]} extra {extra[:3]}")
    for name, group in pred_pos.items():
        arrays = [(a.name, layout) for a in sorted(group.iterdir()) if (layout := _array_layout(a)) is not None]
        if not arrays:
            problems.append(f"{name}: no arrays")
            continue
        for array_name, (shape, chunks, chunk_root) in arrays:
            expected = math.prod(math.ceil(s / c) for s, c in zip(shape, chunks))
            found = _count_chunk_files(chunk_root)
            if found != expected:
                problems.append(f"{name}/{array_name}: {found}/{expected} chunk files")
        gt_layout = _array_layout(gt_pos[name] / "0") if name in gt_pos else None
        full_res = dict(arrays).get("0")
        if gt_layout is not None and full_res is not None:
            pred_tzyx = (full_res[0][0], *full_res[0][2:])
            gt_tzyx = (gt_layout[0][0], *gt_layout[0][2:])
            if pred_tzyx != gt_tzyx:
                problems.append(f"{name}: TZYX {pred_tzyx} != GT {gt_tzyx}")
    return problems


def gt_test_store(parsed: ParsedZarr) -> Path:
    """Return the GT test store the eval compares ``parsed`` against (from its manifest)."""
    ref = benchmark_dataset_ref(parsed)
    with (_MANIFEST_ROOT / ref["dataset"] / "manifest.yaml").open() as f:
        manifest = yaml.safe_load(f)
    return Path(manifest["targets"][ref["target"]]["stores"]["test"])


def parsed_row(organelle: str, model: str, leaf: str, pred_path: Path) -> ParsedZarr:
    """Build the generator's :class:`ParsedZarr` for one roster row."""
    test_set, condition = split_leaf(leaf)
    return ParsedZarr(
        pred_path=pred_path,
        organelle=organelle,
        model=model,
        variant=None,
        train_set=_TRAIN_BUCKET,
        train_set_canonical=TRAIN_SET,
        test_set=test_set,
        condition=condition,
    )


def build_buckets(rows: list[tuple[str, str, str, str, Path]], v2_root: Path = V2_ROOT) -> dict[str, tuple[dict, int]]:
    """Return ``{bucket_name: (leaf_body, n_conditions)}`` for the roster rows."""
    grouped: dict[str, tuple[str, list[ParsedZarr]]] = {}
    for wave, organelle, model, leaf, pred_path in rows:
        name = bucket_name(wave, organelle, leaf)
        grouped.setdefault(name, (organelle, []))[1].append(parsed_row(organelle, model, leaf, pred_path))
    buckets: dict[str, tuple[dict, int]] = {}
    for name, (organelle, members) in sorted(grouped.items()):
        body = build_leaf_yaml(organelle, _TRAIN_BUCKET, members, dynacell_root=v2_root)
        # Replace, never mutate: build_leaf_yaml shallow-copies the generator's
        # module-level overlay, so this dict is shared with every other leaf.
        body["force_recompute"] = dict(PRED_FORCE)
        buckets[name] = (body, len(members))
    return buckets


def leaf_text(name: str, body: dict, n: int, roster: Path) -> str:
    """Render one bucket leaf with its provenance comment."""
    comment = (
        f"# Spotlight-v2 bucket {name} ({n} conditions). Auto-generated by\n"
        f"# tools/generate_spotlight_v2_eval_configs.py from {roster.name}; edit the roster, not this file.\n"
        f"# save_dir / pred_cache_dir live under {V2_ROOT}\n"
        "# (never the canonical dirs the paper reads); the GT caches are the shared manifest\n"
        "# ones. Every pred_* artifact is forced.\n"
    )
    return _HYDRA_HEADER + comment + yaml.safe_dump(body, default_flow_style=False, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    """Gate every roster store, then write one grouped leaf per bucket."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER, help="roster YAML (default: %(default)s)")
    ap.add_argument("--out-root", type=Path, default=_LEAF_OUT_ROOT, help="leaf root (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="gate stores and print buckets; write nothing")
    ap.add_argument(
        "--wave",
        action="append",
        help="gate and write only this wave (repeatable; default: every wave). Waves whose stores "
        "are still being predicted would otherwise fail the gate for the whole roster.",
    )
    args = ap.parse_args(argv)

    # Load the whole roster so the one-wave-per-condition check still spans every wave.
    rows = load_roster(args.roster)
    if args.wave:
        unknown = set(args.wave) - {row[0] for row in rows}
        if unknown:
            raise ValueError(f"--wave {sorted(unknown)} not in roster {args.roster}")
        rows = [row for row in rows if row[0] in args.wave]
    parsed_rows = [parsed_row(organelle, model, leaf, pred) for _, organelle, model, leaf, pred in rows]
    # Listing ~5k chunk files per store is NFS-latency bound; threads overlap it.
    with ThreadPoolExecutor(_GATE_WORKERS) as pool:
        all_problems = list(pool.map(lambda p: store_problems(p.pred_path, gt_test_store(p)), parsed_rows))
    failures: list[str] = []
    for parsed, problems in zip(parsed_rows, all_problems):
        pred_path = parsed.pred_path
        status = "OK  " if not problems else "FAIL"
        print(f"[gate] {status} {condition_name(parsed)} {pred_path}")
        failures += [f"{pred_path}: {p}" for p in problems]
    if failures:
        print("[gate] incomplete prediction stores:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    buckets = build_buckets(rows)
    for name, (body, n) in buckets.items():
        out = args.out_root / name / "eval_grouped.yaml"
        print(f"[gen] {name}: {n} conditions -> {out}")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(leaf_text(name, body, n, args.roster))
    print(f"[gen] {len(rows)} conditions in {len(buckets)} buckets{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
