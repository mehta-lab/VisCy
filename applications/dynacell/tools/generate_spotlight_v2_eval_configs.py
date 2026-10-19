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

ER and mito buckets are built by the same :func:`build_leaf_yaml` as the canonical
``er_*`` / ``mitochondria_*`` grouped leaves, so they score exactly like them:
the default semantic (Dice) masks, no ``segmentation`` override and no instance
AP, GT from the per-organelle iPSC stores (``SEC61B.zarr`` / ``TOMM20.zarr``) and
the A549 ``sec61b`` / ``tomm20`` test sets. Roster keys are the on-disk
organelle tokens (``er``, ``mito``); mito's eval ``target_name`` is
``mitochondria``, as in the canonical leaves.

Every pred-side artifact is force-recomputed (masks, instances, CP and all deep
features) plus ``final_metrics``. The generator predates the per-position
prediction sources the eval caches now record, and forcing keeps these waves
independent of that check: later waves add stores (post-hoc clipped predictions)
whose pixels must never be served from an earlier cache. ``gt_*`` stays cached.

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

``--segmentation-backend {cellpose,supermodel}`` (nucleus waves only) writes a second set of
buckets, ``<bucket>_<backend>``, that re-score the same stores with another segmenter: Cellpose-SAM
instances (``cellpose``, with instance AP) or the AICS SuperModel semantic mask (``supermodel``).
They exist for the P1 hacking check, which scores the training segmenter (cpsam) and a held-out
one (SuperModel) next to cpdino. Only masks are recomputed (no feature metrics), and outputs go to
their own root, ``<DATA_ROOT>/_spotlight_v2_<backend>``, so they never overwrite the cpdino rows.
Their GT masks are cached in the shared manifest GT cache under the backend's own plate name.

A 2-D model predicted on the 2-D benchmark's focus slab only (``predict_z_planes``: the
store holds the GT-anchored focus plane +-h per timepoint and records them as
``viscy_predicted_z_planes``) cannot be scored over the full volume. The roster's top-level
``focus2d_waves: {<wave>: <halfwidth>}`` scores such a wave like the 2-D benchmark
(``focus2d.halfwidth``: every metric on the focus plane / slab), into
``<DATA_ROOT>/_spotlight_v2_focus2d_h<halfwidth>`` so a full-volume row of the same model is
never overwritten. A ``(organelle, model, leaf)`` may then appear once per protocol. The
gate accepts a plane-restricted store only in a ``focus2d`` wave, and requires exactly the
chunks of its recorded planes there.

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
    _CANONICAL_ORG_TO_INTERNAL,
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
# Re-scoring backends (``--segmentation-backend``): name -> whether it produces instances (AP).
ALT_BACKENDS: dict[str, bool] = {"cellpose": True, "supermodel": False}
DEFAULT_ROSTER = Path(__file__).resolve().parent / "spotlight_v2_eval_roster.yaml"
BUCKET_PREFIX = "spotlight_v2_"
TRAIN_SET = "ipsc"
_TRAIN_BUCKET = "ipsc_trained"
# On-disk organelle tokens (the roster keys and prediction-store dirs).
ORGANELLES: frozenset[str] = frozenset({"nucleus", "membrane", "er", "mito"})
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


def focus2d_root(halfwidth: int) -> Path:
    """Return the output root of ``focus2d`` waves scored at ``halfwidth``."""
    return paths.DATA_ROOT / f"_spotlight_v2_focus2d_h{halfwidth}"


def load_focus2d_waves(path: Path) -> dict[str, int]:
    """Return the roster's ``{wave: focus2d halfwidth}`` (empty when it scores every wave in 3-D)."""
    with path.open() as f:
        return dict(yaml.safe_load(f).get("focus2d_waves") or {})


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
        On an unknown organelle/leaf/model, a ``focus2d_waves`` entry naming no wave, or a
        duplicate ``(organelle, model, leaf)`` within one protocol (3-D or a focus2d halfwidth).
    """
    with path.open() as f:
        roster = yaml.safe_load(f)
    default_leaves = roster["leaves"]
    focus2d = dict(roster.get("focus2d_waves") or {})
    if unknown := set(focus2d) - set(roster["waves"]):
        raise ValueError(f"focus2d_waves {sorted(unknown)} are not waves")
    rows: list[tuple[str, str, str, str, Path]] = []
    seen: dict[tuple, str] = {}
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
                    key = (organelle, model, leaf, focus2d.get(wave))
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


def _slab_chunk_keys(group: Path, shape: list[int], chunks: list[int]) -> set[tuple[int, ...]] | None:
    """Return the chunk keys a plane-restricted (2-D benchmark) predict writes, or None for a full one.

    A slab-only predict records ``viscy_predicted_z_planes`` (``{channel: [planes per t]}``) on the
    position and writes only those planes, so the full-array chunk count does not apply. Each
    listed channel needs every ``(t, c, z, y, x)`` chunk of its planes; other channels need all.
    """
    attrs = json.loads((group / "zarr.json").read_text()).get("attributes", {})
    planes = attrs.get("viscy_predicted_z_planes")
    if planes is None:
        return None
    labels = [c["label"] for c in attrs["ome"]["omero"]["channels"]]
    grid = [math.ceil(n / c) for n, c in zip(shape, chunks)]
    yx = [(y, x) for y in range(grid[3]) for x in range(grid[4])]
    keys: set[tuple[int, ...]] = set()
    for ci, label in enumerate(labels):
        per_t = planes.get(label)
        for t in range(shape[0]):
            zs = range(shape[2]) if per_t is None else per_t[t]
            keys |= {(t // chunks[0], ci // chunks[1], z // chunks[2], y, x) for z in zs for y, x in yx}
    return keys


def store_problems(pred_path: Path, gt_store: Path, focus2d: bool = False) -> list[str]:
    """Return why ``pred_path`` is not a complete prediction of ``gt_store`` (empty = complete).

    Checks the position set against the GT test store, each position's full-res
    ``(T, Z, Y, X)`` against the GT's (channels differ by design), and every
    array's chunk-file count against ``prod(ceil(shape / chunks))``. A plane-restricted
    (2-D benchmark) store must instead hold exactly the chunks of its recorded planes
    (:func:`_slab_chunk_keys`); score it with a ``focus2d`` wave.
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
            slab = _slab_chunk_keys(group, shape, chunks) if array_name == "0" else None
            if slab is not None and not focus2d:
                problems.append(
                    f"{name}: plane-restricted predict (viscy_predicted_z_planes); score it in a focus2d wave"
                )
                continue
            expected = len(slab) if slab is not None else math.prod(math.ceil(s / c) for s, c in zip(shape, chunks))
            found = _count_chunk_files(chunk_root)
            if found != expected:
                problems.append(f"{name}/{array_name}: {found}/{expected} chunk files")
            elif slab is not None and (absent := [k for k in slab if not chunk_root.joinpath(*map(str, k)).is_file()]):
                problems.append(f"{name}/{array_name}: {len(absent)} recorded-plane chunks missing, e.g. {absent[0]}")
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


def target_name(organelle: str) -> str:
    """Return the eval ``target_name`` for a roster organelle (``mito`` -> ``mitochondria``)."""
    return _CANONICAL_ORG_TO_INTERNAL.get(organelle, organelle)


def parsed_row(organelle: str, model: str, leaf: str, pred_path: Path) -> ParsedZarr:
    """Build the generator's :class:`ParsedZarr` for one roster row."""
    test_set, condition = split_leaf(leaf)
    return ParsedZarr(
        pred_path=pred_path,
        organelle=target_name(organelle),
        model=model,
        variant=None,
        train_set=_TRAIN_BUCKET,
        train_set_canonical=TRAIN_SET,
        test_set=test_set,
        condition=condition,
    )


def backend_root(backend: str) -> Path:
    """Return the output root of the ``backend`` re-scoring buckets."""
    if backend not in ALT_BACKENDS:
        raise ValueError(f"unknown re-scoring backend {backend!r}; expected one of {sorted(ALT_BACKENDS)}")
    return paths.DATA_ROOT / f"_spotlight_v2_{backend}"


def wave_root(wave: str, focus2d: dict[str, int], backend: str | None = None, v2_root: Path = V2_ROOT) -> Path:
    """Return where a wave's buckets save: the v2 root, a backend root, or a focus2d root."""
    if wave in focus2d:
        if backend is not None:
            raise ValueError(f"wave {wave!r} is a focus2d wave; --segmentation-backend re-scores 3-D waves only")
        return focus2d_root(focus2d[wave])
    return v2_root if backend is None else backend_root(backend)


def build_buckets(
    rows: list[tuple[str, str, str, str, Path]],
    v2_root: Path = V2_ROOT,
    backend: str | None = None,
    focus2d: dict[str, int] | None = None,
) -> dict[str, tuple[dict, int]]:
    """Return ``{bucket_name: (leaf_body, n_conditions)}`` for the roster rows.

    With ``backend``, the buckets re-score nucleus stores with that segmenter into
    :func:`backend_root` instead of ``v2_root``; a ``focus2d`` wave scores on the focus
    plane into :func:`focus2d_root` (see the module docstring).
    """
    focus2d = focus2d or {}
    if backend is not None:
        v2_root = backend_root(backend)
        if bad := {row[1] for row in rows} - {"nucleus"}:
            raise ValueError(f"--segmentation-backend re-scores nucleus only; got {sorted(bad)}")
    grouped: dict[str, tuple[str, str, list[ParsedZarr]]] = {}
    for wave, organelle, model, leaf, pred_path in rows:
        name = bucket_name(wave, organelle, leaf) + (f"_{backend}" if backend else "")
        grouped.setdefault(name, (wave, organelle, []))[2].append(parsed_row(organelle, model, leaf, pred_path))
    buckets: dict[str, tuple[dict, int]] = {}
    for name, (wave, organelle, members) in sorted(grouped.items()):
        root = wave_root(wave, focus2d, backend, v2_root)
        body = build_leaf_yaml(target_name(organelle), _TRAIN_BUCKET, members, dynacell_root=root)
        # Replace, never mutate: build_leaf_yaml shallow-copies the generator's
        # module-level overlay, so this dict is shared with every other leaf.
        body["force_recompute"] = dict(PRED_FORCE)
        if backend is not None:
            body["segmentation"] = {"backend": backend}
            body["compute_instance_ap"] = ALT_BACKENDS[backend]
            body["compute_feature_metrics"] = False
        if wave in focus2d:
            # Before ``conditions``, where the canonical 2-D benchmark leaves put it.
            conditions = body.pop("conditions")
            body["focus2d"] = {"halfwidth": focus2d[wave]}
            body["conditions"] = conditions
        buckets[name] = (body, len(members))
    return buckets


def leaf_text(name: str, body: dict, n: int, roster: Path, root: Path = V2_ROOT) -> str:
    """Render one bucket leaf with its provenance comment."""
    comment = (
        f"# Spotlight-v2 bucket {name} ({n} conditions). Auto-generated by\n"
        f"# tools/generate_spotlight_v2_eval_configs.py from {roster.name}; edit the roster, not this file.\n"
        f"# save_dir / pred_cache_dir live under {root}\n"
        "# (never the canonical dirs the paper reads); the GT caches are the shared manifest\n"
        "# ones. Every pred_* artifact is forced.\n"
    )
    if "focus2d" in body:
        comment += (
            f"# Scored like the 2-D benchmark: focus2d.halfwidth={body['focus2d']['halfwidth']}, every metric on the\n"
            "# GT-anchored focus plane / slab of each FOV and timepoint (the store may hold only that slab).\n"
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
    ap.add_argument(
        "--segmentation-backend",
        choices=sorted(ALT_BACKENDS),
        help="write <bucket>_<backend> re-scoring buckets for the --wave(s) instead (nucleus only)",
    )
    args = ap.parse_args(argv)
    if args.segmentation_backend and not args.wave:
        ap.error("--segmentation-backend needs --wave")

    # Load the whole roster so the one-wave-per-condition check still spans every wave.
    rows = load_roster(args.roster)
    if args.wave:
        unknown = set(args.wave) - {row[0] for row in rows}
        if unknown:
            raise ValueError(f"--wave {sorted(unknown)} not in roster {args.roster}")
        rows = [row for row in rows if row[0] in args.wave]
    focus2d = load_focus2d_waves(args.roster)
    parsed_rows = [parsed_row(organelle, model, leaf, pred) for _, organelle, model, leaf, pred in rows]
    in_focus2d = [row[0] in focus2d for row in rows]
    # Listing ~5k chunk files per store is NFS-latency bound; threads overlap it.
    with ThreadPoolExecutor(_GATE_WORKERS) as pool:
        all_problems = list(
            pool.map(lambda p, f: store_problems(p.pred_path, gt_test_store(p), f), parsed_rows, in_focus2d)
        )
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

    backend = args.segmentation_backend
    buckets = build_buckets(rows, backend=backend, focus2d=focus2d)
    wave_of = {bucket_name(w, o, leaf) + (f"_{backend}" if backend else ""): w for w, o, _, leaf, _ in rows}
    for name, (body, n) in buckets.items():
        out = args.out_root / name / "eval_grouped.yaml"
        print(f"[gen] {name}: {n} conditions -> {out}")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(leaf_text(name, body, n, args.roster, wave_root(wave_of[name], focus2d, backend)))
    print(f"[gen] {len(rows)} conditions in {len(buckets)} buckets{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
