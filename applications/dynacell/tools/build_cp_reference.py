r"""Build a target's CP (GLCM+) reference from existing GT CP feature caches.

The reference fixes the CP feature space every eval of a target is scored in (see
:mod:`dynacell.evaluation.cp_reference`):

* one feature keep-mask selected on GT cells only, pooled over the target's
  ``MASK_FIT_DATASETS`` (iPSC + A549 mock/denv/zikv);
* one GT scaler per test set, for EVERY dataset a grouped leaf of the target
  evaluates on (``benchmark.dataset_ref.dataset`` under ``--leaves-root``),
  HEK included (its own scaler, outside the mask fit);
* lite datasets mapped to their parent's scaler, the parent read from the lite
  split's ``selection_criteria.lite_of``. No scaler is fit on lite cells, but their
  positions, count and GT-matrix sha256 are recorded from the lite GT caches so the
  eval's content gate covers lite evals too.

This tool only READS. Each dataset ref is resolved exactly as an eval resolves it
(``apply_dataset_ref`` on the packaged ``eval.yaml`` defaults), the GT cache is
opened through the pipeline's own ``init_cache_context`` with
``require_complete_cache=true`` (so a cache of another store or CP recipe is
refused), and every ``(position, timepoint)`` of the GT store must already be in
the CP cache. A miss raises; features are never computed. Non-finite GT cells are
dropped and the rest stacked in the canonical order the eval hashes
(``cp_reference.read_gt_cp_cells`` / ``canonical_gt_matrix``).

The write is atomic, and an existing reference with a different content hash is
only replaced under ``--force`` (an identical one is left untouched).

``--verify`` re-reads every dataset's GT CP cells from the current caches and
compares their sha256 and count with the ones recorded in the reference's ``fit``
section; it exits 1 on any mismatch. It is the offline form of the eval's content
gate, which refuses to score a dataset whose staged GT cells no longer hash to the
recorded value.

Run::

    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py --target er --dry-run
    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py --target er   # registry path
    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py --target er --out /tmp/er.json
    uv run --no-sync python applications/dynacell/tools/build_cp_reference.py --target er --verify
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

import dynacell.evaluation
from dynacell.data.manifests import DatasetRef, load_splits
from dynacell.data.manifests import load_manifest as load_dataset_manifest
from dynacell.data.resolver import resolve_dataset_ref
from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.cp_reference import (
    MASK_FIT_DATASETS,
    DatasetFit,
    cp_space,
    fit_cp_reference,
    gt_matrix_sha256,
    load_cp_reference,
    read_gt_cp_cells,
    write_cp_reference,
)
from dynacell.evaluation.paths import cp_reference_path
from dynacell.evaluation.pipeline_cache import check_cp_cache_feature_names, init_cache_context

_EVAL_YAML = Path(dynacell.evaluation.__file__).parent / "_configs" / "eval.yaml"
_LEAVES_ROOT = Path(__file__).resolve().parents[1] / "configs/benchmarks/virtual_staining/_internal/leaf/grouped"

Ref = tuple[str, str]


def eval_config_for(target_name: str, ref: Ref) -> DictConfig:
    """Return the packaged eval defaults resolved for one dataset ref, cache reads mandatory."""
    config = OmegaConf.load(_EVAL_YAML)
    config.target_name = target_name
    config.benchmark = {"dataset_ref": {"dataset": ref[0], "target": ref[1]}}
    config.io.require_complete_cache = True
    apply_dataset_ref(config)
    return config


def leaf_dataset_refs(target_name: str, leaves_root: Path) -> set[Ref]:
    """Return every ``(dataset, manifest target)`` a grouped leaf of ``target_name`` evaluates on.

    Raises
    ------
    ValueError
        If no grouped leaf of the target is found.
    """
    refs: set[Ref] = set()
    for leaf in sorted(leaves_root.glob("*/eval_grouped.yaml")):
        doc = yaml.safe_load(leaf.read_text())
        if doc.get("target_name") != target_name:
            continue
        for cond in doc["conditions"]:
            ref = cond["benchmark"]["dataset_ref"]
            refs.add((ref["dataset"], ref["target"]))
    if not refs:
        raise ValueError(f"no grouped leaf under {leaves_root} evaluates target {target_name!r}")
    return refs


def lite_parent(ref: Ref) -> str | None:
    """Return the parent dataset a lite dataset was subset from, or ``None`` for a non-lite one.

    Read from the recorded provenance -- the dataset split's
    ``selection_criteria.lite_of`` written by ``generate_lite_benchmark_configs.py``
    -- never inferred from the name. Only lite splits carry that key.
    """
    resolved = resolve_dataset_ref(DatasetRef(dataset=ref[0], target=ref[1]))
    manifest = load_dataset_manifest(resolved.manifest_path)
    splits = load_splits(resolved.manifest_path.parent / manifest.targets[ref[1]].splits)
    return (splits.selection_criteria or {}).get("lite_of")


def read_dataset_fit(
    target_name: str, ref: Ref, feature_names: tuple[str, ...], in_mask_fit: bool, parent: str | None = None
) -> DatasetFit:
    """Read every finite GT CP cell of one dataset from its cache, read-only, in canonical order.

    The cells come from :func:`~dynacell.evaluation.cp_reference.read_gt_cp_cells`,
    the ordering the eval's content gate hashes in. A lite dataset (``parent`` set)
    is read the same way from its own GT cache, so its cells are recorded too.

    Raises
    ------
    StaleCacheError
        If the cache belongs to another store or CP recipe, its columns are not
        ``feature_names`` by name and order, a ``(position, timepoint)`` of the GT
        store is missing from it, or an entry has the wrong column count.
    ValueError
        If a GT position is not a 3-D volume.
    """
    config = eval_config_for(target_name, ref)
    ctx = init_cache_context(config, side="gt")
    check_cp_cache_feature_names(ctx, feature_names)
    cells, read = read_gt_cp_cells(ctx, Path(config.io.gt_path), len(feature_names))
    entry = ctx.manifest["artifacts"].get("cp_features")
    record = {
        "target": ref[1],
        "gt_path": str(config.io.gt_path),
        "gt_cache_dir": str(ctx.paths.root),
        "cp_cache_path": str(ctx.paths.cp_features()),
        # Audit only: the eval gates on the cells' sha256, never on this stamp.
        "cp_cache_built_at": entry["built_at"] if entry is not None else None,
        **read,
    }
    return DatasetFit(dataset=ref[0], cells=cells, record=record, in_mask_fit=in_mask_fit, parent=parent)


def build(target_name: str, leaves_root: Path) -> dict[str, Any]:
    """Read every dataset of the target and return the fitted reference payload.

    Raises
    ------
    ValueError
        If the datasets resolve different CP recipes, a lite parent has no scaler,
        or a lite store holds a position its parent's does not.
    """
    mask_refs = set(MASK_FIT_DATASETS[target_name])
    leaf_refs = leaf_dataset_refs(target_name, leaves_root)
    parents = {ref: lite_parent(ref) for ref in sorted(leaf_refs | mask_refs)}
    scaler_refs = sorted(ref for ref, parent in parents.items() if parent is None)
    lite_refs = sorted(ref for ref, parent in parents.items() if parent is not None)
    if mask_refs - set(scaler_refs):
        raise ValueError(f"mask-fit datasets {sorted(mask_refs - set(scaler_refs))} are lite")

    identity, names = cp_space(eval_config_for(target_name, scaler_refs[0]))
    for ref in [*scaler_refs[1:], *lite_refs]:
        if cp_space(eval_config_for(target_name, ref)) != (identity, names):
            raise ValueError(f"{ref[0]} resolves a different CP recipe than {scaler_refs[0][0]}")

    fits = [read_dataset_fit(target_name, ref, names, ref in mask_refs) for ref in scaler_refs]
    positions = {fit.dataset: set(fit.record["positions"]) for fit in fits}
    for ref in lite_refs:
        parent = parents[ref]
        if parent not in positions:
            raise ValueError(f"lite {ref[0]} reuses {parent}, which no grouped leaf or mask fit evaluates")
        lite = read_dataset_fit(target_name, ref, names, False, parent=parent)
        extra = set(lite.record["positions"]) - positions[parent]
        if extra:
            raise ValueError(f"lite {ref[0]} has positions outside its parent {parent}: {sorted(extra)[:5]}")
        fits.append(lite)
    return fit_cp_reference(fits, target_name=target_name, feature_names=names, cp_identity=identity)


def verify(target_name: str, path: Path) -> list[str]:
    """Recompute every dataset's (lite included) GT-matrix sha256 from the caches and compare.

    Parameters
    ----------
    target_name : str
        Eval target of the reference.
    path : pathlib.Path
        Reference JSON to verify.

    Returns
    -------
    list of str
        One message per dataset whose recomputed sha256 or cell count differs from
        the recorded one; empty when all match.
    """
    ref = load_cp_reference(path, target_name=target_name)
    mismatches = []
    for name, record in sorted({**ref.fit["datasets"], **ref.fit["lite"]}.items()):
        fit = read_dataset_fit(target_name, (name, record["target"]), ref.feature_names, record["in_mask_fit"])
        sha256 = gt_matrix_sha256(fit.cells)
        ok = sha256 == record["gt_matrix_sha256"] and fit.cells.shape[0] == record["n_cells"]
        print(f"  {name}: {'OK' if ok else 'MISMATCH'} ({fit.cells.shape[0]} cells, sha256 {sha256[:12]})")
        if not ok:
            mismatches.append(
                f"{name}: recorded {record['n_cells']} cells sha256 {record['gt_matrix_sha256'][:12]}, "
                f"caches now give {fit.cells.shape[0]} cells sha256 {sha256[:12]}"
            )
    return mismatches


def _summary(payload: dict[str, Any]) -> str:
    """Human-readable per-dataset summary of a payload."""
    lines = [
        f"kept {len(payload['kept_feature_names'])}/{len(payload['feature_names'])}: {payload['kept_feature_names']}"
    ]
    fit = payload["fit"]
    lines.append(f"mask fit: {fit['mask_fit']['n_cells']} cells from {fit['mask_fit']['datasets']}")
    for name, scaler in sorted(payload["scalers"].items()):
        d = fit["datasets"][name]
        lines.append(
            f"  {name}{' [mask]' if d['in_mask_fit'] else ''}: {len(d['positions'])} positions, "
            f"{d['n_timepoints']} timepoints, {d['n_cells']} cells ({d['n_cells_dropped_nonfinite']} non-finite "
            f"dropped), floored {scaler['floored_features'] or 'none'}"
        )
    for name, entry in sorted(payload["lite"].items()):
        d = fit["lite"][name]
        lines.append(
            f"  {name} -> scaler of {entry['parent']}: {len(d['positions'])} positions, {d['n_cells']} cells, "
            f"sha256 {d['gt_matrix_sha256'][:12]}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (1 when ``--verify`` finds a mismatch)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=sorted(MASK_FIT_DATASETS))
    parser.add_argument("--leaves-root", type=Path, default=_LEAVES_ROOT, help="grouped eval leaves to scan")
    parser.add_argument("--out", type=Path, default=None, help="output JSON (default: the registry path)")
    parser.add_argument("--dry-run", action="store_true", help="read and fit; write nothing")
    parser.add_argument("--force", action="store_true", help="replace an existing reference with a different hash")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute each dataset's GT-matrix sha256 from the caches; exit 1 on mismatch",
    )
    args = parser.parse_args(argv)

    out = args.out if args.out is not None else cp_reference_path(args.target)
    if args.verify:
        mismatches = verify(args.target, out)
        for line in mismatches:
            print(f"MISMATCH {line}")
        print(f"verify {out}: {'FAILED' if mismatches else 'OK'}")
        return 1 if mismatches else 0
    start = time.perf_counter()
    payload = build(args.target, args.leaves_root)
    print(_summary(payload))
    elapsed = time.perf_counter() - start
    if args.dry_run:
        print(f"dry run: would write {out} sha256={payload['sha256']} ({elapsed:.1f} s)")
        return 0
    written = write_cp_reference(payload, out, force=args.force)
    status = "wrote" if written else "unchanged (identical reference)"
    print(f"{status} {out} sha256={payload['sha256']} ({elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
