"""Generate grouped instance-AP eval leaves for nucleus + whole-cell (membrane).

Companion to ``generate_grouped_eval_configs.py`` (whose parser + path helpers it
reuses). That one drives the pixel/feature/embedding campaign; this one drives the
Cellpose / watershed **instance average-precision** campaign — a distinct metric
set (``AP_0.50..0.95``, ``mAP``, ``instance_dice``) for the two organelles that
have an instance interpretation:

- **nucleus** → ``segmentation.backend=cellpose``: each side segments its own
  nucleus channel into instances.
- **membrane** → ``segmentation.backend=cellpose_watershed``: GT-nuclei-seeded
  membrane EDT watershed → cytoplasm-only whole-cell instances.

Buckets are keyed by ``(organelle, test_set)`` so the 2D slice fraction is uniform
within a grouped run (iPSC mid-slice 0.5, A549 in-focus 0.3 — the GT decides the
plane, applied identically to the prediction). Conditions fold every in-scope
``(train_set, model[, plate])``.

GT nuclei for the membrane watershed: on iPSC they live in the same ``cell.zarr``
as the GT membrane; on A549 they live in the **separate** ``H2B_<cond>.ozx`` store
(membrane GT is ``CAAX_<cond>.ozx``), positions matched 1:1 by name. The A549
membrane conditions therefore set ``io.nuclei_gt_path`` to the H2B test store.

Model scope is **opt-out**: every model the campaign can parse (i.e. registered
in the grouped generator's ``_CODE_TO_PAPER``) gets instance metrics for the two
instance organelles, minus the campaign-wide ``_SKIP_MODELS`` and a (currently
empty) ``_INSTANCE_DENYLIST``. So registering a new model once — in
``_CODE_TO_PAPER`` — is enough; instance metrics follow automatically and a
forgotten model is over-included or loudly rejected, never silently skipped.
CellDiff_r2 ``sliding_window``/``denoise`` variants are kept only on the iPSC
test set (the only place they exist on disk).

Inherits the Cellpose/watershed params + IoU sweep from ``_configs/eval.yaml``;
each leaf overrides only what differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# tools/ is not a Python package; it is placed on sys.path via the root
# pyproject ``[tool.pytest.ini_options] pythonpath`` (tests) and via sys.path[0]
# when this file is run as a script. Either way the sibling generator imports by
# short name.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from generate_grouped_eval_configs import (  # noqa: E402
    _CODE_TO_PAPER,
    _DIR_INFIX,
    _DYNACELL_ROOT,
    _IGNORE_NAMES,
    _LEAF_OUT_ROOT,
    _MANIFEST_ROOT,
    _SKIP_FILENAMES,
    _SKIP_MODELS,
    ParsedZarr,
    _is_ablation_track_zarr,
    benchmark_dataset_ref,
    condition_name,
    parse_zarr_name,
    walk_predictions,
)

_INSTANCE_ORGANELLES: tuple[str, ...] = ("nucleus", "membrane")
# Opt-out model gate (see module docstring): models that genuinely have no
# instance interpretation go here. Empty today — nucleus/membrane instance AP
# is well-defined for every current model. The campaign-wide _SKIP_MODELS
# (e.g. pre-r2 ``celldiff``) is excluded separately by _instance_eligible.
_INSTANCE_DENYLIST: frozenset[str] = frozenset()
# CellDiff_r2 variants kept off the iPSC test set (iterative-only elsewhere).
_IPSC_ONLY_VARIANTS: frozenset[str] = frozenset({"sliding_window", "denoise"})
_SLICE_FRACTION: dict[str, float] = {"ipsc": 0.5, "a549": 0.3}
_BACKEND: dict[str, str] = {"nucleus": "cellpose", "membrane": "cellpose_watershed"}

_HYDRA_HEADER = "# @package _global_\n"
_OUT_ROOT = _LEAF_OUT_ROOT.parent / "instance_ap"


def _instance_eligible(model: str) -> bool:
    """Opt-out model gate: any campaign-registered model that isn't skipped.

    ``model in _CODE_TO_PAPER`` is the campaign's model registry; ``_SKIP_MODELS``
    (e.g. pre-r2 ``celldiff``) and ``_INSTANCE_DENYLIST`` carve out exceptions.
    Deriving eligibility this way means registering a new model once — in
    ``_CODE_TO_PAPER`` — is enough; instance metrics follow automatically,
    closing the gap that silently shipped pix2pix3d without instance metrics.
    """
    return model in _CODE_TO_PAPER and model not in _SKIP_MODELS and model not in _INSTANCE_DENYLIST


def in_scope(p: ParsedZarr) -> bool:
    """Whether a parsed prediction belongs in the instance-AP campaign."""
    if p.organelle not in _INSTANCE_ORGANELLES:
        return False
    if not _instance_eligible(p.model):
        return False
    if p.variant in _IPSC_ONLY_VARIANTS and p.test_set != "ipsc":
        return False
    return True


def a549_nuclei_store(condition: str) -> str:
    """Test-store path of the A549 H2B (nuclei) manifest for ``condition``."""
    manifest = _MANIFEST_ROOT / f"a549-mantis-h2b-{condition}" / "manifest.yaml"
    data = yaml.safe_load(manifest.read_text())
    return data["targets"]["h2b"]["stores"]["test"]


def save_dir_for(p: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Canonical instance-AP save_dir (parallel to the feature-metric campaign)."""
    infix = _DIR_INFIX[p.train_set]
    name = f"eval_{p.paper_variant}{infix}_{p.organelle}"
    if p.test_set != "ipsc":
        name += f"_{p.condition}"
    return dynacell_root / p.test_set / "evaluations_instance_ap" / name


def pred_cache_dir_for(p: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Dedicated pred-side instance cache (kept apart from the mask/feature caches).

    Namespaced by organelle: the cache manifest records a single ``pred.plate_path``
    per dir, so nucleus (``nucl_*.zarr``) and membrane (``memb_*.zarr``) for the same
    (train_set, model, condition) must not share a dir — otherwise the two grouped
    jobs race over the manifest and the loser dies with ``StaleCacheError``.
    """
    cond_seg = "ipsc" if p.test_set == "ipsc" else str(p.condition)
    return (
        dynacell_root
        / p.test_set
        / "eval_cache_pred_instance_ap"
        / p.organelle
        / p.train_set
        / p.model_variant
        / cond_seg
    )


def build_leaf(organelle: str, test_set: str, conditions: list[ParsedZarr]) -> dict:
    """Return the grouped-leaf dict for one (organelle, test_set) bucket."""
    seg: dict = {"backend": _BACKEND[organelle], "slice_fraction": _SLICE_FRACTION[test_set]}
    if organelle == "membrane":
        seg["nuclei_channel_name"] = "Nuclei"
    body: dict = {
        "target_name": organelle,
        "compute_instance_ap": True,
        "compute_feature_metrics": False,
        "compute_microssim": False,
        "use_gpu": True,
        "segmentation": seg,
        "io": {"require_complete_cache": False},
        "runtime": {"executor": "serial", "fov_workers": 1, "threads_per_worker": "auto"},
        "force_recompute": {"final_metrics": True},
    }
    blocks: list[dict] = []
    for p in conditions:
        io_block: dict = {
            "pred_path": str(p.pred_path),
            "pred_cache_dir": str(pred_cache_dir_for(p)),
        }
        if organelle == "membrane" and test_set == "a549":
            io_block["nuclei_gt_path"] = a549_nuclei_store(p.condition)
        blocks.append(
            {
                "name": condition_name(p),
                "benchmark": {"dataset_ref": benchmark_dataset_ref(p)},
                "io": io_block,
                "save": {"save_dir": str(save_dir_for(p))},
            }
        )
    body["conditions"] = blocks
    return body


def bucket_predictions(dynacell_root: Path = _DYNACELL_ROOT) -> dict[tuple[str, str], list[ParsedZarr]]:
    """Group in-scope predictions by ``(organelle, test_set)``, sorted per bucket."""
    buckets: dict[tuple[str, str], list[ParsedZarr]] = {}
    for p in walk_predictions(dynacell_root):
        if not in_scope(p):
            continue
        buckets.setdefault((p.organelle, p.test_set), []).append(p)
    for conds in buckets.values():
        conds.sort(key=condition_name)
    return buckets


def audit_prediction_coverage(dynacell_root: Path = _DYNACELL_ROOT) -> list[str]:
    """Loud coverage guard: actionable errors for predictions the campaign can't fold in.

    A newly trained model whose prediction zarrs sit on disk but whose code-name
    is not registered would otherwise crash :func:`walk_predictions` on the first
    bad name (and, before the opt-out gate, be silently dropped by ``in_scope``).
    This walks the prediction dirs with a tolerant per-zarr parse — applying the
    same skip filters as :func:`walk_predictions` — and returns one message per
    unregistered/unparseable zarr, so :func:`main` can fail with the *complete*
    list pointing at the registry instead of dying on the first one.

    An empty list means every on-disk prediction is registered and will be
    bucketed by the relevant campaign. Returns messages (does not raise) so the
    caller controls exit behavior; pure-string output keeps it unit-testable.
    """
    errors: list[str] = []
    for dataset in ("ipsc", "a549"):
        for subdir in ("predictions", "joint_predictions"):
            root = dynacell_root / dataset / subdir
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                name = entry.name
                if name in _IGNORE_NAMES or name.startswith(("_", ".")):
                    continue
                if not name.endswith(".zarr") or name in _SKIP_FILENAMES:
                    continue
                if _is_ablation_track_zarr(name) or not entry.is_dir():
                    continue
                try:
                    parse_zarr_name(entry, dynacell_root=dynacell_root)
                except ValueError as exc:
                    errors.append(
                        f"unregistered/unparseable prediction {entry} -> {exc}; register its "
                        f"model in generate_grouped_eval_configs (_DETERMINISTIC_MODELS + _CODE_TO_PAPER)"
                    )
    return errors


def emit_leaf(out_path: Path, organelle: str, test_set: str, body: dict) -> None:
    """Write one grouped instance-AP leaf YAML with the Hydra header + comment."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comment = (
        f"# Grouped instance-AP leaf: {organelle} × {test_set} test "
        f"({len(body['conditions'])} conditions, slice_fraction={_SLICE_FRACTION[test_set]}). "
        f"Auto-generated by tools/generate_instance_ap_eval_configs.py.\n"
    )
    text = _HYDRA_HEADER + comment + yaml.safe_dump(body, default_flow_style=False, sort_keys=False)
    out_path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    """Generate the instance-AP grouped leaves under ``_internal/leaf/instance_ap/``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dynacell-root", type=Path, default=_DYNACELL_ROOT)
    ap.add_argument("--out-root", type=Path, default=_OUT_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print bucket summary; write nothing")
    args = ap.parse_args(argv)

    coverage_errors = audit_prediction_coverage(args.dynacell_root)
    if coverage_errors:
        print("[instance-ap] COVERAGE CHECK FAILED — unregistered model predictions on disk:", file=sys.stderr)
        for err in coverage_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    buckets = bucket_predictions(args.dynacell_root)
    total = 0
    for (organelle, test_set), conds in sorted(buckets.items()):
        total += len(conds)
        print(f"{organelle}_{test_set}: {len(conds)} conditions")
        if args.dry_run:
            continue
        body = build_leaf(organelle, test_set, conds)
        emit_leaf(args.out_root / f"{organelle}_{test_set}" / "eval_grouped.yaml", organelle, test_set, body)
    print(f"total: {total} conditions across {len(buckets)} buckets")
    if not args.dry_run:
        print(f"wrote leaves under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
