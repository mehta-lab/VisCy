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

Model scope: F-net, UNeXt2 (scratch), VSCyto3D (FCMAE-pretrained), UNet3DViT,
Pix2Pix3D (unetvit generator), CellDiff_r2. CellDiff_r2
``sliding_window``/``denoise`` variants are kept only on the iPSC test set (the
only place they exist on disk).

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
    _DIR_INFIX,
    _DYNACELL_ROOT,
    _LEAF_OUT_ROOT,
    _MANIFEST_ROOT,
    ParsedZarr,
    benchmark_dataset_ref,
    condition_name,
    walk_predictions,
)

_INSTANCE_ORGANELLES: tuple[str, ...] = ("nucleus", "membrane")
_INSTANCE_MODELS: frozenset[str] = frozenset(
    {
        "fnet3d_paper",
        "fcmae_vscyto3d_scratch",
        "fcmae_vscyto3d_pretrained",
        "unetvit3d",
        "pix2pix3d_unetvit",
        "celldiff_r2",
    }
)
# CellDiff_r2 variants kept off the iPSC test set (iterative-only elsewhere).
_IPSC_ONLY_VARIANTS: frozenset[str] = frozenset({"sliding_window", "denoise"})
_SLICE_FRACTION: dict[str, float] = {"ipsc": 0.5, "a549": 0.3}
_BACKEND: dict[str, str] = {"nucleus": "cellpose", "membrane": "cellpose_watershed"}

_HYDRA_HEADER = "# @package _global_\n"
_OUT_ROOT = _LEAF_OUT_ROOT.parent / "instance_ap"


def in_scope(p: ParsedZarr) -> bool:
    """Whether a parsed prediction belongs in the instance-AP campaign."""
    if p.organelle not in _INSTANCE_ORGANELLES or p.model not in _INSTANCE_MODELS:
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
