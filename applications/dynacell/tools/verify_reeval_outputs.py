#!/usr/bin/env python3
"""Post-run sanity for the re-eval campaign.

Walks every grouped-leaf YAML under
``applications/dynacell/configs/benchmarks/virtual_staining/_internal/leaf/grouped/``
and verifies each condition's ``save_dir`` + ``pred_cache_dir`` against
the 5 checks defined in the campaign plan's Verification section:

1. ``feature_metrics.csv`` exists with ≥70 columns (target: 74).
2. ``embeddings/`` lists 8 ``{gt,pred}_{cp,dinov3,dynaclr,celldino}_single_cell_embeddings.npz``.
3. ``eval_timing.csv`` exists.
4. ``feature_metrics.csv`` carries new probe / fidelity columns.
5. Pred cache populated with ``features/celldino/*.zarr``.

Prints a pass/fail table and exits non-zero if any condition fails any check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Use the generator's helpers for path consistency.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from generate_grouped_eval_configs import _LEAF_OUT_ROOT  # noqa: E402

_REQUIRED_EMBEDDINGS = {
    f"{side}_{backbone}_single_cell_embeddings.npz"
    for side in ("gt", "pred")
    for backbone in ("cp", "dinov3", "dynaclr", "celldino")
}

_REQUIRED_PROBE_COLS = (
    "CellDINO_FID",
    "CellDINO_RealVsPred_AUROC",
    "DynaCLR_RealVsPred_AUROC",
)


@dataclass
class Check:
    """One pass/fail check on a condition's output dir."""

    name: str
    passed: bool
    detail: str = ""


def check_feature_csv(save_dir: Path) -> Check:
    """1. feature_metrics.csv exists with ≥70 columns."""
    csv = save_dir / "feature_metrics.csv"
    if not csv.is_file():
        return Check("csv≥70", False, "feature_metrics.csv missing")
    with csv.open() as f:
        first = f.readline()
    ncol = len(first.split(","))
    return Check("csv≥70", ncol >= 70, f"{ncol} cols")


def check_embeddings(save_dir: Path) -> Check:
    """2. embeddings/ has 8 npz files."""
    emb_dir = save_dir / "embeddings"
    if not emb_dir.is_dir():
        return Check("emb=8", False, "embeddings/ missing")
    present = {p.name for p in emb_dir.iterdir() if p.name.endswith(".npz")}
    missing = _REQUIRED_EMBEDDINGS - present
    if missing:
        return Check("emb=8", False, f"missing: {sorted(missing)}")
    return Check("emb=8", True, f"{len(present)} npz")


def check_eval_timing(save_dir: Path) -> Check:
    """3. eval_timing.csv marker."""
    return Check("timing", (save_dir / "eval_timing.csv").is_file())


def check_probe_columns(save_dir: Path) -> Check:
    """4. feature_metrics.csv carries CellDINO/DynaCLR probe + fidelity entries."""
    csv = save_dir / "feature_metrics.csv"
    if not csv.is_file():
        return Check("probes", False, "csv missing")
    with csv.open() as f:
        header = f.readline()
    missing = [col for col in _REQUIRED_PROBE_COLS if col not in header]
    if missing:
        return Check("probes", False, f"missing: {missing}")
    return Check("probes", True)


def check_pred_cache(pred_cache_dir: Path) -> Check:
    """5. Pred cache populated with celldino features."""
    celldino_dir = pred_cache_dir / "features" / "celldino"
    if not celldino_dir.is_dir():
        return Check("pred_cache", False, "features/celldino/ missing")
    has_zarr = any(p.name.endswith(".zarr") for p in celldino_dir.iterdir())
    return Check("pred_cache", has_zarr, "" if has_zarr else "no .zarr inside")


def iter_conditions(leaf_root: Path):
    """Yield (leaf_path, condition_dict) for every condition in every leaf."""
    for leaf in sorted(leaf_root.glob("*/eval_grouped.yaml")):
        if "_probe" in leaf.parts:
            continue  # skip probe — verifies separately if desired
        with leaf.open() as f:
            body = yaml.safe_load(f)
        for cond in body.get("conditions", []):
            yield leaf, cond


def verify_all(leaf_root: Path) -> int:
    """Walk every leaf's conditions, run the 5 checks, print a table.

    Returns 0 when all conditions pass; 1 otherwise.
    """
    rows: list[tuple[str, str, list[Check]]] = []
    fail_count = 0
    for leaf, cond in iter_conditions(leaf_root):
        bucket = leaf.parent.name
        name = cond["name"]
        save_dir = Path(cond["save"]["save_dir"])
        pred_cache_dir = Path(cond["io"]["pred_cache_dir"])
        checks = [
            check_feature_csv(save_dir),
            check_embeddings(save_dir),
            check_eval_timing(save_dir),
            check_probe_columns(save_dir),
            check_pred_cache(pred_cache_dir),
        ]
        if any(not c.passed for c in checks):
            fail_count += 1
        rows.append((bucket, name, checks))

    # Print summary table.
    header = ["bucket", "condition"] + [c.name for c in rows[0][2]] if rows else []
    if not rows:
        print("no conditions found", file=sys.stderr)
        return 1
    print(" | ".join(f"{h:<14}" for h in header))
    print("-" * (len(header) * 17))
    for bucket, name, checks in rows:
        marks = [("OK " + c.detail if c.passed else "FAIL " + c.detail).strip() for c in checks]
        print(f"{bucket:<30} {name:<60} " + " ".join(f"[{m[:14]:<14}]" for m in marks))

    print(f"\n{len(rows)} conditions, {fail_count} failures")
    return 0 if fail_count == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--leaf-root",
        type=Path,
        default=_LEAF_OUT_ROOT,
        help=f"override leaf output root (default: {_LEAF_OUT_ROOT})",
    )
    args = ap.parse_args(argv)
    return verify_all(args.leaf_root)


if __name__ == "__main__":
    sys.exit(main())
