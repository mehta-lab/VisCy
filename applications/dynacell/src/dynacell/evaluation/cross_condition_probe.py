"""Cross-condition linear-probe CLI.

Post-hoc diagnostic that runs FOV-stratified logistic-regression probes
between two infection conditions for each of the three feature spaces
(CP regionprops, DINOv3, DynaCLR), separately for GT and predicted
embeddings, on the per-cell ``*_single_cell_embeddings.npz`` artifacts
emitted by ``dynacell.evaluation.pipeline._save_embeddings``.

Run from the repository root, after at least two per-plate evals have
finished::

    uv run python -m dynacell.evaluation.cross_condition_probe \\
        --eval_dirs \\
            /hpc/projects/.../eval_celldiff_iterative_membrane_mock \\
            /hpc/projects/.../eval_celldiff_iterative_membrane_denv \\
            /hpc/projects/.../eval_celldiff_iterative_membrane_zikv \\
        --out /tmp/cross_condition_probe.csv

The script auto-detects the conditions (mock/denv/zikv) from the eval
dir's trailing path token. It always emits a CSV with rows for every
(feature_type × pair × source) combination present in the inputs, and
NaN AUROC for pairs that don't reach two unique condition labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import get_args

import numpy as np

from dynacell.evaluation.cache import FeatureKind
from dynacell.evaluation.feature_select import select_features
from dynacell.evaluation.linear_probe import MADScaler, paired_auroc

_FEATURE_TYPES: tuple[str, ...] = get_args(FeatureKind)
_SOURCES = ("pred", "gt")
_CONDITION_TOKENS = ("mock", "denv", "zikv")
_DEFAULT_PAIRS = (("mock", "denv"), ("mock", "zikv"))


def _detect_condition(eval_dir: Path) -> str:
    """Extract ``mock``, ``denv``, or ``zikv`` from the dir name's trailing token."""
    name = eval_dir.name
    for token in _CONDITION_TOKENS:
        if name.endswith(f"_{token}"):
            return token
    raise ValueError(f"cannot infer condition from eval_dir name {name!r}: expected trailing _{{mock,denv,zikv}}")


def _load_embeddings(eval_dir: Path, source: str, feature: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(embeddings, fov_ids)`` from one ``*_single_cell_embeddings.npz``.

    ``np.load`` raises ``FileNotFoundError`` when the NPZ is missing.
    """
    npz_path = eval_dir / "embeddings" / f"{source}_{feature}_single_cell_embeddings.npz"
    with np.load(npz_path) as data:
        return np.asarray(data["embeddings"]), np.asarray(data["fov"])


def _probe_pair(
    eval_dirs_by_condition: dict[str, Path],
    pair: tuple[str, str],
    feature: str,
    source: str,
    n_splits: int,
    rng_seed: int,
) -> dict:
    """Run one ``fov_stratified_auroc`` call for the given (pair, feature, source).

    Returns a row dict ready for the CSV writer. NaN row when either
    side has no embeddings on disk; this is recorded explicitly so the
    consumer can tell "skipped" from "ran and got noisy result".
    """
    c0, c1 = pair
    row = {
        "feature_type": feature,
        "pair": f"{c0}_vs_{c1}",
        "source": source,
        "n_cells_c0": 0,
        "n_cells_c1": 0,
        "n_fovs": 0,
        "n_folds": 0,
        "auroc_mean": float("nan"),
        "auroc_std": float("nan"),
        "skipped_reason": "",
    }
    if c0 not in eval_dirs_by_condition or c1 not in eval_dirs_by_condition:
        row["skipped_reason"] = "missing eval dir for one side of pair"
        return row
    try:
        x0, fov0 = _load_embeddings(eval_dirs_by_condition[c0], source, feature)
        x1, fov1 = _load_embeddings(eval_dirs_by_condition[c1], source, feature)
    except FileNotFoundError as e:
        row["skipped_reason"] = f"missing embeddings file: {e}"
        return row
    if x0.size == 0 or x1.size == 0:
        row["skipped_reason"] = "empty embeddings on one side"
        return row
    if x0.shape[1] != x1.shape[1]:
        raise ValueError(f"feature dim mismatch for {feature} {source}: {c0}={x0.shape[1]} vs {c1}={x1.shape[1]}")

    # CP regionprops: variance + correlation prune on the pooled cohort
    # to drop near-constant or redundant columns. Skipped for deep
    # embeddings (DINOv3/DynaCLR), which are dense learned features.
    if feature == "cp":
        x0, x1, _ = select_features(x0, x1)
        if x0.size == 0 or x1.size == 0:
            row["skipped_reason"] = "all CP columns dropped by select_features"
            return row

    # Per-plate MAD normalization: cancels per-plate intensity offsets
    # (illumination, exposure, gain) that would otherwise dominate the
    # classifier — especially on CP regionprops, where raw intensity
    # columns make different plates trivially separable (AUROC = 1.0).
    x0_scaled = MADScaler().fit_transform(x0.astype(np.float64))
    x1_scaled = MADScaler().fit_transform(x1.astype(np.float64))
    # Tag FOV ids by condition so the two sides cannot collide.
    fov0_tagged = np.asarray([f"{c0}::{f}" for f in fov0])
    fov1_tagged = np.asarray([f"{c1}::{f}" for f in fov1])
    result = paired_auroc(x0_scaled, x1_scaled, fov0_tagged, fov1_tagged, n_splits=n_splits, rng_seed=rng_seed)
    row.update(
        {
            "n_cells_c0": int(len(x0)),
            "n_cells_c1": int(len(x1)),
            "n_fovs": int(len(np.unique(fov0_tagged)) + len(np.unique(fov1_tagged))),
            "n_folds": int(result["n_folds"]),
            "auroc_mean": float(result["auroc_mean"]),
            "auroc_std": float(result["auroc_std"]),
        }
    )
    return row


def run(
    eval_dirs: list[Path],
    out_path: Path,
    pairs: tuple[tuple[str, str], ...] = _DEFAULT_PAIRS,
    n_splits: int = 5,
    rng_seed: int = 2020,
) -> Path:
    """Probe each (pair, feature, source) and dump a long-form CSV.

    Parameters
    ----------
    eval_dirs : list[Path]
        Per-plate eval directories (one per condition). The condition
        is inferred from the dir name's trailing ``_{mock,denv,zikv}``.
    out_path : Path
        Output CSV.
    pairs : sequence[tuple[str, str]]
        Condition pairs to probe. Defaults to ``(mock,denv)`` + ``(mock,zikv)``.
    n_splits, rng_seed : int
        Forwarded to ``fov_stratified_auroc``.
    """
    eval_dirs_by_condition: dict[str, Path] = {}
    for d in eval_dirs:
        cond = _detect_condition(d)
        if cond in eval_dirs_by_condition:
            raise ValueError(f"duplicate condition {cond!r}: {eval_dirs_by_condition[cond]} and {d}")
        eval_dirs_by_condition[cond] = d

    rows = []
    for feature in _FEATURE_TYPES:
        for pair in pairs:
            for source in _SOURCES:
                rows.append(_probe_pair(eval_dirs_by_condition, pair, feature, source, n_splits, rng_seed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_type",
        "pair",
        "source",
        "n_cells_c0",
        "n_cells_c1",
        "n_fovs",
        "n_folds",
        "auroc_mean",
        "auroc_std",
        "skipped_reason",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--eval_dirs",
        nargs="+",
        type=Path,
        required=True,
        help="per-plate eval directories (2 or 3 of mock/denv/zikv)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output CSV path",
    )
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="GroupKFold splits (default: 5)",
    )
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=2020,
        help="seed for LogisticRegression tie-breaking (default: 2020)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        args.eval_dirs,
        args.out,
        n_splits=args.n_splits,
        rng_seed=args.rng_seed,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
