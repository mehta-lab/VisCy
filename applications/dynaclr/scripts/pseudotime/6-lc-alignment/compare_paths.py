"""Stage C — cross-validate Stage B Δt estimators per cell.

Each Stage B per-cell parquet carries side-by-side Δt columns for every
``(path, baseline, tau, label_column)`` combination configured in Stage B
defaults. This script pairs Δt columns **within the same parquet** (same
cell, two estimators) and reports per-cell agreement statistics:

- Spearman ρ, Pearson r
- Mean absolute error in frames and minutes
- Fraction of cells within ±1 / ±2 frames
- Per-pairing scatter plot with the ``y = x`` diagonal

Default pairings (configurable via leaf):

- Path 1 cell_own τ=0.5     vs  Path 2 organelle_state            (embedding-distance vs human annotation)
- Path 1 uninfected_pop τ=0.5 vs Path 2 organelle_state            (population baseline vs human annotation)
- Path 2 organelle_state    vs  Path 2 predicted_organelle_state  (LC trained on human vs LC inference column)

For each pairing, only cells **detected by both estimators** contribute to
the statistic. The output is one CSV row per (dataset, organelle,
pairing) plus one scatter PNG per pairing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("compare_paths")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge ``base:`` recipe imports and resolve per-entry parquet paths.

    Schema::

        timing_dir: /abs/.../zikv_timing_<organelle>
        entries:
          - name: <dataset>
            organelle: <label>
            timing_filename: <dataset>_event_timing.parquet
        pairings:
          - label: "cell_own_tau50_vs_human_LC"
            a_col: path1_cell_own_tau050_delta_t_minutes
            a_det: path1_cell_own_tau050_detected
            a_frame_col: path1_cell_own_tau050_delta_t_frames
            b_col: path2_organelle_state_delta_t_minutes
            b_det: path2_organelle_state_detected
            b_frame_col: path2_organelle_state_delta_t_frames
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        leaf = yaml.safe_load(f) or {}
    merged: dict = {}
    for rel in leaf.pop("base", []):
        with open((config_path.parent / rel).resolve()) as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(leaf)
    timing_dir = merged.get("timing_dir")
    for e in merged.get("entries") or []:
        if "timing_parquet" not in e and "timing_filename" in e:
            if timing_dir is None:
                raise KeyError(f"{e.get('name')}: timing_filename needs timing_dir")
            e["timing_parquet"] = str(Path(timing_dir) / e["timing_filename"])
    return merged


def _agreement_stats(df: pd.DataFrame, pairing: dict) -> dict:
    """Compute Spearman / Pearson / MAE / within-tolerance for one pairing.

    Restricts to rows where both estimators detected the event.
    """
    a_col = pairing["a_col"]
    b_col = pairing["b_col"]
    a_det = pairing["a_det"]
    b_det = pairing["b_det"]
    a_frame_col = pairing.get("a_frame_col")
    b_frame_col = pairing.get("b_frame_col")

    missing_cols = [c for c in (a_col, b_col, a_det, b_det) if c not in df.columns]
    if missing_cols:
        return {"n_both_detected": 0, "missing_columns": "+".join(missing_cols)}

    keep = df[a_det].fillna(False) & df[b_det].fillna(False)
    sub = df[keep].dropna(subset=[a_col, b_col])
    n_both = len(sub)
    if n_both < 3:
        return {
            "n_both_detected": int(n_both),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "mae_minutes": float("nan"),
            "mae_frames": float("nan"),
            "frac_within_1_frame": float("nan"),
            "frac_within_2_frames": float("nan"),
        }
    a = sub[a_col].to_numpy(dtype=float)
    b = sub[b_col].to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        sp = float("nan")
        pr = float("nan")
    else:
        sp = float(spearmanr(a, b).statistic)
        pr = float(pearsonr(a, b).statistic)
    mae_min = float(np.mean(np.abs(a - b)))

    if a_frame_col and b_frame_col and a_frame_col in df.columns and b_frame_col in df.columns:
        af = sub[a_frame_col].to_numpy(dtype=float)
        bf = sub[b_frame_col].to_numpy(dtype=float)
        diff = np.abs(af - bf)
        mae_fr = float(np.mean(diff[np.isfinite(diff)]))
        frac_1 = float(np.mean(diff[np.isfinite(diff)] <= 1.0)) if np.isfinite(diff).any() else float("nan")
        frac_2 = float(np.mean(diff[np.isfinite(diff)] <= 2.0)) if np.isfinite(diff).any() else float("nan")
    else:
        mae_fr = float("nan")
        frac_1 = float("nan")
        frac_2 = float("nan")

    return {
        "n_both_detected": int(n_both),
        "spearman": sp,
        "pearson": pr,
        "mae_minutes": mae_min,
        "mae_frames": mae_fr,
        "frac_within_1_frame": frac_1,
        "frac_within_2_frames": frac_2,
    }


def _plot_scatter(df: pd.DataFrame, pairing: dict, dataset: str, organelle: str, out_png: Path) -> None:
    """Per-cell scatter of Δt_A vs Δt_B with the y=x diagonal."""
    a_col = pairing["a_col"]
    b_col = pairing["b_col"]
    a_det = pairing["a_det"]
    b_det = pairing["b_det"]
    if not all(c in df.columns for c in (a_col, b_col, a_det, b_det)):
        return
    keep = df[a_det].fillna(False) & df[b_det].fillna(False)
    sub = df[keep].dropna(subset=[a_col, b_col])
    if len(sub) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(sub[a_col], sub[b_col], alpha=0.6, s=18, edgecolor="black", linewidth=0.3)
    lo = float(min(sub[a_col].min(), sub[b_col].min(), 0))
    hi = float(max(sub[a_col].max(), sub[b_col].max(), 0))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.7, label="y=x")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"$\\Delta t$ {pairing['label']} A (min)")
    ax.set_ylabel(f"$\\Delta t$ {pairing['label']} B (min)")
    ax.set_title(f"{dataset} {organelle} — {pairing['label']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    """Stage C — cross-validate Path 1 vs Path 2 estimators within each Stage B parquet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    entries = cfg.get("entries") or []
    pairings = cfg.get("pairings") or []
    if not entries or not pairings:
        raise ValueError("config must list entries and pairings")

    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    rows: list[dict] = []
    for entry in entries:
        df = pd.read_parquet(entry["timing_parquet"])
        if "dataset" in df.columns:
            df = df[df["dataset"] == entry["name"]].reset_index(drop=True)
        organelle = entry.get("organelle", "unknown")
        for pairing in pairings:
            stats = _agreement_stats(df, pairing)
            stats.update(
                {
                    "dataset": entry["name"],
                    "organelle": organelle,
                    "pairing": pairing["label"],
                    "a_col": pairing["a_col"],
                    "b_col": pairing["b_col"],
                }
            )
            rows.append(stats)
            png = args.output_dir / f"{entry['name']}_{organelle}_{pairing['label']}.png"
            _plot_scatter(df, pairing, entry["name"], organelle, png)
            _logger.info(
                "[%s/%s] %s: n=%s spearman=%s mae_min=%s",
                entry["name"],
                organelle,
                pairing["label"],
                stats.get("n_both_detected"),
                stats.get("spearman"),
                stats.get("mae_minutes"),
            )

    out_csv = args.output_dir / "path_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
