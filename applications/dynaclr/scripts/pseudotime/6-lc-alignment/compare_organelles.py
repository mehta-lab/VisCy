"""Stage D — sequence of organelle remodeling events within one perturbation.

Compares Stage B per-cell event-timing distributions across two organelle
channels. Auto-detects test design based on whether the same
``(fov_name, track_id)`` keys appear in both parquets:

- **Paired mode**: cells co-imaged in both channels (same FOV/well, both
  markers stained simultaneously). Reports per-cell Δt_A − Δt_B,
  Wilcoxon signed-rank, and fraction A-before-B.
- **Unpaired mode**: the two channels are imaged in different wells of
  the same plate (e.g., A549 ZIKV: SEC61 in column A, G3BP1 in column C).
  Reports independent Δt distributions, Mann-Whitney U, and the
  difference of medians.

By default uses the **predicted_organelle_state** estimator from Stage B
(Path 2 LC inference) — the practitioner column available on every
dataset and not dependent on human annotations.

Inputs (per leaf):
- ``organelle_a / organelle_b`` blocks (label + timing parquet path).
- ``estimator_col`` / ``estimator_detected_col``: which Stage B columns
  to compare.
- ``dataset``: the parquet is restricted to the named dataset before
  joining/pooling.

Outputs (per pairing):
- ``<dataset>_<orgA>_vs_<orgB>_data.parquet``      raw paired or pooled
                                                    Δt values.
- ``<dataset>_<orgA>_vs_<orgB>_scatter.png``       (paired only)
- ``<dataset>_<orgA>_vs_<orgB>_distribution.png``  (unpaired only)
- ``sequence_summary.csv``                          one row per pairing
                                                    with mode + stats.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, wilcoxon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("compare_organelles")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge ``base:`` recipe imports and resolve per-channel parquet paths.

    Each pairing entry under ``pairings:`` declares an ``organelle_a`` and
    ``organelle_b``. Each side carries either ``timing_parquet`` (absolute)
    or ``timing_dir`` + ``timing_filename`` (resolved on read).
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        leaf = yaml.safe_load(f) or {}
    merged: dict = {}
    for rel in leaf.pop("base", []):
        with open((config_path.parent / rel).resolve()) as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(leaf)
    for pairing in merged.get("pairings") or []:
        for side in ("organelle_a", "organelle_b"):
            blk = pairing[side]
            if "timing_parquet" not in blk and "timing_filename" in blk:
                if "timing_dir" not in blk:
                    raise KeyError(f"{side}: timing_filename needs timing_dir")
                blk["timing_parquet"] = str(Path(blk["timing_dir"]) / blk["timing_filename"])
    return merged


def _restrict_dataset(timing: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Restrict a Stage B parquet to one dataset (no-op if already single-dataset)."""
    if "dataset" in timing.columns:
        return timing[timing["dataset"] == dataset].reset_index(drop=True)
    return timing


def _load_one_side(
    parquet: Path, dataset: str, estimator_col: str, estimator_detected_col: str, label: str
) -> pd.DataFrame:
    """Read one timing parquet, restrict to dataset, rename estimator columns to label-suffixed."""
    df = pd.read_parquet(parquet)
    df = _restrict_dataset(df, dataset)
    keep = ["fov_name", "track_id", estimator_col, estimator_detected_col]
    df = df[keep].rename(
        columns={
            estimator_col: f"delta_t_minutes_{label}",
            estimator_detected_col: f"detected_{label}",
        }
    )
    return df


def _compare_one(
    a_parquet: Path,
    b_parquet: Path,
    dataset: str,
    estimator_col: str,
    estimator_detected_col: str,
    label_a: str,
    label_b: str,
) -> tuple[str, pd.DataFrame, dict]:
    """Compare two organelle Δt distributions, auto-detect paired vs unpaired.

    Returns ``(mode, data_df, summary_dict)`` where ``mode`` is one of
    ``"paired"`` or ``"unpaired"``. ``data_df`` is the paired-by-cell
    dataframe in paired mode, else the long-format pooled Δt table.
    """
    a = _load_one_side(a_parquet, dataset, estimator_col, estimator_detected_col, label_a)
    b = _load_one_side(b_parquet, dataset, estimator_col, estimator_detected_col, label_b)

    overlap = a.merge(b, on=["fov_name", "track_id"], how="inner")
    n_overlap = len(overlap)

    if n_overlap >= 5:
        mode = "paired"
        both_det = overlap[f"detected_{label_a}"].fillna(False) & overlap[f"detected_{label_b}"].fillna(False)
        sub = overlap[both_det].copy()
        n_both = len(sub)
        if n_both == 0:
            summary = {
                "mode": mode,
                "dataset": dataset,
                "organelle_a": label_a,
                "organelle_b": label_b,
                "estimator": estimator_col,
                "n_a_total": int(len(a)),
                "n_b_total": int(len(b)),
                "n_cells_joined": n_overlap,
                "n_cells_both_detected": 0,
                "median_delta_t_a_minutes": float("nan"),
                "median_delta_t_b_minutes": float("nan"),
                "iqr_a_low_minutes": float("nan"),
                "iqr_a_high_minutes": float("nan"),
                "iqr_b_low_minutes": float("nan"),
                "iqr_b_high_minutes": float("nan"),
                "frac_a_before_b": float("nan"),
                "test_statistic": float("nan"),
                "test_p_a_lt_b": float("nan"),
            }
            return mode, sub, summary
        sub["delta_t_diff_minutes"] = sub[f"delta_t_minutes_{label_a}"] - sub[f"delta_t_minutes_{label_b}"]
        sub_diff = sub["delta_t_diff_minutes"].dropna()
        median_a = float(sub[f"delta_t_minutes_{label_a}"].median())
        median_b = float(sub[f"delta_t_minutes_{label_b}"].median())
        iqr_a = (
            float(sub[f"delta_t_minutes_{label_a}"].quantile(0.25)),
            float(sub[f"delta_t_minutes_{label_a}"].quantile(0.75)),
        )
        iqr_b = (
            float(sub[f"delta_t_minutes_{label_b}"].quantile(0.25)),
            float(sub[f"delta_t_minutes_{label_b}"].quantile(0.75)),
        )
        frac_a_lt_b = float((sub[f"delta_t_minutes_{label_a}"] < sub[f"delta_t_minutes_{label_b}"]).mean())
        if len(sub_diff) > 5 and sub_diff.abs().sum() > 0:
            try:
                w = wilcoxon(sub_diff, alternative="less")
                wstat, wp = float(w.statistic), float(w.pvalue)
            except ValueError:
                wstat, wp = float("nan"), float("nan")
        else:
            wstat, wp = float("nan"), float("nan")
        summary = {
            "mode": mode,
            "dataset": dataset,
            "organelle_a": label_a,
            "organelle_b": label_b,
            "estimator": estimator_col,
            "n_a_total": int(len(a)),
            "n_b_total": int(len(b)),
            "n_cells_joined": n_overlap,
            "n_cells_both_detected": n_both,
            "median_delta_t_a_minutes": median_a,
            "median_delta_t_b_minutes": median_b,
            "iqr_a_low_minutes": iqr_a[0],
            "iqr_a_high_minutes": iqr_a[1],
            "iqr_b_low_minutes": iqr_b[0],
            "iqr_b_high_minutes": iqr_b[1],
            "frac_a_before_b": frac_a_lt_b,
            "test_statistic": wstat,
            "test_p_a_lt_b": wp,
        }
        return mode, sub, summary

    # Unpaired: cells imaged in different wells. Pool independent Δt distributions.
    mode = "unpaired"
    a_det = a[a[f"detected_{label_a}"].fillna(False)].copy()
    b_det = b[b[f"detected_{label_b}"].fillna(False)].copy()
    a_vals = a_det[f"delta_t_minutes_{label_a}"].dropna()
    b_vals = b_det[f"delta_t_minutes_{label_b}"].dropna()
    long_rows = pd.concat(
        [
            pd.DataFrame({"organelle": label_a, "delta_t_minutes": a_vals.to_numpy()}),
            pd.DataFrame({"organelle": label_b, "delta_t_minutes": b_vals.to_numpy()}),
        ],
        ignore_index=True,
    )
    if len(a_vals) > 5 and len(b_vals) > 5:
        try:
            u = mannwhitneyu(a_vals.to_numpy(), b_vals.to_numpy(), alternative="less")
            ustat, up = float(u.statistic), float(u.pvalue)
        except ValueError:
            ustat, up = float("nan"), float("nan")
    else:
        ustat, up = float("nan"), float("nan")
    summary = {
        "mode": mode,
        "dataset": dataset,
        "organelle_a": label_a,
        "organelle_b": label_b,
        "estimator": estimator_col,
        "n_a_total": int(len(a)),
        "n_b_total": int(len(b)),
        "n_a_detected": int(len(a_vals)),
        "n_b_detected": int(len(b_vals)),
        "n_cells_joined": n_overlap,
        "n_cells_both_detected": 0,
        "median_delta_t_a_minutes": float(a_vals.median()) if len(a_vals) else float("nan"),
        "median_delta_t_b_minutes": float(b_vals.median()) if len(b_vals) else float("nan"),
        "iqr_a_low_minutes": float(a_vals.quantile(0.25)) if len(a_vals) else float("nan"),
        "iqr_a_high_minutes": float(a_vals.quantile(0.75)) if len(a_vals) else float("nan"),
        "iqr_b_low_minutes": float(b_vals.quantile(0.25)) if len(b_vals) else float("nan"),
        "iqr_b_high_minutes": float(b_vals.quantile(0.75)) if len(b_vals) else float("nan"),
        "frac_a_before_b": float("nan"),
        "test_statistic": ustat,
        "test_p_a_lt_b": up,
    }
    return mode, long_rows, summary


def _plot_scatter_paired(paired: pd.DataFrame, label_a: str, label_b: str, dataset: str, out_png: Path) -> None:
    """Per-cell paired scatter: Δt_A vs Δt_B with the y=x diagonal."""
    if paired.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    a = paired[f"delta_t_minutes_{label_a}"]
    b = paired[f"delta_t_minutes_{label_b}"]
    ax.scatter(a, b, alpha=0.6, s=18, edgecolor="black", linewidth=0.3)
    lo = float(np.nanmin([a.min(), b.min(), 0]))
    hi = float(np.nanmax([a.max(), b.max(), 0]))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.7, label="y=x")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    n_below = int((a < b).sum())
    n_total = int((~a.isna() & ~b.isna()).sum())
    ax.set_xlabel(f"$\\Delta t$ {label_a} (min after $t_{{perturb}}$)")
    ax.set_ylabel(f"$\\Delta t$ {label_b} (min after $t_{{perturb}}$)")
    ax.set_title(
        f"{dataset} — {label_a} vs {label_b} (paired)\n{n_below}/{n_total} cells with {label_a} before {label_b}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_distribution_unpaired(
    long_rows: pd.DataFrame,
    label_a: str,
    label_b: str,
    dataset: str,
    median_a: float,
    median_b: float,
    p_value: float,
    out_png: Path,
) -> None:
    """Side-by-side violin + box of Δt for two organelles (unpaired mode)."""
    if long_rows.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        long_rows.loc[long_rows["organelle"] == label_a, "delta_t_minutes"].dropna().to_numpy(),
        long_rows.loc[long_rows["organelle"] == label_b, "delta_t_minutes"].dropna().to_numpy(),
    ]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.7)
    for body in parts["bodies"]:
        body.set_alpha(0.5)
    ax.boxplot(data, widths=0.15, showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{label_a}\n(n={len(data[0])})", f"{label_b}\n(n={len(data[1])})"])
    ax.set_ylabel("$\\Delta t$ (min after $t_{perturb}$)")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    title_p = "n/a" if not np.isfinite(p_value) else f"{p_value:.3g}"
    ax.set_title(
        f"{dataset} — {label_a} vs {label_b} (unpaired)\n"
        f"median(A) = {median_a:.0f} min, median(B) = {median_b:.0f} min, "
        f"Mann-Whitney U p (A<B) = {title_p}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    """Stage D — pairwise organelle sequence comparison via Stage B parquets.

    Config schema::

        estimator_col: path2_predicted_organelle_state_delta_t_minutes
        estimator_detected_col: path2_predicted_organelle_state_detected
        pairings:
          - dataset: 07_24_ZIKV
            organelle_a:
              label: SEC61B
              timing_dir: /abs/.../zikv_timing_sec61
              timing_filename: 07_24_ZIKV_event_timing.parquet
            organelle_b:
              label: G3BP1
              timing_dir: /abs/.../zikv_timing_g3bp1
              timing_filename: 07_24_ZIKV_event_timing.parquet
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    estimator_col = cfg.get("estimator_col", "path2_predicted_organelle_state_delta_t_minutes")
    estimator_detected_col = cfg.get("estimator_detected_col", "path2_predicted_organelle_state_detected")

    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    summary_rows: list[dict] = []
    for pairing in cfg["pairings"]:
        dataset = pairing["dataset"]
        a_blk = pairing["organelle_a"]
        b_blk = pairing["organelle_b"]
        label_a = a_blk["label"]
        label_b = b_blk["label"]
        tag = f"{dataset}_{label_a}_vs_{label_b}"
        mode, data_df, summary = _compare_one(
            a_parquet=Path(a_blk["timing_parquet"]),
            b_parquet=Path(b_blk["timing_parquet"]),
            dataset=dataset,
            estimator_col=estimator_col,
            estimator_detected_col=estimator_detected_col,
            label_a=label_a,
            label_b=label_b,
        )
        data_df.to_parquet(args.output_dir / f"{tag}_data.parquet", index=False)
        if mode == "paired":
            _plot_scatter_paired(data_df, label_a, label_b, dataset, args.output_dir / f"{tag}_scatter.png")
        else:
            _plot_distribution_unpaired(
                data_df,
                label_a,
                label_b,
                dataset,
                median_a=summary["median_delta_t_a_minutes"],
                median_b=summary["median_delta_t_b_minutes"],
                p_value=summary["test_p_a_lt_b"],
                out_png=args.output_dir / f"{tag}_distribution.png",
            )
        summary_rows.append(summary)
        _logger.info(
            "[%s vs %s, %s, %s] median_a=%.0f median_b=%.0f p(A<B)=%.3g",
            label_a,
            label_b,
            dataset,
            mode,
            summary["median_delta_t_a_minutes"],
            summary["median_delta_t_b_minutes"],
            summary["test_p_a_lt_b"],
        )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "sequence_summary.csv", index=False)


if __name__ == "__main__":
    main()
