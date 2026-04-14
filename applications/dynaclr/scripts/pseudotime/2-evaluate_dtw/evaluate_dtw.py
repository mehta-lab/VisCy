"""Evaluate DTW pseudotime alignment against annotations.

Uses the existing alignments from Step 1 and compares pseudotime
against ground truth annotations (infection_state, organelle_state).
Produces AUC scores, onset concordance, and per-timepoint AUC.

These metrics quantify how well the model captures the infection
transition and organelle remodeling.

Usage::

    uv run python evaluate_dtw.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from dynaclr.evaluation.pseudotime.evaluation import (
    evaluate_embedding,
    per_timepoint_auc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def _well_label(dataset_id: str, embedding: str = "sensor") -> str:
    r"""Format dataset ID as 'WELL well\n(EMB PT)' for plot labels."""
    well = dataset_id.replace("2025_07_24_", "").replace("2025_07_22_", "")
    return f"{well} well\n({embedding} PT)"


IOU_TASKS: dict[str, tuple[str, str, str]] = {
    "infection": ("propagated_infection_label", "infection_state", "infected"),
    "organelle": ("propagated_organelle_label", "organelle_state", "remodel"),
}


def _compute_label_metrics(
    merged: pd.DataFrame,
    propagated_col: str,
    annotation_col: str,
    positive_value: str,
    label_threshold: float = 0.5,
) -> tuple[float, float, float, int]:
    """Compute IoU, precision, and recall between propagated template labels and human annotations.

    Parameters
    ----------
    merged : pd.DataFrame
        Must have propagated_col and annotation_col columns.
    propagated_col : str
        Column with propagated label fractions.
    annotation_col : str
        Column with ground truth annotation strings.
    positive_value : str
        Value in annotation_col that is the positive class.
    label_threshold : float
        Threshold on propagated label to binarize.

    Returns
    -------
    tuple[float, float, float, int]
        (IoU, precision, recall, number of valid cells used).
    """
    if propagated_col not in merged.columns or annotation_col not in merged.columns:
        return np.nan, np.nan, np.nan, 0

    valid = merged.dropna(subset=[propagated_col, annotation_col])
    valid = valid[valid[annotation_col] != ""]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan, 0

    pred = (valid[propagated_col] >= label_threshold).astype(int).to_numpy()
    true = (valid[annotation_col] == positive_value).astype(int).to_numpy()

    tp = int((pred & true).sum())
    fp = int((pred & ~true.astype(bool)).sum())
    fn = int((~pred.astype(bool) & true).sum())
    union = tp + fp + fn

    iou = float(tp / union) if union > 0 else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    return iou, precision, recall, len(valid)


def _add_dtw_quality_metrics(result: dict, alignments: pd.DataFrame) -> None:
    """Add DTW-specific quality metrics to the result dict.

    Parameters
    ----------
    result : dict
        Evaluation result dict to update in-place.
    alignments : pd.DataFrame
        Alignment results with dtw_cost, pseudotime, warping_speed columns.
    """
    per_track = alignments.groupby(["fov_name", "track_id"])
    costs = per_track["dtw_cost"].first()
    finite_costs = costs[np.isfinite(costs)]

    # Coverage: fraction of tracks with finite DTW cost
    result["coverage"] = float(len(finite_costs) / len(costs)) if len(costs) > 0 else 0.0

    # Normalized DTW cost: cost / track_length
    track_lengths = per_track.size()
    norm_costs = finite_costs / track_lengths.loc[finite_costs.index]
    result["normalized_dtw_cost_mean"] = float(norm_costs.mean()) if len(norm_costs) > 0 else np.nan
    result["normalized_dtw_cost_std"] = float(norm_costs.std()) if len(norm_costs) > 0 else np.nan

    # Transition sharpness: how many frames does pseudotime take to go from 0.1 to 0.9?
    sharpness_frames = []
    for _, track in per_track:
        track = track.sort_values("t")
        pt = track["pseudotime"].to_numpy()
        above_01 = np.where(pt >= 0.1)[0]
        above_09 = np.where(pt >= 0.9)[0]
        if len(above_01) > 0 and len(above_09) > 0:
            sharpness_frames.append(above_09[0] - above_01[0])
    if sharpness_frames:
        result["transition_sharpness_mean_frames"] = float(np.mean(sharpness_frames))
        result["transition_sharpness_std_frames"] = float(np.std(sharpness_frames))
    else:
        result["transition_sharpness_mean_frames"] = np.nan
        result["transition_sharpness_std_frames"] = np.nan


def main() -> None:
    """Evaluate DTW alignment against annotations."""
    parser = argparse.ArgumentParser(description="Evaluate DTW pseudotime against annotations")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pseudotime_dir = SCRIPT_DIR.parent
    output_dir = SCRIPT_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load all alignments from Step 1 (one parquet per template)
    alignments_dir = pseudotime_dir / "1-align_cells" / "alignments"
    parquet_files = sorted(alignments_dir.glob("alignments_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No alignment parquets found in {alignments_dir}")
    alignments = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    _logger.info(
        f"Loaded {len(alignments)} alignment rows from {len(parquet_files)} file(s): {[p.name for p in parquet_files]}"
    )

    # Evaluate each dataset that has annotations
    all_results = []
    all_timepoint_aucs = []
    all_merged: dict[str, pd.DataFrame] = {}

    for ds in config["alignment"]["datasets"]:
        dataset_id = ds["dataset_id"]
        annotations_path = ds.get("annotations_path")
        if annotations_path is None:
            _logger.info(f"Skipping {dataset_id} — no annotations_path")
            continue

        annotations = pd.read_csv(annotations_path)
        ds_alignments = alignments[alignments["dataset_id"] == dataset_id]

        if len(ds_alignments) == 0:
            _logger.warning(f"No alignments for {dataset_id}")
            continue

        # Run evaluation (AUC, onset concordance)
        eval_result = evaluate_embedding(ds_alignments, annotations, "sensor", dataset_id)

        # Merge with annotations for IoU and per-timepoint AUC
        ann_cols = ["fov_name", "track_id", "t"]
        for col in ["infection_state", "organelle_state"]:
            if col in annotations.columns:
                ann_cols.append(col)
        merged = ds_alignments.merge(
            annotations[ann_cols].drop_duplicates(),
            on=["fov_name", "track_id", "t"],
            how="left",
        )
        all_merged[dataset_id] = merged

        # IoU, precision, recall for each label task
        for task_name, (prop_col, ann_col, pos_val) in IOU_TASKS.items():
            iou, precision, recall, n_cells = _compute_label_metrics(merged, prop_col, ann_col, pos_val)
            eval_result[f"{task_name}_iou"] = iou
            eval_result[f"{task_name}_precision"] = precision
            eval_result[f"{task_name}_recall"] = recall
            eval_result[f"{task_name}_iou_n_cells"] = n_cells
            if np.isfinite(iou):
                _logger.info(
                    f"  {task_name} IoU: {iou:.3f}  precision: {precision:.3f}  recall: {recall:.3f}  ({n_cells} cells)"
                )

        # DTW quality metrics
        _add_dtw_quality_metrics(eval_result, ds_alignments)

        all_results.append(eval_result)

        # Per-timepoint AUC (infection)
        tp_auc = per_timepoint_auc(merged, annotation_col="infection_state", positive_value="infected")
        tp_auc["dataset_id"] = dataset_id
        tp_auc["task"] = "infection"
        all_timepoint_aucs.append(tp_auc)

        # Per-timepoint AUC (organelle)
        if "organelle_state" in merged.columns:
            tp_auc_org = per_timepoint_auc(merged, annotation_col="organelle_state", positive_value="remodel")
            tp_auc_org["dataset_id"] = dataset_id
            tp_auc_org["task"] = "organelle"
            all_timepoint_aucs.append(tp_auc_org)

    # Save results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_parquet(output_dir / "evaluation_summary.parquet", index=False)
        summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
        _logger.info("Evaluation summary:\n%s", summary_df.to_string())

        _plot_summary(summary_df, plots_dir)

    if all_timepoint_aucs:
        tp_df = pd.concat(all_timepoint_aucs, ignore_index=True)
        tp_df.to_parquet(output_dir / "per_timepoint_auc.parquet", index=False)

        _plot_per_timepoint_auc(tp_df, plots_dir)

    if all_merged:
        _plot_pseudotime_by_class(all_merged, plots_dir)
        _plot_example_tracks(all_merged, plots_dir)
        _plot_per_timepoint_auc_with_prevalence(all_merged, plots_dir)

    _save_failed_alignments(alignments, output_dir)

    _logger.info(f"Data saved to {output_dir}, plots saved to {plots_dir}")


def _save_failed_alignments(alignments: pd.DataFrame, output_dir: Path) -> None:
    """Save a CSV of tracks with non-finite DTW cost (alignment failures).

    Parameters
    ----------
    alignments : pd.DataFrame
        Combined alignments from all templates.
    output_dir : Path
        Directory to write failed_alignments.csv.
    """
    per_track = (
        alignments.groupby(["dataset_id", "template_id", "fov_name", "track_id"])
        .agg(
            dtw_cost=("dtw_cost", "first"),
            n_timepoints=("t", "count"),
            t_min=("t", "min"),
            t_max=("t", "max"),
        )
        .reset_index()
    )
    failed = per_track[~np.isfinite(per_track["dtw_cost"])].copy()
    out_path = output_dir / "failed_alignments.csv"
    failed.to_csv(out_path, index=False)
    _logger.info(
        f"Failed alignments: {len(failed)} / {len(per_track)} tracks "
        f"({100 * len(failed) / len(per_track):.1f}%) — saved to {out_path}"
    )
    if len(failed) > 0:
        by_dataset = failed.groupby(["dataset_id", "template_id"]).size().reset_index(name="n_failed")
        _logger.info("Failed tracks by dataset/template:\n%s", by_dataset.to_string(index=False))


def _plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of AUC metrics per dataset."""
    metrics = [
        c
        for c in [
            "infection_auc",
            "infection_ap",
            "infection_iou",
            "infection_precision",
            "infection_recall",
            "organelle_auc",
            "organelle_ap",
            "organelle_iou",
            "organelle_precision",
            "organelle_recall",
        ]
        if c in summary_df.columns
    ]
    metric_labels = {
        "infection_auc": "infection\n(pseudotime AUC)",
        "infection_ap": "infection\n(pseudotime AP)",
        "infection_iou": "infection\n(propagated IoU)",
        "infection_precision": "infection\n(propagated precision)",
        "infection_recall": "infection\n(propagated recall)",
        "organelle_auc": "organelle\n(pseudotime AUC)",
        "organelle_ap": "organelle\n(pseudotime AP)",
        "organelle_iou": "organelle\n(propagated IoU)",
        "organelle_precision": "organelle\n(propagated precision)",
        "organelle_recall": "organelle\n(propagated recall)",
    }

    datasets = summary_df["dataset_id"].unique()
    x = np.arange(len(datasets))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(datasets)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        values = [
            summary_df[summary_df["dataset_id"] == ds][metric].to_numpy()[0]
            if len(summary_df[summary_df["dataset_id"] == ds]) > 0
            else np.nan
            for ds in datasets
        ]
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_well_label(d) for d in datasets], fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11)
        if "auc" in metric:
            ylabel = "AUC"
        elif "ap" in metric:
            ylabel = "AP"
        elif "precision" in metric:
            ylabel = "Precision"
        elif "recall" in metric:
            ylabel = "Recall"
        else:
            ylabel = "IoU"
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        if "auc" in metric:
            ax.axhline(0.5, color="gray", ls="--", lw=0.5, label="chance")
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    fig.suptitle("Sensor Pseudotime vs Human Annotations", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_timepoint_auc(tp_df: pd.DataFrame, output_dir: Path) -> None:
    """Per-timepoint AUC: sensor pseudotime vs infection_state, one subplot per well."""
    inf_data = tp_df[tp_df["task"] == "infection"] if "task" in tp_df.columns else tp_df
    datasets = sorted(inf_data["dataset_id"].unique())
    well_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for i, (ax, ds_id) in enumerate(zip(axes, datasets)):
        ds_data = inf_data[inf_data["dataset_id"] == ds_id].sort_values("t")
        ax.plot(
            ds_data["t"],
            ds_data["auc"],
            color=well_colors.get(i, "#333333"),
            marker=".",
            markersize=4,
            linewidth=1.5,
            alpha=0.85,
        )
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("AUC")
        ax.set_title(_well_label(ds_id), fontsize=11)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Per-timepoint AUC — sensor pseudotime vs infection_state", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "per_timepoint_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pseudotime_by_class(all_merged: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    """KDE/violin of pseudotime distributions split by annotation class, per dataset.

    For each dataset shows uninfected vs infected pseudotime distribution so you can
    see whether the two classes are well-separated and where on [0,1] the transition sits.
    """
    for ann_col, pos_val, title_tag in [
        ("infection_state", "infected", "infection"),
        ("organelle_state", "remodel", "organelle"),
    ]:
        datasets = [ds for ds, df in all_merged.items() if ann_col in df.columns]
        if not datasets:
            continue

        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
        axes = axes[0]

        for ax, ds_id in zip(axes, datasets):
            df = all_merged[ds_id].dropna(subset=["pseudotime", ann_col])
            df = df[df[ann_col] != ""]

            neg = df[df[ann_col] != pos_val]["pseudotime"]
            pos = df[df[ann_col] == pos_val]["pseudotime"]

            ax.hist(neg, bins=30, range=(0, 1), density=True, alpha=0.6, color="#1f77b4", label=f"not {pos_val}")
            ax.hist(pos, bins=30, range=(0, 1), density=True, alpha=0.6, color="#d62728", label=pos_val)
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Density")
            ax.set_title(_well_label(ds_id), fontsize=11)
            ax.legend(fontsize=8)

        fig.suptitle(f"Pseudotime distribution by {ann_col}", fontsize=12)
        fig.tight_layout()
        fig.savefig(plots_dir / f"pseudotime_by_class_{title_tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_example_tracks(all_merged: dict[str, pd.DataFrame], plots_dir: Path, n_tracks: int = 6) -> None:
    """Pseudotime trajectory per track with annotation onset marked.

    Samples n_tracks infected cells per dataset. Each subplot shows pseudotime over
    time with a vertical line at the annotated infection onset frame.
    """
    ann_col = "infection_state"
    pos_val = "infected"

    for ds_id, df in all_merged.items():
        if ann_col not in df.columns:
            continue

        df = df.dropna(subset=["pseudotime", ann_col])
        df = df[df[ann_col] != ""]

        # Pick tracks that have at least one annotated positive frame
        infected_tracks = (
            df[df[ann_col] == pos_val]
            .groupby(["fov_name", "track_id"])
            .filter(lambda g: len(g) >= 1)[["fov_name", "track_id"]]
            .drop_duplicates()
        )
        if len(infected_tracks) == 0:
            continue

        sample = infected_tracks.sample(min(n_tracks, len(infected_tracks)), random_state=42)
        n_cols = min(3, len(sample))
        n_rows = int(np.ceil(len(sample) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)

        for idx, (_, row) in enumerate(sample.iterrows()):
            ax = axes[idx // n_cols][idx % n_cols]
            track = df[(df["fov_name"] == row["fov_name"]) & (df["track_id"] == row["track_id"])].sort_values("t")

            ax.plot(track["t"], track["pseudotime"], color="#1f77b4", linewidth=1.5)

            # Mark annotation onset (first infected frame)
            onset_frames = track[track[ann_col] == pos_val]["t"]
            if len(onset_frames) > 0:
                ax.axvline(onset_frames.iloc[0], color="#d62728", ls="--", lw=1.2, label="annotation onset")

            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Pseudotime")
            ax.set_title(f"fov={row['fov_name']}\ntrack={row['track_id']}", fontsize=8)
            ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(len(sample), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        ds_short = ds_id.replace("2025_07_24_", "").replace("2025_07_22_", "")
        fig.suptitle(f"Example tracks — {ds_short}", fontsize=12)
        fig.tight_layout()
        fig.savefig(plots_dir / f"example_tracks_{ds_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_per_timepoint_auc_with_prevalence(all_merged: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    """Per-timepoint AUC with infection prevalence overlay.

    Primary y-axis: AUC at each frame. Secondary y-axis (right): fraction of cells
    annotated as infected. Helps interpret low early AUC as a prevalence issue.
    """
    ann_col = "infection_state"
    pos_val = "infected"
    well_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    datasets = [ds for ds, df in all_merged.items() if ann_col in df.columns]
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for i, (ax, ds_id) in enumerate(zip(axes, datasets)):
        df = all_merged[ds_id].dropna(subset=["pseudotime", ann_col])
        df = df[df[ann_col] != ""]

        color = well_colors[i % len(well_colors)]

        # Per-timepoint AUC
        tp_rows = []
        for t_val, group in df.groupby("t"):
            y_true = (group[ann_col] == pos_val).astype(int).to_numpy()
            y_score = group["pseudotime"].to_numpy()
            n_pos = int(y_true.sum())
            n_total = len(group)
            if len(np.unique(y_true)) < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(y_true, y_score))
            tp_rows.append({"t": t_val, "auc": auc, "prevalence": n_pos / n_total if n_total > 0 else 0.0})
        if not tp_rows:
            continue
        tp = pd.DataFrame(tp_rows).sort_values("t")

        ax.plot(tp["t"], tp["auc"], color=color, linewidth=1.5, marker=".", markersize=4, label="AUC")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frame")
        ax.set_ylabel("AUC")
        ax.set_title(_well_label(ds_id), fontsize=11)

        ax2 = ax.twinx()
        ax2.fill_between(tp["t"], tp["prevalence"], alpha=0.15, color=color, label="% infected")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Fraction infected", color=color, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Per-timepoint AUC with infection prevalence", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_timepoint_auc_with_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
