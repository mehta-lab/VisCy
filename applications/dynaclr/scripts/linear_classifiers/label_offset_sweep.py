"""Sweep temporal offsets on infection labels and evaluate classifier performance.

Shifts infection onset labels by varying frame offsets, trains cross-validated
classifiers at each offset, and evaluates both accuracy (against original labels)
and trajectory smoothness of predictions.
"""

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.linear_classifier import load_and_combine_datasets

logger = logging.getLogger(__name__)


def shift_infection_labels(adata: AnnData, task: str, dt: int) -> AnnData:
    """Shift infection onset labels forward or backward in time.

    Parameters
    ----------
    adata : AnnData
        Annotated data with ``task`` column, ``fov_name``, ``track_id``, ``t`` in obs.
    task : str
        Column name for infection state labels.
    dt : int
        Frame offset to apply. Negative = label infected earlier,
        positive = label infected later.

    Returns
    -------
    AnnData
        Copy of adata with ``{task}_shifted`` column added.
    """
    adata = adata.copy()
    shifted_col = f"{task}_shifted"
    adata.obs[shifted_col] = adata.obs[task].copy()

    if dt == 0:
        return adata

    for (fov, track), idx in adata.obs.groupby(["fov_name", "track_id"]).groups.items():
        track_obs = adata.obs.loc[idx].sort_values("t")
        infected_mask = track_obs[task] == "infected"

        if not infected_mask.any():
            continue

        t_onset = track_obs.loc[infected_mask, "t"].min()
        new_onset = t_onset + dt

        new_labels = track_obs[task].copy()
        new_labels[:] = "uninfected"
        new_labels[track_obs["t"] >= new_onset] = "infected"
        adata.obs.loc[track_obs.index, shifted_col] = new_labels

    return adata


def compute_smoothness(proba: np.ndarray, t: np.ndarray) -> dict:
    """Compute smoothness metrics for a single track's probability trajectory.

    Parameters
    ----------
    proba : np.ndarray
        Predicted infection probabilities for consecutive frames.
    t : np.ndarray
        Time values corresponding to each probability.

    Returns
    -------
    dict
        Smoothness metrics: ``mean_abs_diff`` and ``n_sign_changes``.
    """
    sort_idx = np.argsort(t)
    proba = proba[sort_idx]

    if len(proba) < 2:
        return {"mean_abs_diff": 0.0, "n_sign_changes": 0}

    diffs = np.diff(proba)
    mean_abs_diff = float(np.mean(np.abs(diffs)))

    signs = np.sign(diffs)
    signs = signs[signs != 0]
    n_sign_changes = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0

    return {"mean_abs_diff": mean_abs_diff, "n_sign_changes": n_sign_changes}


def build_pipeline(X, y, use_scaling, use_pca, n_pca_components, clf_params):
    """Fit preprocessing + classifier and return fitted objects.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    use_scaling : bool
        Whether to apply StandardScaler.
    use_pca : bool
        Whether to apply PCA.
    n_pca_components : int or None
        Number of PCA components.
    clf_params : dict
        LogisticRegression parameters.

    Returns
    -------
    tuple
        (scaler_or_None, pca_or_None, fitted_classifier, transformed_X)
    """
    scaler = None
    pca = None

    if use_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if use_pca and n_pca_components is not None:
        pca = PCA(n_components=n_pca_components)
        X = pca.fit_transform(X)

    clf = LogisticRegression(**clf_params)
    clf.fit(X, y)
    return scaler, pca, clf, X


def transform_features(X, scaler, pca):
    """Apply fitted preprocessing to features.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix.
    scaler : StandardScaler or None
        Fitted scaler.
    pca : PCA or None
        Fitted PCA.

    Returns
    -------
    np.ndarray
        Transformed features.
    """
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X


def run_sweep(config: dict) -> pd.DataFrame:
    """Run the label offset sweep experiment.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Results with one row per offset.
    """
    task = config["task"]
    offsets = config["offsets_frames"]
    frame_interval = config.get("frame_interval_minutes", 1)
    n_folds = config.get("n_cv_folds", 5)
    seed = config.get("random_seed", 42)

    use_scaling = config.get("use_scaling", True)
    use_pca = config.get("use_pca", False)
    n_pca_components = config.get("n_pca_components")
    clf_params = {
        "max_iter": config.get("max_iter", 1000),
        "class_weight": config.get("class_weight", "balanced"),
        "solver": config.get("solver", "liblinear"),
        "random_state": seed,
    }

    adata = load_and_combine_datasets(config["datasets"], task)

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    y_original = adata.obs[task].to_numpy()

    results = []

    for dt in offsets:
        dt_minutes = dt * frame_interval
        logger.info(f"Offset dt={dt} frames ({dt_minutes} min)")

        adata_shifted = shift_infection_labels(adata, task, dt)
        shifted_col = f"{task}_shifted"
        y_shifted = adata_shifted.obs[shifted_col].to_numpy()

        n_infected = np.sum(y_shifted == "infected")
        n_uninfected = np.sum(y_shifted == "uninfected")
        logger.info(f"  Class balance: infected={n_infected}, uninfected={n_uninfected}")

        unique_classes = np.unique(y_shifted)
        if len(unique_classes) < 2:
            logger.warning(f"  Skipping dt={dt}: only class '{unique_classes[0]}' remains after shifting")
            continue

        # --- Cross-validation: train on shifted, evaluate on original ---
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs, fold_f1s, fold_aurocs = [], [], []

        for train_idx, val_idx in skf.split(X, y_shifted):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_shifted = y_shifted[train_idx]
            y_val_original = y_original[val_idx]

            scaler, pca, clf, _ = build_pipeline(
                X_train, y_train_shifted, use_scaling, use_pca, n_pca_components, clf_params
            )
            X_val_t = transform_features(X_val, scaler, pca)

            y_pred = clf.predict(X_val_t)
            fold_accs.append(accuracy_score(y_val_original, y_pred))
            fold_f1s.append(f1_score(y_val_original, y_pred, pos_label="infected"))

            try:
                y_proba = clf.predict_proba(X_val_t)
                infected_idx = list(clf.classes_).index("infected")
                fold_aurocs.append(roc_auc_score(y_val_original, y_proba[:, infected_idx]))
            except ValueError:
                fold_aurocs.append(np.nan)

        # --- Smoothness: refit on full shifted data ---
        scaler_full, pca_full, clf_full, _ = build_pipeline(
            X, y_shifted, use_scaling, use_pca, n_pca_components, clf_params
        )
        X_full_t = transform_features(X, scaler_full, pca_full)
        infected_idx_full = list(clf_full.classes_).index("infected")
        proba_full = clf_full.predict_proba(X_full_t)[:, infected_idx_full]

        adata_shifted.obs["_proba_infected"] = proba_full
        track_smoothness = []
        for _, idx in adata_shifted.obs.groupby(["fov_name", "track_id"]).groups.items():
            track_obs = adata_shifted.obs.loc[idx]
            p = track_obs["_proba_infected"].to_numpy()
            t = track_obs["t"].to_numpy()
            track_smoothness.append(compute_smoothness(p, t))

        smooth_df = pd.DataFrame(track_smoothness)

        row = {
            "offset_frames": dt,
            "offset_minutes": dt_minutes,
            "n_infected": int(n_infected),
            "n_uninfected": int(n_uninfected),
            "cv_accuracy_mean": np.mean(fold_accs),
            "cv_accuracy_std": np.std(fold_accs),
            "cv_f1_mean": np.mean(fold_f1s),
            "cv_f1_std": np.std(fold_f1s),
            "cv_auroc_mean": np.nanmean(fold_aurocs),
            "cv_auroc_std": np.nanstd(fold_aurocs),
            "smoothness_mean_abs_diff": smooth_df["mean_abs_diff"].mean(),
            "smoothness_n_sign_changes": smooth_df["n_sign_changes"].mean(),
        }
        results.append(row)
        logger.info(
            f"  Acc={row['cv_accuracy_mean']:.3f}+-{row['cv_accuracy_std']:.3f}, "
            f"AUROC={row['cv_auroc_mean']:.3f}, "
            f"Smoothness={row['smoothness_mean_abs_diff']:.4f}"
        )

    return pd.DataFrame(results)


def plot_sweep(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy/AUROC and smoothness vs offset.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sweep results.
    output_path : Path
        Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    x = results_df["offset_frames"]

    ax1.errorbar(
        x, results_df["cv_accuracy_mean"], yerr=results_df["cv_accuracy_std"], marker="o", label="Accuracy", capsize=3
    )
    ax1.errorbar(x, results_df["cv_auroc_mean"], yerr=results_df["cv_auroc_std"], marker="s", label="AUROC", capsize=3)
    ax1.set_ylabel("Score")
    ax1.set_title("CV Performance vs Label Offset (evaluated on original labels)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax2.plot(x, results_df["smoothness_mean_abs_diff"], marker="o", label="Mean |dp/dt|")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, results_df["smoothness_n_sign_changes"], marker="s", color="tab:orange", label="Sign changes")
    ax2.set_xlabel("Label offset (frames)")
    ax2.set_ylabel("Mean |dp/dt|")
    ax2_twin.set_ylabel("Mean sign changes")
    ax2.set_title("Trajectory Smoothness vs Label Offset")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {output_path}")


@click.command()
@click.option("-c", "--config", "config_path", required=True, help="Path to YAML config file.")
def main(config_path: str) -> None:
    """Run label offset sweep for infection classifier."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_sweep(config)

    csv_path = output_dir / "label_offset_sweep_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    display_cols = [
        "offset_frames",
        "offset_minutes",
        "n_infected",
        "n_uninfected",
        "cv_accuracy_mean",
        "cv_auroc_mean",
        "smoothness_mean_abs_diff",
        "smoothness_n_sign_changes",
    ]
    table_data = results_df[display_cols].to_dict("records")
    md_table = format_markdown_table(table_data, title="Label Offset Sweep Results")
    print(md_table)

    if config.get("save_plots", False) and len(results_df) > 1:
        plot_path = output_dir / "label_offset_sweep.png"
        plot_sweep(results_df, plot_path)


if __name__ == "__main__":
    main()
