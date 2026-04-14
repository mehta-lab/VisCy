"""Diagnostic plots for DTW alignment results.

Generates:
1. Per-track pseudotime vs real time curves (sample of tracks per dataset)
2. Pseudotime distribution histogram (all cells)
3. DTW cost distribution per dataset
4. Warping speed heatmap (pseudotime vs real time)
5. PCA scatter: PC1 vs PC2 colored by real time and pseudotime

Usage::

    uv run python plotting.py [--n-tracks 10] [--config CONFIG]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _well_label(dataset_id: str, embedding: str = "sensor") -> str:
    """Format dataset ID as 'WELL well (EMB PT)' for plot labels."""
    well = dataset_id.replace("2025_07_24_", "").replace("2025_07_22_", "")
    return f"{well} well ({embedding} PT)"


def plot_pseudotime_curves(
    df: pd.DataFrame,
    output_dir: Path,
    n_tracks: int = 10,
) -> None:
    """Plot pseudotime vs real time for a sample of tracks per dataset."""
    datasets = df["dataset_id"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)
    axes = axes[0]

    for ax, ds_id in zip(axes, datasets):
        ds = df[df["dataset_id"] == ds_id]
        tracks = ds.groupby(["fov_name", "track_id"])

        # Sample tracks: pick a range of DTW costs (good, medium, bad)
        track_costs = tracks["dtw_cost"].first().sort_values()
        n_available = len(track_costs)
        n_sample = min(n_tracks, n_available)
        indices = np.linspace(0, n_available - 1, n_sample, dtype=int)
        sampled = track_costs.iloc[indices]

        for (fov, tid), cost in sampled.items():
            track = ds[(ds["fov_name"] == fov) & (ds["track_id"] == tid)].sort_values("t")
            ax.plot(
                track["t"],
                track["pseudotime"],
                alpha=0.6,
                linewidth=1.5,
                label=f"cost={cost:.1f}",
            )

        ax.set_xlabel("Real time (frame)")
        ax.set_ylabel("Pseudotime [0, 1]")
        ax.set_title(f"{_well_label(ds_id)}\n({n_available} tracks)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="grey", linestyle=":", alpha=0.3)
        ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
        if n_sample <= 10:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Pseudotime vs Real Time (sampled tracks, sorted by DTW cost)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "pseudotime_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pseudotime_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of pseudotime values across all cells, per dataset."""
    datasets = df["dataset_id"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds + 1, figsize=(5 * (n_ds + 1), 4), squeeze=False)
    axes = axes[0]

    # Per-dataset
    for ax, ds_id in zip(axes, datasets):
        ds = df[df["dataset_id"] == ds_id]
        ax.hist(ds["pseudotime"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Count (cell-timepoints)")
        ax.set_title(_well_label(ds_id))

    # Combined
    axes[-1].hist(df["pseudotime"].dropna(), bins=50, edgecolor="black", alpha=0.7, color="grey")
    axes[-1].set_xlabel("Pseudotime")
    axes[-1].set_title("All datasets")

    fig.suptitle("Pseudotime Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "pseudotime_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dtw_cost_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """DTW cost distribution per dataset (one cost per track)."""
    track_costs = df.groupby(["dataset_id", "fov_name", "track_id"])["dtw_cost"].first().reset_index()
    datasets = track_costs["dataset_id"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
    axes = axes[0]

    for ax, ds_id in zip(axes, datasets):
        costs = track_costs[track_costs["dataset_id"] == ds_id]["dtw_cost"]
        costs = costs[np.isfinite(costs)]
        ax.hist(costs, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(costs.median(), color="red", linestyle="--", label=f"median={costs.median():.2f}")
        ax.axvline(costs.quantile(0.75), color="orange", linestyle="--", label=f"75th={costs.quantile(0.75):.2f}")
        ax.set_xlabel("DTW Cost")
        ax.set_ylabel("Count (tracks)")
        ax.set_title(f"{_well_label(ds_id)} ({len(costs)} tracks)")
        ax.legend(fontsize=8)

    fig.suptitle("DTW Cost Distribution (per track)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "dtw_cost_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_warping_speed_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: rows = tracks (sorted by mean pseudotime), columns = real time, color = pseudotime."""
    datasets = df["dataset_id"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 6), squeeze=False)
    axes = axes[0]

    for ax, ds_id in zip(axes, datasets):
        ds = df[df["dataset_id"] == ds_id]
        tracks = ds.groupby(["fov_name", "track_id"])

        # Build matrix: rows = tracks, cols = timeframes
        t_min, t_max = int(ds["t"].min()), int(ds["t"].max())
        t_range = np.arange(t_min, t_max + 1)

        # Sort tracks by their mean pseudotime
        track_means = tracks["pseudotime"].mean().sort_values()
        track_order = list(track_means.index)

        matrix = np.full((len(track_order), len(t_range)), np.nan)
        for i, (fov, tid) in enumerate(track_order):
            track = ds[(ds["fov_name"] == fov) & (ds["track_id"] == tid)]
            for _, row in track.iterrows():
                t_idx = int(row["t"]) - t_min
                if 0 <= t_idx < len(t_range):
                    matrix[i, t_idx] = row["pseudotime"]

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xlabel("Real time (frame)")
        ax.set_ylabel(f"Tracks (n={len(track_order)}, sorted by mean pseudotime)")
        ax.set_title(_well_label(ds_id))

        # Reduce x-tick clutter
        n_ticks = min(10, len(t_range))
        tick_idx = np.linspace(0, len(t_range) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(t_range[tick_idx])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes.tolist(), label="Pseudotime", shrink=0.8)
    fig.suptitle("Pseudotime Heatmap (rows=tracks sorted by mean pseudotime, cols=real time)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "warping_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_pseudotime(
    alignments: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> None:
    """PCA scatter: PC1 vs PC2, colored by real time and by pseudotime.

    For each dataset, loads the sensor embeddings, projects to PC1/PC2,
    and makes a 2-column plot: left = colored by real time, right = colored by pseudotime.
    """
    import glob

    import anndata as ad
    from sklearn.decomposition import PCA

    emb_patterns = config.get("embeddings", {})
    alignment_cfg = config["alignment"]
    template_name = alignment_cfg.get("template", "infection_nondividing")
    template_cfg = config.get("templates", {}).get(template_name, {})
    emb_key = template_cfg.get("embedding", "sensor")
    emb_pattern = emb_patterns.get(emb_key, "timeaware_sensor_*.zarr")

    datasets = alignment_cfg["datasets"]
    n_ds = len(datasets)

    fig, axes = plt.subplots(n_ds, 3, figsize=(18, 5 * n_ds), squeeze=False)

    for row, ds in enumerate(datasets):
        dataset_id = ds["dataset_id"]
        pred_dir = ds["pred_dir"]
        fov_pattern = ds.get("fov_pattern")

        matches = glob.glob(str(Path(pred_dir) / emb_pattern))
        if not matches:
            continue
        adata = ad.read_zarr(matches[0])

        # Filter to FOV pattern
        if fov_pattern:
            mask = adata.obs["fov_name"].astype(str).str.contains(fov_pattern, regex=True)
            adata = adata[mask.to_numpy()].copy()

        emb = adata.X
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        emb = np.asarray(emb, dtype=np.float64)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(emb)
        pc1_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        pc2_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"

        # Match pseudotime from alignments
        ds_align = alignments[alignments["dataset_id"] == dataset_id]
        pt_lookup = ds_align.set_index(["fov_name", "track_id", "t"])["pseudotime"].to_dict()

        obs = adata.obs
        pseudotime = np.array(
            [
                pt_lookup.get((row_obs["fov_name"], row_obs["track_id"], row_obs["t"]), np.nan)
                for _, row_obs in obs.iterrows()
            ]
        )
        real_time = obs["t"].to_numpy().astype(float)

        # Infection state from annotations or predictions
        infection_state = None
        if "predicted_infection_state" in obs.columns:
            infection_state = obs["predicted_infection_state"].to_numpy()
        elif "infection_state" in obs.columns:
            infection_state = obs["infection_state"].to_numpy()

        # Shared limits for all 3 columns
        xlim = (pc[:, 0].min() - 1, pc[:, 0].max() + 1)
        ylim = (pc[:, 1].min() - 1, pc[:, 1].max() + 1)

        # Col 1: colored by real time
        ax_rt = axes[row, 0]
        sc = ax_rt.scatter(pc[:, 0], pc[:, 1], c=real_time, cmap="viridis", s=3, alpha=0.5)
        fig.colorbar(sc, ax=ax_rt, label="Real time (frame)")
        ax_rt.set_title(f"{_well_label(dataset_id)}\nColored by real time")

        # Col 2: colored by pseudotime
        ax_pt = axes[row, 1]
        valid = np.isfinite(pseudotime)
        ax_pt.scatter(pc[~valid, 0], pc[~valid, 1], c="lightgrey", s=3, alpha=0.3)
        sc2 = ax_pt.scatter(
            pc[valid, 0], pc[valid, 1], c=pseudotime[valid], cmap="magma", s=3, alpha=0.5, vmin=0, vmax=1
        )
        fig.colorbar(sc2, ax=ax_pt, label="DTW pseudotime")
        ax_pt.set_title(f"{_well_label(dataset_id)}\nColored by pseudotime")

        # Col 3: colored by infection state (uninfected vs infected)
        ax_inf = axes[row, 2]
        if infection_state is not None:
            colors = {"uninfected": "#3498db", "infected": "#e74c3c"}
            for state, color in colors.items():
                state_mask = infection_state == state
                ax_inf.scatter(
                    pc[state_mask, 0],
                    pc[state_mask, 1],
                    c=color,
                    s=3,
                    alpha=0.4,
                    label=state,
                )
            known = np.isin(infection_state, list(colors.keys()))
            if (~known).any():
                ax_inf.scatter(pc[~known, 0], pc[~known, 1], c="lightgrey", s=2, alpha=0.2, label="other")
            ax_inf.legend(fontsize=8, markerscale=3)
        else:
            ax_inf.text(
                0.5,
                0.5,
                "No infection state\navailable",
                transform=ax_inf.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="grey",
            )
        ax_inf.set_title(f"{_well_label(dataset_id)}\nColored by infection state")

        # Apply shared limits and aspect to all 3 axes
        for ax in axes[row]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_xlabel(pc1_label)
            ax.set_ylabel(pc2_label)

    fig.suptitle("Sensor Embeddings: PC1 vs PC2", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_pseudotime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_template_cell_tracks(
    template_path: Path,
    all_adatas: dict[str, "ad.AnnData"],  # noqa: F821
    t_rel_lookups: dict[str, dict],
    pca: "PCA",  # noqa: F821
    n_pcs: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the template cell tracks and return their mean PC trajectory vs t_rel.

    The template zarr stores template_cell_ids as (dataset_id, fov_name, track_id).
    We look up those tracks in the loaded adatas, project to PC space, align on
    t_relative_minutes, and return (t_grid, mean_pc) for plotting as the template trace.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (t_grid of shape (200,), mean_pc of shape (200, n_pcs)) or (None, None).
    """
    import zarr

    store = zarr.open(str(template_path), mode="r")
    cell_ids = [tuple(c) for c in store.attrs["template_cell_ids"]]
    # cell_ids: list of (dataset_id, fov_name, track_id)

    track_t_rels = []
    track_pcs = []
    n_use = min(n_pcs, pca.components_.shape[0])

    for dataset_id, fov_name, track_id in cell_ids:
        track_id = int(track_id)
        if dataset_id not in all_adatas:
            continue
        adata = all_adatas[dataset_id]
        obs = adata.obs.reset_index(drop=True)
        t_rel_lookup = t_rel_lookups.get(dataset_id, {})

        mask = (obs["fov_name"] == fov_name) & (obs["track_id"] == track_id)
        tidx = np.where(mask.values)[0]
        if len(tidx) == 0:
            continue

        emb = adata.X[tidx]
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        emb = np.asarray(emb, dtype=np.float64)
        pc = (emb - pca.mean_) @ pca.components_[:n_use].T

        t_vals = obs.iloc[tidx]["t"].to_numpy()
        t_rel = np.array([t_rel_lookup.get((fov_name, track_id, t), np.nan) for t in t_vals])
        valid = np.isfinite(t_rel)
        if valid.sum() < 2:
            continue

        sort_order = np.argsort(t_rel[valid])
        track_t_rels.append(t_rel[valid][sort_order])
        track_pcs.append(pc[valid][sort_order])

    if not track_t_rels:
        return None, None

    t_min = min(t.min() for t in track_t_rels)
    t_max = max(t.max() for t in track_t_rels)
    t_grid = np.linspace(t_min, t_max, 200)
    interp_pcs = np.full((len(track_t_rels), n_use, 200), np.nan)
    for i, (t_rel_s, pc_s) in enumerate(zip(track_t_rels, track_pcs)):
        for pc_idx in range(n_use):
            interp_pcs[i, pc_idx] = np.interp(t_grid, t_rel_s, pc_s[:, pc_idx], left=np.nan, right=np.nan)

    mean_pc = np.nanmean(interp_pcs, axis=0).T  # (200, n_use)
    return t_grid, mean_pc


def plot_aligned_pcs(
    alignments: pd.DataFrame,
    config: dict,
    output_dir: Path,
    n_tracks: int = 50,
    n_pcs: int = 5,
) -> None:
    """Aligned tracks overlaid on a real-time axis anchored at infection onset.

    X-axis is t_relative_minutes (0 = infection onset, negative = before,
    positive = after). All tracks are overlaid so the infection event is
    synchronized. The black trace is the mean of the actual template cells
    (the tracks used to build the DBA template), giving a true reference.

    Layout: one column per PC, one row per dataset.
    Tracks colored by DTW cost. Vertical dashed line at t=0.
    """
    import glob

    import anndata as ad
    import zarr
    from sklearn.decomposition import PCA

    emb_patterns = config.get("embeddings", {})
    alignment_cfg = config["alignment"]
    template_name = alignment_cfg.get("template", "infection_nondividing")
    template_cfg = config.get("templates", {}).get(template_name, {})
    emb_key = template_cfg.get("embedding", "sensor")
    emb_pattern = emb_patterns.get(emb_key, "timeaware_sensor_*.zarr")

    # Load template PCA
    template_path = SCRIPT_DIR.parent / "0-build_templates" / "templates" / f"template_{template_name}.zarr"
    template_pca = None
    evr = None
    if template_path.exists():
        store = zarr.open(str(template_path), mode="r")
        if "pca_components" in store:
            n_comp = store.attrs["pca_n_components"]
            template_pca = PCA(n_components=n_comp)
            template_pca.components_ = np.array(store["pca_components"])
            template_pca.mean_ = np.array(store["pca_mean"])
            template_pca.explained_variance_ratio_ = np.array(store["pca_explained_variance_ratio"])
            template_pca.explained_variance_ = np.array(store["pca_explained_variance"])
            template_pca.n_components_ = n_comp
            template_pca.n_features_in_ = store.attrs.get("pca_n_features_in", template_pca.components_.shape[1])
            template_pca.n_samples_ = store.attrs.get("pca_n_samples_seen", 0)
            evr = template_pca.explained_variance_ratio_

    datasets = alignment_cfg["datasets"]
    n_ds = len(datasets)

    # Pre-load all adatas and t_rel lookups (needed for template track lookup)
    all_adatas: dict[str, ad.AnnData] = {}
    all_t_rel_lookups: dict[str, dict] = {}
    all_pc: dict[str, np.ndarray] = {}
    all_obs: dict[str, "pd.DataFrame"] = {}

    for ds in datasets:
        dataset_id = ds["dataset_id"]
        fov_pattern = ds.get("fov_pattern")
        matches = glob.glob(str(Path(ds["pred_dir"]) / emb_pattern))
        if not matches:
            continue
        adata = ad.read_zarr(matches[0])
        if fov_pattern:
            mask = adata.obs["fov_name"].astype(str).str.contains(fov_pattern, regex=True)
            adata = adata[mask.to_numpy()].copy()

        emb = adata.X
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        emb = np.asarray(emb, dtype=np.float64)

        if template_pca is not None:
            n_use = min(n_pcs, template_pca.components_.shape[0])
            pc = (emb - template_pca.mean_) @ template_pca.components_[:n_use].T
        else:
            pca = PCA(n_components=n_pcs)
            pc = pca.fit_transform(emb)

        ds_align = alignments[alignments["dataset_id"] == dataset_id]
        t_rel_lookup = ds_align.set_index(["fov_name", "track_id", "t"])["t_relative_minutes"].to_dict()

        all_adatas[dataset_id] = adata
        all_t_rel_lookups[dataset_id] = t_rel_lookup
        all_pc[dataset_id] = pc
        all_obs[dataset_id] = adata.obs.reset_index(drop=True)

    # Compute template trace from actual template cells
    template_t_grid, template_mean_pc = None, None
    if template_pca is not None and template_path.exists():
        template_t_grid, template_mean_pc = _load_template_cell_tracks(
            template_path, all_adatas, all_t_rel_lookups, template_pca, n_pcs
        )

    fig, axes = plt.subplots(n_ds, n_pcs, figsize=(4 * n_pcs, 4 * n_ds), squeeze=False)

    for row_idx, ds in enumerate(datasets):
        dataset_id = ds["dataset_id"]
        if dataset_id not in all_adatas:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, f"No embeddings\n{dataset_id}", transform=ax.transAxes, ha="center", va="center")
            continue

        pc = all_pc[dataset_id]
        obs = all_obs[dataset_id]
        t_rel_lookup = all_t_rel_lookups[dataset_id]
        ds_align = alignments[alignments["dataset_id"] == dataset_id]

        if template_pca is not None:
            n_use = min(n_pcs, template_pca.components_.shape[0])
            pc_evr = evr[:n_use]
        else:
            n_use = n_pcs
            pc_evr = np.zeros(n_pcs)

        # Sample tracks by DTW cost spread
        track_costs = ds_align.groupby(["fov_name", "track_id"])["dtw_cost"].first().sort_values()
        n_available = len(track_costs)
        n_sample = min(n_tracks, n_available)
        indices = np.linspace(0, n_available - 1, n_sample, dtype=int)
        sampled_costs = track_costs.iloc[indices]
        sampled_keys = list(map(tuple, sampled_costs.index.tolist()))

        cost_vals = sampled_costs.to_numpy().astype(float)
        cost_min, cost_max = cost_vals.min(), cost_vals.max()
        cost_norm = (cost_vals - cost_min) / (cost_max - cost_min + 1e-10)
        track_cmap = plt.get_cmap("plasma")

        region_lookup = (
            ds_align.set_index(["fov_name", "track_id", "t"])["alignment_region"].to_dict()
            if "alignment_region" in ds_align.columns
            else None
        )

        track_data = []
        for s_idx, (fov, tid) in enumerate(sampled_keys):
            track_mask = (obs["fov_name"] == fov) & (obs["track_id"] == tid)
            tidx = np.where(track_mask.values)[0]
            if len(tidx) == 0:
                track_data.append(None)
                continue
            t_vals = obs.iloc[tidx]["t"].to_numpy()
            t_rel = np.array([t_rel_lookup.get((fov, tid, t), np.nan) for t in t_vals])
            valid = np.isfinite(t_rel)
            if valid.sum() < 2:
                track_data.append(None)
                continue
            sort_order = np.argsort(t_rel[valid])
            t_rel_sorted = t_rel[valid][sort_order]
            pc_sorted = pc[tidx[valid], :][sort_order, :]
            color = track_cmap(cost_norm[s_idx])
            if region_lookup is not None:
                regions = np.array([region_lookup.get((fov, tid, t), "aligned") for t in t_vals])
                regions_sorted = regions[valid][sort_order]
            else:
                regions_sorted = np.full(valid.sum(), "aligned")
            track_data.append((t_rel_sorted, pc_sorted, color, regions_sorted))

        for pc_idx in range(n_pcs):
            ax = axes[row_idx, pc_idx]

            for td in track_data:
                if td is None:
                    continue
                t_rel_sorted, pc_sorted, color, regions_sorted = td
                if pc_idx < pc_sorted.shape[1]:
                    pc_vals = pc_sorted[:, pc_idx]
                    # Full track: thin dashed at low alpha (pre + post context)
                    ax.plot(t_rel_sorted, pc_vals, color=color, linewidth=0.6, alpha=0.25, linestyle="--")
                    # Aligned region overdraw: solid at normal weight
                    aligned_mask = regions_sorted == "aligned"
                    if aligned_mask.any():
                        ax.plot(
                            t_rel_sorted,
                            np.where(aligned_mask, pc_vals, np.nan),
                            color=color,
                            linewidth=1.0,
                            alpha=0.6,
                        )

            # Template trace: mean of the actual DBA template cells
            if template_t_grid is not None and template_mean_pc is not None and pc_idx < template_mean_pc.shape[1]:
                valid_tmpl = np.isfinite(template_mean_pc[:, pc_idx])
                ax.plot(
                    template_t_grid[valid_tmpl],
                    template_mean_pc[valid_tmpl, pc_idx],
                    color="black",
                    linewidth=2.5,
                    marker="o",
                    markersize=2,
                    markevery=5,
                    label="template",
                    zorder=5,
                )

            ax.axvline(0, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="infection onset")
            evr_label = f" ({pc_evr[pc_idx]:.1%})" if pc_idx < len(pc_evr) else ""
            ax.set_xlabel("Time relative to infection onset (min)")
            ax.set_ylabel(f"PC{pc_idx + 1}{evr_label}")
            if pc_idx == 0:
                ax.set_title(f"{_well_label(dataset_id)}\n({n_available} tracks, {n_sample} shown)")
                ax.legend(fontsize=7, loc="upper left")
            else:
                ax.set_title(f"PC{pc_idx + 1}{evr_label}")

    pca_src = "template PCA" if template_pca is not None else "PCA"
    fig.suptitle(
        f"Aligned tracks: PCn vs time relative to infection onset ({pca_src})\n"
        "color=DTW cost (low=purple, high=yellow), black=DBA template cells mean",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "aligned_pcs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save colorbar as separate PNG
    fig_cb, ax_cb = plt.subplots(figsize=(1.2, 4))
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig_cb.colorbar(sm, cax=ax_cb, label="DTW cost (normalized)")
    fig_cb.tight_layout()
    fig_cb.savefig(output_dir / "aligned_pcs_colorbar.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cb)


def main() -> None:
    """Run diagnostic plots for DTW alignment results."""
    parser = argparse.ArgumentParser(description="Diagnostic plots for DTW alignments")
    parser.add_argument("--n-tracks", type=int, default=10, help="Tracks to sample per dataset for curves plot")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (for PCA plot)")
    parser.add_argument("--alignments", type=str, default=None, help="Path to alignments parquet file")
    args = parser.parse_args()

    alignments_path = Path(args.alignments) if args.alignments else SCRIPT_DIR / "alignments" / "alignments.parquet"
    output_dir = SCRIPT_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(alignments_path)
    print(f"Loaded {len(df)} rows, {df.groupby(['dataset_id', 'fov_name', 'track_id']).ngroups} tracks")

    plot_pseudotime_curves(df, output_dir, n_tracks=args.n_tracks)
    print("  -> pseudotime_curves.png")

    plot_pseudotime_distribution(df, output_dir)
    print("  -> pseudotime_distribution.png")

    plot_dtw_cost_distribution(df, output_dir)
    print("  -> dtw_cost_distribution.png")

    plot_warping_speed_heatmap(df, output_dir)
    print("  -> warping_heatmap.png")

    # PCA/PC1 plots require config to locate embedding zarrs
    config = None
    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config_path = SCRIPT_DIR.parent.parent.parent / "configs" / "pseudotime" / "multi_template.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

    if config is not None:
        plot_pca_pseudotime(df, config, output_dir)
        print("  -> pca_pseudotime.png")
        plot_aligned_pcs(df, config, output_dir, n_tracks=args.n_tracks)
        print("  -> aligned_pcs.png + aligned_pcs_colorbar.png")
    else:
        print("  (skipping PCA/PC1 plots — no config found, pass --config)")

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
