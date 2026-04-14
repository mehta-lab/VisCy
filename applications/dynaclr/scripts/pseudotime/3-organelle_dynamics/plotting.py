"""Diagnostic plots for organelle dynamics results.

Generates:
1. Per-cell remodeling heatmap aligned to real time (filtered by min pre/post frames)
2. Cell crop montage grids (image heatmap) per organelle per channel

Usage::

    uv run python plotting.py --config CONFIG --data-zarr DATA_ZARR [--min-pre 5] [--min-post 5]
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def _get_cell_info(alignments: pd.DataFrame) -> dict:
    """Compute transition onset, pre/post frame counts, and DTW cost per cell."""
    cell_info = {}
    for uid, track in alignments.groupby("cell_uid"):
        track = track.sort_values("t")
        pt = track["pseudotime"].to_numpy()
        t = track["t"].to_numpy()
        trans = t[pt > 0]
        if len(trans) == 0:
            continue
        onset = int(trans[0])
        pre = int((t < onset).sum())
        post = int((t > onset).sum())
        cost = float(track["dtw_cost"].iloc[0])
        cell_info[uid] = {
            "onset": onset,
            "pre": pre,
            "post": post,
            "cost": cost,
            "dataset_id": track["dataset_id"].iloc[0],
        }
    return cell_info


def _compute_organelle_distances(
    alignments: pd.DataFrame,
    config: dict,
    cell_info: dict,
    min_pre: int = 5,
    min_post: int = 5,
) -> dict[str, pd.DataFrame]:
    """Compute per-cell organelle embedding distance from early-time baseline.

    Returns
    -------
    dict[str, pd.DataFrame]
        One DataFrame per organelle with columns: cell_uid, t, t_relative_min,
        organelle_distance, distance_zscore, cost.
    """
    import anndata as ad
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import normalize

    emb_patterns = config["embeddings"]
    org_cfg = config["organelle_dynamics"]
    frame_interval = 30  # minutes

    organelle_results = {}
    for org_name, org_info in org_cfg["organelles"].items():
        emb_key = org_info["embedding"]
        emb_pattern = emb_patterns[emb_key]
        ds_ids = org_info["dataset_ids"]

        all_rows = []
        for ds_id in ds_ids:
            ds_cfg = None
            for ds in config["alignment"]["datasets"]:
                if ds["dataset_id"] == ds_id:
                    ds_cfg = ds
                    break
            if ds_cfg is None:
                continue

            matches = glob.glob(str(Path(ds_cfg["pred_dir"]) / emb_pattern))
            if not matches:
                continue
            adata = ad.read_zarr(matches[0])
            fov_pattern = ds_cfg.get("fov_pattern")
            if fov_pattern:
                mask = adata.obs["fov_name"].astype(str).str.contains(fov_pattern, regex=True)
                adata = adata[mask.to_numpy()].copy()

            emb = adata.X
            if hasattr(emb, "toarray"):
                emb = emb.toarray()
            emb = np.asarray(emb, dtype=np.float64)
            emb_norm = normalize(emb, norm="l2")

            obs = adata.obs.copy()
            obs["_iloc"] = np.arange(len(obs))
            obs_lookup = obs.set_index(["fov_name", "track_id", "t"])["_iloc"]
            ds_align = alignments[alignments["dataset_id"] == ds_id]

            for uid, track_align in ds_align.groupby("cell_uid"):
                if uid not in cell_info:
                    continue
                ci = cell_info[uid]
                if ci["pre"] < min_pre or ci["post"] < min_post or not np.isfinite(ci["cost"]):
                    continue

                onset_t = ci["onset"]

                # Per-cell baseline: this cell's own pre-onset frames
                pre_onset = track_align[track_align["t"].astype(int) < onset_t]
                bl_idx = []
                for _, r in pre_onset.iterrows():
                    k = (r["fov_name"], r["track_id"], r["t"])
                    if k in obs_lookup.index:
                        bl_idx.append(obs_lookup[k])
                if len(bl_idx) < 2:
                    continue
                baseline = emb_norm[bl_idx].mean(axis=0, keepdims=True)

                for _, row in track_align.iterrows():
                    key = (row["fov_name"], row["track_id"], row["t"])
                    if key not in obs_lookup.index:
                        continue
                    iloc = obs_lookup[key]
                    dist = cdist(emb_norm[iloc : iloc + 1], baseline, metric="cosine")[0, 0]
                    t_rel = (int(row["t"]) - onset_t) * frame_interval
                    all_rows.append(
                        {
                            "cell_uid": uid,
                            "t": int(row["t"]),
                            "t_relative_min": t_rel,
                            "organelle_distance": dist,
                            "cost": ci["cost"],
                        }
                    )

        org_df = pd.DataFrame(all_rows)
        if len(org_df) > 0:
            bl = org_df[org_df["t_relative_min"] < 0]["organelle_distance"]
            bl_mean, bl_std = bl.mean(), bl.std()
            if bl_std < 1e-10:
                bl_std = 1.0
            org_df["distance_zscore"] = (org_df["organelle_distance"] - bl_mean) / bl_std
            organelle_results[org_name] = org_df
            _logger.info(f"{org_name}: {org_df['cell_uid'].nunique()} tracks (pre>={min_pre}, post>={min_post})")

    return organelle_results


def plot_remodeling_realtime(
    alignments: pd.DataFrame,
    config: dict,
    output_dir: Path,
    min_pre: int = 5,
    min_post: int = 5,
    organelle_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-cell remodeling heatmap aligned to real time relative to transition onset.

    Returns
    -------
    dict[str, pd.DataFrame]
        The organelle distance results (for reuse by other plots).
    """
    cell_info = _get_cell_info(alignments)
    org_cfg = config["organelle_dynamics"]

    if organelle_results is None:
        organelle_results = _compute_organelle_distances(
            alignments,
            config,
            cell_info,
            min_pre=min_pre,
            min_post=min_post,
        )

    # Plot
    fig, axes = plt.subplots(
        len(organelle_results),
        2,
        figsize=(16, 4 * len(organelle_results)),
        gridspec_kw={"width_ratios": [1, 2]},
        squeeze=False,
    )

    time_bins = np.arange(-300, 660, 30)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2

    for i, (org_name, org_df) in enumerate(organelle_results.items()):
        color = org_cfg["organelles"][org_name]["color"]
        label = org_cfg["organelles"][org_name]["label"]

        ax_line = axes[i, 0]
        medians, q25s, q75s = [], [], []
        for j in range(len(time_bins) - 1):
            mask = (org_df["t_relative_min"] >= time_bins[j]) & (org_df["t_relative_min"] < time_bins[j + 1])
            vals = org_df.loc[mask, "distance_zscore"]
            if len(vals) >= 3:
                medians.append(vals.median())
                q25s.append(vals.quantile(0.25))
                q75s.append(vals.quantile(0.75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        ax_line.plot(time_centers / 60, medians, color=color, linewidth=2, label=label)
        ax_line.fill_between(time_centers / 60, q25s, q75s, color=color, alpha=0.2)
        ax_line.axvline(0, color="red", linestyle="--", alpha=0.5, label="transition onset")
        ax_line.axhline(0, color="grey", linestyle=":", alpha=0.3)
        ax_line.set_xlabel("Hours relative to transition onset")
        ax_line.set_ylabel("Remodeling z-score")
        n_tracks = org_df["cell_uid"].nunique()
        ax_line.set_title(f"{label} (n={n_tracks})")
        ax_line.legend(fontsize=8)
        ax_line.set_xlim(-5, 11)

        ax_heat = axes[i, 1]
        track_list, track_costs = [], []
        for uid, track in org_df.groupby("cell_uid"):
            binned = np.full(len(time_bins) - 1, np.nan)
            for j in range(len(time_bins) - 1):
                mask = (track["t_relative_min"] >= time_bins[j]) & (track["t_relative_min"] < time_bins[j + 1])
                vals = track.loc[mask, "distance_zscore"]
                if len(vals) > 0:
                    binned[j] = vals.mean()
            track_list.append(binned)
            track_costs.append(track["cost"].iloc[0])

        order = np.argsort(track_costs)
        matrix = np.array(track_list)[order]

        im = ax_heat.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-2,
            vmax=3,
            interpolation="nearest",
            extent=[time_bins[0] / 60, time_bins[-1] / 60, len(matrix), 0],
        )
        ax_heat.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        fig.colorbar(im, ax=ax_heat, label="z-score", shrink=0.8)
        ax_heat.set_xlabel("Hours relative to transition onset")
        ax_heat.set_ylabel("Tracks (sorted by DTW cost)")
        ax_heat.set_title(f"{label} — per-cell heatmap")

    fig.suptitle(
        f"Organelle embedding distance aligned to sensor PT onset (min {min_pre} pre + {min_post} post frames)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "remodeling_realtime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved remodeling_realtime.png")
    return organelle_results


def plot_montage_with_zscore(
    alignments: pd.DataFrame,
    config: dict,
    data_zarr_path: str,
    output_dir: Path,
    organelle_results: dict[str, pd.DataFrame],
    organelles: list[str] | None = None,
    n_cells: int = 8,
    crop_half: int = 80,
) -> None:
    """Per-cell GFP montage + z-score trajectory for selected organelles.

    For each organelle, generates one figure where each cell gets:
    - Top strip: GFP crops at every-other-frame relative to onset
    - Bottom strip: z-score trajectory line over the same time range

    Parameters
    ----------
    organelles : list[str] or None
        Organelle names to plot (e.g. ["G3BP1", "SEC61"]). None = all.
    """
    import anndata as ad
    import zarr

    cell_info = _get_cell_info(alignments)
    store = zarr.open(data_zarr_path, mode="r")
    org_cfg = config["organelle_dynamics"]

    pred_dir = config["alignment"]["datasets"][0]["pred_dir"]
    sensor_pattern = config["embeddings"]["sensor"]
    sensor_matches = glob.glob(str(Path(pred_dir) / sensor_pattern))
    adata = ad.read_zarr(sensor_matches[0])
    adata.obs_names_make_unique()

    frame_offsets = np.arange(-10, 21, 2)
    ch_idx_map = {"Phase": 0}  # default to 1 (GFP) for organelles

    if organelles is None:
        organelles = [k for k in organelle_results if k != "Phase"]

    for org_name in organelles:
        if org_name not in organelle_results:
            continue
        org_df = organelle_results[org_name]
        org_info = org_cfg["organelles"][org_name]
        color = org_info["color"]
        label = org_info["label"]
        ch_idx = ch_idx_map.get(org_name, 1)
        ch_name = "Phase" if ch_idx == 0 else "GFP"

        # Find best cells: have z-score data and enough frames
        scored_uids = set(org_df["cell_uid"].unique())
        candidates = []
        for uid in scored_uids:
            if uid not in cell_info:
                continue
            ci = cell_info[uid]
            if ci["pre"] < 5 or ci["post"] < 5 or not np.isfinite(ci["cost"]):
                continue
            candidates.append((uid, ci["cost"]))
        candidates.sort(key=lambda x: x[1])
        cell_uids = [c[0] for c in candidates[:n_cells]]

        if not cell_uids:
            _logger.warning(f"No cells for {org_name} montage+zscore")
            continue

        n_rows = len(cell_uids)
        n_cols = len(frame_offsets)
        fig_height = n_rows * 2.0
        fig, axes = plt.subplots(
            n_rows * 2,
            n_cols,
            figsize=(n_cols * 1.0, fig_height),
            gridspec_kw={"height_ratios": [3, 1] * n_rows},
        )
        if axes.ndim == 1:
            axes = axes.reshape(-1, n_cols)

        for cell_idx, uid in enumerate(cell_uids):
            img_row = cell_idx * 2
            line_row = cell_idx * 2 + 1
            ci = cell_info[uid]
            onset_t = ci["onset"]

            ds_align = alignments[(alignments["cell_uid"] == uid)].sort_values("t")
            fov_name = ds_align["fov_name"].iloc[0]
            track_id = int(ds_align["track_id"].iloc[0])

            cell_obs = adata.obs[(adata.obs["fov_name"] == fov_name) & (adata.obs["track_id"] == track_id)].sort_values(
                "t"
            )
            parts = fov_name.split("/")
            img_arr = store[parts[0]][parts[1]][parts[2]]["0"]
            xy_lookup = {int(r["t"]): (int(r["x"]), int(r["y"])) for _, r in cell_obs.iterrows()}

            # z-score trajectory for this cell
            cell_zscore = org_df[org_df["cell_uid"] == uid].sort_values("t_relative_min")
            zscore_t_hrs = cell_zscore["t_relative_min"].to_numpy() / 60
            zscore_vals = cell_zscore["distance_zscore"].to_numpy()

            for col, offset in enumerate(frame_offsets):
                ax_img = axes[img_row, col]
                ax_line = axes[line_row, col]
                t_abs = onset_t + offset
                t_hrs = offset * 0.5

                # Image
                if t_abs in xy_lookup and 0 <= t_abs < img_arr.shape[0]:
                    cx, cy = xy_lookup[t_abs]
                    y0 = max(0, cy - crop_half)
                    y1 = min(img_arr.shape[3], cy + crop_half)
                    x0 = max(0, cx - crop_half)
                    x1 = min(img_arr.shape[4], cx + crop_half)
                    img = np.array(img_arr[t_abs, ch_idx, 0, y0:y1, x0:x1])
                    vmin, vmax = np.percentile(img, [2, 98])
                    if vmax <= vmin:
                        vmax = vmin + 1
                    ax_img.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                else:
                    ax_img.set_facecolor("#f0f0f0")

                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)

                if cell_idx == 0:
                    ax_img.set_title(
                        f"{t_hrs:+.0f}h",
                        fontsize=6,
                        fontweight="bold" if offset == 0 else "normal",
                        color="red" if offset == 0 else "black",
                    )

                # Z-score line — draw full trajectory in each subplot, highlight current timepoint
                ax_line.plot(zscore_t_hrs, zscore_vals, color=color, linewidth=0.8, alpha=0.7)
                ax_line.axhline(0, color="grey", ls=":", lw=0.3)
                ax_line.axvline(0, color="red", ls=":", lw=0.3, alpha=0.5)
                # Highlight current frame
                close = np.abs(zscore_t_hrs - t_hrs) < 0.3
                if close.any():
                    ax_line.scatter(
                        zscore_t_hrs[close],
                        zscore_vals[close],
                        color=color,
                        s=15,
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.3,
                    )
                ax_line.set_ylim(-2, 4)
                ax_line.set_xlim(-6, 11)
                ax_line.set_xticks([])
                ax_line.set_yticks([])
                for spine in ax_line.spines.values():
                    spine.set_visible(False)

                if col == 0:
                    ax_line.set_yticks([-1, 0, 1, 2, 3])
                    ax_line.tick_params(labelsize=4)
                    for spine in [ax_line.spines["left"]]:
                        spine.set_visible(True)

        fig.suptitle(f"{label} — {ch_name} + remodeling z-score (sorted by DTW cost, t=0 = onset)", fontsize=10)
        fig.subplots_adjust(wspace=0.03, hspace=0.05)
        out_path = output_dir / f"montage_zscore_{org_name}_{ch_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        _logger.info(f"Saved {out_path.name} ({n_rows} cells x {n_cols} timepoints)")


def plot_cell_montage_grid(
    alignments: pd.DataFrame,
    config: dict,
    data_zarr_path: str,
    output_dir: Path,
    min_pre: int = 5,
    min_post: int = 5,
    n_cells: int = 20,
    crop_half: int = 80,
) -> None:
    """Cell crop montage grid: rows=cells, cols=fixed real time relative to onset.

    Generates one grid per (organelle well, channel).
    Border color encodes pseudotime (blue/orange/red).
    Top bar encodes organelle annotation (green=noremodel, magenta=remodel).
    """
    import anndata as ad
    import zarr
    from matplotlib.patches import Rectangle

    cell_info = _get_cell_info(alignments)
    store = zarr.open(data_zarr_path, mode="r")

    # Load AnnData for x, y coordinates
    pred_dir = config["alignment"]["datasets"][0]["pred_dir"]
    sensor_pattern = config["embeddings"]["sensor"]
    sensor_matches = glob.glob(str(Path(pred_dir) / sensor_pattern))
    adata = ad.read_zarr(sensor_matches[0])
    adata.obs_names_make_unique()

    # Load annotations for organelle_state overlay
    ann_lookup: dict[tuple[str, int, int], str] = {}
    for ds in config["alignment"]["datasets"]:
        ann_path = ds.get("annotations_path")
        if ann_path:
            ann_df = pd.read_csv(ann_path)
            if "organelle_state" in ann_df.columns:
                for _, r in ann_df.iterrows():
                    if pd.notna(r["organelle_state"]):
                        ann_lookup[(r["fov_name"], int(r["track_id"]), int(r["t"]))] = r["organelle_state"]

    # Every other frame: -10 to +20 step 2 = 16 columns
    frame_offsets = np.arange(-10, 21, 2)

    channel_defs = [
        (0, "Phase"),
        (1, "GFP"),
        (2, "mCherry"),
    ]

    for ds in config["alignment"]["datasets"]:
        ds_id = ds["dataset_id"]
        org_label = ds_id.replace("2025_07_24_", "").replace("2025_07_22_", "")
        well_label = f"{org_label} well (sensor PT)"

        # Pick cells with enough pre+post, sorted by most post-transition data then cost
        ds_align = alignments[alignments["dataset_id"] == ds_id]
        candidates = []
        for uid in ds_align["cell_uid"].unique():
            if uid not in cell_info:
                continue
            ci = cell_info[uid]
            if ci["pre"] < min_pre or ci["post"] < min_post or not np.isfinite(ci["cost"]):
                continue
            pt_max = ds_align[ds_align["cell_uid"] == uid]["pseudotime"].max()
            if pt_max < 1.0:
                continue
            candidates.append((uid, ci["cost"], -(ci["pre"] + ci["post"])))
        candidates.sort(key=lambda x: (x[1], x[2]))
        cell_uids = [c[0] for c in candidates[:n_cells]]

        if not cell_uids:
            _logger.warning(f"No cells for {org_label} after filtering")
            continue

        n_rows = len(cell_uids)
        n_cols = len(frame_offsets)

        for ch_idx, ch_name in channel_defs:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.0, n_rows * 1.0))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            for row, uid in enumerate(cell_uids):
                track = ds_align[ds_align["cell_uid"] == uid].sort_values("t")
                onset_t = cell_info[uid]["onset"]
                fov_name = track["fov_name"].iloc[0]
                track_id = int(track["track_id"].iloc[0])

                cell_obs = adata.obs[
                    (adata.obs["fov_name"] == fov_name) & (adata.obs["track_id"] == track_id)
                ].sort_values("t")

                parts = fov_name.split("/")
                img_arr = store[parts[0]][parts[1]][parts[2]]["0"]

                xy_lookup = {int(r["t"]): (int(r["x"]), int(r["y"])) for _, r in cell_obs.iterrows()}
                pt_lookup = {int(r["t"]): r["pseudotime"] for _, r in track.iterrows()}

                for col, offset in enumerate(frame_offsets):
                    ax = axes[row, col]
                    t_abs = onset_t + offset

                    if t_abs in xy_lookup and 0 <= t_abs < img_arr.shape[0]:
                        cx, cy = xy_lookup[t_abs]
                        y0 = max(0, cy - crop_half)
                        y1 = min(img_arr.shape[3], cy + crop_half)
                        x0 = max(0, cx - crop_half)
                        x1 = min(img_arr.shape[4], cx + crop_half)

                        img = np.array(img_arr[t_abs, ch_idx, 0, y0:y1, x0:x1])
                        vmin, vmax = np.percentile(img, [2, 98])
                        if vmax <= vmin:
                            vmax = vmin + 1
                        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

                        # Pseudotime border color
                        pt = pt_lookup.get(t_abs, -1)
                        if pt == 0.0:
                            bc = "#3498db"
                        elif pt >= 1.0:
                            bc = "#e74c3c"
                        elif pt > 0:
                            bc = "#f39c12"
                        else:
                            bc = "#cccccc"
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_color(bc)
                            spine.set_linewidth(1.5)

                        # Organelle annotation top bar
                        org_state = ann_lookup.get((fov_name, track_id, t_abs))
                        if org_state is not None:
                            bar_color = "#e91e9e" if org_state == "remodel" else "#2ecc71"
                            xlim = ax.get_xlim()
                            bar_width = xlim[1] - xlim[0]
                            bar_height = (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.06
                            ax.add_patch(
                                Rectangle(
                                    (xlim[0], ax.get_ylim()[1]),
                                    bar_width,
                                    bar_height,
                                    facecolor=bar_color,
                                    edgecolor="none",
                                    clip_on=True,
                                    zorder=5,
                                )
                            )
                    else:
                        ax.set_facecolor("#f0f0f0")
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_color("#e0e0e0")
                            spine.set_linewidth(0.5)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    if row == 0:
                        ax.set_title(
                            f"{offset * 0.5:+.0f}h",
                            fontsize=6,
                            fontweight="bold" if offset == 0 else "normal",
                            color="red" if offset == 0 else "black",
                        )

            fig.suptitle(
                f"{well_label} — {ch_name}  |  border: blue=pre orange=transition red=post"
                f"  |  top bar: green=noremodel magenta=remodel  |  t=0 = onset",
                fontsize=8,
            )
            fig.subplots_adjust(wspace=0.03, hspace=0.03)
            out_path = output_dir / f"montage_{org_label}_{ch_name}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            _logger.info(f"Saved {out_path.name} ({n_rows} cells x {n_cols} timepoints)")


def main() -> None:
    """Run diagnostic plots for organelle dynamics results."""
    parser = argparse.ArgumentParser(description="Diagnostic plots for organelle dynamics")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-zarr", default=None, help="Path to source image zarr (overrides config)")
    parser.add_argument("--min-pre", type=int, default=10, help="Min pre-transition frames per cell")
    parser.add_argument("--min-post", type=int, default=10, help="Min post-transition frames per cell")
    parser.add_argument("--n-cells", type=int, default=20, help="Max cells per montage grid")
    parser.add_argument("--alignments", type=str, default=None, help="Path to alignments parquet file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pseudotime_dir = SCRIPT_DIR.parent
    alignments_path = (
        Path(args.alignments)
        if args.alignments
        else pseudotime_dir / "1-align_cells" / "alignments" / "alignments.parquet"
    )
    alignments = pd.read_parquet(alignments_path)

    output_dir = SCRIPT_DIR / "organelle_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(alignments)} rows, {alignments.groupby(['dataset_id', 'fov_name', 'track_id']).ngroups} tracks")

    organelle_results = plot_remodeling_realtime(
        alignments,
        config,
        output_dir,
        min_pre=args.min_pre,
        min_post=args.min_post,
    )

    data_zarr = args.data_zarr or config.get("data_zarr")
    if data_zarr:
        plot_cell_montage_grid(
            alignments,
            config,
            data_zarr,
            output_dir,
            min_pre=args.min_pre,
            min_post=args.min_post,
            n_cells=args.n_cells,
        )
        if organelle_results:
            plot_montage_with_zscore(
                alignments,
                config,
                data_zarr,
                output_dir,
                organelle_results=organelle_results,
                organelles=["SEC61", "G3BP1", "TOMM20", "Phase"],
                n_cells=args.n_cells,
            )
    else:
        print("  (skipping montage grids — no data_zarr in config or --data-zarr)")

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
