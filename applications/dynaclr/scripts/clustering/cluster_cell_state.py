"""Proof-of-principle: recover a binary cell state from DynaCLR embeddings.

The state is configurable (``label_column`` + ``positive_class`` / ``negative_class``):
e.g. ``infection_state`` (infected vs uninfected) or ``organelle_state`` (remodel vs
noremodel). Per marker, two annotation-frugal arms:

- **KNN (semi-supervised):** train on the sparse human labels
  (``GroupKFold`` by ``fov_name`` for honest metrics), then propagate to every cell.
- **HDBSCAN (unsupervised):** sweep several representations {``X_pca``, UMAP-2D,
  PHATE-2D} and score clusters against the human labels (ARI / NMI / purity).

Also colors low-dimensional embeddings by time (``t`` / ``hours_post_perturbation``)
and quantifies the time-vs-state trend, and saves qualitative single-cell image crops
split by predicted class into ``<positive>/`` and ``<negative>/`` for visual QC. Build
the combined montage from those crops with ``make_sample_montage.py``.

Usage
-----
python cluster_cell_state.py -c zikv_infection_pop.yml
python cluster_cell_state.py -c zikv_remodel_pop.yml
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import umap
from iohub import open_ome_zarr
from scipy.stats import pointbiserialr, spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from viscy_utils.cli_utils import format_markdown_table, load_config

# The two label values, set from config in main(); defaults are the infection task.
POSITIVE_CLASS = "infected"
NEGATIVE_CLASS = "uninfected"
# Time bins (equal-width over the full t range) used for image sampling.
BIN_LABELS = ("early", "early-mid", "mid", "mid-late", "late")


def _as_str_array(series: pd.Series) -> np.ndarray:
    """Materialize an Arrow-backed obs column as a plain numpy object array.

    sklearn indexing raises ``TypeError: only integer scalar arrays can be
    converted to a scalar index`` on Arrow-backed pandas columns, so convert
    through a Python list first.
    """
    return np.asarray(series.astype(str).tolist())


def _labeled_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of cells with a real (non-missing) human label."""
    return np.isin(labels, [POSITIVE_CLASS, NEGATIVE_CLASS])


def run_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    k_grid: list[int],
    n_splits: int,
) -> tuple[list[dict], KNeighborsClassifier, int]:
    """Evaluate KNN with FOV-grouped CV and return the best refit classifier.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings of the labeled subset, shape (n_labeled, n_features).
    labels : np.ndarray
        Human labels of the labeled subset (``infected`` / ``uninfected``).
    groups : np.ndarray
        Group id per labeled cell (``fov_name``) for ``GroupKFold``.
    k_grid : list[int]
        Neighbor counts to sweep.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    tuple[list[dict], KNeighborsClassifier, int]
        Per-k metric rows, the classifier refit on all labeled data at the best
        k (by balanced accuracy), and that best k.
    """
    y_pos = (labels == POSITIVE_CLASS).astype(int)
    cv = GroupKFold(n_splits=n_splits)
    rows = []
    best = None
    for k in k_grid:
        clf = KNeighborsClassifier(n_neighbors=k)
        classes = clf.fit(embeddings, labels).classes_
        pos_idx = list(classes).index(POSITIVE_CLASS)
        # Single CV pass: derive the predicted class from the out-of-fold proba.
        proba = cross_val_predict(clf, embeddings, labels, cv=cv, groups=groups, method="predict_proba")
        pred = classes[proba.argmax(axis=1)]
        bal = balanced_accuracy_score(labels, pred)
        row = {
            "arm": "knn",
            "config": f"k={k}",
            "balanced_accuracy": bal,
            "f1": f1_score(y_pos, (pred == POSITIVE_CLASS).astype(int)),
            "roc_auc": roc_auc_score(y_pos, proba[:, pos_idx]),
            "n_labeled": int(len(labels)),
        }
        rows.append(row)
        if best is None or bal > best[0]:
            best = (bal, k)
    best_k = best[1]
    refit = KNeighborsClassifier(n_neighbors=best_k).fit(embeddings, labels)
    return rows, refit, best_k


def _compute_space(name: str, X: np.ndarray, X_pca: np.ndarray, seed: int) -> np.ndarray:
    """Return the representation matrix for a named clustering space."""
    if name == "X":
        return X
    if name == "X_pca":
        return X_pca
    if name == "umap2d":
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    if name == "phate2d":
        return phate.PHATE(n_components=2, random_state=seed, verbose=False).fit_transform(X)
    raise ValueError(f"Unknown clustering space: {name}")


def _cluster_purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    """Majority-vote purity of clusters against labels, ignoring noise (-1)."""
    keep = clusters != -1
    if not keep.any():
        return 0.0
    correct = 0
    for c in np.unique(clusters[keep]):
        members = labels[keep][clusters[keep] == c]
        vals, counts = np.unique(members, return_counts=True)
        correct += counts.max()
    return correct / keep.sum()


def _well_separation(clusters: np.ndarray, well_infected: np.ndarray) -> tuple[float, float]:
    """Score how well a clustering sorts cells by infected-vs-control WELL condition.

    Uses the experiment's well layout (``perturbation``) as annotation-free
    biological structure: control wells are clean negatives, infected wells a
    mixture. Returns:

    - ``well_separation``: size-weighted mean |cluster infected-well frac − global
      frac| over non-noise clusters, normalized so 0 = no separation, 1 = every
      cluster is well-pure. Rewards clusterings that split control from infected
      at ANY granularity (not just the 2-cluster ARI winner).
    - ``max_infected_frac``: the highest infected-well fraction of any cluster —
      the most infection-enriched cluster found.
    """
    keep = clusters != -1
    if not keep.any():
        return 0.0, 0.0
    c, w = clusters[keep], well_infected[keep]
    global_frac = float(w.mean())
    denom = 2.0 * global_frac * (1.0 - global_frac)  # max achievable weighted MAD
    wmad, max_frac = 0.0, 0.0
    for cid in np.unique(c):
        m = c == cid
        frac = float(w[m].mean())
        wmad += (m.sum() / keep.sum()) * abs(frac - global_frac)
        max_frac = max(max_frac, frac)
    return (wmad / denom if denom > 0 else 0.0), max_frac


def run_hdbscan(
    reps: dict[str, np.ndarray],
    labeled_mask: np.ndarray,
    labels_labeled: np.ndarray,
    grid_mcs: list[int],
    grid_ms: list[int],
    well_infected: np.ndarray,
) -> list[dict]:
    """Sweep HDBSCAN over representations and parameters; score vs labels + wells.

    Parameters
    ----------
    reps : dict[str, np.ndarray]
        Named representation matrices computed on all cells.
    labeled_mask : np.ndarray
        Boolean mask selecting the human-labeled cells (for label scoring only).
    labels_labeled : np.ndarray
        Human labels of the labeled subset.
    grid_mcs, grid_ms : list[int]
        ``min_cluster_size`` / ``min_samples`` values to sweep.
    well_infected : np.ndarray
        Per-cell bool: True if the cell is from an infected well (``perturbation``).
        Annotation-free biological structure for cluster enrichment scoring.

    Returns
    -------
    list[dict]
        One metric row per (space, min_cluster_size, min_samples), including both
        label-based (ari/nmi/purity) and well-based (well_separation,
        max_infected_frac) scores.
    """
    rows = []
    for space, mat in reps.items():
        for mcs in grid_mcs:
            for ms in grid_ms:
                clusters = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(mat)
                cl_lab = clusters[labeled_mask]
                n_clusters = int(len(set(clusters)) - (1 if -1 in clusters else 0))
                well_sep, max_inf = _well_separation(clusters, well_infected)
                rows.append(
                    {
                        "arm": "hdbscan",
                        "config": f"{space} mcs={mcs} ms={ms}",
                        "ari": adjusted_rand_score(labels_labeled, cl_lab),
                        "nmi": normalized_mutual_info_score(labels_labeled, cl_lab),
                        "purity": _cluster_purity(cl_lab, labels_labeled),
                        "well_separation": well_sep,
                        "max_infected_frac": max_inf,
                        "n_clusters": n_clusters,
                        "noise_frac": float((clusters == -1).mean()),
                    }
                )
    return rows


def time_correlations(
    t: np.ndarray,
    hpp: np.ndarray,
    labeled_mask: np.ndarray,
    labels_labeled: np.ndarray,
    knn_proba_pos: np.ndarray,
) -> list[dict]:
    """Correlate time against human label (labeled subset) and KNN proba (all cells)."""
    y_pos = (labels_labeled == POSITIVE_CLASS).astype(int)
    rows = []
    for tname, tvals in (("t", t), ("hours_post_perturbation", hpp)):
        pb = pointbiserialr(y_pos, tvals[labeled_mask])
        sp = spearmanr(tvals, knn_proba_pos)
        rows.append(
            {
                "time_col": tname,
                "pointbiserial_vs_human_label": pb.statistic,
                "spearman_vs_knn_proba": sp.statistic,
            }
        )
    return rows


def _category_color(category: str, index: int) -> str:
    """Stable color for a category so a label reads the same across every panel.

    Positive class -> red, negative -> blue, missing/noise -> gray; anything else
    (e.g. HDBSCAN cluster ids) falls back to tab10 by index. Reads the module
    class globals so it tracks whatever ``main()`` bound from config.
    """
    fixed = {POSITIVE_CLASS: "tab:red", NEGATIVE_CLASS: "tab:blue", "nan": "lightgray", "-1": "lightgray"}
    if category in fixed:
        return fixed[category]
    return plt.cm.tab10(index % 10)


def plot_scatter(coords: np.ndarray, color_by: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    """Save a multi-panel 2D scatter, one panel per coloring.

    Categorical panels use a fixed label -> color map (``CLASS_COLORS``) so the
    same class is colored identically in every panel regardless of how many
    categories that panel happens to contain.
    """
    n = len(color_by)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (name, values) in zip(axes[0], color_by.items(), strict=True):
        if values.dtype.kind in "OU":
            categories = sorted(set(values.tolist()))
            color_map = {c: _category_color(c, i) for i, c in enumerate(categories)}
            point_colors = [color_map[v] for v in values]
            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=2, alpha=0.5)
            handles = [plt.Line2D([], [], marker="o", ls="", color=color_map[c]) for c in categories]
            ax.legend(handles, categories, fontsize=6, markerscale=1.5)
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=2, alpha=0.5)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Channels shown for each sampled cell (biological context), matching the
# witness/prob-sample figures. (display label, zarr channel, reduction). Absent
# channels are skipped per dataset. Phase = per-FOV focus slice; fluor = full MIP.
COMPANION_CHANNELS = [
    ("phase", "Phase3D", "center"),
    ("organelle", "raw GFP EX488 EM525-45", "mip"),
    ("sensor", "raw mCherry EX561 EM600-37", "mip"),
]


def _focus_z(pos, n_z: int) -> int:
    """Per-FOV focus slice from zattrs (``focus_slice.<ch>.fov_statistics``), else mid-stack."""
    fs = dict(pos.zattrs).get("focus_slice", {})
    for ch in fs.values():
        mean = ch.get("fov_statistics", {}).get("z_focus_mean")
        if mean is not None:
            return int(np.clip(round(mean), 0, n_z - 1))
    return n_z // 2


def _crop_channel(pos, ch_idx: int, t: int, y: int, x: int, half: int, reduction: str) -> np.ndarray:
    """Crop patch_size around (y,x); ``center`` = per-FOV focus slice, else full-stack MIP."""
    arr = pos.data  # (T, C, Z, Y, X)
    ymax, xmax = arr.shape[-2], arr.shape[-1]
    y0, x0 = max(0, y - half), max(0, x - half)
    y1, x1 = min(ymax, y0 + 2 * half), min(xmax, x0 + 2 * half)
    y0, x0 = y1 - 2 * half, x1 - 2 * half
    stack = np.asarray(arr[t, ch_idx, :, y0:y1, x0:x1])  # (Z, Y, X)
    if reduction == "center":
        return stack[_focus_z(pos, stack.shape[0])]
    return stack.max(axis=0)


def _norm01(a: np.ndarray) -> np.ndarray:
    """Min-max normalize a crop to [0, 1] for display (per-crop contrast)."""
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=float)


def save_image_samples(
    obs: pd.DataFrame,
    knn_pred: np.ndarray,
    knn_proba_pos: np.ndarray,
    image_zarr: str,
    n_samples: int,
    patch_size: int,
    out_dir: Path,
) -> None:
    """Crop and save top-confidence KNN-predicted cells across 5 time bins.

    For each class, the full ``t`` range is split into ``len(BIN_LABELS)`` bins;
    within each bin the highest-confidence cells of that class are chosen, so the
    montage spans early -> late. ``n_samples`` is divided evenly across bins.

    Each cell is saved once **per available companion channel** (phase / organelle
    / sensor) under ``<class>/<channel_label>/``, filename prefixed
    ``bin{idx}_{label}_`` so the montage step can group by (bin, channel). Phase =
    per-FOV focus slice; fluorescence = full-stack MIP. Channels absent from the
    zarr are skipped. The channel set comes from COMPANION_CHANNELS, so the
    marker's own ``channel`` no longer needs to be passed.
    """
    half = patch_size // 2
    n_bins = len(BIN_LABELS)
    per_bin = max(1, n_samples // n_bins)
    t_all = obs["t"].to_numpy(dtype=float)
    edges = np.linspace(t_all.min(), t_all.max(), n_bins + 1)
    # Right-closed on the last edge so the max-t cells fall in the final bin.
    bin_of = np.clip(np.digitize(t_all, edges[1:-1]), 0, n_bins - 1)
    with open_ome_zarr(image_zarr, mode="r") as plate:
        available = set(plate.channel_names)
        channels = [(lbl, ch, red) for (lbl, ch, red) in COMPANION_CHANNELS if ch in available]
        for cls, want_pos in ((POSITIVE_CLASS, True), (NEGATIVE_CLASS, False)):
            conf = knn_proba_pos if want_pos else 1.0 - knn_proba_pos
            for b, label in enumerate(BIN_LABELS):
                sel = np.flatnonzero((knn_pred == cls) & (bin_of == b))
                chosen = sel[np.argsort(conf[sel])[::-1][:per_bin]]
                for i in chosen:
                    r = obs.iloc[int(i)]
                    fov = str(r["fov_name"])
                    t = int(r["t"])
                    y, x = int(r["y"]), int(r["x"])
                    pos = plate[fov]
                    name = f"bin{b}_{label}_{fov.replace('/', '_')}_track{int(r['track_id'])}_t{t}.png"
                    for ch_lbl, ch, red in channels:
                        ch_dir = out_dir / cls / ch_lbl
                        ch_dir.mkdir(parents=True, exist_ok=True)
                        crop = _crop_channel(pos, plate.channel_names.index(ch), t, y, x, half, red)
                        plt.imsave(ch_dir / name, _norm01(crop), cmap="gray")


def save_cluster_samples(
    obs: pd.DataFrame,
    clusters: np.ndarray,
    coords: np.ndarray,
    well_infected: np.ndarray,
    image_zarr: str,
    n_samples: int,
    patch_size: int,
    out_dir: Path,
    top_k: int = 5,
) -> None:
    """Crop representative cells for the top-K most infection-enriched clusters.

    Clusters are ranked by **infected-well fraction** (§5, annotation-free) and the
    ``top_k`` most enriched are kept — the candidate infected/remodeled phenotypes.
    Within a cluster the cells closest to the cluster centroid in the clustering
    space (``coords``) are the most representative. Each cell is saved once **per
    available companion channel** (phase / organelle / sensor) under
    ``rank{r}_cluster{id}_inf{frac}/<channel>/`` so the montage can order clusters
    by enrichment and stack channels. Noise (``-1``) is skipped.
    """
    half = patch_size // 2
    ids = [c for c in sorted(set(clusters)) if c != -1]
    enrich = {c: float(well_infected[clusters == c].mean()) for c in ids}
    top = sorted(ids, key=lambda c: enrich[c], reverse=True)[:top_k]
    with open_ome_zarr(image_zarr, mode="r") as plate:
        available = set(plate.channel_names)
        channels = [(lbl, ch, red) for (lbl, ch, red) in COMPANION_CHANNELS if ch in available]
        for rank_c, cid in enumerate(top):
            members = np.flatnonzero(clusters == cid)
            centroid = coords[members].mean(axis=0)
            order = np.argsort(((coords[members] - centroid) ** 2).sum(axis=1))
            chosen = members[order[:n_samples]]
            cl_dir = out_dir / f"rank{rank_c}_cluster{cid}_inf{enrich[cid]:.2f}"
            for i in chosen:
                r = obs.iloc[int(i)]
                fov = str(r["fov_name"])
                t = int(r["t"])
                y, x = int(r["y"]), int(r["x"])
                pos = plate[fov]
                name = f"{fov.replace('/', '_')}_track{int(r['track_id'])}_t{t}.png"
                for ch_lbl, ch, red in channels:
                    ch_dir = cl_dir / ch_lbl
                    ch_dir.mkdir(parents=True, exist_ok=True)
                    crop = _crop_channel(pos, plate.channel_names.index(ch), t, y, x, half, red)
                    plt.imsave(ch_dir / name, _norm01(crop), cmap="gray")


def process_marker(entry: dict, cfg: dict, out_root: Path) -> dict:
    """Run both arms + plots + samples for one marker; return summary rows."""
    marker = entry["marker"]
    out_dir = out_root / marker
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["random_seed"]

    adata = ad.read_zarr(entry["embeddings"])
    adata.obs_names_make_unique()
    obs = adata.obs
    X = np.asarray(adata.X, dtype=np.float64)
    X_pca = np.asarray(obs_pca) if (obs_pca := adata.obsm.get("X_pca")) is not None else X

    labels = _as_str_array(obs[cfg["label_column"]])
    groups = _as_str_array(obs[cfg["group_column"]])
    baseline = _as_str_array(obs[cfg["baseline_column"]])
    t = obs["t"].to_numpy(dtype=float)
    hpp = obs["hours_post_perturbation"].to_numpy(dtype=float)
    # Well condition (annotation-free biological structure, §5): control wells are
    # clean negatives; infected wells a mixture. `perturbation` == "infected".
    perturbation = _as_str_array(obs["perturbation"])
    well_infected = perturbation == "infected"

    lm = _labeled_mask(labels)
    labels_lab = labels[lm]
    print(f"### {marker}: {lm.sum()} / {len(labels)} human-labeled cells", flush=True)

    # --- Supervised KNN arm (metrics + propagation) ---
    print(f"[{marker}] KNN grouped-CV ...", flush=True)
    knn_rows, refit, best_k = run_knn(X[lm], labels_lab, groups[lm], cfg["knn"]["k_grid"], cfg["knn"]["n_splits"])
    knn_pred = refit.predict(X)
    pos_idx = list(refit.classes_).index(POSITIVE_CLASS)
    knn_proba_pos = refit.predict_proba(X)[:, pos_idx]

    # Well-stratified eval: accuracy of the propagated labels within control wells
    # (where truth is ~all-negative) vs infected wells (the hard, mixed case).
    ctrl = ~well_infected
    knn_rows.append(
        {
            "arm": "knn_well_stratified",
            "config": "control_well (frac predicted positive)",
            "balanced_accuracy": float((knn_pred[ctrl] == POSITIVE_CLASS).mean()),
            "n_labeled": int(ctrl.sum()),
        }
    )
    knn_rows.append(
        {
            "arm": "knn_well_stratified",
            "config": "infected_well (frac predicted positive)",
            "balanced_accuracy": float((knn_pred[well_infected] == POSITIVE_CLASS).mean()),
            "n_labeled": int(well_infected.sum()),
        }
    )

    # Control-negatives KNN (annotation-free): negatives = control-well cells,
    # positives = human-positive cells. Grouped-CV, reported alongside the
    # human-label KNN. Helps the imbalanced remodel task (few human positives).
    cn_pos = well_infected & (labels == POSITIVE_CLASS)
    cn_mask = ctrl | cn_pos
    if cn_pos.sum() >= cfg["knn"]["n_splits"] and ctrl.sum() >= cfg["knn"]["n_splits"]:
        cn_labels = np.where(cn_pos, POSITIVE_CLASS, NEGATIVE_CLASS)[cn_mask]
        cn_rows, _, _ = run_knn(X[cn_mask], cn_labels, groups[cn_mask], cfg["knn"]["k_grid"], cfg["knn"]["n_splits"])
        for r in cn_rows:
            r["arm"] = "knn_control_neg"
        knn_rows.extend(cn_rows)

    # Baseline row: existing linear classifier vs human labels on the same subset.
    baseline_row = {
        "arm": "baseline_lc",
        "config": cfg["baseline_column"],
        "balanced_accuracy": balanced_accuracy_score(labels_lab, baseline[lm]),
        "f1": f1_score((labels_lab == POSITIVE_CLASS).astype(int), (baseline[lm] == POSITIVE_CLASS).astype(int)),
        "n_labeled": int(lm.sum()),
    }

    # --- Unsupervised HDBSCAN arm ---
    print(f"[{marker}] computing spaces {cfg['hdbscan']['spaces']} (UMAP/PHATE ~1-2 min each) ...", flush=True)
    reps = {name: _compute_space(name, X, X_pca, seed) for name in cfg["hdbscan"]["spaces"]}
    print(f"[{marker}] HDBSCAN sweep ...", flush=True)
    hdb_rows = run_hdbscan(
        reps, lm, labels_lab, cfg["hdbscan"]["min_cluster_size"], cfg["hdbscan"]["min_samples"], well_infected
    )

    # --- Time evidence ---
    time_rows = time_correlations(t, hpp, lm, labels_lab, knn_proba_pos)
    pd.DataFrame(time_rows).to_csv(out_dir / "time_correlation.csv", index=False)

    # --- Plots (UMAP + PHATE), colored by every signal ---
    # Select the montaged/plotted clustering by WELL SEPARATION (annotation-free,
    # §5) among USABLE configs — this avoids the ARI-vs-binary-label collapse to
    # ~2 clusters and surfaces infection-enriched structure, while a noise cap and
    # cluster-count cap keep the pick displayable (a 72-cluster / 80%-noise config
    # can win raw well_separation via many tiny pure clusters). Progressive
    # fallback: usable → any ≥2-cluster → best ARI.
    sel_cfg = cfg.get("hdbscan", {})
    max_noise = sel_cfg.get("select_max_noise_frac", 0.5)
    max_k = sel_cfg.get("select_max_clusters", 20)
    usable = [r for r in hdb_rows if 2 <= r["n_clusters"] <= max_k and r["noise_frac"] <= max_noise]
    multi = [r for r in hdb_rows if r["n_clusters"] >= 2]
    if usable:
        best_hdb = max(usable, key=lambda r: r["well_separation"])
    elif multi:
        best_hdb = max(multi, key=lambda r: r["well_separation"])
    else:
        best_hdb = max(hdb_rows, key=lambda r: r["ari"])
    best_space = best_hdb["config"].split()[0]
    best_clusters = HDBSCAN(
        min_cluster_size=int(best_hdb["config"].split("mcs=")[1].split()[0]),
        min_samples=int(best_hdb["config"].split("ms=")[1]),
    ).fit_predict(reps[best_space])
    for space in ("umap2d", "phate2d"):
        coords = reps.get(space)
        if coords is None:
            coords = _compute_space(space, X, X_pca, seed)
        color_by = {
            "human_label": labels,
            "predicted_lc": baseline,
            "knn_propagated": knn_pred,
            "hdbscan_cluster": best_clusters.astype(str),
            "t": t,
            "hours_post_perturbation": hpp,
        }
        plot_scatter(coords, color_by, out_dir / f"scatter_{space}.png", f"{marker} — {space}")

    # Persist the best HDBSCAN assignment per cell so downstream steps (e.g. the
    # cluster montage) need not recompute clustering. Space recorded in the header.
    pd.DataFrame(
        {
            "fov_name": obs["fov_name"].astype(str).to_numpy(),
            "track_id": obs["track_id"].to_numpy(),
            "t": obs["t"].to_numpy(),
            "hdbscan_cluster": best_clusters,
            "cluster_space": best_space,
        }
    ).to_csv(out_dir / "cluster_assignments.csv", index=False)

    # --- Qualitative image samples ---
    print(f"[{marker}] cropping image samples ...", flush=True)
    save_image_samples(
        obs,
        knn_pred,
        knn_proba_pos,
        entry["image_zarr"],
        cfg["samples"]["n_samples"],
        cfg["samples"]["patch_size"],
        out_dir / "samples",
    )

    # Representative crops for the top-K most infection-enriched HDBSCAN clusters
    # (multi-channel), so the candidate infected phenotypes can be inspected.
    print(f"[{marker}] cropping cluster samples ...", flush=True)
    save_cluster_samples(
        obs,
        best_clusters,
        reps[best_space],
        well_infected,
        entry["image_zarr"],
        cfg["samples"]["n_samples"],
        cfg["samples"]["patch_size"],
        out_dir / "cluster_samples",
        top_k=cfg["samples"].get("top_clusters", 5),
    )

    for r in knn_rows + [baseline_row] + hdb_rows:
        r["marker"] = marker
    return {"metrics": knn_rows + [baseline_row] + hdb_rows, "knn_best_k": best_k, "best_hdb": best_hdb}


def main() -> None:
    """Parse config, run every marker, and write the combined summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Bind the two label values from config (defaults keep the infection task).
    global POSITIVE_CLASS, NEGATIVE_CLASS
    POSITIVE_CLASS = cfg.get("positive_class", POSITIVE_CLASS)
    NEGATIVE_CLASS = cfg.get("negative_class", NEGATIVE_CLASS)

    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for entry in cfg["datasets"]:
        result = process_marker(entry, cfg, out_root)
        all_metrics.extend(result["metrics"])

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(out_root / "metrics_summary.csv", index=False)

    headers = [
        "marker",
        "arm",
        "config",
        "balanced_accuracy",
        "roc_auc",
        "ari",
        "nmi",
        "purity",
        "well_separation",
        "max_infected_frac",
        "n_clusters",
    ]
    md = format_markdown_table(
        [{h: row.get(h, "") for h in headers} for row in all_metrics],
        title=cfg.get("title", f"{cfg['label_column']} clustering — proof of principle"),
        headers=headers,
    )
    (out_root / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
