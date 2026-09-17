"""PHATE + diffusion-pseudotime + velocity map of phenotype flow over time.

Tests whether cells follow an ordered control -> remodel progression (§6):

- **PHATE** layout — preserves trajectory geometry (UMAP fragmented it, giving a
  scrambled time gradient and artifact-dominated arrows; see PLAN §6 build log).
- **Diffusion pseudotime** rooted at the control-well centroid on the PHATE kNN
  graph (annotation-free): per-cell distance-along-the-manifold from control.
- **Velocity** measured from single-cell tracks in the FULL 768-d embedding
  (``X(t+1) - X(t)``), then projected to the PHATE layout scVelo-style (cosine of
  the high-dim velocity against neighbor offsets) — NOT ``PHATE(t+1)-PHATE(t)``,
  which measured layout jitter. Drawn as a grid-averaged streamplot.
- **Validation:** Spearman(pseudotime, real ``t``) — a real progression is
  strongly positive.

Renders per marker a 3-panel figure (perturbation + arrows / pseudotime + stream /
time) and writes pseudotime + velocity CSVs.

Usage
-----
python velocity_map.py -c zikv_velocity.yml
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
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph

from viscy_utils.cli_utils import load_config


def _control_centroids(X: np.ndarray, obs: pd.DataFrame, min_cells: int = 5) -> dict[int, np.ndarray]:
    """Per-timepoint mean embedding of control (uninfected-well) cells.

    Captures the background drift shared by all cells at each timepoint (imaging,
    media, general time-of-experiment / cell-cycle effects). Subtracting it
    isolates the infection-driven component of a cell's motion.
    """
    ctrl = obs["perturbation"].astype(str).to_numpy() == "uninfected"
    t = obs["t"].to_numpy(dtype=int)
    cents = {}
    for tt in np.unique(t[ctrl]):
        m = ctrl & (t == tt)
        if m.sum() >= min_cells:
            cents[int(tt)] = X[m].mean(axis=0)
    return cents


def high_dim_velocity(X: np.ndarray, obs: pd.DataFrame, control_ref: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell velocity in the FULL embedding space, measured from tracks.

    ``v_i = X(cell, t+1) − X(cell, t)`` in the 768-d embedding (not the 2-D
    layout) for every cell that has a next-frame in its track. Computing velocity
    where the biology lives — then projecting to 2-D (``project_velocity``) —
    avoids the layout-jitter that plain ``PHATE(t+1)−PHATE(t)`` suffered.

    If ``control_ref`` is True, each frame's embedding is first referenced to the
    per-timepoint control centroid (``X - control_centroid(t)``) before
    differencing, so shared background drift is removed and only the
    infection-driven displacement remains. A segment is dropped if either of its
    two timepoints lacks a control centroid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``src`` (n_seg,) = row index of the origin cell; ``v`` (n_seg, n_dim) =
        high-dim velocity for each such cell.
    """
    cents = _control_centroids(X, obs) if control_ref else None
    cell = obs["fov_name"].astype(str) + "|" + obs["track_id"].astype(str)
    t = obs["t"].to_numpy(dtype=int)
    idx = {(c, int(tt)): i for i, (c, tt) in enumerate(zip(cell, t, strict=True))}
    src, v = [], []
    for (c, tt), i in idx.items():
        j = idx.get((c, tt + 1))
        if j is None:
            continue
        if cents is None:
            src.append(i)
            v.append(X[j] - X[i])
        elif tt in cents and (tt + 1) in cents:
            src.append(i)
            v.append((X[j] - cents[tt + 1]) - (X[i] - cents[tt]))
    return np.asarray(src), np.asarray(v)


def project_velocity(
    X: np.ndarray, coords: np.ndarray, src: np.ndarray, v: np.ndarray, n_neighbors: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Project high-dim velocity onto the 2-D layout (scVelo ``velocity_embedding``).

    For each origin cell, its 2-D arrow is the neighbor-displacement-weighted mean
    of directions to its kNN, where the weight is the cosine similarity between the
    cell's HIGH-DIM velocity ``v_i`` and the high-dim offset to each neighbor
    (mean-centered, softmax-free correlation kernel). The 2-D arrow thus points
    toward the neighbors the cell is actually moving toward in 768-d — robust to
    the L2-normalized sphere (direction-based, not magnitude-based).

    Returns ``origins`` (n_seg, 2) and 2-D ``arrows`` (n_seg, 2).
    """
    knn = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    origins = coords[src]
    arrows = np.zeros((len(src), 2))
    for k, i in enumerate(src):
        nbr = knn[i].indices
        dX = X[nbr] - X[i]  # high-dim offsets to neighbors
        dP = coords[nbr] - coords[i]  # 2-D offsets to neighbors
        # cosine(v_i, dX) — correlation kernel, mean-centered as in scVelo.
        dX_n = dX / (np.linalg.norm(dX, axis=1, keepdims=True) + 1e-8)
        cos = dX_n @ (v[k] / (np.linalg.norm(v[k]) + 1e-8))
        w = cos - cos.mean()
        arrows[k] = w @ dP
    # scale arrows to a readable common size relative to the layout
    med = np.median(np.linalg.norm(arrows, axis=1))
    if med > 0:
        span = np.hypot(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        arrows *= 0.02 * span / med
    return origins, arrows


def grid_field(origins: np.ndarray, vectors: np.ndarray, n_grid: int = 25):
    """Average velocity vectors onto a regular grid for a streamplot.

    Returns (gx, gy, U, V) with NaN where a grid cell has too few segments.
    """
    xmin, ymin = origins.min(axis=0)
    xmax, ymax = origins.max(axis=0)
    gx = np.linspace(xmin, xmax, n_grid)
    gy = np.linspace(ymin, ymax, n_grid)
    ix = np.clip(np.searchsorted(gx, origins[:, 0]) - 1, 0, n_grid - 1)
    iy = np.clip(np.searchsorted(gy, origins[:, 1]) - 1, 0, n_grid - 1)
    U = np.full((n_grid, n_grid), np.nan)
    V = np.full((n_grid, n_grid), np.nan)
    for a in range(n_grid):
        for b in range(n_grid):
            m = (ix == a) & (iy == b)
            if m.sum() >= 3:  # require a few segments per cell for a stable mean
                U[b, a] = vectors[m, 0].mean()
                V[b, a] = vectors[m, 1].mean()
    return gx, gy, U, V


def diffusion_pseudotime(X: np.ndarray, root_mask: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Graph geodesic distance from the root population, normalized to [0, 1].

    Builds a kNN graph on the high-dim embedding, then takes the shortest-path
    (Dijkstra) distance from every cell to the nearest root (control) cell. This
    is an annotation-free pseudotime rooted at the controls — small = control-like,
    large = far along the manifold from control.
    """
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance", include_self=False)
    graph = graph.maximum(graph.T)  # symmetrize for an undirected geodesic
    roots = np.flatnonzero(root_mask)
    dist = dijkstra(graph, directed=False, indices=roots).min(axis=0)
    finite = dist[np.isfinite(dist)]
    if finite.size:  # disconnected cells -> max finite distance
        dist[~np.isfinite(dist)] = finite.max()
    rng = dist.max() - dist.min()
    return (dist - dist.min()) / rng if rng > 0 else np.zeros_like(dist)


def _clip_arrows(origins: np.ndarray, vectors: np.ndarray, pct: float = 99.0):
    """Drop segments whose length exceeds the ``pct`` percentile (UMAP/PHATE jitter)."""
    mag = np.hypot(vectors[:, 0], vectors[:, 1])
    keep = mag <= np.percentile(mag, pct)
    return origins[keep], vectors[keep]


def make_velocity_map(entry: dict, cfg: dict, out_dir: Path) -> None:
    """Compute + render the PHATE + pseudotime + velocity map for one marker."""
    marker = entry["marker"]
    seed = cfg.get("random_seed", 42)
    adata = ad.read_zarr(entry["embeddings"])
    adata.obs_names_make_unique()
    obs = adata.obs.reset_index(drop=True)
    X = np.asarray(adata.X, dtype=np.float64)
    pert = obs["perturbation"].astype(str).to_numpy()
    tvals = obs["t"].to_numpy(dtype=float)

    print(f"[{marker}] PHATE over {len(obs)} cells ...", flush=True)
    coords = phate.PHATE(n_components=2, random_state=seed, verbose=False).fit_transform(X)

    print(f"[{marker}] diffusion pseudotime rooted at control ...", flush=True)
    ptime = diffusion_pseudotime(X, pert == "uninfected")
    rho, _ = spearmanr(ptime, tvals)

    # control_ref subtracts the per-timepoint control centroid before differencing,
    # removing background drift shared by all cells. Helps entangled channels
    # (organelle) more than the reporter — see PLAN §6 build log.
    control_ref = entry.get("control_ref", cfg.get("control_ref", False))
    ref_tag = " (control-referenced)" if control_ref else ""
    print(f"[{marker}] high-dim velocity{ref_tag} + scVelo projection to PHATE ...", flush=True)
    src, v = high_dim_velocity(X, obs, control_ref=control_ref)
    coherence = float(np.linalg.norm(v.mean(axis=0)) / (np.linalg.norm(v, axis=1).mean() + 1e-9))
    origins, vectors = project_velocity(X, coords, src, v)
    origins, vectors = _clip_arrows(origins, vectors)
    gx, gy, U, V = grid_field(origins, vectors)
    print(
        f"[{marker}] {len(origins)} segments | Spearman(ptime,t)={rho:.3f} | "
        f"global velocity coherence={coherence:.3f}{ref_tag}",
        flush=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel 1: perturbation (control = source) + velocity streamlines.
    ax = axes[0]
    for cond, color in (("uninfected", "tab:blue"), ("infected", "tab:red")):
        m = pert == cond
        ax.scatter(coords[m, 0], coords[m, 1], s=3, c=color, alpha=0.3, label=cond)
    ax.streamplot(gx, gy, U, V, color="k", density=1.3, linewidth=0.8, arrowsize=0.9)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("well condition + velocity field (control = source)")

    # Panel 2: diffusion pseudotime + velocity streamlines.
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=4, c=ptime, cmap="viridis", alpha=0.6)
    ax.streamplot(gx, gy, U, V, color="k", density=1.3, linewidth=0.8, arrowsize=0.9)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="pseudotime (from control)")
    ax.set_title("diffusion pseudotime + velocity")

    # Panel 3: real time (validation: pseudotime should track t).
    ax = axes[2]
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=4, c=tvals, cmap="plasma", alpha=0.6)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="t")
    ax.set_title(f"real time t  (Spearman ptime~t = {rho:.2f})")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{marker} — PHATE phenotype flow (pseudotime rooted at control, velocity from tracks)", fontsize=13)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{marker}_velocity_map.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(
        {
            "fov_name": obs["fov_name"].astype(str),
            "track_id": obs["track_id"],
            "t": tvals,
            "perturbation": pert,
            "phate_0": coords[:, 0],
            "phate_1": coords[:, 1],
            "pseudotime": ptime,
        }
    ).to_csv(out_dir / f"{marker}_pseudotime.csv", index=False)
    print(f"[{marker}] wrote velocity map + pseudotime CSV (Spearman={rho:.3f})", flush=True)


def main() -> None:
    """Build velocity maps for every marker in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg["output_dir"])
    for entry in cfg["datasets"]:
        make_velocity_map(entry, cfg, out_root / entry["marker"] / "velocity")


if __name__ == "__main__":
    main()
