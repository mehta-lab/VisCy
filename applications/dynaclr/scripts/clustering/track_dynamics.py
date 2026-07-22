"""Per-track / per-phenotype dynamics probe (heterogeneity-aware go/no-go for straightening).

The population velocity map (``velocity_map.py``) found velocity ⊥ the *global*
pseudotime gradient — but that is exactly what **heterogeneous responses** predict:
if cells split into several asynchronous remodeling phenotypes, there is no single
population drift direction, so the pooled field cancels. That population null does
NOT bear on a *per-track* straightening objective. This script measures the
quantities that actually decide whether directed dynamics exist:

1. **Per-track straightness** — net displacement / path length in the FULL 768-d
   embedding, one value per (fov, track). Drift ≈ 1, random walk ≈ 0. Distribution,
   NOT a pooled mean. This is the headroom question for straightening.
2. **Curvature vs stencil spacing τ** — ``1 − cos(v_t, v_{t+τ})`` for τ ∈ {1,2,3,5}
   averaged per track. Falling with τ ⇒ directed drift the single step is too noisy
   to show (build); flat/high at all τ ⇒ no drift (stop).
3. **Velocity-direction clustering (the "3–4 phenotypes" test)** — cluster tracks by
   their mean unit velocity direction in 768-d; count coherent phenotype bundles and
   report per-cluster straightness.
4. **Biology check** — do the velocity-direction clusters map to distinct human
   ``infection_state``? (contingency + purity). A real phenotype bundle is enriched
   for a fate.

Reuses ``high_dim_velocity`` and ``diffusion_pseudotime`` from ``velocity_map.py``.

Usage
-----
python track_dynamics.py -c zikv_velocity.yml
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from velocity_map import high_dim_velocity

from viscy_utils.cli_utils import format_markdown_table, load_config

TAUS = (1, 2, 3, 5)
MIN_TRACK_LEN = 4  # need ≥4 frames for a τ=1 curvature triple + a net/path ratio
VELOCITY_CLUSTERS = range(2, 7)  # sweep k for the velocity-direction phenotype count


def _track_key(obs: pd.DataFrame) -> np.ndarray:
    """Stable per-track key ``fov_name|track_id`` (lineage_id is absent here)."""
    return (obs["fov_name"].astype(str) + "|" + obs["track_id"].astype(str)).to_numpy()


def per_track_straightness(X: np.ndarray, obs: pd.DataFrame) -> pd.DataFrame:
    """Net-displacement / path-length per track in the full embedding.

    For a track with ordered embeddings ``z_0..z_L``: ``net = ‖z_L − z_0‖``,
    ``path = Σ ‖z_{k+1} − z_k‖``; straightness = net / path ∈ [0, 1]. Drift → 1,
    random walk → ~1/√L. Also returns mean per-track step size for context.
    """
    key = _track_key(obs)
    t = obs["t"].to_numpy(dtype=int)
    rows = []
    for k in pd.unique(key):
        m = key == k
        order = np.argsort(t[m])
        z = X[m][order]
        if len(z) < MIN_TRACK_LEN:
            continue
        steps = np.linalg.norm(np.diff(z, axis=0), axis=1)
        path = float(steps.sum())
        net = float(np.linalg.norm(z[-1] - z[0]))
        rows.append(
            {
                "track": k,
                "length": len(z),
                "net": net,
                "path": path,
                "straightness": net / path if path > 0 else np.nan,
                "mean_step": float(steps.mean()),
            }
        )
    return pd.DataFrame(rows)


def per_track_curvature(X: np.ndarray, obs: pd.DataFrame, taus=TAUS) -> pd.DataFrame:
    """Mean ``1 − cos(v_t, v_{t+τ})`` per track, for each stencil spacing τ.

    ``v_t = z_{t+τ} − z_t`` on the ordered track; curvature is the mean over the
    sliding pair of consecutive velocity vectors. Lower = straighter (more directed).
    A drop as τ grows means directed drift is present but buried under single-step noise.
    """
    key = _track_key(obs)
    t = obs["t"].to_numpy(dtype=int)
    rows = []
    for k in pd.unique(key):
        m = key == k
        order = np.argsort(t[m])
        z = X[m][order]
        rec = {"track": k, "length": len(z)}
        for tau in taus:
            if len(z) < 2 * tau + 1:
                rec[f"curv_tau{tau}"] = np.nan
                continue
            v = z[tau:] - z[:-tau]  # velocity at spacing tau
            v1, v2 = v[:-tau], v[tau:]  # consecutive (non-overlapping) velocities
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            good = (n1 > 0) & (n2 > 0)
            if not good.any():
                rec[f"curv_tau{tau}"] = np.nan
                continue
            cos = (v1[good] * v2[good]).sum(1) / (n1[good] * n2[good])
            rec[f"curv_tau{tau}"] = float(np.mean(1.0 - cos))
        rows.append(rec)
    return pd.DataFrame(rows)


def velocity_direction_clusters(X: np.ndarray, obs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cluster tracks by their MEAN UNIT velocity direction in 768-d.

    One vector per track = the mean of its unit per-frame velocities (direction of
    travel, magnitude-free — robust on the L2-normalized sphere). KMeans over a k
    sweep; pick k by silhouette. Coherent bundles ⇒ heterogeneous directed
    phenotypes; a single blob ⇒ no directional structure.
    """
    src, v = high_dim_velocity(X, obs)
    key = _track_key(obs)
    # mean unit velocity per track
    unit = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    by_track: dict[str, list] = {}
    for i, s in enumerate(src):
        by_track.setdefault(key[s], []).append(unit[i])
    tracks = [k for k, u in by_track.items() if len(u) >= MIN_TRACK_LEN - 1]
    if len(tracks) < max(VELOCITY_CLUSTERS) + 1:
        return pd.DataFrame(columns=["track", "vel_cluster"])
    D = np.vstack([np.mean(by_track[k], axis=0) for k in tracks])
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)  # re-normalize mean direction

    best_k, best_sil, best_lbl = None, -1.0, None
    for k in VELOCITY_CLUSTERS:
        lbl = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(D)
        sil = silhouette_score(D, lbl, metric="cosine")
        if sil > best_sil:
            best_k, best_sil, best_lbl = k, sil, lbl
    print(f"    velocity-direction clustering: best k={best_k} (cosine silhouette={best_sil:.3f})", flush=True)
    return pd.DataFrame({"track": tracks, "vel_cluster": best_lbl})


def biology_check(clusters: pd.DataFrame, straight: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per velocity-cluster: size, mean straightness, and dominant infection_state.

    Maps each track's velocity-direction cluster to the modal human ``infection_state``
    over that track's frames — tests whether a direction bundle is a real fate bundle.
    """
    key = _track_key(obs)
    inf = obs["infection_state"].astype(str).to_numpy()
    labeled = inf != "nan"  # "nan" is a STRING here (unlabeled), not real NaN
    # modal infection_state per track over LABELED frames only; tracks with no
    # labeled frame are "unlabeled" and excluded from purity.
    track_state = {}
    for k in pd.unique(key):
        vals = pd.Series(inf[(key == k) & labeled])
        track_state[k] = vals.mode().iloc[0] if not vals.empty else "unlabeled"
    df = clusters.merge(straight[["track", "straightness", "length"]], on="track", how="left")
    df["infection_state"] = df["track"].map(track_state)
    rows = []
    for c, g in df.groupby("vel_cluster"):
        lab = g[g["infection_state"] != "unlabeled"]
        counts = lab["infection_state"].value_counts()
        rows.append(
            {
                "vel_cluster": int(c),
                "n_tracks": len(g),
                "n_labeled": int(len(lab)),
                "mean_straightness": float(g["straightness"].mean()),
                "dominant_state": counts.index[0] if len(counts) else "n/a",
                "purity": float(counts.iloc[0] / counts.sum()) if len(counts) else np.nan,
                "state_breakdown": ", ".join(f"{k}:{v}" for k, v in counts.items())
                if len(counts)
                else "no labeled tracks",
            }
        )
    return pd.DataFrame(rows).sort_values("vel_cluster")


def analyze(entry: dict, cfg: dict, out_dir: Path) -> dict:
    """Run the four probes for one marker; write CSVs; return summary metrics."""
    marker = entry["marker"]
    seed = cfg.get("random_seed", 42)
    print(f"[{marker}] reading {entry['embeddings']}", flush=True)
    adata = ad.read_zarr(entry["embeddings"])
    adata.obs_names_make_unique()
    obs = adata.obs.reset_index(drop=True)
    X = np.asarray(adata.X, dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)

    straight = per_track_straightness(X, obs)
    curv = per_track_curvature(X, obs)
    straight.to_csv(out_dir / f"{marker}_per_track_straightness.csv", index=False)
    curv.to_csv(out_dir / f"{marker}_per_track_curvature.csv", index=False)

    clusters = velocity_direction_clusters(X, obs, seed)
    bio = pd.DataFrame()
    if not clusters.empty:
        clusters.to_csv(out_dir / f"{marker}_velocity_clusters.csv", index=False)
        bio = biology_check(clusters, straight, obs)
        bio.to_csv(out_dir / f"{marker}_velocity_cluster_biology.csv", index=False)

    curv_means = {tau: float(curv[f"curv_tau{tau}"].mean(skipna=True)) for tau in TAUS}
    summary = {
        "marker": marker,
        "n_tracks": len(straight),
        "straightness_median": float(straight["straightness"].median()),
        "straightness_p90": float(straight["straightness"].quantile(0.90)),
        **{f"curv_tau{tau}": curv_means[tau] for tau in TAUS},
        "curv_drop_1to5": curv_means[1] - curv_means[5],
        "n_vel_clusters": 0 if clusters.empty else int(clusters["vel_cluster"].nunique()),
        "max_cluster_purity": float(bio["purity"].max()) if not bio.empty else np.nan,
    }
    print(f"[{marker}] {summary}", flush=True)
    if not bio.empty:
        print(format_markdown_table(bio.to_dict("records"), headers=list(bio.columns)), flush=True)
    return summary


def main() -> None:
    """Run the per-track / per-phenotype probe for every marker in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg["output_dir"])
    summaries = [analyze(entry, cfg, out_root / entry["marker"] / "track_dynamics") for entry in cfg["datasets"]]

    print("\n" + "=" * 80)
    print("PER-TRACK / PER-PHENOTYPE DYNAMICS SUMMARY")
    print("=" * 80 + "\n")
    cols = [
        "marker",
        "n_tracks",
        "straightness_median",
        "straightness_p90",
        "curv_tau1",
        "curv_tau5",
        "curv_drop_1to5",
        "n_vel_clusters",
        "max_cluster_purity",
    ]
    rows = [{c: s[c] for c in cols} for s in summaries]
    print(format_markdown_table(rows, headers=cols))
    print("\n**Read:** straightness_median≈1/√L → random walk; curv falling from τ1→τ5 → directed drift")
    print("buried under step noise; high max_cluster_purity → velocity bundles = real fate phenotypes.")
    pd.DataFrame(summaries).to_csv(out_root / "track_dynamics_summary.csv", index=False)


if __name__ == "__main__":
    main()
