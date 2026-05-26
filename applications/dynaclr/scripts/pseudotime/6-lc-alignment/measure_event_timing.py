"""Per-cell organelle event timing, anchored to the viral_sensor LC.

Stage B of the LC-registered event-timing DAG (see
``.ed_planning/dynaclr/DAG/lc_registered_event_timing.md``).

For each (dataset, organelle channel) entry in the recipe, this script:

1. Loads the organelle embedding zarr + Stage A registration parquet.
2. Restricts to cells with ``crossed_tau == True`` (have a valid t_perturb).
3. Computes the event time T_event per cell via two independent paths:

   **Path 1 (embedding cosine distance, two baselines)**
   - ``cell_own``:        mean of cell's first ``n_init_frames`` embeddings.
   - ``uninfected_pop``:  mean of organelle embeddings of cells with
                          ``perturbation == 'uninfected'`` in the same dataset.
   For each baseline: ``d(t) = cos_dist(organelle(t), baseline)``,
   normalize per cell to [0, 1] by dividing by the cell's observed max.
   For each tau in ``tau_sweep``: ``T_50 = first t where d_norm >= tau``.

   **Path 2 (organelle LC threshold)**
   - For each ``label_column`` in ``label_columns`` (default
     ``[organelle_state]``):
     If the column carries categorical labels (``remodel`` / ``noremodel``)
     a logistic regression is trained group-aware on it. If the column is
     a predicted label (e.g., ``predicted_organelle_state``) it's read
     directly and converted to a 0/1 indicator. In both cases
     ``T_LC = first t where p_remodel >= tau_lc`` (default 0.5).

4. Computes ``delta_t = T_event - t_perturb`` per cell, in frames and
   minutes (via ``hours_post_perturbation`` on obs).
5. Emits one parquet row per cell with every (path, baseline, tau,
   label_column) combination side-by-side, plus a ``timing_summary.csv``
   aggregating across cells.

Recipe layout mirrors the rest of 6-lc-alignment (see DAG doc Stage B for
the schema).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("measure_event_timing")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge ``base:`` recipe imports; resolve per-dataset zarr/parquet paths."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        leaf = yaml.safe_load(f) or {}
    merged: dict = {}
    for rel in leaf.pop("base", []):
        with open((config_path.parent / rel).resolve()) as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(leaf)
    org_dir = merged.get("organelle_embedding_dir")
    reg_dir = merged.get("registration_dir")
    for d in merged.get("datasets") or []:
        if "organelle_embedding_zarr" not in d and "organelle_embedding_filename" in d:
            d["organelle_embedding_zarr"] = str(Path(org_dir) / d["organelle_embedding_filename"])
        if "registration_parquet" not in d and "registration_filename" in d:
            d["registration_parquet"] = str(Path(reg_dir) / d["registration_filename"])
    return merged


@dataclass
class OrganelleData:
    """Per-dataset organelle embeddings + joined registration anchors + obs."""

    name: str
    obs: pd.DataFrame
    embeddings: np.ndarray  # (N, D)
    uninfected_embeddings: np.ndarray | None  # (M, D) or None if no uninfected well
    marker: str
    organelle: str


def _load_dataset(
    name: str,
    organelle_zarr: Path,
    registration_parquet: Path,
    marker: str,
    organelle: str,
) -> OrganelleData:
    """Load organelle embeddings, attach Stage A anchors, split uninfected pool.

    The annotation columns we use for Path 2 live on the zarr's ``obs``
    directly (``organelle_state``, ``predicted_organelle_state``, etc.), so
    no annotation CSV is required at this stage.
    """
    _logger.info("[%s] loading %s", name, organelle_zarr.name)
    a = ad.read_zarr(organelle_zarr)
    obs = a.obs.copy().reset_index(drop=True)
    obs["track_id"] = obs["track_id"].astype(int)
    obs["t"] = obs["t"].astype(int)
    obs["_iloc"] = np.arange(len(obs))
    obs = obs.drop_duplicates(subset=["fov_name", "track_id", "t"]).reset_index(drop=True)
    X = a.X[obs["_iloc"].to_numpy()]
    if hasattr(X, "toarray"):
        X = X.toarray()
    obs = obs.drop(columns=["_iloc"])

    # Split uninfected population for Path 1 baseline
    if "perturbation" in obs.columns:
        is_uninf = obs["perturbation"].astype(str) == "uninfected"
        uninf_X = X[is_uninf.to_numpy()].astype(np.float32) if is_uninf.any() else None
        if uninf_X is None:
            _logger.warning("[%s] no uninfected rows on obs; cell_own baseline only", name)
        else:
            _logger.info("[%s] %d uninfected rows for population baseline", name, len(uninf_X))
    else:
        _logger.warning("[%s] obs lacks 'perturbation' column; cell_own baseline only", name)
        uninf_X = None

    # Join Stage A registration on (fov_name, track_id)
    reg = pd.read_parquet(registration_parquet)
    reg["track_id"] = reg["track_id"].astype(int)
    reg = reg[["fov_name", "track_id", "t_LC_star", "crossed_tau", "p_max"]]
    merged = obs.merge(reg, on=["fov_name", "track_id"], how="left")

    keep = merged["crossed_tau"].fillna(False).to_numpy().astype(bool)
    _logger.info("[%s] %d/%d frames in LC-registered cohort", name, int(keep.sum()), len(merged))
    obs = merged.loc[keep].reset_index(drop=True)
    X = X[keep].astype(np.float32)
    return OrganelleData(
        name=name, obs=obs, embeddings=X, uninfected_embeddings=uninf_X, marker=marker, organelle=organelle
    )


def _make_groups(obs: pd.DataFrame) -> np.ndarray:
    """Lineage-aware group ids when parent_track_id is present, else track-level."""
    if "parent_track_id" not in obs.columns:
        return (obs["fov_name"].astype(str) + "/" + obs["track_id"].astype(str)).to_numpy()
    p_map = (
        obs[["fov_name", "track_id", "parent_track_id"]]
        .drop_duplicates(["fov_name", "track_id"])
        .set_index(["fov_name", "track_id"])["parent_track_id"]
        .to_dict()
    )

    def root(fov: str, tid: int) -> tuple[str, int]:
        cur = (fov, int(tid))
        seen: set[tuple[str, int]] = set()
        while cur in p_map and pd.notna(p_map[cur]) and int(p_map[cur]) >= 0:
            if cur in seen:
                break
            seen.add(cur)
            parent_tid = int(p_map[cur])
            if (fov, parent_tid) not in p_map:
                break
            cur = (fov, parent_tid)
        return cur

    roots = [root(r.fov_name, r.track_id) for r in obs.itertuples()]
    return np.array([f"{f}/{t}" for f, t in roots])


def _cosine_distance(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-row cosine distance ``1 - cos_sim(X[i], ref)`` (ref is 1D)."""
    Xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    rn = float(np.linalg.norm(ref) + 1e-12)
    return 1.0 - (X @ ref) / (Xn.ravel() * rn)


def _path1_per_track_t50(
    obs: pd.DataFrame,
    X: np.ndarray,
    uninfected_pool: np.ndarray | None,
    n_init_frames: int,
    tau_sweep: list[float],
) -> dict[tuple[str, float], pd.DataFrame]:
    """For each (baseline, tau), compute per-track T_50 + Δt placeholder.

    Returns a dict keyed by ``(baseline_name, tau)`` mapping to a DataFrame
    with one row per (fov_name, track_id): ``T_50_frames`` (NaN if
    threshold never crossed), ``detected`` boolean.
    """
    # Pre-compute cell_own and uninfected_pop baselines per track in one pass
    out: dict[tuple[str, float], list[dict]] = {("cell_own", t): [] for t in tau_sweep}
    if uninfected_pool is not None and len(uninfected_pool) > 0:
        for t in tau_sweep:
            out[("uninfected_pop", t)] = []
        pop_baseline = uninfected_pool.mean(axis=0).astype(np.float32)
    else:
        pop_baseline = None

    obs_idx = obs.reset_index(drop=False).rename(columns={"index": "_row"})
    for (fov, tid), grp in obs_idx.groupby(["fov_name", "track_id"], sort=False):
        grp_sorted = grp.sort_values("t")
        rows = grp_sorted["_row"].to_numpy()
        t_vals = grp_sorted["t"].to_numpy()
        X_track = X[rows]
        n = len(rows)
        if n == 0:
            continue

        # cell_own baseline: mean of first n_init_frames embeddings
        k = min(n_init_frames, n)
        cell_own_ref = X_track[:k].mean(axis=0)
        d_own = _cosine_distance(X_track, cell_own_ref)
        d_max = float(d_own.max()) if d_own.size else 0.0
        d_norm = d_own / d_max if d_max > 0 else np.zeros_like(d_own)
        for tau in tau_sweep:
            crossed = d_norm >= tau
            if crossed.any():
                T_50 = int(t_vals[int(np.argmax(crossed))])
                det = True
            else:
                T_50 = np.nan
                det = False
            out[("cell_own", tau)].append(
                {"fov_name": fov, "track_id": int(tid), "T_50_frames": T_50, "detected": det, "d_max_raw": d_max}
            )

        # uninfected_pop baseline
        if pop_baseline is not None:
            d_pop = _cosine_distance(X_track, pop_baseline)
            d_pop_max = float(d_pop.max()) if d_pop.size else 0.0
            d_pop_norm = d_pop / d_pop_max if d_pop_max > 0 else np.zeros_like(d_pop)
            for tau in tau_sweep:
                crossed = d_pop_norm >= tau
                if crossed.any():
                    T_50 = int(t_vals[int(np.argmax(crossed))])
                    det = True
                else:
                    T_50 = np.nan
                    det = False
                out[("uninfected_pop", tau)].append(
                    {
                        "fov_name": fov,
                        "track_id": int(tid),
                        "T_50_frames": T_50,
                        "detected": det,
                        "d_max_raw": d_pop_max,
                    }
                )

    return {k: pd.DataFrame(v) for k, v in out.items()}


def _train_organelle_lc(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float, random_state: int
) -> tuple[StandardScaler, LogisticRegression, np.ndarray]:
    """Group-aware 80/20 split, fit LR (same recipe as register_by_lc.py).

    Returns scaler, classifier, and a boolean ``in_train`` mask for the
    inputs. The caller scores the entire input array, so train-row scores
    are present but ``in_train`` allows downstream filtering if needed.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, _test_idx = next(gss.split(X, y, groups=groups))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[train_idx])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(Xs, y[train_idx])
    in_train = np.zeros(len(y), dtype=bool)
    in_train[train_idx] = True
    return scaler, clf, in_train


def _path2_per_track_t_lc(obs: pd.DataFrame, p_remodel: np.ndarray, tau_lc: float) -> pd.DataFrame:
    """Per track, T_LC = first frame where ``p_remodel >= tau_lc``."""
    work = obs[["fov_name", "track_id", "t"]].copy()
    work["p"] = p_remodel
    work = work.sort_values(["fov_name", "track_id", "t"]).reset_index(drop=True)
    rows: list[dict] = []
    for (fov, tid), grp in work.groupby(["fov_name", "track_id"], sort=False):
        t_vals = grp["t"].to_numpy()
        p_vals = grp["p"].to_numpy()
        crossed = p_vals >= tau_lc
        if crossed.any():
            T_LC = int(t_vals[int(np.argmax(crossed))])
            det = True
        else:
            T_LC = np.nan
            det = False
        rows.append(
            {
                "fov_name": fov,
                "track_id": int(tid),
                "T_LC_frames": T_LC,
                "detected": det,
                "p_max": float(p_vals.max()) if p_vals.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _frames_to_minutes(t_frames: np.ndarray, t_to_min: pd.DataFrame) -> np.ndarray:
    """Map a per-track integer frame to minutes using the (fov, track, t) -> minutes lookup.

    NaN frames pass through as NaN minutes.
    """
    out = np.full(len(t_frames), np.nan, dtype=np.float64)
    for i, t in enumerate(t_frames):
        if pd.isna(t):
            continue
        row = t_to_min[t_to_min["t"] == int(t)]
        if not row.empty:
            out[i] = float(row.iloc[0]["minutes"])
    return out


def _attach_minutes(per_track: pd.DataFrame, obs: pd.DataFrame, t_col: str, out_col: str) -> pd.DataFrame:
    """Join per-track frame-time onto minutes via per-cell hours_post_perturbation."""
    if "hours_post_perturbation" not in obs.columns:
        per_track[out_col] = np.nan
        return per_track
    # Build a (fov, track) -> {t: minutes} via the cell's own obs subset
    obs_sub = obs[["fov_name", "track_id", "t", "hours_post_perturbation"]].copy()
    obs_sub["minutes"] = obs_sub["hours_post_perturbation"].astype(float) * 60.0
    merged_min: list[float] = []
    for _, row in per_track.iterrows():
        t = row[t_col]
        if pd.isna(t):
            merged_min.append(np.nan)
            continue
        sub = obs_sub[
            (obs_sub["fov_name"] == row["fov_name"])
            & (obs_sub["track_id"] == row["track_id"])
            & (obs_sub["t"] == int(t))
        ]
        merged_min.append(float(sub.iloc[0]["minutes"]) if not sub.empty else np.nan)
    per_track[out_col] = merged_min
    return per_track


def _t_perturb_minutes(obs: pd.DataFrame) -> pd.DataFrame:
    """Per track, look up the minutes value of ``t_LC_star`` from obs."""
    if "hours_post_perturbation" not in obs.columns:
        return obs.drop_duplicates(["fov_name", "track_id"])[["fov_name", "track_id", "t_LC_star"]].assign(
            t_perturb_minutes=np.nan
        )
    obs_sub = obs[["fov_name", "track_id", "t", "hours_post_perturbation"]].copy()
    obs_sub["minutes"] = obs_sub["hours_post_perturbation"].astype(float) * 60.0
    rows: list[dict] = []
    for (fov, tid), grp in obs.groupby(["fov_name", "track_id"], sort=False):
        t_star = int(grp["t_LC_star"].iloc[0])
        sub = obs_sub[(obs_sub["fov_name"] == fov) & (obs_sub["track_id"] == int(tid)) & (obs_sub["t"] == t_star)]
        rows.append(
            {
                "fov_name": fov,
                "track_id": int(tid),
                "t_perturb_frames": t_star,
                "t_perturb_minutes": float(sub.iloc[0]["minutes"]) if not sub.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run Stage B per organelle channel: emit per-cell event-timing parquet + summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    n_init_frames = int(cfg.get("n_init_frames", 2))
    tau_sweep = list(cfg.get("tau_sweep", [0.3, 0.4, 0.5, 0.6, 0.7]))
    tau_lc = float(cfg.get("tau_lc", 0.5))
    label_columns = list(cfg.get("label_columns", ["organelle_state"]))
    positive_label = cfg.get("positive_label", "remodel")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    summary_rows: list[dict] = []
    for d in cfg["datasets"]:
        data = _load_dataset(
            name=d["name"],
            organelle_zarr=Path(d["organelle_embedding_zarr"]),
            registration_parquet=Path(d["registration_parquet"]),
            marker=d.get("marker", "unknown"),
            organelle=d.get("organelle", "unknown"),
        )
        obs = data.obs
        X = data.embeddings
        if len(obs) == 0:
            _logger.warning("[%s] no LC-registered cells; skipping", data.name)
            continue

        # Stage B Path 1
        path1_t50 = _path1_per_track_t50(
            obs=obs,
            X=X,
            uninfected_pool=data.uninfected_embeddings,
            n_init_frames=n_init_frames,
            tau_sweep=tau_sweep,
        )

        # Stage B Path 2 — for each label column
        path2_t_lc: dict[str, pd.DataFrame] = {}
        for lab_col in label_columns:
            if lab_col not in obs.columns:
                _logger.warning("[%s] obs missing %r; skipping that label_column", data.name, lab_col)
                continue
            lab = obs[lab_col].astype(str)
            if lab_col == "predicted_organelle_state":
                # LC inference output — already discrete; treat ``remodel`` == 1.
                p_remodel = (lab == positive_label).to_numpy().astype(np.float32)
            else:
                # Human (or any other) annotation — retrain organelle LC.
                # Restrict to rows with a valid annotation, train group-aware, score all rows.
                mask = lab.isin([positive_label, "noremodel"]).to_numpy()
                if mask.sum() < 50:
                    _logger.warning("[%s] %s has only %d annotated rows; skipping", data.name, lab_col, int(mask.sum()))
                    continue
                X_lab = X[mask]
                y_lab = lab.to_numpy()[mask]
                groups_lab = _make_groups(obs.loc[mask])
                if len(set(groups_lab)) < 5:
                    _logger.warning(
                        "[%s] %s has only %d groups; cannot train LC",
                        data.name,
                        lab_col,
                        len(set(groups_lab)),
                    )
                    continue
                scaler, clf, _ = _train_organelle_lc(
                    X_lab,
                    y_lab,
                    groups_lab,
                    test_size=test_size,
                    random_state=random_state,
                )
                Xs = scaler.transform(X)
                proba = clf.predict_proba(Xs)
                if positive_label not in clf.classes_:
                    _logger.warning("[%s] %s LC never saw %r; skipping", data.name, lab_col, positive_label)
                    continue
                pos_idx = list(clf.classes_).index(positive_label)
                p_remodel = proba[:, pos_idx]

            path2_t_lc[lab_col] = _path2_per_track_t_lc(obs=obs, p_remodel=p_remodel, tau_lc=tau_lc)

        # Build per-cell timing parquet: one row per (fov_name, track_id)
        per_track_anchor = _t_perturb_minutes(obs)

        timing = per_track_anchor.copy()
        timing["marker"] = data.marker
        timing["organelle"] = data.organelle
        timing["perturbation"] = "infected"  # by construction: only crossed_tau cells
        timing["dataset"] = data.name

        # Path 1 columns
        for (baseline, tau), df_tau in path1_t50.items():
            tag = f"path1_{baseline}_tau{int(tau * 100):03d}"
            df_attached = _attach_minutes(
                df_tau.copy(),
                obs,
                t_col="T_50_frames",
                out_col="T_50_minutes",
            )
            df_attached = df_attached.rename(
                columns={
                    "T_50_frames": f"{tag}_T_50_frames",
                    "T_50_minutes": f"{tag}_T_50_minutes",
                    "detected": f"{tag}_detected",
                    "d_max_raw": f"{tag}_d_max_raw",
                }
            )
            timing = timing.merge(df_attached, on=["fov_name", "track_id"], how="left")
            # Δt columns
            timing[f"{tag}_delta_t_frames"] = timing[f"{tag}_T_50_frames"] - timing["t_perturb_frames"]
            timing[f"{tag}_delta_t_minutes"] = timing[f"{tag}_T_50_minutes"] - timing["t_perturb_minutes"]
            # Summary row
            det = timing[f"{tag}_detected"].fillna(False).astype(bool)
            dt = timing.loc[det, f"{tag}_delta_t_minutes"]
            summary_rows.append(
                {
                    "dataset": data.name,
                    "organelle": data.organelle,
                    "marker": data.marker,
                    "perturbation": "infected",
                    "path": "path1",
                    "baseline": baseline,
                    "tau": tau,
                    "label_column": "",
                    "n_cells": int(len(timing)),
                    "n_detected": int(det.sum()),
                    "frac_detected": float(det.mean()) if len(timing) else float("nan"),
                    "delta_t_median_minutes": float(dt.median()) if len(dt) else float("nan"),
                    "delta_t_q25_minutes": float(dt.quantile(0.25)) if len(dt) else float("nan"),
                    "delta_t_q75_minutes": float(dt.quantile(0.75)) if len(dt) else float("nan"),
                }
            )

        # Path 2 columns
        for lab_col, df_lc in path2_t_lc.items():
            tag = f"path2_{lab_col}"
            df_attached = _attach_minutes(
                df_lc.copy(),
                obs,
                t_col="T_LC_frames",
                out_col="T_LC_minutes",
            )
            df_attached = df_attached.rename(
                columns={
                    "T_LC_frames": f"{tag}_T_LC_frames",
                    "T_LC_minutes": f"{tag}_T_LC_minutes",
                    "detected": f"{tag}_detected",
                    "p_max": f"{tag}_p_max",
                }
            )
            timing = timing.merge(df_attached, on=["fov_name", "track_id"], how="left")
            timing[f"{tag}_delta_t_frames"] = timing[f"{tag}_T_LC_frames"] - timing["t_perturb_frames"]
            timing[f"{tag}_delta_t_minutes"] = timing[f"{tag}_T_LC_minutes"] - timing["t_perturb_minutes"]
            det = timing[f"{tag}_detected"].fillna(False).astype(bool)
            dt = timing.loc[det, f"{tag}_delta_t_minutes"]
            summary_rows.append(
                {
                    "dataset": data.name,
                    "organelle": data.organelle,
                    "marker": data.marker,
                    "perturbation": "infected",
                    "path": "path2",
                    "baseline": "",
                    "tau": tau_lc,
                    "label_column": lab_col,
                    "n_cells": int(len(timing)),
                    "n_detected": int(det.sum()),
                    "frac_detected": float(det.mean()) if len(timing) else float("nan"),
                    "delta_t_median_minutes": float(dt.median()) if len(dt) else float("nan"),
                    "delta_t_q25_minutes": float(dt.quantile(0.25)) if len(dt) else float("nan"),
                    "delta_t_q75_minutes": float(dt.quantile(0.75)) if len(dt) else float("nan"),
                }
            )

        out_parq = args.output_dir / f"{data.name}_event_timing.parquet"
        timing.to_parquet(out_parq, index=False)
        _logger.info("[%s] wrote %s (%d cells)", data.name, out_parq, len(timing))

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "timing_summary.csv"
    summary.to_csv(summary_path, index=False)
    _logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
