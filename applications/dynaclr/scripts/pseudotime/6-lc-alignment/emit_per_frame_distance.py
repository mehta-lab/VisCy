r"""Stage B addendum — emit per-frame organelle distance from baseline.

Decoupled companion to ``measure_event_timing.py``. Computes the per-frame
cosine distance ``d(t)`` of each organelle embedding from two baselines:

- ``d_cell_own``: distance from the cell's own first ``n_init_frames``
  embeddings (post-LC-registration only; uninfected cells have NaN).
- ``d_uninfected_pop``: distance from the mean of the same dataset's
  uninfected-well organelle embeddings (defined for all cells).

The output parquet is the input for Panel C and any downstream
per-frame trajectory analysis. Emitted next to the per-cell Stage B
parquet so the two artifacts can be regenerated independently.

Schema of ``<dataset>_per_frame_distance.parquet`` (one row per
(fov_name, track_id, t)):

    fov_name, track_id, t, t_perturb, t_reg, t_reg_minutes,
    perturbation,                              # 'infected' (LC-registered) or 'uninfected'
    marker, organelle, dataset,
    d_cell_own, d_uninfected_pop,              # raw cosine distances
    p_remodel_predicted,                        # from obs predicted_organelle_state
    p_remodel_human_LC                          # from obs organelle_state (mapped 1/0/NaN)

Reuses the same recipe as ``measure_event_timing.py``. CLI:

    python emit_per_frame_distance.py \\
        --config configs/<MODEL>/zikv_timing_<channel>.yml \\
        --output-dir out/<MODEL>/zikv_timing_<channel>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("emit_per_frame_distance")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge ``base:`` recipe imports and resolve per-dataset zarr/parquet paths."""
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


def _cosine_distance(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-row cosine distance ``1 - cos_sim(X[i], ref)`` (ref is 1D)."""
    Xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    rn = float(np.linalg.norm(ref) + 1e-12)
    return 1.0 - (X @ ref) / (Xn.ravel() * rn)


def _compute_per_frame(
    obs: pd.DataFrame,
    X: np.ndarray,
    cell_baseline_per_track: dict[tuple[str, int], np.ndarray],
    uninfected_pop_baseline: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row cosine distance vectors (d_cell_own, d_uninfected_pop).

    ``d_cell_own[i]`` is NaN if no per-cell baseline was provided for
    ``(obs.iloc[i].fov_name, obs.iloc[i].track_id)`` — that's the case
    for uninfected cells (we don't anchor them to t_perturb so cell-own
    is undefined).
    """
    n = len(obs)
    d_own = np.full(n, np.nan, dtype=np.float32)
    d_pop = np.full(n, np.nan, dtype=np.float32)
    if uninfected_pop_baseline is not None:
        d_pop_full = _cosine_distance(X, uninfected_pop_baseline)
        d_pop[:] = d_pop_full
    # Cell-own baselines per (fov, track)
    obs_idx = obs.reset_index(drop=False).rename(columns={"index": "_row"})
    for (fov, tid), grp in obs_idx.groupby(["fov_name", "track_id"], sort=False):
        ref = cell_baseline_per_track.get((fov, int(tid)))
        if ref is None:
            continue
        rows = grp["_row"].to_numpy()
        d_own[rows] = _cosine_distance(X[rows], ref).astype(np.float32)
    return d_own, d_pop


def _emit_for_dataset(
    name: str,
    organelle_zarr: Path,
    registration_parquet: Path,
    marker: str,
    organelle: str,
    n_init_frames: int,
    positive_label: str,
    output_path: Path,
) -> None:
    """Load one organelle zarr, compute per-frame distances, write parquet."""
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
    X = X.astype(np.float32)
    obs = obs.drop(columns=["_iloc"])

    # Uninfected pool baseline (population-level)
    if "perturbation" in obs.columns:
        is_uninf = obs["perturbation"].astype(str) == "uninfected"
        if is_uninf.any():
            uninfected_pop = X[is_uninf.to_numpy()].mean(axis=0)
        else:
            _logger.warning("[%s] no uninfected rows; d_uninfected_pop will be NaN", name)
            uninfected_pop = None
    else:
        _logger.warning("[%s] obs lacks 'perturbation' column", name)
        uninfected_pop = None

    # Stage A registration — joins t_perturb only to LC-registered cells
    reg = pd.read_parquet(registration_parquet)
    reg["track_id"] = reg["track_id"].astype(int)
    reg = reg[["fov_name", "track_id", "t_LC_star", "crossed_tau"]]
    merged = obs.merge(reg, on=["fov_name", "track_id"], how="left")

    # Cell-own baseline: mean of first n_init_frames embeddings per
    # LC-registered cell (only infected cohort)
    cell_baselines: dict[tuple[str, int], np.ndarray] = {}
    for (fov, tid), grp in merged.groupby(["fov_name", "track_id"], sort=False):
        if not bool(grp["crossed_tau"].fillna(False).iloc[0]):
            continue
        grp_sorted = grp.sort_values("t")
        ilocs = grp_sorted.index.to_numpy()
        k = min(n_init_frames, len(ilocs))
        cell_baselines[(fov, int(tid))] = X[ilocs[:k]].mean(axis=0)

    d_own, d_pop = _compute_per_frame(merged, X, cell_baselines, uninfected_pop)

    # Build output rows
    t_perturb = merged["t_LC_star"].astype(float)
    out = pd.DataFrame(
        {
            "fov_name": merged["fov_name"].astype(str),
            "track_id": merged["track_id"].astype(int),
            "t": merged["t"].astype(int),
            "t_perturb": t_perturb,
            "t_reg": merged["t"].astype(float) - t_perturb,
        }
    )
    if "hours_post_perturbation" in merged.columns:
        # Build per-cell t_perturb_minutes via the obs's hours_post_perturbation
        # at the cell's t_LC_star frame.
        obs_min = merged[["fov_name", "track_id", "t", "hours_post_perturbation"]].copy()
        obs_min["minutes"] = obs_min["hours_post_perturbation"].astype(float) * 60.0
        anchor_rows = []
        for (fov, tid), grp in merged.groupby(["fov_name", "track_id"], sort=False):
            t_star = grp["t_LC_star"].iloc[0]
            if pd.isna(t_star):
                continue
            row = obs_min[
                (obs_min["fov_name"] == fov) & (obs_min["track_id"] == int(tid)) & (obs_min["t"] == int(t_star))
            ]
            if not row.empty:
                anchor_rows.append(
                    {
                        "fov_name": fov,
                        "track_id": int(tid),
                        "t_perturb_minutes": float(row.iloc[0]["minutes"]),
                    }
                )
        anchors = pd.DataFrame(anchor_rows)
        out = out.merge(anchors, on=["fov_name", "track_id"], how="left")
        out["t_reg_minutes"] = merged["hours_post_perturbation"].astype(float) * 60.0 - out["t_perturb_minutes"].astype(
            float
        )
    else:
        out["t_reg_minutes"] = np.nan
        out["t_perturb_minutes"] = np.nan

    out["perturbation"] = merged.get("perturbation", "").astype(str)
    out["crossed_tau"] = merged["crossed_tau"].fillna(False).astype(bool)
    out["marker"] = marker
    out["organelle"] = organelle
    out["dataset"] = name
    out["d_cell_own"] = d_own
    out["d_uninfected_pop"] = d_pop

    # Per-frame organelle-state predictions
    if "predicted_organelle_state" in merged.columns:
        out["p_remodel_predicted"] = (merged["predicted_organelle_state"].astype(str) == positive_label).astype(
            np.float32
        )
    else:
        out["p_remodel_predicted"] = np.nan
    if "organelle_state" in merged.columns:
        lab = merged["organelle_state"].astype(str)
        mapped = np.where(
            lab == positive_label,
            1.0,
            np.where(lab.isin(["", "nan"]), np.nan, 0.0),
        )
        out["p_remodel_human_LC"] = mapped.astype(np.float32)
    else:
        out["p_remodel_human_LC"] = np.nan

    out.to_parquet(output_path, index=False)
    _logger.info(
        "[%s] wrote %s (%d rows; LC-registered cells=%d, uninfected cells=%d)",
        name,
        output_path,
        len(out),
        int(out["crossed_tau"].sum()),
        int((out["perturbation"] == "uninfected").sum()),
    )


def main() -> None:
    """Emit Stage B per-frame distance parquet per dataset in the recipe.

    Reuses the same YAML recipe as ``measure_event_timing.py``.
    Output filename pattern: ``<dataset>_per_frame_distance.parquet``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    n_init_frames = int(cfg.get("n_init_frames", 2))
    positive_label = cfg.get("positive_label", "remodel")

    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    for d in cfg["datasets"]:
        _emit_for_dataset(
            name=d["name"],
            organelle_zarr=Path(d["organelle_embedding_zarr"]),
            registration_parquet=Path(d["registration_parquet"]),
            marker=d.get("marker", "unknown"),
            organelle=d.get("organelle", "unknown"),
            n_init_frames=n_init_frames,
            positive_label=positive_label,
            output_path=args.output_dir / f"{d['name']}_per_frame_distance.parquet",
        )


if __name__ == "__main__":
    main()
