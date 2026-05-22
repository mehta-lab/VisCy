r"""Compute per-cell registration anchors from a viral_sensor LC.

For each dataset in the recipe:

1. Load viral_sensor embeddings + annotations.
2. Retrain a logistic regression (same recipe as ``5-eval/multi_frame_lc.py``)
   on cells that have a ground-truth ``infection_state`` label, using a
   group-aware split (track-level 80/20 by default; lineage-level if
   ``parent_track_id`` is available on the annotation CSV).
3. Score the held-out cells' viral_sensor embeddings to produce a per-frame
   ``p(infected | t)`` trajectory.
4. Per cell track, compute the registration anchor ``t_LC_star`` as the
   first frame where ``p >= tau`` (default 0.5).
5. Write a parquet ``<dataset>_lc_registration.parquet`` with one row per
   cell track:

   - ``fov_name``, ``track_id``
   - ``t_LC_star``        — first crossing of tau (NaN if never crossed)
   - ``t_LC_steep``       — argmax of dp/dt (alternative anchor)
   - ``p_max``            — max p across the track
   - ``n_frames_observed`` — track length in the embedding zarr
   - ``n_frames_infected`` — frames with p >= tau
   - ``crossed_tau``      — boolean; if False, cell stayed uninfected per LC

Recipe schema mirrors ``5-eval/`` (recipe + per-model leaf with
``embedding_dir`` / ``annotation_path``-style filenames so adding datasets
is just a new entry in the dataset roster).

Usage::

    python register_by_lc.py --config configs/<MODEL>/zikv_register.yml \\
                             --output-dir out/<MODEL>/zikv_register/
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
_logger = logging.getLogger("register_by_lc")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Load a leaf YAML and merge ``base:`` recipe imports.

    Resolves ``embedding_filename`` + ``embedding_dir`` to absolute
    ``embedding_zarr`` per dataset, mirroring ``5-eval/multi_frame_lc.py``.
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        leaf = yaml.safe_load(f) or {}

    merged: dict = {}
    for rel in leaf.pop("base", []):
        recipe_path = (config_path.parent / rel).resolve()
        with open(recipe_path) as f:
            piece = yaml.safe_load(f) or {}
        merged.update(piece)
    merged.update(leaf)

    embedding_dir = merged.get("embedding_dir")
    datasets = merged.get("datasets") or []
    for d in datasets:
        if "embedding_zarr" not in d and "embedding_filename" in d:
            if embedding_dir is None:
                raise KeyError(
                    f"Dataset {d.get('name')} uses embedding_filename but the leaf did not set embedding_dir."
                )
            d["embedding_zarr"] = str(Path(embedding_dir) / d["embedding_filename"])
        if "annotation_csv" not in d and "annotation_path" in d:
            d["annotation_csv"] = d["annotation_path"]
    return merged


@dataclass
class DatasetData:
    """Per-dataset pre-loaded data for the registration step.

    Attributes
    ----------
    name : str
        Dataset name.
    obs_ann : pandas.DataFrame
        Annotated subset — one row per ``(fov_name, track_id, t)`` that
        has a ground-truth label, in the same row order as the legacy
        inner-join. Columns: ``fov_name``, ``track_id``, ``t``, ``gt``,
        and optionally ``parent_track_id`` (suffixed by ``_x``/``_y``
        when the embedding zarr's ``obs`` and the annotation CSV both
        carry a ``parent_track_id`` column).
    embeddings_ann : numpy.ndarray
        ``(N_ann, D)`` embeddings aligned with ``obs_ann``.
    obs_unann : pandas.DataFrame
        Unannotated cell-frames — rows from the embedding zarr that
        have no matching annotation. Same columns as ``obs_ann`` with
        annotation columns (``gt``, ``parent_track_id_y``) set to NaN.
    embeddings_unann : numpy.ndarray
        ``(N_unann, D)`` embeddings aligned with ``obs_unann``.
    """

    name: str
    obs_ann: pd.DataFrame
    embeddings_ann: np.ndarray
    obs_unann: pd.DataFrame
    embeddings_unann: np.ndarray


def _load_dataset(name: str, embedding_zarr: Path, annotation_csv: Path, task: str) -> DatasetData:
    """Load viral_sensor embeddings and split into annotated/unannotated subsets.

    Builds two disjoint subsets of the embedding zarr:

    1. ``obs_ann`` / ``embeddings_ann`` — preserves the exact row order
       of the legacy inner-join with the annotation CSV, so LC training
       and downstream per-track registration on this subset reproduce
       the previous outputs bit-for-bit.
    2. ``obs_unann`` / ``embeddings_unann`` — every cell-frame in the
       zarr whose ``(fov_name, track_id, t)`` is absent from the
       (deduplicated, non-empty) annotation CSV. Annotation columns are
       NaN. Used to score wells lacking per-channel annotations (e.g.
       TOMM20 wells).
    """
    _logger.info("[%s] loading %s", name, embedding_zarr.name)
    a = ad.read_zarr(embedding_zarr)
    obs = a.obs.copy().reset_index(drop=True)
    obs["track_id"] = obs["track_id"].astype(int)
    obs["t"] = obs["t"].astype(int)
    obs["_iloc"] = np.arange(len(obs))
    obs = obs.drop_duplicates(subset=["fov_name", "track_id", "t"]).reset_index(drop=True)
    X = a.X[obs["_iloc"].to_numpy()]
    if hasattr(X, "toarray"):
        X = X.toarray()
    obs = obs.drop(columns=["_iloc"])

    ann = pd.read_csv(annotation_csv)
    if task not in ann.columns:
        raise KeyError(f"[{name}] annotation CSV {annotation_csv} missing column {task!r}")
    keep_cols = ["fov_name", "track_id", "t", task]
    if "parent_track_id" in ann.columns:
        keep_cols.append("parent_track_id")
    ann = ann[ann["fov_name"].isin(obs["fov_name"].unique())]
    ann = ann[keep_cols].rename(columns={task: "gt"})
    ann["track_id"] = ann["track_id"].astype(int)
    ann["t"] = ann["t"].astype(int)
    ann = ann[ann["gt"].notna() & (ann["gt"] != "")]
    ann = ann.drop_duplicates(subset=["fov_name", "track_id", "t"])

    # Annotated subset (legacy inner-join row order; keeps left-frame order).
    merged = obs.merge(ann, on=["fov_name", "track_id", "t"], how="inner")
    obs_to_idx = {(r.fov_name, r.track_id, r.t): i for i, r in enumerate(obs.itertuples())}
    ann_iloc = np.array(
        [obs_to_idx[(r.fov_name, r.track_id, r.t)] for r in merged.itertuples()],
        dtype=np.int64,
    )
    X_ann = X[ann_iloc].astype(np.float32)
    obs_ann = merged.reset_index(drop=True)

    # Unannotated subset — every cell-frame in the zarr not in `merged`.
    is_annotated = np.zeros(len(obs), dtype=bool)
    is_annotated[ann_iloc] = True
    unann_iloc = np.flatnonzero(~is_annotated)
    obs_unann = obs.iloc[unann_iloc].reset_index(drop=True)
    # Match annotated columns so downstream concat works cleanly.
    # NaN-filled columns must use object/float dtype — int columns can't
    # hold NaN, so we leave dtype unset for any newly-added column.
    for col in obs_ann.columns:
        if col not in obs_unann.columns:
            obs_unann[col] = np.nan
    obs_unann = obs_unann[list(obs_ann.columns)]
    X_unann = X[unann_iloc].astype(np.float32)

    return DatasetData(
        name=name,
        obs_ann=obs_ann,
        embeddings_ann=X_ann,
        obs_unann=obs_unann,
        embeddings_unann=X_unann,
    )


def _train_logreg(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """Fit a balanced logistic regression on standardized embeddings.

    Mirrors the recipe in ``5-eval/multi_frame_lc.py:_train_logreg``.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(Xs, y)
    return scaler, clf


def _score(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray, positive: str) -> np.ndarray:
    """Return ``p(positive | x)`` per row."""
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)
    pos_idx = list(clf.classes_).index(positive)
    return prob[:, pos_idx]


def _make_groups(obs: pd.DataFrame) -> np.ndarray:
    """Return per-row group ids for GroupShuffleSplit.

    Uses ``lineage_id = root(parent_track_id chain)`` when
    ``parent_track_id`` is present, else falls back to ``(fov_name, track_id)``.
    """
    if "parent_track_id" not in obs.columns:
        return obs["fov_name"].astype(str) + "/" + obs["track_id"].astype(str)
    # Walk parent_track_id chain to a root per (fov, track)
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


def _per_track_registration(obs: pd.DataFrame, p_infected: np.ndarray, tau: float) -> pd.DataFrame:
    """Per cell track, compute ``t_LC_star`` = first frame where p >= tau.

    Also computes ``t_LC_steep`` (argmax of dp/dt) as a robustness anchor.
    """
    work = obs[["fov_name", "track_id", "t"]].copy()
    work["p"] = p_infected
    work = work.sort_values(["fov_name", "track_id", "t"]).reset_index(drop=True)
    rows: list[dict] = []
    for (fov, tid), grp in work.groupby(["fov_name", "track_id"], sort=False):
        t_vals = grp["t"].to_numpy()
        p_vals = grp["p"].to_numpy()
        crossed = p_vals >= tau
        if crossed.any():
            t_star = int(t_vals[np.argmax(crossed)])
        else:
            t_star = np.nan
        if len(p_vals) >= 2:
            dp = np.diff(p_vals)
            # midpoint t for the derivative; report the t at the right edge
            t_steep = int(t_vals[1 + int(np.argmax(dp))])
        else:
            t_steep = int(t_vals[0])
        rows.append(
            {
                "fov_name": fov,
                "track_id": int(tid),
                "t_LC_star": t_star,
                "t_LC_steep": t_steep,
                "p_max": float(np.max(p_vals)),
                "n_frames_observed": int(len(p_vals)),
                "n_frames_infected": int(crossed.sum()),
                "crossed_tau": bool(crossed.any()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Score viral_sensor LC per cell, emit per-track registration parquet.

    Config schema::

        task: infection_state
        positive_class: infected
        tau: 0.5
        test_size: 0.2
        random_state: 42
        embedding_dir: /abs/path/.../embeddings
        datasets:
          - name: 07_22_ZIKV
            embedding_filename: 2025_07_22_A549_viral_sensor_ZIKV.zarr
            annotation_path: /abs/.../*_combined_annotations.csv
          - name: 07_24_ZIKV
            ...
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    task = cfg.get("task", "infection_state")
    positive = cfg.get("positive_class", "infected")
    tau = float(cfg.get("tau", 0.5))
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    # Snapshot config for reproducibility
    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    summary_rows: list[dict] = []
    for d in cfg["datasets"]:
        data = _load_dataset(
            name=d["name"],
            embedding_zarr=Path(d["embedding_zarr"]),
            annotation_csv=Path(d["annotation_csv"]),
            task=task,
        )
        obs_ann = data.obs_ann
        X_ann = data.embeddings_ann
        obs_unann = data.obs_unann
        X_unann = data.embeddings_unann
        y_ann = obs_ann["gt"].to_numpy()

        # Group-aware 80/20 to retrain the LC on this dataset's annotated cells.
        # We split at lineage level (or track level if no parent_track_id) so the
        # held-out cells' p(infected | t) trajectories are leak-free.
        groups = _make_groups(obs_ann)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_ann, y_ann, groups=groups))
        _logger.info(
            "[%s] LC train=%d (%d groups), eval=%d (%d groups)",
            d["name"],
            len(train_idx),
            len(np.unique(groups[train_idx])),
            len(test_idx),
            len(np.unique(groups[test_idx])),
        )
        scaler, clf = _train_logreg(X_ann[train_idx], y_ann[train_idx])

        # Score annotated and unannotated subsets separately. Scoring the
        # annotated subset alone (rather than a combined matrix) preserves
        # bit-identical p_infected values for the annotated rows — BLAS
        # ordering of the matrix product otherwise introduces ~1 ULP drift
        # when the input shape changes.
        p_inf_ann = _score(scaler, clf, X_ann, positive)
        p_inf_unann = _score(scaler, clf, X_unann, positive) if len(X_unann) > 0 else np.empty(0, dtype=p_inf_ann.dtype)

        # Mark train/eval membership on each subset.
        in_train_ann = np.zeros(len(obs_ann), dtype=bool)
        in_train_ann[train_idx] = True
        obs_ann = obs_ann.copy()
        obs_ann["lc_in_train"] = in_train_ann
        obs_unann = obs_unann.copy()
        obs_unann["lc_in_train"] = np.zeros(len(obs_unann), dtype=bool)

        # Per-track registration: compute on the annotated subset (matching
        # the legacy behavior — tracks restricted to their annotated frames)
        # and on the unannotated subset for tracks lacking any annotation
        # (e.g. TOMM20 wells). Concatenate; the two sets are disjoint by
        # construction (annotated tracks never appear in obs_unann's
        # (fov_name, track_id, t) keys, but a track that has both annotated
        # and unannotated frames is treated as annotated to preserve legacy
        # output).
        reg_ann = _per_track_registration(obs_ann, p_inf_ann, tau=tau)
        ann_track_keys = set(zip(obs_ann["fov_name"], obs_ann["track_id"].astype(int)))
        if len(obs_unann) > 0:
            unann_key = list(zip(obs_unann["fov_name"], obs_unann["track_id"].astype(int)))
            keep_mask = np.array([k not in ann_track_keys for k in unann_key])
            obs_unann_tracks = obs_unann.iloc[np.flatnonzero(keep_mask)].reset_index(drop=True)
            p_unann_tracks = p_inf_unann[keep_mask]
            reg_unann = _per_track_registration(obs_unann_tracks, p_unann_tracks, tau=tau)
        else:
            reg_unann = pd.DataFrame(columns=reg_ann.columns)
        reg = pd.concat([reg_ann, reg_unann], ignore_index=True)
        reg["dataset"] = d["name"]

        # Per-frame parquet covers both subsets so downstream timing
        # analyses can use unannotated-cell trajectories directly.
        per_frame_cols = ["fov_name", "track_id", "t", "lc_in_train", "gt"]
        per_frame_ann = obs_ann[per_frame_cols].copy()
        per_frame_ann["p_infected"] = p_inf_ann
        per_frame_unann = obs_unann[per_frame_cols].copy()
        per_frame_unann["p_infected"] = p_inf_unann
        per_frame = pd.concat([per_frame_ann, per_frame_unann], ignore_index=True)
        per_frame["dataset"] = d["name"]

        out_reg = args.output_dir / f"{d['name']}_lc_registration.parquet"
        reg.to_parquet(out_reg, index=False)
        per_frame.to_parquet(args.output_dir / f"{d['name']}_per_frame.parquet", index=False)

        n_tracks = len(reg)
        n_crossed = int(reg["crossed_tau"].sum())
        summary_rows.append(
            {
                "dataset": d["name"],
                "n_cells": int(len(obs_ann) + len(obs_unann)),
                "n_cells_annotated": int(len(obs_ann)),
                "n_cells_unannotated": int(len(obs_unann)),
                "n_tracks": n_tracks,
                "n_tracks_annotated": int(len(reg_ann)),
                "n_tracks_unannotated": int(len(reg_unann)),
                "n_tracks_crossed_tau": n_crossed,
                "frac_tracks_crossed_tau": n_crossed / max(1, n_tracks),
                "lc_train_size": int(in_train_ann.sum()),
                "lc_eval_size": int((~in_train_ann).sum()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "registration_summary.csv", index=False)
    _logger.info("Wrote summary to %s", args.output_dir / "registration_summary.csv")


if __name__ == "__main__":
    main()
