"""Evaluate LC-derived registration of organelle/Phase3D embeddings.

For each dataset in the recipe:

1. Load the per-cell registration parquet produced by ``register_by_lc.py``
   (one row per cell track with ``t_LC_star``).
2. Load the organelle channel embedding zarr (Phase3D, SEC61, G3BP1, ...)
   for the same cells. Look up infection-state annotations.
3. Restrict to cells with ``crossed_tau == True`` (LC says these got
   infected); exclude cells that never crossed (these are "uninfected"
   per the LC and have no event time to register).
4. Compute the LC-shifted time ``t_reg = t - t_LC_star`` per frame.
5. Compare two classifiers on the **organelle channel embeddings**:
   - **Unshifted LC** — per-frame LC on raw organelle embeddings.
   - **LC-registered HPI** — per-frame LC where the feature is augmented
     with ``t_reg`` (i.e. the model sees the embedding + the LC-derived
     hours-post-onset). This is the practitioner recipe: use the strong
     channel's LC to pin event time, then everything downstream is
     pseudotime-aligned.
6. Score F1 / ROC-AUC on a group-aware 80/20 held-out split (lineage when
   parent_track_id is available, else track-level).

Outputs (per leaf):
- ``metrics.csv``               long-form metrics
- ``tables/by_dataset_*.csv``   wide format
- ``report.md``                 human-readable
- ``<dataset>_per_frame.parquet`` predictions + t_reg for downstream plots
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("eval_registered_organelle")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge leaf config with recipes the same way as ``register_by_lc.py``.

    Per-dataset path resolution: each dataset entry may declare
    ``organelle_embedding_filename`` (resolved against
    ``organelle_embedding_dir``) and ``registration_filename`` (resolved
    against ``registration_dir`` — the output of ``register_by_lc.py``).
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

    org_dir = merged.get("organelle_embedding_dir")
    reg_dir = merged.get("registration_dir")
    for d in merged.get("datasets") or []:
        if "organelle_embedding_zarr" not in d and "organelle_embedding_filename" in d:
            if org_dir is None:
                raise KeyError(f"{d.get('name')}: organelle_embedding_filename needs organelle_embedding_dir")
            d["organelle_embedding_zarr"] = str(Path(org_dir) / d["organelle_embedding_filename"])
        if "registration_parquet" not in d and "registration_filename" in d:
            if reg_dir is None:
                raise KeyError(f"{d.get('name')}: registration_filename needs registration_dir")
            d["registration_parquet"] = str(Path(reg_dir) / d["registration_filename"])
        if "annotation_csv" not in d and "annotation_path" in d:
            d["annotation_csv"] = d["annotation_path"]
    return merged


@dataclass
class RegisteredData:
    """Per-dataset organelle embeddings joined with LC registration anchors."""

    name: str
    obs: pd.DataFrame
    embeddings: np.ndarray


def _load_dataset(
    name: str,
    organelle_zarr: Path,
    registration_parquet: Path,
    annotation_csv: Path,
    task: str,
) -> RegisteredData:
    """Load organelle embeddings, ground truth, and LC registration anchor."""
    _logger.info("[%s] loading organelle embeddings from %s", name, organelle_zarr.name)
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

    # Annotations
    ann = pd.read_csv(annotation_csv)
    if task not in ann.columns:
        raise KeyError(f"[{name}] annotation CSV {annotation_csv} missing {task!r}")
    keep_cols = ["fov_name", "track_id", "t", task]
    if "parent_track_id" in ann.columns:
        keep_cols.append("parent_track_id")
    ann = ann[ann["fov_name"].isin(obs["fov_name"].unique())]
    ann = ann[keep_cols].rename(columns={task: "gt"})
    ann["track_id"] = ann["track_id"].astype(int)
    ann["t"] = ann["t"].astype(int)
    ann = ann[ann["gt"].notna() & (ann["gt"] != "")]
    ann = ann.drop_duplicates(subset=["fov_name", "track_id", "t"])

    # Registration anchors
    reg = pd.read_parquet(registration_parquet)
    reg["track_id"] = reg["track_id"].astype(int)
    reg_cols = ["fov_name", "track_id", "t_LC_star", "t_LC_steep", "crossed_tau", "p_max"]
    reg = reg[reg_cols]

    merged = obs.merge(ann, on=["fov_name", "track_id", "t"], how="inner")
    merged = merged.merge(reg, on=["fov_name", "track_id"], how="left")
    _logger.info("[%s] %d annotated frames before registration filter", name, len(merged))

    # Embeddings restricted to merged rows
    obs_to_idx = {(r.fov_name, r.track_id, r.t): i for i, r in enumerate(obs.itertuples())}
    iloc = np.array([obs_to_idx[(r.fov_name, r.track_id, r.t)] for r in merged.itertuples()])
    X = X[iloc]
    obs = merged.reset_index(drop=True)
    return RegisteredData(name=name, obs=obs, embeddings=X.astype(np.float32))


def _make_groups(obs: pd.DataFrame) -> np.ndarray:
    """Lineage when parent_track_id present, else (fov, track)."""
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


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_pos: np.ndarray | None, positive: str) -> dict:
    """Accuracy / F1 / ROC-AUC; ROC-AUC is None when only one class is present."""
    gt = (y_true == positive).astype(int)
    pr = (y_pred == positive).astype(int)
    out = {
        "accuracy": float((gt == pr).mean()) if len(gt) else float("nan"),
        "f1": float(f1_score(gt, pr, zero_division=0)) if len(gt) else float("nan"),
    }
    if p_pos is not None and len(np.unique(gt)) > 1:
        out["roc_auc"] = float(roc_auc_score(gt, p_pos))
    else:
        out["roc_auc"] = float("nan")
    return out


def _train_eval(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, positive: str
) -> dict:
    """Standardize + L2 LR + balanced class weights (mirrors ``5-eval``)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(Xs, y_train)
    Xt = scaler.transform(X_test)
    pred = clf.predict(Xt)
    prob = clf.predict_proba(Xt)
    pos_idx = list(clf.classes_).index(positive) if positive in clf.classes_ else None
    p_pos = prob[:, pos_idx] if pos_idx is not None else None
    return {"pred": pred, "p_pos": p_pos, **_binary_metrics(y_test, pred, p_pos, positive)}


def main() -> None:
    """Eval registered-vs-unshifted organelle classification.

    Config schema::

        task: infection_state
        positive_class: infected
        test_size: 0.2
        random_state: 42
        organelle_embedding_dir: /abs/.../embeddings
        registration_dir: /abs/.../6-lc-alignment/out/.../zikv_register/
        datasets:
          - name: 07_22_ZIKV
            organelle_embedding_filename: 2025_07_22_A549_Phase3D_ZIKV.zarr
            registration_filename: 07_22_ZIKV_lc_registration.parquet
            annotation_path: /abs/.../*_combined_annotations.csv
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    task = cfg.get("task", "infection_state")
    positive = cfg.get("positive_class", "infected")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    import shutil

    shutil.copy2(args.config, args.output_dir / args.config.name)

    all_results: list[dict] = []
    for d in cfg["datasets"]:
        data = _load_dataset(
            name=d["name"],
            organelle_zarr=Path(d["organelle_embedding_zarr"]),
            registration_parquet=Path(d["registration_parquet"]),
            annotation_csv=Path(d["annotation_csv"]),
            task=task,
        )
        obs = data.obs
        X = data.embeddings

        # Restrict to cells the LC says crossed tau — they have an event time
        keep = obs["crossed_tau"].fillna(False).to_numpy().astype(bool)
        _logger.info(
            "[%s] %d/%d frames in LC-registered cohort (crossed_tau=True)",
            d["name"],
            int(keep.sum()),
            len(obs),
        )
        if keep.sum() == 0:
            _logger.warning("[%s] no cells crossed tau; skipping", d["name"])
            continue
        obs_keep = obs.loc[keep].reset_index(drop=True)
        X_keep = X[keep]
        y = obs_keep["gt"].to_numpy()

        # Group-aware split
        groups = _make_groups(obs_keep)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_keep, y, groups=groups))

        # Feature builders
        t_reg = (obs_keep["t"] - obs_keep["t_LC_star"]).to_numpy(dtype=np.float32).reshape(-1, 1)
        X_unshifted = X_keep
        X_registered = np.concatenate([X_keep, t_reg], axis=1)

        res_unshifted = _train_eval(
            X_unshifted[train_idx],
            y[train_idx],
            X_unshifted[test_idx],
            y[test_idx],
            positive,
        )
        res_registered = _train_eval(
            X_registered[train_idx],
            y[train_idx],
            X_registered[test_idx],
            y[test_idx],
            positive,
        )

        for method, r in (("LC_unshifted", res_unshifted), ("LC_registered", res_registered)):
            all_results.append(
                {
                    "dataset": d["name"],
                    "method": method,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "accuracy": r["accuracy"],
                    "f1": r["f1"],
                    "roc_auc": r["roc_auc"],
                }
            )

        # Per-frame predictions for downstream plotting
        per_frame = obs_keep.iloc[test_idx][["fov_name", "track_id", "t", "gt", "t_LC_star"]].copy()
        per_frame["t_reg"] = t_reg[test_idx].ravel()
        per_frame["pred_unshifted"] = res_unshifted["pred"]
        per_frame["pred_registered"] = res_registered["pred"]
        if res_unshifted["p_pos"] is not None:
            per_frame["p_pos_unshifted"] = res_unshifted["p_pos"]
        if res_registered["p_pos"] is not None:
            per_frame["p_pos_registered"] = res_registered["p_pos"]
        per_frame["dataset"] = d["name"]
        per_frame.to_parquet(args.output_dir / f"{d['name']}_per_frame.parquet", index=False)

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)

    # Wide tables
    wide_dir = args.output_dir / "tables"
    wide_dir.mkdir(exist_ok=True)
    for metric in ["accuracy", "f1", "roc_auc"]:
        pivot = metrics_df.pivot_table(index="dataset", columns="method", values=metric)
        pivot.to_csv(wide_dir / f"by_dataset__{metric}.csv")

    # Markdown report
    lines = [f"# LC-registered vs unshifted organelle classification — {task} (positive={positive})", ""]
    lines.append(f"_Config: `{args.config}`_")
    lines.append("")
    lines.append(
        "Both classifiers run on the same organelle/Phase3D embeddings of cells whose viral_sensor LC "
        "crossed τ. `LC_registered` augments the per-frame embedding with the LC-derived "
        "hours-post-onset (t − t_LC_star); `LC_unshifted` is the raw per-frame embedding baseline. "
        "Group-aware 80/20 holdout (lineage when available, else track)."
    )
    lines.append("")
    for metric in ["f1", "roc_auc", "accuracy"]:
        lines.append(f"## {metric}")
        pivot = metrics_df.pivot_table(index="dataset", columns="method", values=metric)
        lines.append("| dataset | LC_unshifted | LC_registered | Δ |")
        lines.append("| --- | --- | --- | --- |")
        for ds, row in pivot.iterrows():
            u = row.get("LC_unshifted", float("nan"))
            r = row.get("LC_registered", float("nan"))
            delta = (r - u) if pd.notna(u) and pd.notna(r) else float("nan")
            fu = f"{u:.3f}" if pd.notna(u) else "n/a"
            fr = f"{r:.3f}" if pd.notna(r) else "n/a"
            fd = f"{delta:+.3f}" if pd.notna(delta) else "n/a"
            lines.append(f"| {ds} | {fu} | {fr} | {fd} |")
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines))
    _logger.info("Wrote report.md to %s", args.output_dir)


if __name__ == "__main__":
    main()
