"""Fig 6 attempt 25: Alex Lin's OPS set classifier, unchanged, on cached TOMM20 embeddings — cell cycle vs infection.

Applies the pinned ``set_classifier`` package (czbiohub-sf/ops_monorepo@ab11b56, ``train``/``eval``/``score``) to
DynaCLR TOMM20 ``obsm/X_normalized_pca80`` (49 PCs) of 2026_07_01 and 2026_04_10. Model I classifies bags of
TCN-onset lineage cells by 2 h pseudotime bin against bystander and control-well references, with every pool
phase-matched (equal G1/S/G2); Model C classifies the cycle phase of uninfected (control + bystander) cells.
Stages: ``export`` (representative frames, FOV folds, pools, dumps), ``run`` (pinned train/eval/score on the
reserved GPU node), ``analyze`` (sweep, ranking, AUROCs with baselines, distortion map, cross-scoring), ``render``
(crops, panels, montage, UMAP) and ``report`` (claims the attempt-25 figure directory, README and manifest).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from anndata.io import read_elem
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

CHANNEL = "TOMM20"
TRACK = ["dataset", "fov_name", "track_id"]
KEYS = ["fov_name", "track_id", "t"]
SPLITS = ["train", "val", "test"]
MODELS = ["model_i", "model_c"]
MODEL_TITLES = {
    "model_i": "Model I (τ bins + references, phase-matched)",
    "model_c": "Model C (cycle phase, uninfected)",
}
REFERENCES = ["BYSTANDER", "CONTROL"]
GROUP_OF_STATUS = {"infected": "onset", "pre_onset": "onset", "not_infected": "BYSTANDER", "uninfected": "CONTROL"}
ALL_PHASES = ["G1", "S", "G2", "M"]
PHASE_COLORS = {"G1": "#0072b2", "S": "#e69f00", "G2": "#9564a7", "M": "#00846b"}
REFERENCE_COLORS = {"BYSTANDER": "#000000", "CONTROL": "#8c8c8c"}
CLASS_TITLES = {
    "BYSTANDER": "ZIKV-well lineages without a TCN onset (phase-matched pool)",
    "CONTROL": "control-well cells (phase-matched pool)",
}
RC = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
META_COLUMNS = ["experiment", "well", "x_pheno", "y_pheno", "segmentation_id", "channel_type", "index"]
EXTRA_COLUMNS = ["lineage_uid", "tau_h", "hpi", "cc_state", "group"]


def sha(path):
    """Return the sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(array):
    """Return the sha256 of a contiguous array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, payload):
    """Write ``payload`` as indented JSON."""
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def load_config(path):
    """Read the attempt-25 YAML config.

    Parameters
    ----------
    path : pathlib.Path
        Config path.

    Returns
    -------
    dict
        Config with its own path under ``config_path``.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


def tau_class(tau_bin):
    """Return the Model I class name of a τ bin start in hours, e.g. ``-8 -> TAU_m08`` and ``0 -> TAU_p00``."""
    return f"TAU_{'m' if tau_bin < 0 else 'p'}{abs(int(tau_bin)):02d}"


def tau_bins(cfg):
    """Return the τ bin starts in hours."""
    return list(range(cfg["tau_start"], cfg["tau_stop"], cfg["bin_hours"]))


def model_i_classes(cfg):
    """Return the Model I class names in display order (τ bins, then references)."""
    return [*(tau_class(b) for b in tau_bins(cfg)), *REFERENCES]


def pc_columns(cfg):
    """Return the embedding column names of ``cells.parquet``."""
    return [f"pc{i:02d}" for i in range(cfg["embedding_dims"])]


def load_module(name, path):
    """Import a source file as a module registered under ``name`` without touching ``sys.path``.

    Parameters
    ----------
    name : str
        Module name to register in ``sys.modules``.
    path : pathlib.Path
        Source file.

    Returns
    -------
    module
        The executed module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pinned_train(cfg):
    """Import the pinned ``set_classifier/train.py`` (no relative imports) for ``load_dataset`` and ``build_model``."""
    return load_module("set_classifier_ab11b56_train", Path(cfg["pinned_root"]) / "set_classifier" / "train.py")


def labels_path(cfg):
    """Return the attempt-23 label snapshot after checking its sha256 against the config."""
    path = Path(cfg["labels"])
    digest = sha(path)
    if digest != cfg["labels_sha256"]:
        raise ValueError(f"{path} sha256 {digest} != expected {cfg['labels_sha256']}")
    return path


def load_labels(cfg):
    """Read the attempt-23 label snapshot (sha-checked), drop Mantis death and ambiguous frames and assign groups.

    Parameters
    ----------
    cfg : dict
        Attempt config.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Live, non-ambiguous frames with ``well``, ``well_type``, ``group`` and ``fov_key``, and an input record.
    """
    path = labels_path(cfg)
    d = pd.read_parquet(path)
    if len(d) != cfg["expected_frames"]:
        raise ValueError(f"{path}: {len(d)} frames, expected {cfg['expected_frames']}")
    if d.duplicated(["dataset", *KEYS]).any():
        raise ValueError("duplicate (dataset, fov_name, track_id, t) in the label table")
    status = d.infection_status.value_counts().to_dict()
    death = int(d.mantis_death.sum())
    d = d[~d.mantis_death].copy()
    d["well"] = d.fov_name.str.rsplit("/", n=1).str[0]
    control = d.dataset.map(cfg["control_wells"]).eq(d.well)
    if not control.eq(d.infection_status.eq("uninfected")).all():
        raise ValueError("uninfected frames must be exactly the control-well frames")
    d["well_type"] = np.where(control, "control", "zikv")
    ambiguous = int(d.infection_status.eq("ambiguous").sum())
    d = d[d.infection_status.ne("ambiguous")].copy()
    d["group"] = d.infection_status.map(GROUP_OF_STATUS)
    if d.group.isna().any():
        raise ValueError(f"unknown infection_status values: {sorted(d.infection_status[d.group.isna()].unique())}")
    onset = d.group.eq("onset")
    if d.tau_h[onset].isna().any() or d.tau_h[~onset].notna().any():
        raise ValueError("tau_h must be set exactly for onset-lineage frames")
    if not d.cc_state.isin(ALL_PHASES).all():
        raise ValueError(f"unexpected cc_state values: {sorted(d.cc_state.unique())}")
    d["fov_key"] = d.dataset + "::" + d.fov_name
    record = dict(
        path=str(path),
        origin=cfg["labels_origin"],
        sha256=cfg["labels_sha256"],
        frames=cfg["expected_frames"],
        infection_status=status,
        mantis_death_excluded=death,
        ambiguous_excluded=ambiguous,
        used_frames=len(d),
        used_by_group=pd.crosstab(d.dataset, d.group).to_dict(),
    )
    return d.reset_index(drop=True), record


def attach_embeddings(d, cfg):
    """Join every used frame to its nonblank TOMM20 ``obsm/X_normalized_pca80`` row and obs ``x``/``y``.

    Parameters
    ----------
    d : pandas.DataFrame
        Used frames.
    cfg : dict
        Attempt config.

    Returns
    -------
    tuple of (pandas.DataFrame, numpy.ndarray, list)
        Frames with ``x``/``y``, the aligned ``(n, embedding_dims)`` float32 embeddings and a per-dataset audit.
    """
    n_dims, key = cfg["embedding_dims"], cfg["embedding_key"]
    emb = np.full((len(d), n_dims), np.nan, np.float32)
    xy = np.full((len(d), 2), np.nan)
    audit = []
    for ds, experiment in cfg["datasets"].items():
        path = Path(cfg["embedding_root"]) / experiment / cfg["embedding_suffix"]
        g = zarr.open_group(str(path), mode="r")
        meta = read_elem(g["uns"][key])
        checksum = str(meta["pca_model_checksums"][CHANNEL])
        if not checksum.startswith(cfg["pca_checksum_prefix"]):
            raise ValueError(
                f"{path}: {CHANNEL} PCA checksum {checksum} does not start with {cfg['pca_checksum_prefix']}"
            )
        uns = {k: str(read_elem(g["uns"][k])) for k in ("channel", "marker")}
        if uns["marker"] != CHANNEL:
            raise ValueError(f"{path}: uns marker {uns}")
        obs = read_elem(g["obs"]).reset_index(drop=True)
        obs["fov_name"] = obs.fov_name.astype(str)
        x = np.asarray(g["obsm"][key][:], dtype=np.float32)
        if x.shape != (len(obs), n_dims) or int(meta["n_dimensions"]) != n_dims:
            raise ValueError(f"{path}: {key} shape {x.shape}, n_dimensions {meta['n_dimensions']}")
        if obs.duplicated(KEYS).any() or not obs.experiment.astype(str).eq(experiment).all():
            raise ValueError(f"{path}: duplicate keys or foreign experiment rows")
        obs["row"] = np.arange(len(obs))
        nonblank = obs[~obs.is_blank.astype(bool)]
        sel = d.dataset.eq(ds).to_numpy()
        m = d.loc[sel, KEYS].merge(
            nonblank[[*KEYS, "row", "x", "y", "hours_post_perturbation"]], on=KEYS, how="left", validate="one_to_one"
        )
        if m.row.isna().any():
            raise ValueError(f"{CHANNEL} {ds}: {int(m.row.isna().sum())} used frames have no nonblank embedding row")
        if not np.allclose(m.hours_post_perturbation.to_numpy(float), d.hpi[sel].to_numpy(float)):
            raise ValueError(f"{ds}: label hpi disagrees with the store's hours_post_perturbation")
        rows = m.row.astype(int).to_numpy()
        emb[sel] = x[rows]
        xy[sel] = m[["x", "y"]].to_numpy(float)
        audit.append(
            dict(
                dataset=ds,
                path=str(path),
                pca_model_checksum=checksum,
                uns=uns,
                n_dimensions=n_dims,
                obsm_sha256=sha_bytes(x),
                store_rows=len(obs),
                nonblank_rows=len(nonblank),
                matched_rows=int(sel.sum()),
            )
        )
    if not np.isfinite(emb).all() or not np.isfinite(xy).all():
        raise ValueError("non-finite embeddings or coordinates after the join")
    d = d.assign(x=xy[:, 0], y=xy[:, 1])
    return d, emb, audit


def nearest_to_center(frames, bin_column, value_column, width):
    """Keep one frame per (dataset, fov_name, track_id, bin): nearest the bin centre, earliest ``t`` on ties.

    Parameters
    ----------
    frames : pandas.DataFrame
        Frames with ``bin_column`` (bin start) and ``value_column``.
    bin_column, value_column : str
        Bin-start and binned-value columns.
    width : float
        Bin width in hours.

    Returns
    -------
    pandas.DataFrame
        The representative frames.
    """
    distance = (frames[value_column] - (frames[bin_column] + width / 2)).abs()
    f = frames.assign(_distance=distance).sort_values([*TRACK, bin_column, "_distance", "t"], kind="mergesort")
    return f.drop_duplicates([*TRACK, bin_column]).drop(columns="_distance")


def representatives(d, emb, cfg):
    """Pick the representative frames of Model I and Model C.

    Onset-lineage frames with τ in [tau_start, tau_stop) get 2 h τ bins and one frame per track and bin. Bystander
    and control frames are limited to each dataset's HPI span of those onset frames and get one frame per track
    and 2 h HPI bin.

    Parameters
    ----------
    d : pandas.DataFrame
        Used frames with coordinates.
    emb : numpy.ndarray
        Embeddings aligned to ``d``.
    cfg : dict
        Attempt config.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Representatives with ``row_id`` (= row position), ``tau_bin``, ``hpi_bin``, ``class_i`` and the PCs, and a
        record of the HPI spans and counts.
    """
    width = cfg["bin_hours"]
    d = d.assign(_emb=np.arange(len(d)))
    window = d.group.eq("onset") & d.tau_h.ge(cfg["tau_start"]) & d.tau_h.lt(cfg["tau_stop"])
    onset = d[window].copy()
    onset["tau_bin"] = np.floor((onset.tau_h - cfg["tau_start"]) / width) * width + cfg["tau_start"]
    onset = nearest_to_center(onset, "tau_bin", "tau_h", width)
    onset["class_i"] = onset.tau_bin.map(tau_class)
    span = d[window].groupby("dataset").hpi.agg(["min", "max"])
    refs = d[d.group.isin(REFERENCES)].copy()
    in_span = refs.hpi.ge(refs.dataset.map(span["min"])) & refs.hpi.le(refs.dataset.map(span["max"]))
    n_refs = len(refs)
    refs = refs[in_span].copy()
    refs["hpi_bin"] = np.floor(refs.hpi / width) * width
    refs = nearest_to_center(refs, "hpi_bin", "hpi", width)
    refs["tau_bin"] = np.nan
    refs["class_i"] = refs.group
    onset["hpi_bin"] = np.floor(onset.hpi / width) * width
    reps = pd.concat([onset, refs], ignore_index=True).sort_values(["dataset", "fov_name", "track_id", "t"])
    reps = reps.reset_index(drop=True)
    if reps.duplicated(["dataset", *KEYS]).any():
        raise ValueError("a frame represents two classes")
    reps.insert(0, "row_id", np.arange(len(reps)))
    pcs = pd.DataFrame(emb[reps._emb.to_numpy()], columns=pc_columns(cfg))
    reps = pd.concat([reps.drop(columns="_emb"), pcs], axis=1)
    record = dict(
        hpi_span={ds: dict(min=float(r["min"]), max=float(r["max"])) for ds, r in span.iterrows()},
        onset_window_frames=int(window.sum()),
        reference_frames=n_refs,
        reference_frames_in_span=int(in_span.sum()),
        representatives=pd.crosstab(reps.dataset, reps.class_i).to_dict(),
        representatives_by_phase=pd.crosstab(reps.class_i, reps.cc_state).to_dict(),
    )
    return reps, record


def assign_folds(reps, n_folds, seed):
    """Deal FOVs into folds, shuffled within dataset x well type and dealt round-robin within each well type.

    Parameters
    ----------
    reps : pandas.DataFrame
        Cells with ``dataset``, ``fov_key`` and ``well_type``.
    n_folds : int
        Number of folds.
    seed : int
        Seed of the one shuffling generator.

    Returns
    -------
    dict
        ``{fov_key: fold index}``.
    """
    rng = np.random.default_rng(seed)
    fovs = reps[["dataset", "fov_key", "well_type"]].drop_duplicates()
    fold_of = {}
    for well_type in ["control", "zikv"]:
        k = 0
        for ds in sorted(fovs.dataset.unique()):
            names = np.sort(fovs.fov_key[fovs.dataset.eq(ds) & fovs.well_type.eq(well_type)].to_numpy())
            for name in rng.permutation(names):
                fold_of[str(name)] = k % n_folds
                k += 1
    return fold_of


def cv_splits(fold_of, n_folds):
    """Return the CV folds: ``test`` = fold k, ``val`` = fold k+1, ``train`` = the rest.

    Parameters
    ----------
    fold_of : dict
        ``{fov_key: fold index}``.
    n_folds : int
        Number of folds.

    Returns
    -------
    dict
        ``{fold name: {split: sorted fov_keys}}``.
    """
    out = {}
    for k in range(n_folds):
        val_k = (k + 1) % n_folds
        out[f"fold{k}"] = dict(
            train=sorted(f for f, v in fold_of.items() if v not in (k, val_k)),
            val=sorted(f for f, v in fold_of.items() if v == val_k),
            test=sorted(f for f, v in fold_of.items() if v == k),
        )
    return out


def lodo_splits(reps, val_fraction, seed):
    """Return the leave-one-dataset-out folds: train/val FOVs from one dataset (val per well type), test the other.

    Parameters
    ----------
    reps : pandas.DataFrame
        Cells with ``dataset``, ``fov_key`` and ``well_type``.
    val_fraction : float
        Fraction of each well type's FOVs held out for validation (at least one).
    seed : int
        Seed of the split generator.

    Returns
    -------
    dict
        ``{"lodo_<train>_to_<test>": {split: sorted fov_keys}}``.
    """
    rng = np.random.default_rng(seed)
    fovs = reps[["dataset", "fov_key", "well_type"]].drop_duplicates()
    datasets = sorted(fovs.dataset.unique())
    out = {}
    for train_ds in datasets:
        test_ds = [ds for ds in datasets if ds != train_ds]
        train, val = [], []
        for well_type in ["control", "zikv"]:
            names = rng.permutation(np.sort(fovs.fov_key[fovs.dataset.eq(train_ds) & fovs.well_type.eq(well_type)]))
            n_val = max(1, int(round(val_fraction * len(names))))
            val += [str(v) for v in names[:n_val]]
            train += [str(v) for v in names[n_val:]]
        test = fovs.fov_key[fovs.dataset.isin(test_ds)].tolist()
        out[f"lodo_{train_ds}_to_{'_'.join(test_ds)}"] = dict(train=sorted(train), val=sorted(val), test=sorted(test))
    return out


def check_disjoint(folds):
    """Raise if any FOV appears in two splits of one fold.

    Parameters
    ----------
    folds : dict
        ``{fold name: {split: fov_keys}}``.
    """
    for name, f in folds.items():
        seen = Counter(k for split in SPLITS for k in f[split])
        repeated = [k for k, n in seen.items() if n > 1]
        if repeated:
            raise ValueError(f"{name}: FOVs in two splits: {repeated[:5]}")


def phase_match(cells, phases, rng):
    """Subsample each phase of one pool to the smallest phase count.

    Parameters
    ----------
    cells : pandas.DataFrame
        One class pool in one split, with ``row_id`` and ``cc_state``.
    phases : list of str
        Phases to keep (others are dropped).
    rng : numpy.random.Generator
        Generator of this pool.

    Returns
    -------
    tuple of (numpy.ndarray, dict, int)
        Sorted kept ``row_id``, counts per phase before matching (all four phases) and the per-phase count after.
    """
    before = cells.cc_state.value_counts().reindex(ALL_PHASES, fill_value=0)
    n = int(before[phases].min())
    keep = [rng.permutation(np.sort(cells.row_id[cells.cc_state.eq(p)].to_numpy()))[:n] for p in phases]
    return np.sort(np.concatenate(keep)).astype(np.int64), {p: int(before[p]) for p in ALL_PHASES}, n


def source_balance(cells, rng):
    """Subsample the larger of the bystander and control sources of one Model C pool to the smaller one.

    Parameters
    ----------
    cells : pandas.DataFrame
        Reference cells of one phase in one split.
    rng : numpy.random.Generator
        Generator of this pool.

    Returns
    -------
    tuple of (numpy.ndarray, dict, int)
        Sorted kept ``row_id``, counts per source before balancing and the per-source count after.
    """
    before = cells.group.value_counts().reindex(REFERENCES, fill_value=0)
    n = int(before.min())
    keep = [rng.permutation(np.sort(cells.row_id[cells.group.eq(g)].to_numpy()))[:n] for g in REFERENCES]
    return np.sort(np.concatenate(keep)).astype(np.int64), {g: int(before[g]) for g in REFERENCES}, n


def build_pools(reps, folds, cfg):
    """Build the phase-matched Model I pools and the source-balanced Model C pools of every fold and split.

    A class is dropped from a fold when its train pool has fewer than ``min_train_cells`` cells or its val or
    test pool is empty; a dropped class is absent from all three splits of that fold. Model C gets an ``M`` class
    only if every training split has at least ``min_m_cells`` source-balanced M cells.

    Parameters
    ----------
    reps : pandas.DataFrame
        Representatives.
    folds : dict
        ``{fold name: {split: fov_keys}}``.
    cfg : dict
        Attempt config.

    Returns
    -------
    tuple of (dict, pandas.DataFrame, dict)
        ``{(model, fold): {"classes": [...], split: {class: row_ids}}}``, the count table and the M decision.
    """
    seed, phases = cfg["seed"], cfg["phases"]
    classes_i = model_i_classes(cfg)
    m_train = {}
    for fold, f in folds.items():
        refs = reps[reps.fov_key.isin(f["train"]) & reps.group.isin(REFERENCES) & reps.cc_state.eq("M")]
        m_train[fold] = int(refs.group.value_counts().reindex(REFERENCES, fill_value=0).min())
    use_m = min(m_train.values()) >= cfg["min_m_cells"]
    classes_c = [*phases, *(["M"] if use_m else [])]
    pools, rows = {}, []
    for fi, (fold, f) in enumerate(folds.items()):
        for mi, (model, classes) in enumerate([("model_i", classes_i), ("model_c", classes_c)]):
            p = {split: {} for split in SPLITS}
            for si, split in enumerate(SPLITS):
                in_split = reps.fov_key.isin(f[split])
                for ci, c in enumerate(classes):
                    rng = np.random.default_rng([seed, mi, fi, si, ci])
                    if model == "model_i":
                        keep, before, n = phase_match(reps[in_split & reps.class_i.eq(c)], phases, rng)
                        row = dict(**{f"before_{k}": v for k, v in before.items()}, per_stratum_after=n)
                    else:
                        cells = reps[in_split & reps.group.isin(REFERENCES) & reps.cc_state.eq(c)]
                        keep, before, n = source_balance(cells, rng)
                        row = dict(**{f"before_{k}": v for k, v in before.items()}, per_stratum_after=n)
                    p[split][c] = keep
                    rows.append(dict(model=model, fold=fold, split=split, cls=c, after=len(keep), **row))
            dropped = {
                c: reason
                for c in classes
                for reason in [
                    f"train {len(p['train'][c])} < {cfg['min_train_cells']}"
                    if len(p["train"][c]) < cfg["min_train_cells"]
                    else "val empty"
                    if len(p["val"][c]) == 0
                    else "test empty"
                    if len(p["test"][c]) == 0
                    else None
                ]
                if reason is not None
            }
            kept = [c for c in classes if c not in dropped]
            pools[(model, fold)] = dict(
                classes=kept, dropped=dropped, **{s: {c: p[s][c] for c in kept} for s in SPLITS}
            )
            for r in rows:
                if r["model"] == model and r["fold"] == fold:
                    r["kept"] = r["cls"] not in dropped
                    r["drop_reason"] = dropped.get(r["cls"], "")
    counts = pd.DataFrame(rows)
    decision = dict(m_train_balanced=m_train, min_m_cells=cfg["min_m_cells"], include_m=use_m, classes=classes_c)
    return pools, counts, decision


def application_sets(reps, folds):
    """List every fold's held-out onset-lineage representatives grouped by (τ bin x cc_state) for Model C.

    Parameters
    ----------
    reps : pandas.DataFrame
        Representatives.
    folds : dict
        ``{fold name: {split: fov_keys}}``.

    Returns
    -------
    pandas.DataFrame
        ``fold, row_id, tau_bin, cc_state`` of onset cells in each fold's test FOVs.
    """
    out = []
    for fold, f in folds.items():
        cells = reps[reps.fov_key.isin(f["test"]) & reps.group.eq("onset")]
        out.append(cells[["row_id", "tau_bin", "cc_state"]].assign(fold=fold))
    return pd.concat(out, ignore_index=True)[["fold", "row_id", "tau_bin", "cc_state"]]


def cell_metadata(cells):
    """Return the dump ``cell_metadata`` of one pool: one segment per column, our extra columns appended.

    Parameters
    ----------
    cells : pandas.DataFrame
        Pool cells in dump order.

    Returns
    -------
    dict
        ``{column: [list]}``.
    """
    flat = dict(
        experiment=cells.dataset.astype(str).tolist(),
        well=cells.fov_name.astype(str).tolist(),
        x_pheno=[float(v) for v in cells.x],
        y_pheno=[float(v) for v in cells.y],
        segmentation_id=[f"{a}:{b}" for a, b in zip(cells.track_id, cells.t)],
        channel_type=["fluor"] * len(cells),
        index=[int(v) for v in cells.row_id],
        lineage_uid=cells.lineage_uid.astype(str).tolist(),
        tau_h=[float(v) for v in cells.tau_h],
        hpi=[float(v) for v in cells.hpi],
        cc_state=cells.cc_state.astype(str).tolist(),
        group=cells.group.astype(str).tolist(),
    )
    return {k: [v] for k, v in flat.items()}


def write_dump(directory, classes, pool, reps, emb, n_cells):
    """Write one split in the pinned ``train.py`` dump format: ``<class>.pt`` per class plus ``metadata.pt``.

    Parameters
    ----------
    directory : pathlib.Path
        New dump directory.
    classes : list of str
        Classes of the fold, in label order.
    pool : dict
        ``{class: row_ids}`` of this split.
    reps : pandas.DataFrame
        Representatives indexed by position = ``row_id``.
    emb : numpy.ndarray
        Embeddings aligned to ``reps``.
    n_cells : int
        ``metadata.pt`` default cells per set (each run overrides it).
    """
    directory.mkdir(parents=True, exist_ok=False)
    for c in classes:
        rows = pool[c]
        payload = dict(
            embeddings=torch.from_numpy(np.ascontiguousarray(emb[rows])),
            channel_ids=torch.zeros(len(rows), dtype=torch.long),
            cell_metadata=cell_metadata(reps.iloc[rows]),
        )
        torch.save(payload, directory / f"{c}.pt")
    torch.save(
        dict(
            gene_to_idx={c: i for i, c in enumerate(classes)},
            channel_to_idx={CHANNEL: 0},
            emb_dim=int(emb.shape[1]),
            perturbation_list=list(classes),
            n_cells=int(n_cells),
        ),
        directory / "metadata.pt",
    )


def verify_dump(directory, classes, pool, emb, load_dataset):
    """Load a dump with the pinned ``load_dataset`` and check classes, embeddings, channel ids and ``index``.

    Parameters
    ----------
    directory : pathlib.Path
        Dump directory.
    classes : list of str
        Expected classes in label order.
    pool : dict
        ``{class: row_ids}`` written.
    emb : numpy.ndarray
        Embeddings aligned to ``row_id``.
    load_dataset : callable
        Pinned ``set_classifier.train.load_dataset``.

    Returns
    -------
    dict
        Cells per class as loaded.
    """
    ds = load_dataset(str(directory))
    if ds.perturbation_list != list(classes) or ds.gene_to_idx != {c: i for i, c in enumerate(classes)}:
        raise ValueError(f"{directory}: classes {ds.perturbation_list}")
    loaded = {}
    for c in classes:
        g = ds.gene_to_idx[c]
        rows = ds._gene_dump_meta[g]["index"]
        if rows != [int(v) for v in pool[c]]:
            raise ValueError(f"{directory}/{c}: index does not round-trip")
        if not torch.equal(ds._gene_pools[g], torch.from_numpy(emb[pool[c]])):
            raise ValueError(f"{directory}/{c}: embeddings do not round-trip")
        if ds._gene_ch_ids[g].ne(0).any():
            raise ValueError(f"{directory}/{c}: channel ids")
        loaded[c] = len(rows)
    return loaded


def export(cfg):
    """Stage ``export``: labels + embeddings, representatives, folds, pools and verified dumps."""
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    (work / "dumps").mkdir(exist_ok=False)
    train_module = pinned_train(cfg)
    d, label_record = load_labels(cfg)
    d, emb, join_audit = attach_embeddings(d, cfg)
    reps, rep_record = representatives(d, emb, cfg)
    fold_of = assign_folds(reps, cfg["n_folds"], cfg["seed"])
    reps["cv_fold"] = reps.fov_key.map(fold_of).astype(int)
    folds = {**cv_splits(fold_of, cfg["n_folds"]), **lodo_splits(reps, cfg["lodo_val_fraction"], cfg["seed"])}
    check_disjoint(folds)
    for fold, f in folds.items():
        split_of = {k: s for s in SPLITS for k in f[s]}
        reps[f"split_{fold}"] = reps.fov_key.map(split_of).fillna("")
    fov_table = reps.groupby(["dataset", "fov_key", "well_type", "cv_fold"]).size().rename("representatives")
    write_json(
        work / "folds.json",
        dict(
            seed=cfg["seed"],
            rule="CV: FOVs shuffled within dataset x well type and dealt round-robin within well type; test = fold k, "
            "val = fold k+1, train = rest. LODO: train/val from one dataset (val = round(0.2 n) FOVs per well type, "
            "at least 1), test = the other dataset.",
            fold_of=fold_of,
            folds=folds,
            fovs=fov_table.reset_index().to_dict(orient="records"),
        ),
    )
    pools, counts, m_decision = build_pools(reps, folds, cfg)
    counts.to_csv(work / "pool_counts.csv", index=False)
    print(counts.to_string(), flush=True)
    members = []
    emb_reps = reps[pc_columns(cfg)].to_numpy(np.float32)
    if not np.array_equal(reps.row_id.to_numpy(), np.arange(len(reps))):
        raise ValueError("row_id must equal row position")
    loaded = {}
    for (model, fold), p in pools.items():
        for split in SPLITS:
            directory = work / "dumps" / model / fold / split
            write_dump(directory, p["classes"], p[split], reps, emb_reps, cfg["dump_n_cells"])
            loaded[f"{model}/{fold}/{split}"] = verify_dump(
                directory, p["classes"], p[split], emb_reps, train_module.load_dataset
            )
            for c in p["classes"]:
                members.append(pd.DataFrame(dict(model=model, fold=fold, split=split, cls=c, row_id=p[split][c])))
    members = pd.concat(members, ignore_index=True)
    check_members(members, reps, cfg)
    members.to_parquet(work / "pool_members.parquet", index=False)
    application_sets(reps, folds).to_parquet(work / "application.parquet", index=False)
    reps.to_parquet(work / "cells.parquet", index=False)
    write_json(
        work / "export.json",
        dict(
            labels=label_record,
            embeddings=join_audit,
            representatives=rep_record,
            model_c_m_decision=m_decision,
            classes={f"{m}/{f}": p["classes"] for (m, f), p in pools.items()},
            dropped={f"{m}/{f}": p["dropped"] for (m, f), p in pools.items() if p["dropped"]},
            dumps_verified_with="pinned set_classifier.train.load_dataset",
            loaded_cells=loaded,
            cells_parquet_sha256=sha(work / "cells.parquet"),
            pool_members_sha256=sha(work / "pool_members.parquet"),
        ),
    )
    print("export done", json.dumps(dict(m_decision=m_decision, cells=len(reps)), default=str), flush=True)


def check_members(members, reps, cfg):
    """Assert the pool invariants: FOVs inside their split, equal phases (Model I) and references only (Model C).

    Parameters
    ----------
    members : pandas.DataFrame
        ``model, fold, split, cls, row_id``.
    reps : pandas.DataFrame
        Representatives.
    cfg : dict
        Attempt config.
    """
    m = members.merge(reps[["row_id", "fov_key", "cc_state", "group", "class_i"]], on="row_id", validate="many_to_one")
    for fold in m.fold.unique():
        f = m[m.fold.eq(fold)]
        fovs = {s: set(f.fov_key[f.split.eq(s)]) for s in SPLITS}
        if fovs["train"] & fovs["val"] or fovs["train"] & fovs["test"] or fovs["val"] & fovs["test"]:
            raise ValueError(f"{fold}: a FOV is in two splits")
    mi = m[m.model.eq("model_i")]
    if not mi.cls.eq(mi.class_i).all() or not mi.cc_state.isin(cfg["phases"]).all():
        raise ValueError("Model I pool holds a cell of another class or phase")
    per_phase = pd.crosstab([mi.fold, mi.split, mi.cls], mi.cc_state)
    if not per_phase.eq(per_phase.iloc[:, 0], axis=0).all().all():
        raise ValueError("Model I pools are not phase-matched")
    mc = m[m.model.eq("model_c")]
    if not mc.group.isin(REFERENCES).all() or not mc.cls.eq(mc.cc_state).all():
        raise ValueError("Model C pool holds an onset-lineage cell or a mislabeled phase")
    per_source = pd.crosstab([mc.fold, mc.split, mc.cls], mc.group)
    if not per_source.eq(per_source.iloc[:, 0], axis=0).all().all():
        raise ValueError("Model C pools are not source-balanced")


def run_env(cfg):
    """Return the subprocess environment of the pinned classifier (set on the subprocess only)."""
    threads = str(cfg["run"]["threads_per_job"])
    env = dict(os.environ)
    env.update(
        PYTHONPATH=cfg["pinned_root"],
        CONFIG_PATH=cfg["hydra_config_dir"],
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
        NUMEXPR_NUM_THREADS=threads,
        HYDRA_FULL_ERROR="1",
    )
    return env


def run_dir(root, model, fold, n_cells):
    """Return the directory of one (model, fold, training bag size) run."""
    return Path(root) / "runs" / model / fold / f"n{n_cells:03d}"


def dump_dir(cfg, model, fold, split):
    """Return one dump directory (always under the full ``work_dir``, also for the smoke run)."""
    return Path(cfg["work_dir"]) / "dumps" / model / fold / split


def train_job(cfg, root, model, fold, n_cells, num_epochs=None):
    """Return the job spec of one pinned ``train.py`` run.

    Parameters
    ----------
    cfg : dict
        Attempt config.
    root : pathlib.Path
        Output root (``work_dir`` or its ``smoke`` folder).
    model, fold : str
        Model and fold.
    n_cells : int
        Training ``n_cells_per_set``.
    num_epochs : int, optional
        Override of the shipped 100 epochs (smoke run only).

    Returns
    -------
    dict
        Job spec.
    """
    out = run_dir(root, model, fold, n_cells)
    overrides = [
        f"name=attempt25_{model}_{fold}_n{n_cells:03d}",
        f"load_train_dir={dump_dir(cfg, model, fold, 'train')}",
        f"load_val_dir={dump_dir(cfg, model, fold, 'val')}",
        f"n_cells_per_set={n_cells}",
        f"save_path={out / 'checkpoint.pt'}",
        f"metrics_out={out / 'metrics.json'}",
        f"hydra.run.dir={out / 'hydra_train'}",
    ]
    if num_epochs is not None:
        overrides.append(f"num_epochs={num_epochs}")
    cmd = [cfg["run"]["python"], "-m", "set_classifier.train", "--config-name", cfg["hydra_train_config"], *overrides]
    return dict(name=f"{model}_{fold}_n{n_cells:03d}_train", kind="train", dir=out, cmd=cmd)


def eval_job(cfg, root, model, fold, n_cells, n_cells_list):
    """Return the job spec of one pinned ``eval.py`` run on the fold's held-out ``test/`` dump."""
    out = run_dir(root, model, fold, n_cells)
    overrides = [
        f"checkpoint_path={out / 'checkpoint.pt'}",
        f"val_dump_dir={dump_dir(cfg, model, fold, 'test')}",
        f"output_plot_path={out / 'eval' / 'accuracy_vs_n.png'}",
        f"per_class_output_file={out / 'eval' / 'per_class.csv'}",
        f"n_cells_list=[{','.join(str(n) for n in n_cells_list)}]",
        f"n_repetitions={cfg['run']['eval_repetitions']}",
        "sample_without_replacement=true",
        f"hydra.run.dir={out / 'eval' / 'hydra'}",
    ]
    cmd = [cfg["run"]["python"], "-m", "set_classifier.eval", "--config-name", cfg["hydra_eval_config"], *overrides]
    return dict(name=f"{model}_{fold}_n{n_cells:03d}_eval", kind="eval", dir=out, cmd=cmd)


def score_job(cfg, root, model, fold, n_cells, classes):
    """Return the job spec of one pinned ``score.py`` run on the fold's held-out ``test/`` dump."""
    out = run_dir(root, model, fold, n_cells)
    cmd = [
        cfg["run"]["python"],
        "-m",
        "set_classifier.score",
        "--checkpoint",
        str(out / "checkpoint.pt"),
        "--dump_dir",
        str(dump_dir(cfg, model, fold, "test")),
        "--channel",
        CHANNEL,
        "--genes",
        *classes,
        "--bag_sizes",
        *(str(b) for b in cfg["run"]["score_bag_sizes"]),
        "--reps",
        str(cfg["run"]["score_reps"]),
        "--out_csv",
        str(out / "score.csv"),
        "--device",
        "cuda",
    ]
    return dict(name=f"{model}_{fold}_n{n_cells:03d}_score", kind="score", dir=out, cmd=cmd)


class Runner:
    """Run job specs as subprocesses, at most ``parallel`` at a time, with per-job logs, PIDs and done markers."""

    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, Path(root)
        self.env = run_env(cfg)
        self.lock = threading.Lock()
        self.log = self.root / "run_log.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, entry):
        """Append one JSON line to ``run_log.jsonl`` and echo it."""
        with self.lock:
            with open(self.log, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
            print(json.dumps(entry, default=str), flush=True)

    def execute(self, job):
        """Run one job unless its done marker exists; raise on a non-zero exit."""
        marker = job["dir"] / f"{job['kind']}.done"
        if marker.exists():
            self.record(dict(job=job["name"], event="skip_done"))
            return
        job["dir"].mkdir(parents=True, exist_ok=True)
        tag = "smoke_" if self.root.name == "smoke" else ""
        log = Path(self.cfg["run"]["log_dir"]) / f"{self.cfg['run']['log_prefix']}_{tag}{job['name']}.log"
        start = time.time()
        with open(log, "w") as fh:
            proc = subprocess.Popen(job["cmd"], env=self.env, cwd=job["dir"], stdout=fh, stderr=subprocess.STDOUT)
            self.record(dict(job=job["name"], event="start", pid=proc.pid, host=socket.gethostname(), log=str(log)))
            rc = proc.wait()
        entry = dict(
            job=job["name"],
            event="end",
            pid=proc.pid,
            returncode=rc,
            seconds=round(time.time() - start, 1),
            log=str(log),
        )
        self.record(entry)
        if rc != 0:
            raise RuntimeError(f"{job['name']} exited {rc}; see {log}")
        write_json(marker, dict(entry, cmd=job["cmd"]))

    def run_all(self, jobs):
        """Run a batch of independent jobs and raise after the batch if any failed."""
        with ThreadPoolExecutor(max_workers=self.cfg["run"]["parallel"]) as pool:
            futures = [pool.submit(self.execute, job) for job in jobs]
        errors = [f.exception() for f in futures if f.exception() is not None]
        if errors:
            raise RuntimeError(f"{len(errors)} job(s) failed: {errors}")


def fold_classes(cfg, model, fold):
    """Return the classes of one fold's dump (label order)."""
    meta = torch.load(dump_dir(cfg, model, fold, "train") / "metadata.pt", map_location="cpu", weights_only=False)
    return list(meta["perturbation_list"])


def eval_sizes(cfg, model, fold):
    """Cap the eval bag-size list at the fold's smallest held-out test pool.

    ``eval.py`` pads a pool smaller than ``n_cells`` and uses all of its cells, so every repetition repeats the
    same bag; sizes above the smallest test pool are dropped for that fold.

    Returns
    -------
    tuple of (list, int)
        Kept sizes and the smallest test pool.
    """
    counts = pd.read_csv(Path(cfg["work_dir"]) / "pool_counts.csv")
    c = counts[counts.model.eq(model) & counts.fold.eq(fold) & counts.split.eq("test") & counts.kept]
    smallest = int(c["after"].min())
    return [n for n in cfg["run"]["eval_n_cells"] if n <= smallest], smallest


def select_bag_sizes(cfg, root, cv_folds):
    """Choose each model's training ``n_cells_per_set`` by mean inner-val accuracy over the CV folds.

    Reads the ``metrics_out`` JSON of every run (the val accuracy of the epoch ``train.py`` checkpointed); ties go
    to the smaller size. Test results are never read.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Per-run table with ``chosen`` and ``{model: n_cells}``.
    """
    rows = []
    for model in MODELS:
        for fold in cv_folds:
            for n in cfg["run"]["n_cells_per_set"]:
                out = run_dir(root, model, fold, n)
                m = json.loads((out / "metrics.json").read_text())
                ckpt = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
                rows.append(
                    dict(
                        model=model,
                        fold=fold,
                        n_cells_per_set=n,
                        best_epoch=int(ckpt["epoch"]),
                        **{k: v for k, v in m.items() if k != "n_cells_per_set"},
                    )
                )
    df = pd.DataFrame(rows)
    summary = df.groupby(["model", "n_cells_per_set"]).val_accuracy.agg(["mean", "std"]).reset_index()
    chosen = {}
    for model, g in summary.groupby("model"):
        g = g.sort_values("n_cells_per_set")
        chosen[model] = int(g.n_cells_per_set.iloc[int(np.argmax(g["mean"].to_numpy()))])
    df = df.merge(summary.rename(columns={"mean": "mean_val_accuracy", "std": "sd_val_accuracy"}))
    df["chosen"] = df.n_cells_per_set.eq(df.model.map(chosen))
    return df, chosen


def check_node(cfg):
    """Raise unless running on the reserved GPU node."""
    host = socket.gethostname()
    if not host.startswith(cfg["run"]["host"]):
        raise RuntimeError(f"stage run must execute on {cfg['run']['host']} (ssh), not {host}")


def run(cfg, smoke):
    """Stage ``run``: pinned train -> bag-size choice -> eval + score, then leave-one-dataset-out, on the GPU node."""
    check_node(cfg)
    torch.set_num_threads(1)
    work = Path(cfg["work_dir"])
    folds = json.loads((work / "folds.json").read_text())["folds"]
    cv_folds = [f for f in folds if f.startswith("fold")]
    lodo_folds = [f for f in folds if f.startswith("lodo")]
    if smoke:
        s = cfg["run"]["smoke"]
        root = work / "smoke"
        runner = Runner(cfg, root)
        jobs = [train_job(cfg, root, m, s["fold"], s["n_cells_per_set"], s["num_epochs"]) for m in MODELS]
        runner.run_all(jobs)
        jobs = []
        for m in MODELS:
            jobs.append(eval_job(cfg, root, m, s["fold"], s["n_cells_per_set"], eval_sizes(cfg, m, s["fold"])[0]))
            jobs.append(score_job(cfg, root, m, s["fold"], s["n_cells_per_set"], fold_classes(cfg, m, s["fold"])))
        runner.run_all(jobs)
        print("smoke run done", flush=True)
        return
    runner = Runner(cfg, work)
    sizes = cfg["run"]["n_cells_per_set"]
    runner.run_all([train_job(cfg, work, m, f, n) for m in MODELS for f in cv_folds for n in sizes])
    selection, chosen = select_bag_sizes(cfg, work, cv_folds)
    selection.to_csv(work / "bag_size_selection.csv", index=False)
    print(selection.to_string(), "\nchosen", chosen, flush=True)
    caps = []
    jobs = []
    for m in MODELS:
        for f in [*cv_folds, *lodo_folds]:
            kept, smallest = eval_sizes(cfg, m, f)
            caps.append(dict(model=m, fold=f, smallest_test_pool=smallest, eval_n_cells=kept))
    pd.DataFrame(caps).to_csv(work / "eval_caps.csv", index=False)
    cap = {(c["model"], c["fold"]): c["eval_n_cells"] for c in caps}
    lodo_train = [train_job(cfg, work, m, f, chosen[m]) for m in MODELS for f in lodo_folds]
    jobs = [*lodo_train]
    jobs += [eval_job(cfg, work, m, f, n, cap[(m, f)]) for m in MODELS for f in cv_folds for n in sizes]
    jobs += [score_job(cfg, work, m, f, chosen[m], fold_classes(cfg, m, f)) for m in MODELS for f in cv_folds]
    runner.run_all(jobs)
    jobs = [eval_job(cfg, work, m, f, chosen[m], cap[(m, f)]) for m in MODELS for f in lodo_folds]
    jobs += [score_job(cfg, work, m, f, chosen[m], fold_classes(cfg, m, f)) for m in MODELS for f in lodo_folds]
    runner.run_all(jobs)
    write_json(work / "run_summary.json", dict(chosen=chosen, cv_folds=cv_folds, lodo_folds=lodo_folds, caps=caps))
    print("run done", flush=True)


def tau_of(cls):
    """Return the τ bin start (hours) of a ``TAU_*`` class."""
    return int(cls[5:]) * (-1 if cls[4] == "m" else 1)


def tau_colors(cfg):
    """Return one colour per τ class: a blue arm before onset (dark = early) and a red arm from onset (dark = late)."""
    pre = [b for b in tau_bins(cfg) if b < 0]
    post = [b for b in tau_bins(cfg) if b >= 0]
    blue = LinearSegmentedColormap.from_list("pre", ["#0d366b", "#86b6ef"])
    red = LinearSegmentedColormap.from_list("post", ["#f4a582", "#67001f"])
    colors = {tau_class(b): to_hex(blue(i / max(1, len(pre) - 1))) for i, b in enumerate(pre)}
    colors.update({tau_class(b): to_hex(red(i / max(1, len(post) - 1))) for i, b in enumerate(post)})
    return colors


def class_colors(cfg):
    """Return the display colour of every class of both models."""
    return {**tau_colors(cfg), **REFERENCE_COLORS, **PHASE_COLORS}


def signed(value):
    """Return an hour offset as ``0``, ``+2`` or ``−2`` (true minus sign)."""
    return "0" if value == 0 else f"{value:+d}".replace("-", "−")


def tau_label(cls, width):
    """Return a τ bin as display text, e.g. ``TAU_p00 -> τ 0…+2 h`` and ``TAU_m02 -> τ −2…0 h``."""
    b = tau_of(cls)
    return f"τ {signed(b)}…{signed(b + width)} h"


def save(fig, stem, formats=("png", "pdf")):
    """Write a figure as ``<stem>.<ext>`` (png at 200 dpi) and close it."""
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fold_names(work):
    """Return the CV and leave-one-dataset-out fold names."""
    folds = json.loads((Path(work) / "folds.json").read_text())["folds"]
    return [f for f in folds if f.startswith("fold")], [f for f in folds if f.startswith("lodo")]


def pools_of(members, model, fold, split):
    """Return ``{class: row_ids}`` of one pool split."""
    m = members[members.model.eq(model) & members.fold.eq(fold) & members.split.eq(split)]
    return {c: g.row_id.to_numpy() for c, g in m.groupby("cls")}


def load_classifier(train_module, path, device):
    """Rebuild a checkpoint with the pinned ``build_model``.

    Returns
    -------
    tuple of (torch.nn.Module, list, int)
        Model in eval mode, classes in output order and the checkpointed epoch.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = train_module.build_model(ckpt, device)
    classes = [c for c, _ in sorted(ckpt["gene_to_idx"].items(), key=lambda kv: kv[1])]
    return model, classes, int(ckpt["epoch"])


@torch.no_grad()
def bag_probs(model, emb, bags, device, batch=512):
    """Return the full softmax of the pinned classifier for an ``(n_bags, size)`` array of row ids (no padding)."""
    out = []
    for i in range(0, len(bags), batch):
        x = torch.from_numpy(emb[bags[i : i + batch]]).to(device)
        ch = torch.zeros(x.shape[:2], dtype=torch.long, device=device)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        out.append(torch.softmax(model(x, ch, mask).double(), -1).cpu().numpy())
    return np.concatenate(out)


def analysis_size(cfg, counts, model, chosen, cv_folds):
    """Return the bag size of our own test bags.

    The largest eval size that is at most the chosen training size and at most half of the smallest CV test pool,
    so every class (smallest pool included) yields distinct bags drawn without replacement.
    """
    c = counts[counts.model.eq(model) & counts.split.eq("test") & counts.kept & counts.fold.isin(cv_folds)]
    bound = min(chosen, int(c["after"].min()) // 2)
    return max(n for n in cfg["run"]["eval_n_cells"] if n <= bound)


def draw_bags(pool, classes, size, n_bags, rng, cap=False):
    """Draw ``n_bags`` bags per class, without replacement inside a bag.

    Parameters
    ----------
    pool : dict
        ``{class: row_ids}``.
    classes : list of str
        Classes in output order.
    size : int
        Bag size.
    n_bags : int
        Bags per class.
    rng : numpy.random.Generator
        Generator.
    cap : bool, optional
        Cap the size at a smaller pool (validation bags only); otherwise a smaller pool raises.

    Returns
    -------
    tuple of (list, numpy.ndarray)
        Bags (row-id arrays) and their class labels.
    """
    bags, labels = [], []
    for c in classes:
        rows = pool[c]
        if len(rows) < size and not cap:
            raise ValueError(f"{c}: pool of {len(rows)} < bag size {size}")
        k = min(size, len(rows))
        bags += [rng.choice(rows, k, replace=False) for _ in range(n_bags)]
        labels += [c] * n_bags
    return bags, np.array(labels)


def partition_bags(rows, size, n_partitions, rng):
    """Split ``n_partitions`` random permutations of ``rows`` into full bags of ``size`` (remainders dropped)."""
    k = len(rows) // size
    return np.concatenate([rng.permutation(rows)[: k * size].reshape(k, size) for _ in range(n_partitions)])


def bag_features(bags, emb, hpi):
    """Return the mean-pooled embedding and the (mean, SD) HPI of every bag."""
    mean = np.stack([emb[b].mean(0) for b in bags])
    time_ = np.stack([[hpi[b].mean(), hpi[b].std()] for b in bags])
    return mean, time_


def fit_baseline(x_train, y_train, x_val, y_val, classes, grid):
    """Fit standardized multinomial logistic regressions and keep the C with the lowest validation log-loss.

    Returns
    -------
    tuple of (callable, dict)
        ``predict(x) -> probabilities in class order`` and the selected C with its validation log-loss.
    """
    scaler = StandardScaler().fit(x_train)
    best = None
    for c in grid:
        lr = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x_train), y_train)
        loss = log_loss(y_val, lr.predict_proba(scaler.transform(x_val)), labels=lr.classes_)
        if best is None or loss < best[0]:
            best = (loss, c, lr)
    loss, c, lr = best
    order = [list(lr.classes_).index(k) for k in classes]
    return (lambda x: lr.predict_proba(scaler.transform(x))[:, order]), dict(C=c, val_log_loss=float(loss))


def pair_auroc(p, labels, classes, pos, neg):
    """AUROC of ``pos`` bags vs ``neg`` bags on q = P(pos) / (P(pos) + P(neg))."""
    sel = np.isin(labels, [pos, neg])
    i, j = classes.index(pos), classes.index(neg)
    q = p[sel, i] / (p[sel, i] + p[sel, j])
    return float(roc_auc_score(labels[sel] == pos, q))


def model_i_fold(cfg, ctx, fold, fi):
    """Score one fold's identical seeded test bags with Model I and the two logistic-regression baselines.

    Test bags: ``test_bags_per_class`` per class of ``ctx['size']['model_i']`` cells, without replacement inside a
    bag. Baselines train on ``baseline_train_bags_per_class`` bags per class drawn the same way from the train pools
    and select C on validation bags.

    Returns
    -------
    dict
        Classes, epoch, bags, labels, probabilities per method and baseline fits.
    """
    a = cfg["analyze"]
    path = run_dir(ctx["work"], "model_i", fold, ctx["chosen"]["model_i"]) / "checkpoint.pt"
    model, classes, epoch = load_classifier(ctx["train"], path, ctx["device"])
    rng = np.random.default_rng([cfg["seed"], 25, 1, fi])
    size = ctx["size"]["model_i"]
    pool = {s: pools_of(ctx["members"], "model_i", fold, s) for s in SPLITS}
    test, labels = draw_bags(pool["test"], classes, size, a["test_bags_per_class"], rng)
    train, y_train = draw_bags(pool["train"], classes, size, a["baseline_train_bags_per_class"], rng)
    val, y_val = draw_bags(pool["val"], classes, size, a["test_bags_per_class"], rng, cap=True)
    feats = {k: bag_features(v, ctx["emb"], ctx["hpi"]) for k, v in dict(train=train, val=val, test=test).items()}
    predict = {"set_classifier": lambda bags: bag_probs(model, ctx["emb"], np.stack(bags), ctx["device"])}
    fits = {}
    for method, j in [("mean_logreg", 0), ("hpi_logreg", 1)]:
        fn, fits[method] = fit_baseline(
            feats["train"][j], y_train, feats["val"][j], y_val, classes, a["baseline_c_grid"]
        )
        predict[method] = lambda bags, fn=fn, j=j: fn(bag_features(bags, ctx["emb"], ctx["hpi"])[j])
    probs = {method: fn(test) for method, fn in predict.items()}
    matched = {}
    for pos in [c for c in classes if c.startswith("TAU_")] + ["BYSTANDER"]:
        for ref in [r for r in REFERENCES if r != pos]:
            pos_bags = [test[i] for i in np.flatnonzero(labels == pos)]
            bags, gap, crossed = matched_reference_bags(pos_bags, pool["test"][ref], ctx["cells"], rng)
            matched[(pos, ref)] = dict(
                bags=bags,
                hpi_gap=gap,
                cross_dataset=crossed,
                probs={method: fn(bags) for method, fn in predict.items()},
            )
    return dict(
        fold=fold, classes=classes, epoch=epoch, bags=test, labels=labels, probs=probs, fits=fits, matched=matched
    )


def matched_reference_bags(bags, ref_rows, cells, rng):
    """Draw, for every bag, a reference bag with the same (dataset, 2 h HPI bin) composition.

    Each cell of a bag is matched to a distinct reference cell of the same dataset, taking the nearest HPI bin
    first (random within a bin) and moving to neighbouring bins only when a bin runs out. When the fold's reference
    test pool holds no cell of that dataset (e.g. a fold without a control FOV of one dataset), the cell is matched
    on HPI bin across datasets and counted as cross-dataset.

    Parameters
    ----------
    bags : list of numpy.ndarray
        Row-id bags to match.
    ref_rows : numpy.ndarray
        Reference test pool.
    cells : pandas.DataFrame
        Representatives (``dataset``, ``hpi_bin``, ``hpi``) indexed by position = ``row_id``.
    rng : numpy.random.Generator
        Generator.

    Returns
    -------
    tuple of (list, list, list)
        Matched bags, the absolute difference of bag-mean HPI per pair and the cross-dataset cells per pair.
    """
    dataset, hpi_bin, hpi = cells.dataset.to_numpy(), cells.hpi_bin.to_numpy(float), cells.hpi.to_numpy(float)
    ref_rows = np.asarray(ref_rows)
    out, gaps, crossed = [], [], []
    for bag in bags:
        free = np.ones(len(ref_rows), bool)
        picks, n_cross = [], 0
        for (ds, b), n in Counter(zip(dataset[bag], hpi_bin[bag])).items():
            same = dataset[ref_rows] == ds
            if not same.any():
                same = np.ones(len(ref_rows), bool)
                n_cross += n
            cand = np.flatnonzero(same & free)
            if len(cand) < n:
                raise ValueError(f"reference pool too small to match {n} cells of {ds}")
            order = np.lexsort((rng.random(len(cand)), np.abs(hpi_bin[ref_rows[cand]] - b)))
            take = cand[order[:n]]
            free[take] = False
            picks.append(ref_rows[take])
        matched = np.concatenate(picks)
        out.append(matched)
        gaps.append(float(abs(hpi[matched].mean() - hpi[bag].mean())))
        crossed.append(n_cross)
    return out, gaps, crossed


def model_c_fold(cfg, ctx, fold, fi):
    """Score one fold's seeded reference test bags with Model C (``ctx['size']['model_c']`` cells per bag)."""
    path = run_dir(ctx["work"], "model_c", fold, ctx["chosen"]["model_c"]) / "checkpoint.pt"
    model, classes, epoch = load_classifier(ctx["train"], path, ctx["device"])
    rng = np.random.default_rng([cfg["seed"], 25, 2, fi])
    pool = pools_of(ctx["members"], "model_c", fold, "test")
    test, labels = draw_bags(pool, classes, ctx["size"]["model_c"], cfg["analyze"]["test_bags_per_class"], rng)
    probs = bag_probs(model, ctx["emb"], np.stack(test), ctx["device"])
    return dict(fold=fold, classes=classes, epoch=epoch, bags=test, labels=labels, probs=probs, model=model)


def result_tables(res, model):
    """Flatten one fold result into bag, probability and accuracy rows."""
    bags, probs, acc = [], [], []
    for b, (rows, label) in enumerate(zip(res["bags"], res["labels"])):
        bags.append(
            dict(model=model, fold=res["fold"], bag_id=b, cls=label, size=len(rows), rows=",".join(map(str, rows)))
        )
    methods = res["probs"] if isinstance(res["probs"], dict) else {"set_classifier": res["probs"]}
    for method, p in methods.items():
        pred = np.array(res["classes"])[p.argmax(1)]
        frame = pd.DataFrame(p, columns=[f"p_{c}" for c in res["classes"]])
        probs.append(
            frame.assign(
                model=model, fold=res["fold"], method=method, bag_id=np.arange(len(p)), cls=res["labels"], pred=pred
            )
        )
        for c in res["classes"]:
            sel = res["labels"] == c
            acc.append(dict(model=model, fold=res["fold"], method=method, cls=c, top1=float((pred[sel] == c).mean())))
        acc.append(
            dict(
                model=model, fold=res["fold"], method=method, cls="__all__", top1=float((pred == res["labels"]).mean())
            )
        )
    return bags, probs, acc


def auroc_rows(res):
    """Per-τ AUROC vs each reference and the BYSTANDER-vs-CONTROL AUROC for every method of one fold.

    ``reference_bags = "pool"``: the class's test bags vs the reference's own test bags (as specified).
    ``reference_bags = "hpi_matched"``: the same class bags vs reference bags matched to each class bag's
    (dataset, 2 h HPI bin) composition.
    """
    rows = []
    classes, labels = res["classes"], res["labels"]
    for (pos, ref), m in res["matched"].items():
        i, j = classes.index(pos), classes.index(ref)
        tau_bin = tau_of(pos) if pos.startswith("TAU_") else np.nan
        for method, p in res["probs"].items():
            base = dict(fold=res["fold"], cls=pos, tau_bin=tau_bin, reference=ref, method=method)
            rows.append(dict(base, reference_bags="pool", auroc=pair_auroc(p, labels, classes, pos, ref)))
            q_pos = p[labels == pos, i] / (p[labels == pos, i] + p[labels == pos, j])
            pm = m["probs"][method]
            q_neg = pm[:, i] / (pm[:, i] + pm[:, j])
            y = np.r_[np.ones(len(q_pos)), np.zeros(len(q_neg))]
            auroc = float(roc_auc_score(y, np.r_[q_pos, q_neg]))
            cross = float(np.sum(m["cross_dataset"]) / sum(len(b) for b in m["bags"]))
            rows.append(
                dict(
                    base,
                    reference_bags="hpi_matched",
                    auroc=auroc,
                    mean_hpi_gap=float(np.mean(m["hpi_gap"])),
                    cross_dataset_fraction=cross,
                )
            )
    return rows


def confusion(results, model):
    """Row-normalized set-classifier confusion averaged over folds (identical class lists required)."""
    classes = results[0]["classes"]
    mats = []
    for res in results:
        if res["classes"] != classes:
            raise ValueError("folds disagree on classes; confusion cannot be averaged")
        p = res["probs"]["set_classifier"] if isinstance(res["probs"], dict) else res["probs"]
        pred = p.argmax(1)
        true = np.array([classes.index(c) for c in res["labels"]])
        m = np.zeros((len(classes), len(classes)))
        np.add.at(m, (true, pred), 1)
        mats.append(m / m.sum(1, keepdims=True))
    mean = np.mean(mats, 0)
    rows = [
        dict(model=model, true=t, pred=p, rate=float(mean[i, j]))
        for i, t in enumerate(classes)
        for j, p in enumerate(classes)
    ]
    return mean, classes, rows


def distortion_rows(cfg, ctx, fold, fi, model, classes):
    """Model C's mean predicted cycle distribution for held-out onset bags per (τ bin x called phase) vs references.

    Each group of ``n`` onset cells (``n >= distortion_min_cells``) is split by ``distortion_partitions`` random
    permutations into bags of ``min(distortion_max_bag, n)`` cells, so each cell is in up to that many bags. The
    reference uses the same partitions of the fold's Model C test pool of the same called phase, at the same size.
    """
    a = cfg["analyze"]
    app = ctx["application"][ctx["application"].fold.eq(fold)]
    refs = pools_of(ctx["members"], "model_c", fold, "test")
    rng = np.random.default_rng([cfg["seed"], 25, 3, fi])
    rows = []
    for (tau_bin, phase), g in app.groupby(["tau_bin", "cc_state"]):
        n = len(g)
        row = dict(fold=fold, tau_bin=int(tau_bin), cc_state=phase, n_cells=n)
        if n < a["distortion_min_cells"]:
            rows.append(dict(row, skipped=f"n < {a['distortion_min_cells']}"))
            continue
        size = min(a["distortion_max_bag"], n)
        bags = partition_bags(np.sort(g.row_id.to_numpy()), size, a["distortion_partitions"], rng)
        p = bag_probs(model, ctx["emb"], bags, ctx["device"]).mean(0)
        row.update(bag_size=size, n_bags=len(bags), skipped="", **{f"p_{c}": float(v) for c, v in zip(classes, p)})
        if phase in refs:
            rb = partition_bags(refs[phase], size, a["distortion_partitions"], rng)
            rp = bag_probs(model, ctx["emb"], rb, ctx["device"]).mean(0)
            k = classes.index(phase)
            row.update(
                ref_cells=len(refs[phase]),
                ref_bags=len(rb),
                **{f"ref_p_{c}": float(v) for c, v in zip(classes, rp)},
                p_called=float(p[k]),
                ref_p_called=float(rp[k]),
                delta_called=float(p[k] - rp[k]),
            )
        rows.append(row)
    return rows


def load_scores(cfg, work, cells, members, chosen, folds):
    """Read every ``score.py`` CSV at the chosen size and join it to ``cells`` on ``index`` = ``row_id``.

    Raises if a row fails to join, its dataset/FOV/segmentation id disagree, or a test pool is not scored exactly.
    """
    out = []
    for model in MODELS:
        for fold in folds:
            s = pd.read_csv(
                run_dir(work, model, fold, chosen[model]) / "score.csv",
                dtype={"experiment": str, "well": str, "segmentation_id": str},
            )
            out.append(s.assign(model=model, fold=fold))
    s = pd.concat(out, ignore_index=True).rename(columns={"gene": "cls", "zarr_channel_index": "row_id"})
    keep = ["row_id", "dataset", "fov_name", "track_id", "t", "lineage_uid", "group", "class_i", "tau_bin", "tau_h"]
    keep += ["hpi", "cc_state", "p_G1", "p_S", "p_G2", "p_M", "x", "y", "umap1", "umap2", "well_type"]
    j = s.merge(cells[keep], on="row_id", how="left", validate="many_to_one")
    seg = j.track_id.astype(str) + ":" + j.t.astype(str)
    if (
        j.dataset.isna().any()
        or not (j.experiment.eq(j.dataset) & j.well.eq(j.fov_name) & j.segmentation_id.eq(seg)).all()
    ):
        raise ValueError("score rows do not join back to cells on index")
    for (model, fold, cls), g in j.groupby(["model", "fold", "cls"]):
        expected = np.sort(
            members.row_id[
                members.model.eq(model) & members.fold.eq(fold) & members.split.eq("test") & members.cls.eq(cls)
            ]
        )
        if not np.array_equal(np.sort(g.row_id.to_numpy()), expected):
            raise ValueError(f"{model}/{fold}/{cls}: scored cells differ from the test pool")
    return j


def single_cell_probs(models, cells_by_fold, emb, device):
    """Bag-of-1 softmax of each fold's model for the rows listed per fold; returns probabilities in input order."""
    out = []
    for fold, rows in cells_by_fold:
        model, classes = models[fold]
        out.append(bag_probs(model, emb, np.asarray(rows)[:, None], device))
    return np.concatenate(out) if out else np.zeros((0, 0))


def matched_random(top, candidates, rng, by=("fold", "cc_state")):
    """Draw one random candidate per top cell with the same values of ``by``, without replacement.

    Parameters
    ----------
    top : pandas.DataFrame
        Top cells with the ``by`` columns.
    candidates : callable
        ``(*values of by) -> row_ids`` to draw from.
    rng : numpy.random.Generator
        Generator.
    by : tuple of str, optional
        Matching columns; ``fold`` and ``cc_state`` must be among them.

    Returns
    -------
    pandas.DataFrame
        ``fold, row_id, cc_state`` (fewer rows when a candidate pool is smaller than the request).
    """
    out = []
    for key, g in top.groupby(list(by)):
        rows = candidates(*key)
        k = min(len(g), len(rows))
        out.append(pd.DataFrame(dict(zip(by, key), row_id=rng.choice(rows, k, replace=False))))
    return pd.concat(out, ignore_index=True)[["fold", "row_id", "cc_state"]]


def bootstrap_diff(a, b, n, rng):
    """Return the mean difference ``mean(a) - mean(b)`` and its percentile 95% bootstrap interval over cells."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cross_scoring(cfg, ctx, scores, models_i, models_c, classes_c, cv_folds):
    """Score Model I's top infected cells with Model C (bag of 1), and Model C's top cycle cells with Model I.

    Forward: the ``cross_top`` highest-scoring cells of each post-onset τ class (pooled over CV folds) get Model C's
    single-cell softmax from their own fold; compared with random reference cells and random same-class cells of the
    same fold and called phase. Reverse: Model C's top cells of each phase get Model I's single-cell infection
    score q_inf = Σ P(τ ≥ 0) / (Σ P(τ ≥ 0) + P(BYSTANDER) + P(CONTROL)); compared with random reference cells of
    the same fold, called phase and source (bystander or control, since Model I separates the two) and random
    post-onset cells of the same fold and called phase.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Per-cell rows and the summary table with bootstrap intervals of top minus random reference.
    """
    a = cfg["analyze"]
    rng = np.random.default_rng([cfg["seed"], 25, 4])
    s = scores[scores.fold.isin(cv_folds)]
    ref_c = {f: pools_of(ctx["members"], "model_c", f, "test") for f in cv_folds}
    pool_i = {f: pools_of(ctx["members"], "model_i", f, "test") for f in cv_folds}
    phase = ctx["cells"].cc_state.to_numpy()
    source = ctx["cells"].group.to_numpy()
    post = [c for c in model_i_classes(cfg) if c.startswith("TAU_p")]
    per_cell = []

    def evaluate(cells, models, direction, cls, name):
        cells = cells.sort_values(["fold", "row_id"]).reset_index(drop=True)
        p = single_cell_probs(
            models, [(f, g.row_id.to_numpy()) for f, g in cells.groupby("fold", sort=True)], ctx["emb"], ctx["device"]
        )
        out = cells[["fold", "row_id", "cc_state"]].assign(direction=direction, cls=cls, set=name)
        out["group"] = source[out.row_id.to_numpy()]
        if direction == "model_i_top_to_model_c":
            out["argmax"] = np.array(classes_c)[p.argmax(1)]
            out["argmax_is_called"] = out.argmax.eq(out.cc_state)
            out["p_called"] = p[np.arange(len(p)), [classes_c.index(c) for c in out.cc_state]]
            for k, c in enumerate(classes_c):
                out[f"p_{c}"] = p[:, k]
        else:
            classes_i = models[cv_folds[0]][1]
            inf = [classes_i.index(c) for c in classes_i if c.startswith("TAU_p")]
            ref = [classes_i.index(c) for c in REFERENCES]
            out["q_inf"] = p[:, inf].sum(1) / (p[:, inf].sum(1) + p[:, ref].sum(1))
            out["p_reference"] = p[:, ref].sum(1)
        per_cell.append(out)

    def reference(f, ph):
        return ref_c[f][ph]

    def same_class(f, ph, c):
        return pool_i[f][c][phase[pool_i[f][c]] == ph]

    def post_onset(f, ph):
        return np.concatenate([same_class(f, ph, c) for c in post])

    def same_source(f, ph, grp):
        return ref_c[f][ph][source[ref_c[f][ph]] == grp]

    for cls in post:
        top = s[s.model.eq("model_i") & s.cls.eq(cls)].sort_values("score", ascending=False).head(a["cross_top"])
        evaluate(top, models_c, "model_i_top_to_model_c", cls, "top")
        evaluate(matched_random(top, reference, rng), models_c, "model_i_top_to_model_c", cls, "random_reference")
        evaluate(
            matched_random(top, lambda f, ph, c=cls: same_class(f, ph, c), rng),
            models_c,
            "model_i_top_to_model_c",
            cls,
            "random_same_class",
        )
    for cls in classes_c:
        top = s[s.model.eq("model_c") & s.cls.eq(cls)].sort_values("score", ascending=False).head(a["cross_top"])
        evaluate(top, models_i, "model_c_top_to_model_i", cls, "top")
        evaluate(
            matched_random(top, same_source, rng, ("fold", "cc_state", "group")),
            models_i,
            "model_c_top_to_model_i",
            cls,
            "random_reference",
        )
        if cls in cfg["phases"]:
            evaluate(matched_random(top, post_onset, rng), models_i, "model_c_top_to_model_i", cls, "random_post_onset")
    cells = pd.concat(per_cell, ignore_index=True)
    rows = []
    groups = [(d, c, g) for (d, c), g in cells.groupby(["direction", "cls"], sort=False)]
    for direction in cells.direction.unique():
        d = cells[cells.direction.eq(direction)]
        pooled = "TAU_p* pooled" if direction == "model_i_top_to_model_c" else "all phases pooled"
        groups.append((direction, pooled, d))
    for direction, cls, g in groups:
        metrics = ["argmax_is_called", "p_called"] if direction == "model_i_top_to_model_c" else ["q_inf"]
        for name, h in g.groupby("set", sort=False):
            row = dict(
                direction=direction,
                cls=cls,
                set=name,
                n=len(h),
                bystander_fraction=float(h.group.eq("BYSTANDER").mean()),
                phases=h.cc_state.value_counts().to_dict(),
            )
            for metric in metrics:
                row[metric] = float(h[metric].astype(float).mean())
                if name != "random_reference":
                    base = g[g.set.eq("random_reference")][metric].astype(float)
                    diff, lo, hi = bootstrap_diff(h[metric].astype(float), base, a["bootstrap"], rng)
                    row.update({f"{metric}_minus_reference": diff, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
            rows.append(row)
    return cells, pd.DataFrame(rows)


def plot_sweep(summary, chosen, path, sizes):
    """Accuracy vs evaluation bag size, one line per training size, mean ± fold SD, with chance."""
    ramp = dict(zip(sizes, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, model in zip(axes, MODELS):
        s = summary[summary.model.eq(model)]
        for n_train in sizes:
            g = s[s.n_train.eq(n_train)].sort_values("n_eval")
            lw = 2.0 if n_train == chosen[model] else 1.0
            label = f"train {n_train}" + (" (chosen by val)" if n_train == chosen[model] else "")
            ax.plot(g.n_eval, g["mean"], "-o", color=ramp[n_train], lw=lw, ms=3.5, label=label)
            ax.fill_between(g.n_eval, g["mean"] - g["sd"], g["mean"] + g["sd"], color=ramp[n_train], alpha=0.12, lw=0)
        folds = s.groupby("n_eval").n_folds.max()
        for n_eval, k in folds[folds < folds.max()].items():
            ax.text(n_eval, 0.02, f"{int(k)}/{int(folds.max())} folds", ha="center", fontsize=5.8, color="#52514e")
        chance = float(s.chance.mean())
        ax.axhline(chance, ls="--", lw=0.8, color="#52514e")
        ax.text(s.n_eval.min(), chance + 0.015, f"chance 1/{round(1 / chance)}", fontsize=6.5, color="#52514e")
        ax.set_xscale("log")
        ticks = sorted(s.n_eval.unique())
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.set_xticks([], minor=True)
        ax.set_ylim(0, 1)
        ax.set_xlabel("cells per evaluation bag (eval.py, held-out test FOVs)")
        ax.set_ylabel("top-1 accuracy (mean ± SD over 5 folds)")
        ax.set_title(MODEL_TITLES[model], fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.text(
        0.01,
        -0.03,
        "Sizes above a fold's smallest test pool are dropped for that fold (eval.py would reuse the "
        "whole pool); 'k/5 folds' marks sizes averaged over fewer folds.",
        fontsize=6.5,
        color="#52514e",
    )
    fig.tight_layout()
    save(fig, path)


def plot_ranking(rank, colors, chance, size, n_train, path):
    """Horizontal bars of eval.py per-class top-1 (mean ± SD over folds), sorted: the gKO-ranking analogue."""
    rank = rank.sort_values("top1_mean")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    y = np.arange(len(rank))
    ax.barh(
        y,
        rank.top1_mean,
        xerr=rank.top1_sd,
        color=[colors[c] for c in rank.cls],
        height=0.7,
        edgecolor="white",
        linewidth=1,
        error_kw=dict(lw=0.8, capsize=2, ecolor="#52514e"),
    )
    for k, (_, r) in enumerate(rank.iterrows()):
        ax.scatter(r.fold_values, np.full(len(r.fold_values), k), s=6, color="#52514e", zorder=3, lw=0)
    ax.axvline(chance, ls="--", lw=0.8, color="#52514e")
    ax.text(chance + 0.01, len(rank) - 0.4, f"chance 1/{round(1 / chance)}", fontsize=6.5, color="#52514e")
    ax.set_yticks(y, [r.label for _, r in rank.iterrows()])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"held-out per-class top-1 (eval.py, bags of {size}, 30 reps; trained at {n_train})")
    ax.set_title("Model I class ranking (dots = folds)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, path)


def plot_auroc(summary, width, size, path):
    """Per-τ infection AUROC vs BYSTANDER and vs CONTROL, and the BYSTANDER-vs-CONTROL AUROC, for all methods."""
    methods = {
        "set_classifier": ("#2a78d6", "o", "set classifier (pinned)"),
        "mean_logreg": ("#eb6834", "s", "mean-pooled logistic regression"),
        "hpi_logreg": ("#1baf7a", "^", "HPI-only logistic regression (bag HPI mean, SD)"),
    }
    rows = {
        "pool": "reference bags drawn from the whole reference test pool (as specified)",
        "hpi_matched": "reference bags matched to each bag's (dataset, 2 h HPI bin) composition",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 7.0),
        gridspec_kw=dict(
            width_ratios=[3, 3, 1.4], left=0.06, right=0.99, top=0.87, bottom=0.14, wspace=0.28, hspace=0.62
        ),
    )
    for r, (variant, title) in enumerate(rows.items()):
        s = summary[summary.reference_bags.eq(variant)]
        for ax, ref in zip(axes[r, :2], REFERENCES):
            for k, (method, (color, marker, label)) in enumerate(methods.items()):
                g = s[s.reference.eq(ref) & s.method.eq(method) & s.cls.str.startswith("TAU_")].sort_values("tau_bin")
                x = g.tau_bin + width / 2 + (k - 1) * 0.25
                ax.errorbar(
                    x, g["mean"], yerr=g["sd"], color=color, marker=marker, ms=4, lw=1.2, capsize=2, label=label
                )
            ax.axhline(0.5, ls="--", lw=0.8, color="#52514e")
            ax.axvline(0, ls=":", lw=0.8, color="#52514e")
            ax.text(0.15, 0.53, "TCN onset", fontsize=6.5, color="#52514e", transform=ax.get_xaxis_transform())
            ax.set_ylim(0.2, 1.02)
            ax.set_xlabel("τ bin centre (h from TCN onset)")
            ax.set_ylabel("AUROC (mean ± SD over 5 folds)")
            ax.set_title(f"τ-bin bags vs {ref} bags", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, lw=0.5)
        ax = axes[r, 2]
        g = s[s.cls.eq("BYSTANDER")].set_index("method").loc[list(methods)]
        ax.bar(
            range(3),
            g["mean"],
            yerr=g["sd"],
            color=[m[0] for m in methods.values()],
            width=0.65,
            edgecolor="white",
            error_kw=dict(lw=0.8, capsize=2),
        )
        ax.axhline(0.5, ls="--", lw=0.8, color="#52514e")
        ax.set_xticks(range(3), ["set", "mean", "HPI"], fontsize=7)
        ax.set_ylim(0.2, 1.02)
        ax.set_title("BYSTANDER vs CONTROL\n(well effect)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        top = axes[r, 0].get_position().y1
        fig.text(0.06, top + 0.055, f"{'AB'[r]}  {title}", fontsize=8.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Model I infection AUROC on identical seeded held-out bags of {size} cells, q = P(a) / (P(a) + P(b))",
        fontsize=9.5,
        y=0.985,
    )
    fig.savefig(f"{path}.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_confusion(mats, path):
    """Row-normalized confusion heatmaps of both models on our held-out test bags."""
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[12, 5]))
    for ax, (model, (m, classes, size)) in zip(axes, mats.items()):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=5 if len(classes) > 6 else 7,
                    color="white" if m[i, j] > 0.55 else "#0b0b0b",
                )
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes, fontsize=6.5)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{MODEL_TITLES[model]}\nbags of {size}, 30 per class per fold, mean of 5 folds", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="row fraction")
    fig.tight_layout()
    save(fig, path, ("png",))


def plot_distortion(dist, classes_c, cfg, path):
    """Model C mean predicted cycle distribution of held-out onset bags per τ bin and called phase, vs references."""
    phases = [p for p in ALL_PHASES if p in dist.cc_state.unique() and p in classes_c]
    bins = tau_bins(cfg)
    diverging = LinearSegmentedColormap.from_list("div", ["#184f95", "#f0efec", "#b2182b"])
    fig = plt.figure(figsize=(11, 1.5 + 1.55 * len(phases)))
    grid = fig.add_gridspec(len(phases), 2, width_ratios=[2.2, 1.2], wspace=0.25, hspace=0.55)
    delta = np.full((len(phases), len(bins)), np.nan)
    for i, phase in enumerate(phases):
        ax = fig.add_subplot(grid[i, 0])
        g = dist[dist.cc_state.eq(phase)].set_index("tau_bin")
        labels, stacks, ns = ["ref"], [[g[f"ref_p_{c}"].mean() for c in classes_c]], [""]
        for b in bins:
            labels.append(signed(b))
            if b in g.index:
                stacks.append([g.loc[b, f"p_{c}"] for c in classes_c])
                ns.append(f"{int(g.loc[b, 'n_cells'])}")
                delta[i, bins.index(b)] = g.loc[b, "delta_called"]
            else:
                stacks.append([np.nan] * len(classes_c))
                ns.append("–")
        stacks = np.nan_to_num(np.array(stacks, float))
        bottom = np.zeros(len(labels))
        for k, c in enumerate(classes_c):
            ax.bar(
                range(len(labels)),
                stacks[:, k],
                bottom=bottom,
                color=PHASE_COLORS[c],
                width=0.78,
                edgecolor="white",
                linewidth=1,
                label=c,
            )
            bottom += stacks[:, k]
        for x, n in enumerate(ns):
            ax.text(x, 1.02, n, ha="center", fontsize=5.5, color="#52514e")
        ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"called {phase}", color=PHASE_COLORS[phase], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(
                ncol=len(classes_c),
                fontsize=6.5,
                frameon=False,
                loc="lower left",
                bbox_to_anchor=(0, 1.12),
                title="Model C predicted class",
                title_fontsize=6.5,
            )
        if i == len(phases) - 1:
            ax.set_xlabel("τ bin start (h from TCN onset)")
    fig.text(
        0.125,
        0.035,
        "ref = reference (control + bystander) test bags of that called phase; numbers above bars = held-out onset "
        "cells "
        "summed over folds ('–' = fewer than 5 in every fold).",
        fontsize=6.5,
        color="#52514e",
    )
    ax = fig.add_subplot(grid[:, 1])
    lim = max(0.05, np.nanmax(np.abs(delta)))
    im = ax.imshow(delta, cmap=diverging, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if np.isfinite(delta[i, j]):
                ax.text(j, i, f"{delta[i, j]:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(len(bins)), [signed(b) for b in bins], fontsize=6.5)
    ax.set_yticks(range(len(phases)), phases)
    ax.set_xlabel("τ bin start (h from TCN onset)")
    ax.set_title("Δ P(called phase): onset − reference\n(mean over folds)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.09, orientation="horizontal", label="Δ P(called phase)")
    fig.suptitle(
        f"Model C cycle distortion: bags of ≤ {cfg['analyze']['distortion_max_bag']} held-out onset cells per τ bin "
        "x PCNA→Phase called phase",
        fontsize=9,
    )
    save(fig, path, ("png",))


def analyze(cfg):
    """Stage ``analyze``: sweep, class ranking, AUROCs with baselines, confusion, distortion map, cross-scoring."""
    work = Path(cfg["work_dir"])
    out = work / "analysis"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    cv_folds, lodo_folds = fold_names(work)
    chosen = json.loads((work / "run_summary.json").read_text())["chosen"]
    counts = pd.read_csv(work / "pool_counts.csv")
    cells = pd.read_parquet(work / "cells.parquet")
    members = pd.read_parquet(work / "pool_members.parquet")
    sizes = cfg["run"]["n_cells_per_set"]
    ctx = dict(
        work=work,
        cells=cells,
        members=members,
        application=pd.read_parquet(work / "application.parquet"),
        emb=cells[pc_columns(cfg)].to_numpy(np.float32),
        hpi=cells.hpi.to_numpy(float),
        train=pinned_train(cfg),
        chosen=chosen,
        size={m: analysis_size(cfg, counts, m, chosen[m], cv_folds) for m in MODELS},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    colors = class_colors(cfg)
    record = dict(chosen=chosen, analysis_bag_size=ctx["size"], device=str(ctx["device"]))

    rows = []
    for model in MODELS:
        for fold in [*cv_folds, *lodo_folds]:
            for n_train in sizes if fold in cv_folds else [chosen[model]]:
                j = json.loads((run_dir(work, model, fold, n_train) / "eval" / "accuracy_vs_n.json").read_text())
                for k, n_eval in enumerate(j["n_cells_list"]):
                    rows.append(
                        dict(
                            model=model,
                            fold=fold,
                            n_train=n_train,
                            n_eval=n_eval,
                            top1=j["mean_accuracy_top1"][k],
                            top1_sem=j["stderr_top1"][k],
                            top5=j["mean_accuracy_top5"][k],
                            n_classes=j["n_classes"],
                        )
                    )
    evals = pd.DataFrame(rows)
    evals.to_csv(out / "eval_runs.csv", index=False)
    cv = evals[evals.fold.isin(cv_folds)]
    sweep = (
        cv.groupby(["model", "n_train", "n_eval"])
        .agg(
            mean=("top1", "mean"),
            sd=("top1", "std"),
            n_folds=("top1", "size"),
            top5_mean=("top5", "mean"),
            mean_eval_sem=("top1_sem", "mean"),
            chance=("n_classes", lambda x: float(np.mean(1 / x))),
        )
        .reset_index()
    )
    sweep.to_csv(out / "bag_size_sweep.csv", index=False)
    plot_sweep(sweep, chosen, out / "bag_size_sweep", sizes)
    lodo_eval = evals[evals.fold.isin(lodo_folds)]
    lodo_eval.to_csv(out / "lodo_eval.csv", index=False)

    per_class = []
    for fold in cv_folds:
        p = pd.read_csv(run_dir(work, "model_i", fold, chosen["model_i"]) / "eval" / "per_class.csv")
        per_class.append(p.assign(fold=fold))
    per_class = pd.concat(per_class, ignore_index=True)
    sel = per_class[per_class.n_cells.eq(ctx["size"]["model_i"])]
    rank = (
        sel.groupby("gene_name")
        .top1_acc.agg(top1_mean="mean", top1_sd="std", n_folds="size")
        .reset_index()
        .rename(columns={"gene_name": "cls"})
    )
    rank["fold_values"] = rank.cls.map(sel.groupby("gene_name").top1_acc.apply(list))
    rank["label"] = [tau_label(c, cfg["bin_hours"]) if c.startswith("TAU_") else c for c in rank.cls]
    rank = rank.sort_values("top1_mean", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["eval_n_cells"], rank["train_n_cells"] = ctx["size"]["model_i"], chosen["model_i"]
    rank.drop(columns="fold_values").to_csv(out / "class_ranking.csv", index=False)
    per_class.to_csv(out / "eval_per_class_model_i.csv", index=False)
    plot_ranking(rank, colors, 1 / len(rank), ctx["size"]["model_i"], chosen["model_i"], out / "class_ranking")

    res_i = {f: model_i_fold(cfg, ctx, f, fi) for fi, f in enumerate([*cv_folds, *lodo_folds])}
    res_c = {f: model_c_fold(cfg, ctx, f, fi) for fi, f in enumerate([*cv_folds, *lodo_folds])}
    bag_rows, prob_frames, acc_rows, au_rows = [], [], [], []
    for model, results in [("model_i", res_i), ("model_c", res_c)]:
        for res in results.values():
            b, p, a = result_tables(res, model)
            bag_rows += b
            prob_frames += p
            acc_rows += a
    matched_rows = []
    for res in res_i.values():
        au_rows += auroc_rows(res)
        for (pos, ref), m in res["matched"].items():
            for k, (rows_, gap, cross) in enumerate(zip(m["bags"], m["hpi_gap"], m["cross_dataset"])):
                matched_rows.append(
                    dict(
                        fold=res["fold"],
                        cls=pos,
                        reference=ref,
                        pair=k,
                        rows=",".join(map(str, rows_)),
                        hpi_gap=gap,
                        cross_dataset_cells=cross,
                    )
                )
    pd.DataFrame(matched_rows).to_parquet(out / "matched_reference_bags.parquet", index=False)
    pd.DataFrame(bag_rows).to_parquet(out / "test_bags.parquet", index=False)
    pd.concat(prob_frames, ignore_index=True).to_parquet(out / "test_bag_probs.parquet", index=False)
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(out / "test_bag_accuracy.csv", index=False)
    au = pd.DataFrame(au_rows)
    au["split"] = np.where(au.fold.isin(cv_folds), "cv", "lodo")
    au.to_csv(out / "infection_auroc_folds.csv", index=False)
    cv_au = au[au.split.eq("cv")]
    au_sum = (
        cv_au.groupby(["reference_bags", "cls", "tau_bin", "reference", "method"], dropna=False)
        .agg(
            mean=("auroc", "mean"),
            sd=("auroc", "std"),
            n_folds=("auroc", "size"),
            mean_hpi_gap=("mean_hpi_gap", "mean"),
            cross_dataset_fraction=("cross_dataset_fraction", "mean"),
        )
        .reset_index()
    )
    au_sum["bag_size"] = ctx["size"]["model_i"]
    au_sum.to_csv(out / "infection_auroc.csv", index=False)
    plot_auroc(au_sum, cfg["bin_hours"], ctx["size"]["model_i"], out / "infection_auroc")
    record["baseline_fits"] = {f: r["fits"] for f, r in res_i.items()}
    record["checkpoint_epochs"] = {
        "model_i": {f: r["epoch"] for f, r in res_i.items()},
        "model_c": {f: r["epoch"] for f, r in res_c.items()},
    }

    mi, ci, conf_i = confusion([res_i[f] for f in cv_folds], "model_i")
    mc, cc, conf_c = confusion([res_c[f] for f in cv_folds], "model_c")
    pd.DataFrame(conf_i + conf_c).to_csv(out / "confusion.csv", index=False)
    plot_confusion(
        {"model_i": (mi, ci, ctx["size"]["model_i"]), "model_c": (mc, cc, ctx["size"]["model_c"])}, out / "confusion"
    )

    dist = []
    for fi, f in enumerate(cv_folds):
        dist += distortion_rows(cfg, ctx, f, fi, res_c[f]["model"], res_c[f]["classes"])
    dist = pd.DataFrame(dist)
    dist.to_csv(out / "cycle_distortion_folds.csv", index=False)
    used = dist[dist.skipped.eq("") & dist.delta_called.notna()]
    value_cols = [c for c in used.columns if c.startswith(("p_", "ref_p_", "delta_"))]
    dist_sum = used.groupby(["tau_bin", "cc_state"])[value_cols].mean()
    dist_sum = dist_sum.join(
        used.groupby(["tau_bin", "cc_state"]).agg(
            n_cells=("n_cells", "sum"), n_folds=("fold", "size"), mean_bag_size=("bag_size", "mean")
        )
    ).reset_index()
    dist_sum["delta_called_sd"] = used.groupby(["tau_bin", "cc_state"]).delta_called.std().to_numpy()
    dist_sum.to_csv(out / "cycle_distortion.csv", index=False)
    plot_distortion(dist_sum, cc, cfg, out / "cycle_distortion")

    scores = load_scores(cfg, work, cells, members, chosen, [*cv_folds, *lodo_folds])
    scores.to_parquet(out / "scores.parquet", index=False)
    models_i = {
        f: (
            load_classifier(
                ctx["train"], run_dir(work, "model_i", f, chosen["model_i"]) / "checkpoint.pt", ctx["device"]
            )[:2]
        )
        for f in cv_folds
    }
    models_c = {f: (res_c[f]["model"], res_c[f]["classes"]) for f in cv_folds}
    cross_cells, cross = cross_scoring(cfg, ctx, scores, models_i, models_c, cc, cv_folds)
    cross_cells.to_csv(out / "cross_scoring_cells.csv", index=False)
    cross.to_csv(out / "cross_scoring.csv", index=False)

    lodo = acc[acc.fold.isin(lodo_folds) & acc.cls.eq("__all__")]
    record.update(
        sweep_at_analysis_size={
            m: sweep[sweep.model.eq(m) & sweep.n_eval.eq(ctx["size"][m])].to_dict(orient="records") for m in MODELS
        },
        class_ranking=rank.drop(columns="fold_values").to_dict(orient="records"),
        test_bag_top1=acc[acc.cls.eq("__all__")]
        .groupby(["model", "method", acc.fold.isin(cv_folds).map({True: "cv", False: "lodo"})])
        .top1.agg(["mean", "std"])
        .reset_index()
        .to_dict(orient="records"),
        lodo_test_bag_top1=lodo.to_dict(orient="records"),
        lodo_auroc=au[au.split.eq("lodo")].to_dict(orient="records"),
        auroc=au_sum.to_dict(orient="records"),
        distortion=dist_sum.to_dict(orient="records"),
        distortion_skipped=dist[dist.skipped.ne("")].to_dict(orient="records"),
        cross_scoring=cross.to_dict(orient="records"),
    )
    write_json(out / "analysis.json", record)
    print("analyze done", json.dumps(dict(chosen=chosen, size=ctx["size"]), default=str), flush=True)


def crop_helpers(cfg):
    """Import the attempt-13 crop helpers by file path, registering their two script dependencies first."""
    path = Path(cfg["crop_helpers"])
    for name in cfg["crop_helper_dependencies"]:
        load_module(name, path.parent / f"{name}.py")
    return load_module(path.stem, path)


class Crops:
    """Attempt-13 crop reader with the attempt-13 TOMM20 and pAL QC gates (thresholds from its manifest), cached."""

    def __init__(self, cfg, a13, cells):
        self.a13 = a13
        m13 = json.loads(Path(cfg["attempt13_manifest"]).read_text())
        self.thresholds, self.paths = {}, {}
        for ds, unit in cfg["units13"].items():
            experiment = cfg["datasets"][ds]
            path = a13.IMAGES / experiment / f"{experiment}.zarr"
            if a13.DATES[unit] != experiment or m13["image_zarrs"][unit] != str(path):
                raise ValueError(f"{ds}: attempt-13 unit {unit} does not point at {path}")
            self.thresholds[ds] = {k: float(m13["crop_qc"]["thresholds"][unit][k]) for k in ("tomm20", "pal")}
            self.paths[ds] = str(path)
        markers = {ds: [(a13.PAL_MARKERS[unit], a13.PAL), (CHANNEL, a13.TOMM20)] for ds, unit in cfg["units13"].items()}
        self.channels = a13.confirm_channels(cells, markers, cfg["datasets"])
        self.pal_marker = {ds: a13.PAL_MARKERS[unit] for ds, unit in cfg["units13"].items()}
        self.stores = {ds: zarr.open_group(p, mode="r") for ds, p in self.paths.items()}
        self.cache, self.results, self.audit = {}, {}, []

    def get(self, r, figure):
        """Return ``(reason, crop)`` for one cell; ``reason`` is ``None`` if geometry and both gates pass."""
        key = int(r.row_id)
        if key not in self.results:
            reason, out = self.a13.read_crop(self.stores, self.cache, r)
            for gate in ("tomm20", "pal"):
                if reason is None and out["meta"][f"{gate}_center_p99"] < self.thresholds[r.dataset][gate]:
                    reason = f"{gate}_dim"
            self.results[key] = (reason, out)
            meta = {} if out is None else out["meta"]
            self.audit.append(
                dict(
                    row_id=key,
                    first_figure=figure,
                    dataset=r.dataset,
                    fov_name=r.fov_name,
                    track_id=int(r.track_id),
                    t=int(r.t),
                    outcome=reason or "accepted",
                    **meta,
                )
            )
        return self.results[key]


def pick(cands, k, crops, figure, used_lineages, fov_counts, per_fov, stats):
    """Take the first ``k`` candidates (already sorted by score) that pass diversity and crop QC.

    Parameters
    ----------
    cands : pandas.DataFrame
        Candidates sorted by descending score.
    k : int
        Cells wanted.
    crops : Crops
        Crop reader.
    figure : str
        Figure name for the audit.
    used_lineages : set
        ``lineage_uid`` already used in this figure (updated).
    fov_counts : collections.Counter
        Picks per (dataset, fov_name) in this figure (updated).
    per_fov : int
        Maximum picks per FOV in this figure.
    stats : collections.Counter
        Skip and rejection counts (updated).

    Returns
    -------
    list of pandas.Series
        Accepted candidates.
    """
    picks = []
    for _, r in cands.iterrows():
        if len(picks) == k:
            break
        if r.lineage_uid in used_lineages:
            stats["skip_lineage_used"] += 1
            continue
        if fov_counts[(r.dataset, r.fov_name)] >= per_fov:
            stats["skip_fov_full"] += 1
            continue
        reason, _ = crops.get(r, figure)
        if reason:
            stats[f"reject_{reason}"] += 1
            continue
        used_lineages.add(r.lineage_uid)
        fov_counts[(r.dataset, r.fov_name)] += 1
        picks.append(r)
    stats["selected"] += len(picks)
    stats["shortfall"] += k - len(picks)
    return picks


def ranked(scores, model, cls, phase=None):
    """Return one class's CV-fold score rows (each held-out cell scored by its own fold), best first."""
    s = scores[scores.model.eq(model) & scores.cls.eq(cls) & scores.fold.str.startswith("fold")]
    if phase is not None:
        s = s[s.cc_state.eq(phase)]
    return s.sort_values(["score", "row_id"], ascending=[False, True])


def draw_cell(fig, size, x, y_top, t, gap, r, crops, limits, vertical, first):
    """Draw one cell as a Phase tile and a TOMM20 + pAL merge tile (5 µm bar on Phase); inches from top-left."""
    width, height = size
    _, out = crops.results[int(r.row_id)]
    crop, lim = out["crop"], limits[r.dataset]
    boxes = [(x, y_top), (x, y_top + t + gap) if vertical else (x + t + gap, y_top)]
    axes = [fig.add_axes([bx / width, 1 - (by + t) / height, t / width, t / height]) for bx, by in boxes]
    for ax in axes:
        ax.set_axis_off()
    axes[0].imshow(crop["phase"], cmap="gray", vmin=lim["phase"][0], vmax=lim["phase"][1])
    axes[0].plot([12, 12 + 5 / out["meta"]["pixel_um"]], [208, 208], c="white", lw=1.6, solid_capstyle="butt")
    axes[1].imshow(crops.a13.merge(crop["pal"], crop["tomm20"], lim))
    if first:
        axes[0].text(8, 8, "Phase", color="white", size=5.5, va="top", path_effects=crops.a13.DARK)
        axes[0].text(12, 196, "5 µm", color="white", size=5, path_effects=crops.a13.DARK)
        axes[1].text(8, 8, CHANNEL, color="#00ff66", size=5.5, va="top", path_effects=crops.a13.DARK)
        axes[1].text(
            8, 34, crops.pal_marker[r.dataset], color="#ff00ff", size=5.5, va="top", path_effects=crops.a13.DARK
        )


def cell_caption(r):
    """Short caption of a picked cell: dataset, HPI, τ (onset lineages) and score."""
    tau = f" · τ {r.tau_h:+.1f}".replace("-", "−") if np.isfinite(r.tau_h) else ""
    return f"{r.dataset} · {r.hpi:.0f} HPI{tau} · s {r.score:.3f}"


def render_cycle_panel(picks, crops, limits, cfg, stem):
    """Rows G1 / S / G2; columns cycle-typical uninfected | infected τ 0–2 h | bystander; 2 x 2 cells per group."""
    columns = [
        (
            "cycle_typical",
            "cycle-typical, uninfected",
            "Model C top cells of that phase class\n(control + bystander cells)",
        ),
        ("infected", "infected τ 0–2 h", "Model I TAU_p00 top cells\nwith that PCNA→Phase call"),
        ("bystander", "bystander", "Model I BYSTANDER top cells\nwith that PCNA→Phase call"),
    ]
    phases = cfg["phases"]
    t, pair_gap, cell_gap, group_gap, caption, row_gap = 0.62, 0.02, 0.07, 0.24, 0.13, 0.2
    left, top, bottom = 0.42, 0.62, 0.3
    pair = 2 * t + pair_gap
    group_w = 2 * pair + cell_gap
    cell_h = t + caption
    group_h = 2 * cell_h + 0.04
    width = left + len(columns) * group_w + (len(columns) - 1) * group_gap + 0.1
    height = top + len(phases) * group_h + (len(phases) - 1) * row_gap + bottom
    fig = plt.figure(figsize=(width, height))
    first = True
    for j, (key, title, sub) in enumerate(columns):
        x0 = left + j * (group_w + group_gap)
        fig.text(
            (x0 + group_w / 2) / width, 1 - 0.08 / height, title, ha="center", va="top", fontsize=8.5, fontweight="bold"
        )
        fig.text((x0 + group_w / 2) / width, 1 - 0.27 / height, sub, ha="center", va="top", fontsize=6, color="#52514e")
    for i, phase in enumerate(phases):
        y0 = top + i * (group_h + row_gap)
        fig.text(
            0.17 / width,
            1 - (y0 + group_h / 2) / height,
            phase,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PHASE_COLORS[phase],
        )
        for j, (key, _, _) in enumerate(columns):
            x0 = left + j * (group_w + group_gap)
            cells = picks.get((key, phase), [])
            for k in range(4):
                x = x0 + (k % 2) * (pair + cell_gap)
                y = y0 + (k // 2) * (cell_h + 0.04)
                if k >= len(cells):
                    fig.text(
                        (x + pair / 2) / width,
                        1 - (y + t / 2) / height,
                        "no QC-passing cell",
                        ha="center",
                        fontsize=6,
                        color="#8c8c8c",
                    )
                    continue
                r = cells[k]
                draw_cell(fig, (width, height), x, y, t, pair_gap, r, crops, limits, False, first)
                first = False
                fig.text(
                    x / width, 1 - (y + t + 0.015) / height, cell_caption(r), fontsize=4.6, color="#52514e", va="top"
                )
    fig.text(
        left / width,
        0.2 / height,
        "Ranked by pinned score.py (leave-one-out contribution, bags 1–100) on held-out test FOVs; ≤ 1 cell per "
        "lineage "
        "and ≤ 2 per FOV in the figure;\nattempt-13 crop QC (TOMM20 and pAL gates). Phase focus slice | TOMM20 green + "
        "pAL magenta 16-slice MIP; 5 µm bar. s = score.",
        fontsize=5.5,
        color="#52514e",
        va="top",
    )
    save(fig, stem)


def render_pseudotime(picks, crops, limits, cfg, stem):
    """Rows G1 / S / G2 / M x columns τ bins (Phase above merge).

    G1 / S / G2: the top Model I cell (score.py) of each τ class with that call. M: the top Mantis-confirmed M cell
    of each τ bin by Model I single-cell q_inf (M is outside Model I's training classes).
    """
    phases, bins = [*cfg["phases"], "M"], tau_bins(cfg)
    t, gap, col_gap, caption, row_gap, left, top, m_note = 0.72, 0.02, 0.07, 0.15, 0.16, 0.38, 0.5, 0.2
    width = left + len(bins) * (t + col_gap) + 0.05
    height = top + len(phases) * (2 * t + gap + caption + row_gap) + m_note + 0.45
    fig = plt.figure(figsize=(width, height))
    colors = tau_colors(cfg)
    first = True
    for j, b in enumerate(bins):
        x = left + j * (t + col_gap)
        fig.text(
            (x + t / 2) / width,
            1 - 0.2 / height,
            tau_label(tau_class(b), cfg["bin_hours"]),
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color=colors[tau_class(b)],
        )
    fig.text(
        left / width,
        1 - 0.04 / height,
        "Model I top cell per τ class and PCNA→Phase call (held-out test FOVs; τ from the Phase→pAL TCN onset)",
        fontsize=7.5,
        va="top",
    )
    for i, phase in enumerate(phases):
        y = top + i * (2 * t + gap + caption + row_gap) + (m_note if phase == "M" else 0)
        if phase == "M":
            fig.text(
                left / width,
                1 - (y - 0.07) / height,
                "M (Mantis-confirmed division, same track ±1 frame) — outside Model I's training classes (M excluded "
                "from phase-matched pools); ranked by Model I infection score",
                fontsize=6.5,
                color=PHASE_COLORS["M"],
                va="bottom",
            )
        fig.text(
            0.16 / width,
            1 - (y + t + gap / 2) / height,
            phase,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=PHASE_COLORS[phase],
        )
        for j, b in enumerate(bins):
            x = left + j * (t + col_gap)
            cells = picks.get((tau_class(b), phase), [])
            if not cells:
                note = "no confirmed M" if phase == "M" else "none"
                fig.text((x + t / 2) / width, 1 - (y + t) / height, note, ha="center", fontsize=6, color="#8c8c8c")
                continue
            r = cells[0]
            draw_cell(fig, (width, height), x, y, t, gap, r, crops, limits, True, first)
            first = False
            value = f"q_inf {r.score:.3f}" if phase == "M" else f"s {r.score:.3f}"
            fig.text(
                x / width,
                1 - (y + 2 * t + gap + 0.015) / height,
                f"{r.dataset} · {r.hpi:.0f} HPI\n{value}",
                fontsize=4.6,
                color="#52514e",
                va="top",
            )
    m_cfg = cfg["render"]["m_row"]
    fig.text(
        left / width,
        0.36 / height,
        "G1 / S / G2: ranked by pinned score.py within each phase-matched τ pool. M: onset-lineage frames called M "
        f"with a Mantis division event on the same track at t−1…t+1 (division ≥ {m_cfg['division_min']}, death ≤ "
        f"{m_cfg['death_max']}), one per track and τ bin,\nranked by Model I single-cell q_inf = ΣP(τ ≥ 0) / "
        "(ΣP(τ ≥ 0) + P(BYSTANDER) + P(CONTROL)) from the fold that holds the FOV out. All rows: held-out test FOVs; "
        "≤ 1 cell per lineage and ≤ 2 per FOV in the figure; attempt-13 crop QC.\nPhase focus slice above TOMM20 "
        "green + pAL magenta MIP; 5 µm bar; display limits as in the other attempt-25 figures.",
        fontsize=5.5,
        color="#52514e",
        va="top",
    )
    save(fig, stem)


def render_montage(pages_spec, crops, limits, a13, pdf_path, pages_dir):
    """One page per class (top 10 by score, Phase | merge strips) with attempt-13 ``render_strips(n_tiles=2)``."""
    with PdfPages(pdf_path) as pdf:
        for page, (stem, header, cells, info) in enumerate(pages_spec, start=1):
            strips = []
            for r in cells:
                _, out = crops.results[int(r.row_id)]
                crop, lim = out["crop"], limits[r.dataset]
                tau = (
                    f"τ {r.tau_h:+.1f} h".replace("-", "−") if np.isfinite(r.tau_h) else f"{r.group.lower()} (no onset)"
                )
                captions = [
                    f"#{int(r['rank'])} of {int(r.n_cells)} in {r.fold} · score {r.score:.4f} · P(class | cell) "
                    f"{r.bag1:.3f}",
                    f"{r.dataset} · {r.fov_name} / track {int(r.track_id)} / t {int(r.t)} · {r.hpi:.1f} HPI · {tau} · "
                    f"called {r.cc_state} (p {r[f'p_{r.cc_state}']:.2f})",
                ]
                strips.append(
                    dict(
                        captions=[
                            (0.08 + n * 0.15, s, dict(size=6.6, weight="bold") if n == 0 else dict(size=6.3))
                            for n, s in enumerate(captions)
                        ],
                        tiles=[
                            (
                                "Phase",
                                crop["phase"],
                                dict(cmap="gray", vmin=lim["phase"][0], vmax=lim["phase"][1]),
                                True,
                            ),
                            (
                                f"TOMM20 + {crops.pal_marker[r.dataset]}",
                                a13.merge(crop["pal"], crop["tomm20"], lim),
                                {},
                                False,
                            ),
                        ],
                        pixel_um=out["meta"]["pixel_um"],
                    )
                )
            footer = [
                (
                    0.1,
                    "Contrast: 1st–99.5th percentile per dataset and channel over every crop selected for attempt 25.",
                    dict(size=7),
                )
            ]
            for n, ds in enumerate(sorted(limits)):
                lim = limits[ds]
                footer.append(
                    (
                        0.27 + n * 0.17,
                        f"{ds} ({lim['crops']} crops): Phase [{lim['phase'][0]:.4g}, {lim['phase'][1]:.4g}] · pAL "
                        f"[{lim['pal'][0]:.4g}, {lim['pal'][1]:.4g}] · TOMM20 [{lim['tomm20'][0]:.4g}, "
                        f"{lim['tomm20'][1]:.4g}]  |  QC: TOMM20 p99 ≥ {crops.thresholds[ds]['tomm20']:.4g}, pAL p99 ≥ "
                        f"{crops.thresholds[ds]['pal']:.4g}",
                        dict(size=6.8),
                    )
                )
            with mock.patch.object(plt, "close"):
                a13.render_strips(
                    header,
                    strips,
                    footer,
                    pages_dir / f"p{page:02d}_{stem}",
                    n_tiles=2,
                    caption=0.56,
                    header_height=1.5,
                )
            fig = plt.gcf()
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            (pages_dir / f"p{page:02d}_{stem}.pdf").unlink()


def render_umap(cfg, scores, path):
    """Top cells of every class on the attempt-23 Phase UMAP over all frames: Model I by τ, Model C by phase."""
    allframes = pd.read_parquet(labels_path(cfg), columns=["umap1", "umap2"])
    colors = class_colors(cfg)
    top = cfg["render"]["umap_top"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    panels = [
        ("model_i", model_i_classes(cfg), "Model I top cells (colour = τ bin; references black / grey)"),
        ("model_c", [*cfg["phases"], "M"], "Model C top cells (colour = phase class)"),
    ]
    for ax, (model, classes, title) in zip(axes, panels):
        ax.scatter(allframes.umap1, allframes.umap2, s=0.6, color="#e3e3e3", lw=0, rasterized=True)
        handles = []
        for c in classes:
            g = ranked(scores, model, c).head(top)
            if g.empty:
                continue
            ax.scatter(g.umap1, g.umap2, s=10, color=colors[c], edgecolor="white", linewidth=0.3, zorder=3)
            label = tau_label(c, cfg["bin_hours"]) if c.startswith("TAU_") else c
            handles.append(
                Line2D([], [], marker="o", ls="", color=colors[c], markeredgecolor="white", ms=5, label=label)
            )
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_title(f"{title}\ntop {top} per class by score.py, pooled over held-out folds", fontsize=8)
        ax.set_xlabel("UMAP 1 (attempt-23 Phase3D UMAP)")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, path, ("png",))


def confirmed_divisions(d, m_cfg):
    """Flag frames whose track has a Mantis division event at t−1, t or t+1 (attempt-23 rule).

    A frame is a division event when its event evidence is available (non-null probabilities), its
    ``probability__division`` is at least ``division_min`` and its ``probability__death`` at most ``death_max``.

    Parameters
    ----------
    d : pandas.DataFrame
        Label-snapshot frames (all frames, so neighbours of any frame are present).
    m_cfg : dict
        ``render.m_row`` config.

    Returns
    -------
    numpy.ndarray
        Boolean flag per row of ``d``.
    """
    keys = ["dataset", *KEYS]
    event = (
        d.probability__division.notna()
        & d.probability__death.notna()
        & d.probability__division.ge(m_cfg["division_min"])
        & d.probability__death.le(m_cfg["death_max"])
    )
    events = d.loc[event, keys].assign(_event=True)
    confirmed = np.zeros(len(d), dtype=bool)
    for k in m_cfg["window"]:
        m = d[keys].merge(events.assign(t=events.t - k), on=keys, how="left", validate="one_to_one")
        confirmed |= m._event.eq(True).to_numpy()
    return confirmed


def infection_score(p, classes):
    """Single-cell infection score q_inf = Σ P(τ ≥ 0) / (Σ P(τ ≥ 0) + P(BYSTANDER) + P(CONTROL)) (cross-scoring)."""
    post = [classes.index(c) for c in classes if c.startswith("TAU_p")]
    ref = [classes.index(c) for c in REFERENCES]
    return p[:, post].sum(1) / (p[:, post].sum(1) + p[:, ref].sum(1))


def m_row_candidates(cfg, train_module, device):
    """Build the pseudotime M-row candidates: Mantis-confirmed M frames of onset lineages, scored by Model I.

    Onset-lineage frames (pre_onset + infected, not Mantis death) with τ in [tau_start, tau_stop) and ``cc_state``
    M are kept only with a Mantis-confirmed division on the same track within the window, then reduced to one frame
    per (track, τ bin) nearest the bin centre. Each candidate is joined to its TOMM20 embedding and scored with the
    single-cell q_inf of the Model I checkpoint (chosen size) of the CV fold that holds its FOV out.

    Returns
    -------
    tuple of (pandas.DataFrame, list)
        Candidates (negative ``row_id`` keys, ``score`` = q_inf) and per-bin counts before and after the Mantis rule.
    """
    work, width, m_cfg = Path(cfg["work_dir"]), cfg["bin_hours"], cfg["render"]["m_row"]
    d = pd.read_parquet(labels_path(cfg))
    d["confirmed"] = confirmed_divisions(d, m_cfg)
    window = (
        d.infection_status.isin(["pre_onset", "infected"]) & d.tau_h.ge(cfg["tau_start"]) & d.tau_h.lt(cfg["tau_stop"])
    )
    m = d[window & ~d.mantis_death & d.cc_state.eq("M")].copy()
    m["tau_bin"] = np.floor((m.tau_h - cfg["tau_start"]) / width) * width + cfg["tau_start"]
    before = nearest_to_center(m, "tau_bin", "tau_h", width)
    after = nearest_to_center(m[m.confirmed], "tau_bin", "tau_h", width).reset_index(drop=True)
    counts = []
    for b in tau_bins(cfg):
        counts.append(
            dict(
                tau_bin=b,
                m_frames=int(m.tau_bin.eq(b).sum()),
                m_frames_confirmed=int((m.tau_bin.eq(b) & m.confirmed).sum()),
                tracks_before_mantis=int(before.tau_bin.eq(b).sum()),
                tracks_after_mantis=int(after.tau_bin.eq(b).sum()),
            )
        )
    fold_of = json.loads((work / "folds.json").read_text())["fold_of"]
    after["fold"] = "fold" + (after.dataset + "::" + after.fov_name).map(fold_of).astype(int).astype(str)
    after, emb, _ = attach_embeddings(after, cfg)
    chosen = json.loads((work / "run_summary.json").read_text())["chosen"]["model_i"]
    after["score"] = np.nan
    for fold, g in after.groupby("fold"):
        model, classes, _ = load_classifier(
            train_module, run_dir(work, "model_i", fold, chosen) / "checkpoint.pt", device
        )
        p = bag_probs(model, emb, g.index.to_numpy()[:, None], device)
        after.loc[g.index, "score"] = infection_score(p, classes)
    after["row_id"] = -(np.arange(len(after)) + 1)
    after["group"] = "onset"
    return after, counts


def render(cfg):
    """Stage ``render``: diverse QC-passing top cells -> cycle-vs-infection panel, pseudotime grid, montage, UMAP."""
    work = Path(cfg["work_dir"])
    out = work / "figures"
    pages = out / "pages"
    out.mkdir(exist_ok=True)
    if pages.exists():
        shutil.rmtree(pages)
    pages.mkdir()
    r_cfg = cfg["render"]
    scores = pd.read_parquet(work / "analysis" / "scores.parquet")
    cells = pd.read_parquet(work / "cells.parquet")
    a13 = crop_helpers(cfg)
    crops = Crops(cfg, a13, cells)
    phases = cfg["phases"]
    report, selection = {}, []

    def take(figure, slot, cands, k, used, fovs, per_fov):
        stats = Counter()
        chosen = pick(cands, k, crops, figure, used, fovs, per_fov, stats)
        report[f"{figure}/{slot}"] = dict(candidates=len(cands), **stats)
        for order, r in enumerate(chosen, start=1):
            selection.append(dict(figure=figure, slot=slot, order=order, **r.to_dict()))
        return chosen

    used, fovs = set(), Counter()
    panel = {}
    for phase in phases:
        panel[("cycle_typical", phase)] = take(
            "panel_cycle_vs_infection",
            f"cycle_typical_{phase}",
            ranked(scores, "model_c", phase),
            r_cfg["per_column"],
            used,
            fovs,
            r_cfg["per_fov"],
        )
        panel[("infected", phase)] = take(
            "panel_cycle_vs_infection",
            f"infected_TAU_p00_{phase}",
            ranked(scores, "model_i", "TAU_p00", phase),
            r_cfg["per_column"],
            used,
            fovs,
            r_cfg["per_fov"],
        )
        panel[("bystander", phase)] = take(
            "panel_cycle_vs_infection",
            f"bystander_{phase}",
            ranked(scores, "model_i", "BYSTANDER", phase),
            r_cfg["per_column"],
            used,
            fovs,
            r_cfg["per_fov"],
        )
    used, fovs = set(), Counter()
    pseudo = {}
    for phase in phases:
        for b in tau_bins(cfg):
            c = tau_class(b)
            pseudo[(c, phase)] = take(
                "pseudotime_representatives",
                f"{c}_{phase}",
                ranked(scores, "model_i", c, phase),
                1,
                used,
                fovs,
                r_cfg["per_fov"],
            )
    pages_spec = []
    montage_classes = [("model_i", c) for c in model_i_classes(cfg)] + [
        ("model_c", c) for c in scores[scores.model.eq("model_c")].cls.unique()
    ]
    for model, c in montage_classes:
        cands = ranked(scores, model, c)
        chosen = take(
            "montage_top_cells", f"{model}_{c}", cands, r_cfg["montage_per_class"], set(), Counter(), r_cfg["per_fov"]
        )
        info = report[f"montage_top_cells/{model}_{c}"]
        if c.startswith("TAU_"):
            title = f"onset lineages, {tau_label(c, cfg['bin_hours'])} from TCN onset (phase-matched pool)"
        elif model == "model_i":
            title = CLASS_TITLES[c]
        else:
            title = f"{c}, uninfected (control + bystander, source-balanced pool)"
        rejected = (
            ", ".join(f"{k.removeprefix('reject_')} {v}" for k, v in info.items() if k.startswith("reject_")) or "none"
        )
        lines = [
            "Ranked by pinned score.py score (mean leave-one-out P(class | bag) − P(class | bag∖cell) over bag sizes "
            "1–100) on each CV fold's test FOVs.",
            f"Diversity: ≤ 1 per lineage_uid, ≤ {r_cfg['per_fov']} per FOV on this page. Crop QC (attempt 13): in "
            "bounds, "
            "focus z ≥ 5 and z+11 ≤ Z,",
            "central TOMM20 and pAL MIP p99 ≥ dataset threshold; QC-rejected and skipped cells are replaced by the "
            "next "
            "ranked cell.",
            f"Skipped: lineage {info.get('skip_lineage_used', 0)}, FOV full {info.get('skip_fov_full', 0)}; QC "
            f"rejections: {rejected}.",
            "Tiles: Phase focus slice | TOMM20 green + pAL magenta 16-slice MIP; 5 µm bar.",
        ]
        header = [
            (
                0.16,
                f"{MODEL_TITLES[model].split(' (')[0]} · {c} — top {len(chosen)} of {info['candidates']:,} held-out "
                "cells",
                dict(size=12, weight="bold"),
            ),
            (0.44, title, dict(size=8.5)),
            *((0.66 + n * 0.14, s, dict(size=7)) for n, s in enumerate(lines)),
        ]
        pages_spec.append((f"{model}_{c}", header, chosen, info))
    accepted = pd.DataFrame(selection).drop_duplicates("row_id")
    images = {}
    for rid in accepted.row_id:
        crop = crops.results[int(rid)][1]["crop"]
        for ch in a13.CHANNELS:
            images[f"r{int(rid)}_{ch}"] = crop[ch]
    limits = a13.contrast_limits(accepted.assign(candidate_id=[f"r{int(r)}" for r in accepted.row_id]), images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cands, m_counts = m_row_candidates(cfg, pinned_train(cfg), device)
    outcomes = [crops.get(r, "pseudotime_representatives_M")[0] or "accepted" for _, r in m_cands.iterrows()]
    m_cands["crop_outcome"] = outcomes
    m_cands["picked"] = False
    for row in m_counts:
        b = row["tau_bin"]
        cands = m_cands[m_cands.tau_bin.eq(b)].sort_values(["score", "row_id"], ascending=[False, True])
        stats = Counter()
        chosen = pick(cands, 1, crops, "pseudotime_representatives", used, fovs, r_cfg["per_fov"], stats)
        pseudo[(tau_class(b), "M")] = chosen
        report[f"pseudotime_representatives/{tau_class(b)}_M"] = dict(candidates=len(cands), **stats)
        m_cands.loc[[r.name for r in chosen], "picked"] = True
        row.update(after_qc=int(cands.crop_outcome.eq("accepted").sum()), picked=len(chosen))
        if chosen:
            r = chosen[0]
            row.update(picked_cell=f"{r.dataset} {r.fov_name} track {int(r.track_id)} t {int(r.t)} q_inf {r.score:.3f}")
    render_cycle_panel(panel, crops, limits, cfg, out / "panel_cycle_vs_infection")
    render_pseudotime(pseudo, crops, limits, cfg, out / "pseudotime_representatives")
    render_montage(pages_spec, crops, limits, a13, out / "montage_top_cells.pdf", pages)
    render_umap(cfg, scores, out / "umap_top_cells")
    pd.DataFrame(selection).to_csv(out / "selection.csv", index=False)
    audit = pd.DataFrame(crops.audit)
    m_meta = audit[audit.row_id.lt(0)].drop(columns=["first_figure", "dataset", "fov_name", "track_id", "t", "outcome"])
    m_cands.drop(columns=["confirmed"]).merge(m_meta, on="row_id", how="left", validate="one_to_one").to_csv(
        out / "pseudotime_m_candidates.csv", index=False
    )
    audit = audit[audit.row_id.ge(0)]
    audit.to_csv(out / "crop_audit.csv", index=False)
    write_json(
        out / "render.json",
        dict(
            thresholds=crops.thresholds,
            image_zarrs=crops.paths,
            channel_confirmation=crops.channels,
            contrast_limits=limits,
            selection=report,
            crops_read=len(audit),
            crop_outcomes=audit.outcome.value_counts().to_dict(),
            m_row=dict(
                rule=(
                    "onset-lineage frames (pre_onset + infected, not Mantis death) in each τ bin with cc_state M and "
                    "a Mantis division event on the same track at t-1, t or t+1 (evidence available, "
                    f"probability__division >= {r_cfg['m_row']['division_min']}, probability__death <= "
                    f"{r_cfg['m_row']['death_max']}); one frame per (track, τ bin) nearest the bin centre; ranked by "
                    "Model I single-cell q_inf from the chosen-size checkpoint of the fold holding the FOV out; same "
                    "diversity (shared with the G1/S/G2 rows) and crop QC; the M crops do not enter the contrast limits"
                ),
                per_bin=m_counts,
                crop_outcomes=m_cands.crop_outcome.value_counts().to_dict(),
            ),
        ),
    )
    print("render done", json.dumps({k: v for k, v in report.items() if v.get("shortfall")}, default=str), flush=True)


def md_table(df, digits=3):
    """Render a DataFrame as a GitHub markdown table (floats rounded)."""

    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pm(mean, sd, digits=3):
    """Format ``mean ± sd``."""
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def source_shas(cfg):
    """Recompute the pinned files' sha256 and check them against ``SOURCE.md``."""
    root = Path(cfg["pinned_root"])
    table = {}
    for line in (root / "SOURCE.md").read_text().splitlines():
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 3 and parts[0].startswith("set_classifier/"):
            table[parts[0]] = parts[2]
    found = {f: sha(root / f) for f in table}
    if found != table:
        raise ValueError(f"pinned files differ from SOURCE.md: {[f for f in table if found[f] != table[f]]}")
    return found


def run_records(work):
    """Return the end records of the full and smoke run logs."""
    out = []
    for name, log in [("full", work / "run_log.jsonl"), ("smoke", work / "smoke" / "run_log.jsonl")]:
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry["event"] == "end":
                out.append(dict(run=name, **entry))
    return out


def cross_verdict(cross):
    """Return one sentence per direction from the pooled cross-scoring rows (bootstrap 95% interval vs references)."""
    lines = []
    f = cross[
        (cross.direction.eq("model_i_top_to_model_c")) & cross.cls.eq("TAU_p* pooled") & cross.set.eq("top")
    ].iloc[0]
    ref = cross[
        (cross.direction.eq("model_i_top_to_model_c"))
        & cross.cls.eq("TAU_p* pooled")
        & cross.set.eq("random_reference")
    ].iloc[0]

    def side(lo, hi):
        return "no detectable difference" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")

    lines.append(
        f"Model I's top post-onset cells (n = {f.n}) get Model C's argmax = their PCNA→Phase call in "
        f"{f.argmax_is_called:.1%} "
        f"vs {ref.argmax_is_called:.1%} for random reference cells of the same fold and call (difference "
        f"{f.argmax_is_called_minus_reference:+.3f}, "
        f"95% bootstrap [{f.argmax_is_called_ci_low:+.3f}, {f.argmax_is_called_ci_high:+.3f}]: "
        f"{side(f.argmax_is_called_ci_low, f.argmax_is_called_ci_high)}); "
        f"mean P(called phase) {f.p_called:.3f} vs {ref.p_called:.3f} ({f.p_called_minus_reference:+.3f} "
        f"[{f.p_called_ci_low:+.3f}, {f.p_called_ci_high:+.3f}]: "
        f"{side(f.p_called_ci_low, f.p_called_ci_high)})."
    )
    r = cross[(cross.direction.eq("model_c_top_to_model_i")) & cross.cls.eq("all phases pooled")].set_index("set")
    t = r.loc["top"]
    lines.append(
        f"Model C's top cycle cells (n = {t.n}) get Model I's single-cell infection score q_inf {t.q_inf:.3f} vs "
        f"{r.loc['random_reference'].q_inf:.3f} "
        f"for random reference cells of the same fold and call ({t.q_inf_minus_reference:+.3f} [{t.q_inf_ci_low:+.3f}, "
        f"{t.q_inf_ci_high:+.3f}]: "
        f"{side(t.q_inf_ci_low, t.q_inf_ci_high)}) and {r.loc['random_post_onset'].q_inf:.3f} for random post-onset "
        "cells."
    )
    return lines


def write_readme(cfg, ctx):
    """Write the attempt-25 README from the export, run, analysis and render records."""
    e, a, rj = ctx["export"], ctx["analysis"], ctx["render"]
    sel, sweep, rank = ctx["selection"], ctx["sweep"], ctx["rank"]
    au, dist, cross, counts = ctx["auroc"], ctx["distortion"], ctx["cross"], ctx["counts"]
    chosen, size = a["chosen"], a["analysis_bag_size"]
    cv = counts[counts.fold.str.startswith("fold")]
    lines = [
        "# Attempt 25 — OPS set classifier (unchanged) on TOMM20 embeddings: cell cycle vs infection",
        "",
        "Alex Lin's set classifier (`czbiohub-sf/ops_monorepo@ab11b56`, "
        "`ops_model/src/ops_model/models/interpretability/classifier/`, run unchanged as `python -m "
        "set_classifier.{train,eval,score}`) on cached DynaCLR TOMM20 `obsm/X_normalized_pca80` (49 PCs, control-MAD "
        "normalized per experiment × HPI, PCA checksum `c81187f1…`) of 2026_07_01 and 2026_04_10.",
        "",
        "- **Model I** — classes: TCN τ bins of 2 h over [−8, 12) for onset lineages (`TAU_m08` … `TAU_p10`), "
        "`BYSTANDER` (ZIKV-well lineages without TCN onset) and `CONTROL` (control wells 07_01 A/3, 04_10 A/2). Every "
        "pool is phase-matched (equal G1/S/G2, M excluded) within each fold × split, so cycle composition cannot drive "
        "the call.",
        f"- **Model C** — classes {', '.join(e['model_c_m_decision']['classes'])} of uninfected cells only (control + "
        "bystander, each phase pool balanced between the two sources). M is included because every training split has "
        f"≥ {e['model_c_m_decision']['min_m_cells']} source-balanced M cells (min "
        f"{min(e['model_c_m_decision']['m_train_balanced'].values())}).",
        "- Both use the same FOV folds and representative frames.",
        "",
        "## Inputs",
        "",
        f"- Labels: snapshot `{ctx['labels']['path']}` of `{ctx['labels']['origin']}`, sha256 "
        f"`{ctx['labels']['sha256']}` (asserted at export, render and report)."
        + (
            f" The export stage read the identical bytes (same sha256) from the origin path "
            f"`{ctx['labels']['export_read']['path']}` before the snapshot existed."
            if ctx["labels"]["export_read"]["path"] != ctx["labels"]["path"]
            else ""
        )
        + f" {e['labels']['frames']:,} frames; Mantis-death frames excluded "
        f"({e['labels']['mantis_death_excluded']:,}); ambiguous lineages excluded "
        f"({e['labels']['ambiguous_excluded']:,} live frames); {e['labels']['used_frames']:,} frames used.",
    ]
    for j in e["embeddings"]:
        lines.append(
            f"- {j['dataset']}: `{j['path']}` `obsm/X_normalized_pca80` sha256 `{j['obsm_sha256'][:16]}…`, checksum "
            f"`{j['pca_model_checksum'][:16]}…`; {j['matched_rows']:,} used frames all joined on (fov_name, track_id, "
            f"t) to nonblank rows ({j['nonblank_rows']:,} nonblank of {j['store_rows']:,})."
        )
    lines += [
        "- Representatives: one frame per (dataset, FOV, track, 2 h bin) nearest the bin centre (earliest t on ties); "
        "τ bins for onset lineages, HPI bins for references limited to each dataset's onset HPI span "
        f"({', '.join(f'{k} {v["min"]:.1f}–{v["max"]:.1f} h' for k, v in e['representatives']['hpi_span'].items())}). "
        f"{ctx['n_cells']:,} representatives.",
        f"- Pinned source: `{cfg['pinned_root']}/SOURCE.md` (repo, commit, per-file sha256; re-verified by this "
        "report). Hydra configs: `code/train_tomm20_set_classifier.yaml`, `code/eval_tomm20_set_classifier.yaml` "
        "(shipped `train_set_classifier_phase_1K.yaml` values: d_model 512, 4 heads, 2 ISAB layers, 32 inducing "
        "points, dropout 0, cosine head, concat conditioning, lr 1e-4, wd 0, warmup 20, 100 epochs, batch 64, 32 / 5 "
        "sets per class per epoch, `val_n_cells_per_set` [500, 200]; changed only `num_workers` 0 and, in eval, "
        "`sample_without_replacement` true).",
        "",
        "## Folds",
        "",
        "5 CV folds by FOV (seed 17): FOVs shuffled within dataset × well type and dealt round-robin within well type; "
        "fold k test = its FOVs, val = fold k+1 (used only by `train.py` checkpoint selection and the baselines' C), "
        "train = the rest. Two leave-one-dataset-out (LODO) folds: train/val from one dataset (val = 20% of FOVs per "
        "well type), test on the other. No FOV is in two splits of a fold (asserted). Each CV fold has 2 control FOVs; "
        "fold0's test has no 07_01 control FOV (07_01 has only 4).",
        "",
        "## Pools (phase matching)",
        "",
        "Counts before matching are representative frames per called phase; after = phase-matched pool (Model I) or "
        "source-balanced pool (Model C). CV test pools partition the FOVs, so their sum is every matched cell once. "
        "Per fold × split × class counts: `source_data/pool_counts.csv`. No class was dropped (train ≥ 20 cells and "
        "non-empty val/test everywhere).",
        "",
    ]
    rep = pd.DataFrame(e["representatives"]["representatives_by_phase"]).fillna(0).astype(int)
    rows = []
    for c in model_i_classes(cfg):
        g = cv[cv.model.eq("model_i") & cv.cls.eq(c)]
        rows.append(
            dict(
                cls=c,
                **{f"reps {p}": int(rep.loc[c, p]) for p in ALL_PHASES},
                **{
                    f"{s} after (min–max over folds)": (
                        f"{g[g.split.eq(s)]['after'].min()}–{g[g.split.eq(s)]['after'].max()}"
                    )
                    for s in SPLITS
                },
                matched_total=int(g[g.split.eq("test")]["after"].sum()),
            )
        )
    lines += [md_table(pd.DataFrame(rows)), ""]
    rows = []
    for c in e["model_c_m_decision"]["classes"]:
        g = cv[cv.model.eq("model_c") & cv.cls.eq(c)]
        rows.append(
            dict(
                cls=c,
                **{
                    f"{s} after (min–max)": f"{g[g.split.eq(s)]['after'].min()}–{g[g.split.eq(s)]['after'].max()}"
                    for s in SPLITS
                },
                matched_total=int(g[g.split.eq("test")]["after"].sum()),
            )
        )
    lines += ["Model C (bystander = control within every pool):", "", md_table(pd.DataFrame(rows)), ""]
    s = (
        sel.groupby(["model", "n_cells_per_set"])
        .agg(
            mean_val=("val_accuracy", "mean"),
            sd_val=("val_accuracy", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            chosen=("chosen", "first"),
        )
        .reset_index()
    )
    lines += [
        "## Bag-size choice (inner validation only)",
        "",
        "`train.py` for `n_cells_per_set` ∈ {10, 25, 50, 100} × 5 folds × 2 models (40 runs). Each run keeps the epoch "
        "with the best primary val accuracy (5 val bags per class at the training size; `metrics_out`). The size with "
        "the highest mean over folds is chosen (ties → smaller). Test data are never read. `bag_size_selection.csv` "
        "has every run.",
        "",
        md_table(s),
        "",
        f"Chosen: Model I **{chosen['model_i']}**, Model C **{chosen['model_c']}**. "
        + "; ".join(
            f"{m}: mean val spans {g.mean_val.min():.3f}–{g.mean_val.max():.3f} across sizes, fold SD "
            f"{g.sd_val.min():.3f}–{g.sd_val.max():.3f}"
            for m, g in s.groupby("model")
        )
        + ". Val = 5 bags per class, and the stored value is the best epoch's, so it is noisy and optimistic.",
        "",
        "## Accuracy vs bag size (eval.py on held-out test FOVs)",
        "",
        "`bag_size_sweep.{png,pdf,csv}`: every trained size evaluated by the pinned `eval.py` (30 repetitions, "
        "`sample_without_replacement` true). eval.py pads a pool smaller than `n_cells` and uses all its cells (every "
        "repetition then repeats one bag), so each fold's list is capped at its smallest test pool "
        "(`source_data/eval_caps.csv`): "
        + "; ".join(
            f"{r.model} {r.fold} ≤ {r.smallest_test_pool} → max {max(r.eval_n_cells)}" for r in ctx["caps"].itertuples()
        )
        + ". Mean ± SD over folds:",
        "",
    ]
    for model in MODELS:
        g = sweep[sweep.model.eq(model) & sweep.n_train.eq(chosen[model])].sort_values("n_eval")
        lines += [
            f"{MODEL_TITLES[model]}, trained at {chosen[model]} (chance {g.chance.iloc[0]:.3f}):",
            "",
            md_table(
                pd.DataFrame(
                    dict(
                        eval_bag=g.n_eval,
                        top1=[pm(m, sd) for m, sd in zip(g["mean"], g["sd"].fillna(0))],
                        top5=g.top5_mean.round(3),
                        folds=g.n_folds,
                        mean_eval_sem=g.mean_eval_sem.round(3),
                    )
                )
            ),
            "",
        ]
    lines += [
        "## Model I class ranking (gKO-ranking analogue)",
        "",
        f"`class_ranking.{{png,csv}}`: eval.py per-class held-out top-1 at training size {chosen['model_i']} and "
        f"evaluation bags of {size['model_i']} (the largest size in every fold), mean ± SD over 5 folds:",
        "",
        md_table(rank[["rank", "cls", "label", "top1_mean", "top1_sd", "n_folds"]]),
        "",
        "## Infection AUROC (identical seeded bags, three methods)",
        "",
        f"Our own held-out bags: 30 per class of **{size['model_i']} cells** (without replacement inside a bag), per "
        "fold; bag size = largest eval size ≤ the chosen size and ≤ half the smallest CV test pool (21), so the "
        "smallest τ pools still give distinct bags. q = P(τ) / (P(τ) + P(ref)) from the full softmax (pinned "
        "`build_model`). Baselines on the identical bags: mean-pooled multinomial logistic regression on the 49 PCs "
        "and an HPI-only multinomial logistic regression (bag HPI mean, SD), both trained on 200 bags per class drawn "
        "the same way from the train pools, C ∈ {0.01, 0.1, 1, 10} chosen by val log-loss. "
        "`infection_auroc.{png,csv}`.",
        "",
        "**A (as specified):** reference bags from the whole reference test pool. **B (added, see deviations):** each "
        "τ bag paired with a reference bag of the same (dataset, 2 h HPI bin) composition, which removes the HPI "
        "difference between τ bins and references.",
        "",
    ]
    for variant in ["pool", "hpi_matched"]:
        v = au[au.reference_bags.eq(variant) & au.cls.str.startswith("TAU_")]
        rows = []
        for c in model_i_classes(cfg)[:-2]:
            row = dict(cls=c)
            for ref in REFERENCES:
                for method, short in [("set_classifier", "set"), ("mean_logreg", "mean LR"), ("hpi_logreg", "HPI LR")]:
                    r = v[v.cls.eq(c) & v.reference.eq(ref) & v.method.eq(method)].iloc[0]
                    row[f"vs {ref[:3]} {short}"] = pm(r["mean"], r["sd"], 2)
            rows.append(row)
        lines += [
            f"{'A' if variant == 'pool' else 'B'} — reference bags: {variant}",
            "",
            md_table(pd.DataFrame(rows)),
            "",
        ]
        b = au[au.reference_bags.eq(variant) & au.cls.eq("BYSTANDER")].set_index("method")
        lines += [
            f"BYSTANDER vs CONTROL (well effect, {variant}): set "
            f"{pm(b.loc['set_classifier', 'mean'], b.loc['set_classifier', 'sd'], 2)}, mean LR "
            f"{pm(b.loc['mean_logreg', 'mean'], b.loc['mean_logreg', 'sd'], 2)}, HPI LR "
            f"{pm(b.loc['hpi_logreg', 'mean'], b.loc['hpi_logreg', 'sd'], 2)}.",
            "",
        ]
    m = au[au.reference_bags.eq("hpi_matched")]
    lines += [
        f"HPI matching: mean |Δ bag-mean HPI| {m.mean_hpi_gap.mean():.2f} h; cross-dataset matched cells "
        f"{m.cross_dataset_fraction.mean():.1%} (folds whose reference test pool lacks one dataset's FOVs).",
        "",
    ]
    acc = ctx["accuracy"]
    acc_all = acc[acc.cls.eq("__all__")].assign(split=lambda x: np.where(x.fold.str.startswith("fold"), "cv", x.fold))
    t = acc_all.groupby(["model", "method", "split"]).top1.agg(["mean", "std"]).reset_index()
    lines += ["Top-1 on these bags (CV mean ± SD; LODO per fold):", "", md_table(t), ""]
    lodo = ctx["lodo_eval"]
    lines += [
        "## Leave-one-dataset-out transfer",
        "",
        "Trained at the chosen size on one dataset (val 20% of its FOVs), tested on the other. eval.py top-1 (30 "
        "repetitions):",
        "",
        md_table(lodo.pivot_table(index=["model", "fold"], columns="n_eval", values="top1").reset_index()),
        "",
    ]
    la = ctx["auroc_folds"]
    la = la[~la.fold.str.startswith("fold")]
    lt = (
        la[la.cls.str.startswith("TAU_p")]
        .pivot_table(index=["fold", "reference_bags", "reference"], columns="method", values="auroc", aggfunc="mean")
        .reset_index()
    )
    lines += ["Mean AUROC over post-onset τ bins (τ ≥ 0) on LODO test bags:", "", md_table(lt), ""]
    lb = (
        la[la.cls.eq("BYSTANDER")]
        .pivot_table(index=["fold", "reference_bags"], columns="method", values="auroc")
        .reset_index()
    )
    lines += ["LODO BYSTANDER vs CONTROL AUROC:", "", md_table(lb), ""]
    lines += [
        "## Model C",
        "",
        f"Confusion on reference test bags of {size['model_c']} cells (`confusion.png`, right): diagonal "
        f"{', '.join(f'{c} {v:.2f}' for c, v in ctx['diag_c'].items())}. Model I confusion (left): diagonal "
        f"{', '.join(f'{c} {v:.2f}' for c, v in ctx['diag_i'].items())}.",
        "",
        "Distortion map (`cycle_distortion.{png,csv}`): Model C's mean predicted distribution for held-out "
        "onset-lineage bags (all called phases, not phase-matched) per τ bin × called phase, bags of ≤ "
        f"{cfg['analyze']['distortion_max_bag']} cells from {cfg['analyze']['distortion_partitions']} random "
        "partitions (each cell in up to 30 bags), vs reference test bags of the same called phase and size. Groups "
        f"with < {cfg['analyze']['distortion_min_cells']} cells skipped. Δ P(called phase), onset − reference, mean "
        "over folds:",
        "",
        md_table(dist.pivot_table(index="cc_state", columns="tau_bin", values="delta_called").reset_index(), 2),
        "",
        "## Cross-scoring (the decoupling test)",
        "",
        f"`cross_scoring.csv` (`source_data/cross_scoring_cells.csv` per cell). Top {cfg['analyze']['cross_top']} "
        "cells per class by score.py, pooled over CV folds, each scored with its own fold's other model on a bag of 1.",
        "",
        *[f"- {s}" for s in cross_verdict(cross)],
        "",
        md_table(cross[cross.cls.str.contains("pooled")].drop(columns=["phases"]).dropna(axis=1, how="all"), 3),
        "",
        "## Montages",
        "",
        "Cells ranked by the pinned `score.py` (mean over bag sizes 1, 2, 5, 10, 20, 50, 100 of the leave-one-out "
        "contribution P(class | bag) − P(class | bag∖cell); `--reps 50` tapered as shipped) on each CV fold's test "
        "dump at the chosen size; joined back on `index` = `row_id` (dataset, FOV and segmentation id asserted). "
        "Diverse picks: ≤ 1 per lineage_uid per figure (per page in the montage), ≤ 2 per FOV; attempt-13 crop reader "
        "and QC with its manifest thresholds (TOMM20 and pAL gates); QC failures are skipped and recorded "
        "(`source_data/crop_audit.csv`).",
        "",
        "- `panel_cycle_vs_infection.{png,pdf}`, `pseudotime_representatives.{png,pdf}`, `montage_top_cells.pdf` "
        f"(pages in `pages/`), `umap_top_cells.png` (top {cfg['render']['umap_top']} per class on the attempt-23 Phase "
        "UMAP).",
        f"- Crops read {rj['crops_read']}; outcomes {rj['crop_outcomes']}.",
    ]
    short = {k: v for k, v in rj["selection"].items() if v.get("shortfall") and not k.endswith("_M")}
    lines.append(f"- Shortfalls (G1/S/G2 slots): {json.dumps(short) if short else 'none'}.")
    m_row = rj["m_row"]
    lines += [
        "",
        "### Pseudotime M row",
        "",
        f"Rule: {m_row['rule']}. Counts per τ bin (held-out test FOVs of the 5 CV folds = every FOV): live onset M "
        "frames, those with a confirmed division, (track, τ bin) units before and after the Mantis rule, candidates "
        "passing crop QC and cells shown. Per-candidate q_inf, fold and crop outcome: "
        "`source_data/pseudotime_m_candidates.csv`.",
        "",
        md_table(pd.DataFrame(m_row["per_bin"])),
        "",
        f"- M crop outcomes: {m_row['crop_outcomes']}. A bin without a QC-passing, diversity-compatible confirmed M "
        "cell is left blank ('no confirmed M'); unconfirmed M frames are never used.",
    ]
    lines += [
        "",
        "## Deviations from the brief",
        "",
        *[f"- {d}" for d in ctx["deviations"]],
        "",
        "## Caveats",
        "",
        '- Cycle labels are PCNA→Phase calls, not PCNA ground truth; infected "G2" is partly a calling artifact '
        "(attempt 22). Phase matching is only as good as those calls.",
        '- The TCN onset lags sensor translocation by ~5.5 h; "2 hpi" here means τ [0, 2) after the TCN onset, so '
        "pre-onset bins (τ −6…0) already include sensor-positive cells.",
        "- HPI confound: τ bins sit at later and narrower HPI than the references. With pool references (A) the "
        f"HPI-only regression reaches {ctx['hpi_pool']:.2f} mean AUROC over τ bins and references, the set classifier "
        f"{ctx['set_pool']:.2f}; with HPI-matched references (B) the HPI-only regression drops to "
        f"{ctx['hpi_matched']:.2f} while the set classifier gives {ctx['set_matched']:.2f} and the mean-pooled "
        f"regression {ctx['mean_matched']:.2f}.",
        "- Only 2 datasets, 46 ZIKV and 10 control FOVs: fold SD measures split sensitivity, not biological "
        "replication; test bags overlap within a pool.",
        "- eval.py's SEM is across repetitions of one fold's test pools and understates uncertainty; there is no "
        "permutation null. Training is not bit-reproducible (`train.py` does not seed Python's `random`, which "
        "shuffles bag order).",
        "- Late τ pools are small (21–48 cells per CV test fold), which caps evaluation bag sizes; checkpoint "
        "selection uses 60 val bags.",
        "- The montage ranking is predictive attribution of the classifier on these embeddings, not causality.",
        "- Crop QC selects against dim TOMM20: TOMM20-gate rejections while filling the 10 montage cells per class: "
        + ", ".join(
            f"{k.split('/')[1].removeprefix('model_i_').removeprefix('model_c_')} {v.get('reject_tomm20_dim', 0)}"
            for k, v in rj["selection"].items()
            if k.startswith("montage_top_cells/")
        )
        + ". Where many top cells are rejected (late τ bins), the montage shows lower-ranked cells (see each caption's "
        "rank and score).",
        "- Visual check of the montages: a few top BYSTANDER and Model C G2 cells, including one control-well G2 cell, "
        "show diffuse nuclear pAL17. Whether these are infected cells without a TCN onset call or cycle-linked sensor "
        "redistribution is not resolved here; the reference pools are not sensor-verified.",
        "",
        "## Files",
        "",
        "Figures and tables here; `source_data/` (pool counts, folds, eval caps, per-fold AUROC, test-bag "
        "probabilities, matched bags, per-class eval, scores, selection, crop audit); `code/` (wrapper, config, hydra "
        "configs, SOURCE.md, test, imported crop scripts); `manifest.json`. Dumps, checkpoints and run outputs: `"
        + cfg["work_dir"]
        + "`.",
    ]
    (Path(cfg["fig_dir"]) / "README.md").write_text("\n".join(lines) + "\n")


DEVIATIONS = [
    "Infection AUROC: added variant B with reference bags matched to each τ bag's (dataset, 2 h HPI bin) composition "
    "(cross-dataset only where a fold's reference test pool lacks that dataset), because the HPI-only baseline "
    "separates τ bags from pool references almost perfectly; variant A is the specified analysis.",
    "Analysis bags (AUROC, confusion, cross-checks) use the largest eval size ≤ the chosen size and ≤ half the "
    "smallest CV test pool (10 cells), not the chosen training size, because late τ test pools (21–48 cells) are "
    "smaller than the chosen size; eval.py lists are capped per fold at the smallest test pool.",
    "Classes are also dropped from a fold if their val or test pool is empty (an empty pool would give all-masked "
    "bags); no class was dropped.",
    "Hydra configs: num_workers 0 (≤ 8 threads on the shared node) and eval show_progress false; everything else as "
    "shipped. Training seed 42 as shipped; our sampling uses seed 17.",
    "Cross-scoring adds a random same-class (forward) and random post-onset (reverse) comparator next to the specified "
    "random reference cells; the reverse random reference is also matched on source (bystander or control); the UMAP "
    "figure adds a Model C panel.",
    "Provenance: the export and run stages ran with an earlier revision of this wrapper (same export/run logic; later "
    "edits changed analyze/render/report, export-record tabulation, the label-path check and line wrapping); "
    "analyze, render and report were re-run with the revision recorded in manifest.json.",
    "Labels are read from the fixed snapshot fig6_inputs/attempt23_phase_embedding_e017ba39.parquet (sha256 asserted) "
    "because attempt 23 is being rewritten; the export stage had read the identical bytes from attempt 23's "
    "source_data/phase_embedding.parquet (sha256 checked against attempt 23's manifest at that time).",
    "Folds deal FOVs round-robin within well type across datasets, so each fold has 2 control FOVs but 07_01's 4 "
    "control FOVs cannot reach all 5 folds (fold0 test lacks one).",
]


def report(cfg):
    """Stage ``report``: claim the attempt-25 figure directory and write figures, tables, code, README and manifest."""
    work = Path(cfg["work_dir"])
    fig_dir = Path(cfg["fig_dir"])
    analysis, figures = work / "analysis", work / "figures"
    pinned = source_shas(cfg)
    fig_dir.mkdir(exist_ok=False)
    for sub in ["code", "source_data", "pages"]:
        (fig_dir / sub).mkdir()
    for name in [
        "bag_size_sweep.png",
        "bag_size_sweep.pdf",
        "bag_size_sweep.csv",
        "class_ranking.png",
        "class_ranking.csv",
        "infection_auroc.png",
        "infection_auroc.csv",
        "confusion.png",
        "cycle_distortion.png",
        "cycle_distortion.csv",
        "cross_scoring.csv",
    ]:
        shutil.copyfile(analysis / name, fig_dir / name)
    for name in [
        "panel_cycle_vs_infection.png",
        "panel_cycle_vs_infection.pdf",
        "pseudotime_representatives.png",
        "pseudotime_representatives.pdf",
        "montage_top_cells.pdf",
        "umap_top_cells.png",
    ]:
        shutil.copyfile(figures / name, fig_dir / name)
    for p in sorted((figures / "pages").glob("*.png")):
        shutil.copyfile(p, fig_dir / "pages" / p.name)
    shutil.copyfile(work / "bag_size_selection.csv", fig_dir / "bag_size_selection.csv")
    for p in [work / "pool_counts.csv", work / "folds.json", work / "eval_caps.csv", work / "export.json"]:
        shutil.copyfile(p, fig_dir / "source_data" / p.name)
    for name in [
        "eval_runs.csv",
        "lodo_eval.csv",
        "eval_per_class_model_i.csv",
        "infection_auroc_folds.csv",
        "test_bag_accuracy.csv",
        "test_bags.parquet",
        "test_bag_probs.parquet",
        "matched_reference_bags.parquet",
        "confusion.csv",
        "cycle_distortion_folds.csv",
        "cross_scoring_cells.csv",
        "scores.parquet",
        "analysis.json",
    ]:
        shutil.copyfile(analysis / name, fig_dir / "source_data" / name)
    for name in ["selection.csv", "crop_audit.csv", "render.json", "pseudotime_m_candidates.csv"]:
        shutil.copyfile(figures / name, fig_dir / "source_data" / name)
    hydra_dir = Path(cfg["hydra_config_dir"])
    test_path = Path(__file__).resolve().parents[1] / "tests" / "test_tomm20_set_classifier.py"
    scripts = Path(cfg["crop_helpers"]).parent
    code = [
        Path(__file__).resolve(),
        Path(cfg["config_path"]),
        hydra_dir / f"{cfg['hydra_train_config']}.yaml",
        hydra_dir / f"{cfg['hydra_eval_config']}.yaml",
        Path(cfg["pinned_root"]) / "SOURCE.md",
        test_path,
        Path(cfg["crop_helpers"]),
        *(scripts / f"{n}.py" for n in cfg["crop_helper_dependencies"]),
    ]
    for p in code:
        shutil.copyfile(p, fig_dir / "code" / p.name)

    a = json.loads((analysis / "analysis.json").read_text())
    au = pd.read_csv(analysis / "infection_auroc.csv")
    tau = au[au.cls.str.startswith("TAU_")]
    conf = pd.read_csv(analysis / "confusion.csv")

    def mean_of(variant, method):
        return float(tau[tau.reference_bags.eq(variant) & tau.method.eq(method)]["mean"].mean())

    def diag(model):
        return {r.true: r.rate for r in conf[conf.model.eq(model) & conf.true.eq(conf.pred)].itertuples()}

    export_record = json.loads((work / "export.json").read_text())
    if export_record["labels"]["sha256"] != cfg["labels_sha256"]:
        raise ValueError("the exported labels differ from the configured snapshot sha256")
    labels_input = dict(
        path=str(labels_path(cfg)),
        origin=cfg["labels_origin"],
        sha256=cfg["labels_sha256"],
        export_read={
            k: v for k, v in export_record["labels"].items() if k in ("path", "sha256", "manifest", "manifest_sha256")
        },
        **{
            k: v
            for k, v in export_record["labels"].items()
            if k not in ("path", "sha256", "origin", "manifest", "manifest_sha256")
        },
    )
    ctx = dict(
        export=export_record,
        labels=labels_input,
        analysis=a,
        render=json.loads((figures / "render.json").read_text()),
        selection=pd.read_csv(work / "bag_size_selection.csv"),
        sweep=pd.read_csv(analysis / "bag_size_sweep.csv"),
        rank=pd.read_csv(analysis / "class_ranking.csv"),
        auroc=au,
        auroc_folds=pd.read_csv(analysis / "infection_auroc_folds.csv"),
        accuracy=pd.read_csv(analysis / "test_bag_accuracy.csv"),
        lodo_eval=pd.read_csv(analysis / "lodo_eval.csv"),
        distortion=pd.read_csv(analysis / "cycle_distortion.csv"),
        cross=pd.read_csv(analysis / "cross_scoring.csv"),
        counts=pd.read_csv(work / "pool_counts.csv"),
        caps=pd.read_csv(work / "eval_caps.csv", converters=dict(eval_n_cells=json.loads)),
        n_cells=len(pd.read_parquet(work / "cells.parquet", columns=["row_id"])),
        diag_i=diag("model_i"),
        diag_c=diag("model_c"),
        hpi_pool=mean_of("pool", "hpi_logreg"),
        set_pool=mean_of("pool", "set_classifier"),
        hpi_matched=mean_of("hpi_matched", "hpi_logreg"),
        set_matched=mean_of("hpi_matched", "set_classifier"),
        mean_matched=mean_of("hpi_matched", "mean_logreg"),
        deviations=DEVIATIONS,
    )
    write_readme(cfg, ctx)
    outputs = {str(p.relative_to(fig_dir)): sha(p) for p in sorted(fig_dir.rglob("*")) if p.is_file()}
    manifest = dict(
        attempt=cfg["attempt"],
        inputs=dict(
            labels=labels_input,
            embeddings=export_record["embeddings"],
            attempt13_manifest=dict(path=cfg["attempt13_manifest"], sha256=sha(cfg["attempt13_manifest"])),
            image_zarrs=ctx["render"]["image_zarrs"],
            cells_parquet=dict(path=str(work / "cells.parquet"), sha256=sha(work / "cells.parquet")),
            pool_members=dict(path=str(work / "pool_members.parquet"), sha256=sha(work / "pool_members.parquet")),
        ),
        code=dict(
            wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).resolve())),
            config=dict(path=cfg["config_path"], sha256=sha(cfg["config_path"])),
            hydra={p.name: sha(p) for p in code[2:4]},
            test=dict(path=str(test_path), sha256=sha(test_path)),
            crop_helpers={p.name: sha(p) for p in code[6:]},
            pinned=dict(
                repo="czbiohub-sf/ops_monorepo",
                commit="ab11b56d70e09356847898b12e85d8c594e56f74",
                root=cfg["pinned_root"],
                sha256=pinned,
            ),
        ),
        params=cfg,
        folds=json.loads((work / "folds.json").read_text())["folds"],
        chosen=a["chosen"],
        analysis_bag_size=a["analysis_bag_size"],
        checkpoint_epochs=a["checkpoint_epochs"],
        baseline_fits=a["baseline_fits"],
        model_c_m_decision=export_record["model_c_m_decision"],
        runs=run_records(work),
        run_context=dict(
            host=cfg["run"]["host"],
            allocation=cfg["run"]["allocation"],
            report_host=socket.gethostname(),
            launch="ssh + nohup on the nomachine allocation (not sbatch)",
            cuda_visible_devices="0",
            threads_per_job=cfg["run"]["threads_per_job"],
            parallel=cfg["run"]["parallel"],
        ),
        deviations=DEVIATIONS,
        versions=dict(
            python=platform.python_version(),
            **{
                k: version(k)
                for k in [
                    "torch",
                    "hydra-core",
                    "omegaconf",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "zarr",
                    "anndata",
                    "matplotlib",
                ]
            },
        ),
        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        outputs=outputs,
    )
    write_json(fig_dir / "manifest.json", manifest)
    print("report done", fig_dir, flush=True)


def main():
    """Parse ``--config`` and ``--stage`` and run the requested stage(s)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["export", "run", "analyze", "render", "report", "all"], required=True)
    parser.add_argument("--smoke", action="store_true", help="stage run only: 1 fold, n_cells 10, 2 epochs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    plt.rcParams.update(RC)
    stages = ["export", "run", "analyze", "render", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "run":
            run(cfg, args.smoke)
        else:
            globals()[stage](cfg)


if __name__ == "__main__":
    main()
