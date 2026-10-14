"""DynaFormer evaluator: self-supervised temporal head plus the raw-sequence baseline ladder.

Per split (leave-one-experiment-out, then FOV folds) the head is trained on
training-fold lineage paths without labels, every representation in the ladder is
computed for all rows, and linear probes are fitted afterwards on the frozen
representations. Prospective readouts use only past-visible states; bidirectional
states are reported for retrospective readouts only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dynaclr.evaluation.geometry import embedding_geometry
from dynaclr.evaluation.temporal.config_dynaformer import DynaFormerConfig, load_dynaformer_config
from dynaclr.evaluation.temporal.dynaformer_data import (
    EvalWindowSpec,
    FoldStandardizer,
    LineageTable,
    MixedMaskSampler,
    build_eval_windows,
    build_lineage_table,
    causal_history_stack,
    infer_frame_interval_hours,
    smooth_along_paths,
)
from dynaclr.evaluation.temporal.dynaformer_model import (
    DynaFormerHead,
    HorizonDeltaMLP,
    anchor_states,
    batch_to_tensors,
    encode_rows,
    multi_horizon_infonce,
)
from dynaclr.evaluation.temporal.forecast import (
    ForecastTargets,
    assign_fov_folds,
    build_forecast_targets,
    load_forecast_data,
)

PROSPECTIVE_REPRESENTATIONS = ("z", "z_smooth_causal", "h_causal", "h_causal_nolineage", "h_causal_shuffled")
RETROSPECTIVE_REPRESENTATIONS = ("z", "z_smooth", "h_causal", "h_bidir", "h_bidir_shuffled")
SCORE_REPRESENTATIONS = ("z", "h_causal", "h_bidir")


@dataclass(frozen=True)
class Split:
    name: str
    kind: str
    train_mask: np.ndarray
    test_mask: np.ndarray


def next_run_dir(root: str | Path, tag: str) -> Path:
    """Create the next append-only ``NN_<tag>`` attempt directory under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        numbers = [int(p.name.split("_", 1)[0]) for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit()]
        candidate = root / f"{(max(numbers, default=-1) + 1):02d}_{tag}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate


def make_splits(labels: pd.DataFrame, config: DynaFormerConfig) -> tuple[list[Split], np.ndarray]:
    """Leave-one-experiment-out splits followed by dataset-stratified FOV folds."""
    experiments = labels["experiment"].astype(str).to_numpy()
    splits: list[Split] = []
    if len(np.unique(experiments)) > 1:
        for experiment in sorted(np.unique(experiments)):
            test = experiments == experiment
            splits.append(Split(f"loeo_{experiment}", "loeo", ~test, test))
    folds = assign_fov_folds(labels, config.n_folds, config.random_seed, split_group_columns=config.split_group_columns)
    for fold in range(config.n_folds):
        splits.append(Split(f"fov_fold{fold}", "fov", folds != fold, folds == fold))
    return splits, folds


def train_head(
    table: LineageTable,
    features: torch.Tensor,
    train_mask: np.ndarray,
    config: DynaFormerConfig,
    seed: int,
    device: torch.device,
    *,
    log: Any = None,
) -> tuple[DynaFormerHead, HorizonDeltaMLP, pd.DataFrame]:
    """Train the head and delta predictor with mixed masks and multi-horizon InfoNCE."""
    features_np = features.cpu().numpy()
    sampler = MixedMaskSampler(
        table,
        features_np,
        allowed_rows=train_mask,
        k_frames=config.k_frames,
        n_max_frames=config.n_max_frames,
        frame_dropout=config.frame_dropout,
        causal_probability=config.causal_probability,
        n_negatives=config.train.n_negatives,
        seed=seed,
    )
    torch.manual_seed(seed)
    settings = config.head
    head = DynaFormerHead(
        in_dim=features.shape[1],
        d_model=settings.d_model,
        n_heads=settings.n_heads,
        n_layers=settings.n_layers,
        feedforward_multiplier=settings.feedforward_multiplier,
        dropout=settings.dropout,
        rope_min_period_hours=settings.rope_min_period_hours,
        rope_max_period_hours=settings.rope_max_period_hours,
    ).to(device)
    delta_mlp = HorizonDeltaMLP(settings.d_model, len(config.k_frames), features.shape[1]).to(device)
    params = list(head.parameters()) + list(delta_mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    steps = config.train.steps
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / steps))),
    )
    history: list[dict[str, Any]] = []
    head.train()
    delta_mlp.train()
    started = time.time()
    for step in range(steps):
        batch = batch_to_tensors(sampler.sample_batch(config.train.batch_size), device)
        h = anchor_states(head, batch, features)
        anchor_z = features[batch["anchor_row"]]
        prediction = anchor_z[:, None, :] + delta_mlp(h)
        target = features[batch["target_row"].clamp_min(0)]
        negatives = features[batch["negative_row"].clamp_min(0)]
        loss, top1, n_valid = multi_horizon_infonce(
            prediction,
            target,
            batch["target_valid"],
            negatives,
            batch["negative_row"] >= 0,
            config.train.temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        if (step + 1) % config.train.log_every == 0 or step == steps - 1:
            record = {
                "step": step + 1,
                "loss": float(loss.item()),
                "seconds": time.time() - started,
                **{f"top1_k{k}": float(v) for k, v in zip(config.k_frames, top1.cpu().numpy())},
                **{f"n_valid_k{k}": int(v) for k, v in zip(config.k_frames, n_valid.cpu().numpy())},
            }
            history.append(record)
            if log is not None:
                log(
                    f"  step {step + 1}/{steps} loss {record['loss']:.3f} top1 "
                    + " ".join(f"{v:.2f}" for v in top1.cpu().numpy())
                )
    return head, delta_mlp, pd.DataFrame(history)


def _probe(X_train: np.ndarray, y_train: np.ndarray, C: float):
    model = make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=5000))
    model.fit(X_train, y_train)
    return model


def _score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "average_precision": np.nan, "balanced_accuracy": np.nan}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, score >= 0)),
    }


def _current_state_target(labels: pd.DataFrame, config: DynaFormerConfig) -> np.ndarray:
    event = config.primary
    states = labels[event.column].astype("string").str.strip().str.lower()
    y = np.full(len(labels), np.nan, dtype=np.float32)
    y[(states == event.negative.lower()).fillna(False).to_numpy(dtype=bool)] = 0.0
    y[(states == event.positive.lower()).fillna(False).to_numpy(dtype=bool)] = 1.0
    return y


def _representations(
    table: LineageTable,
    standardized: np.ndarray,
    features_t: torch.Tensor,
    head: DynaFormerHead,
    config: DynaFormerConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Compute every ladder representation for all rows."""
    all_rows = np.arange(table.n_rows)
    sigma = config.probe.smoothing_sigma_hours
    reps: dict[str, np.ndarray] = {
        "z": standardized,
        "z_smooth_causal": smooth_along_paths(table, standardized, sigma, causal=True),
        "z_smooth": smooth_along_paths(table, standardized, sigma, causal=False),
    }
    for k in config.probe.last_k:
        stack = causal_history_stack(table, standardized, k)
        reps[f"z_mean_k{k}"] = stack.mean(axis=1)
        reps[f"z_concat_k{k}"] = stack.reshape(len(stack), -1)
    n_max = config.n_max_frames
    hide = tuple(int(k) for k in config.k_frames)
    specs = {
        "h_causal": EvalWindowSpec(history_budget=n_max),
        "h_causal_nolineage": EvalWindowSpec(history_budget=n_max, no_lineage=True),
        "h_causal_shuffled": EvalWindowSpec(history_budget=n_max, shuffle=True),
        "h_bidir": EvalWindowSpec(history_budget=n_max, future_budget=n_max, hide_offsets=hide),
        "h_bidir_shuffled": EvalWindowSpec(history_budget=n_max, future_budget=n_max, hide_offsets=hide, shuffle=True),
    }
    for name, spec in specs.items():
        reps[name], _ = encode_rows(
            head,
            features_t,
            all_rows,
            lambda rows, spec=spec: build_eval_windows(table, rows, spec, rng=rng),
            device=device,
            batch_size=512,
        )
    return reps


def _fit_and_score(
    reps: dict[str, np.ndarray],
    names: Sequence[str],
    y: np.ndarray,
    split: Split,
    C: float,
    *,
    subsets: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit one probe per representation on training rows and score the test rows.

    Returns metric rows and the decision scores on **all** rows for each representation
    (needed for onset timing along whole tracks).
    """
    finite = np.isfinite(y)
    train_rows = np.flatnonzero(split.train_mask & finite)
    test_rows = np.flatnonzero(split.test_mask & finite)
    metric_rows: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    if len(np.unique(y[train_rows])) < 2 or len(test_rows) == 0:
        return metric_rows, scores
    for name in names:
        X = reps[name]
        model = _probe(X[train_rows], y[train_rows], C)
        score = model.decision_function(X).astype(np.float32)
        scores[name] = score
        groups = {"all": test_rows}
        if subsets:
            for subset_name, mask in subsets.items():
                groups[subset_name] = test_rows[mask[test_rows]]
        for subset_name, rows in groups.items():
            metric_rows.append(
                {
                    "representation": name,
                    "subset": subset_name,
                    "n_test": int(len(rows)),
                    "n_positive": int((y[rows] == 1).sum()),
                    **_score_metrics(y[rows], score[rows]),
                }
            )
    return metric_rows, scores


def _onset_timing_and_flicker(
    labels: pd.DataFrame,
    table: LineageTable,
    targets: ForecastTargets,
    scores: dict[str, np.ndarray],
    split: Split,
    config: DynaFormerConfig,
) -> list[dict[str, Any]]:
    """Onset-time error on fated test tracks and decision flicker on all test tracks."""
    hysteresis = config.probe.hysteresis_frames
    lead = targets.lead_hours[:, 0]
    onset_true = np.where(np.isfinite(lead), table.hours + lead, np.nan)
    groups = labels.groupby(list(config.group_columns), observed=True, sort=False).indices
    rows_out: list[dict[str, Any]] = []
    for name, score in scores.items():
        errors: list[float] = []
        n_fated = n_detected = 0
        flicker: list[float] = []
        for raw_rows in groups.values():
            rows = np.asarray(raw_rows, dtype=int)
            if not split.test_mask[rows].all():
                continue
            rows = rows[np.argsort(table.hours[rows], kind="stable")]
            s = score[rows]
            duration = table.hours[rows[-1]] - table.hours[rows[0]]
            if len(rows) >= 4 and duration > 0:
                flicker.append(float((np.diff(s >= 0) != 0).sum() / duration))
            true_onset = onset_true[rows]
            true_onset = true_onset[np.isfinite(true_onset)]
            if not len(true_onset):
                continue
            n_fated += 1
            positive = s >= 0
            for i in range(len(rows) - hysteresis + 1):
                if positive[i : i + hysteresis].all():
                    n_detected += 1
                    errors.append(float(table.hours[rows[i]] - true_onset[0]))
                    break
        errors_arr = np.asarray(errors)
        rows_out.append(
            {
                "split": split.name,
                "split_kind": split.kind,
                "representation": name,
                "n_fated_tracks": n_fated,
                "detected_fraction": (n_detected / n_fated) if n_fated else np.nan,
                "median_abs_error_hours": float(np.median(np.abs(errors_arr))) if len(errors_arr) else np.nan,
                "median_signed_error_hours": float(np.median(errors_arr)) if len(errors_arr) else np.nan,
                "n_tracks_flicker": len(flicker),
                "flicker_per_hour_mean": float(np.mean(flicker)) if flicker else np.nan,
            }
        )
    return rows_out


def _forecast_skill(
    table: LineageTable,
    standardized: np.ndarray,
    predicted: np.ndarray,
    test_rows: np.ndarray,
    config: DynaFormerConfig,
    split: Split,
) -> list[dict[str, Any]]:
    """Top-1 retrieval of the true future embedding among same-FOV same-frame cells."""
    rows_out: list[dict[str, Any]] = []
    test_set = np.zeros(table.n_rows, dtype=bool)
    test_set[test_rows] = True
    for j, k in enumerate(config.k_frames):
        anchors, targets_, prev_rows = [], [], []
        for row in test_rows:
            path, p = table.path_of(int(row))
            t_path = table.t[path]
            q = int(np.searchsorted(t_path, t_path[p] + k))
            if q < len(path) and t_path[q] == t_path[p] + k:
                anchors.append(row)
                targets_.append(path[q])
                prev_rows.append(path[p - 1] if p > 0 else -1)
        if not anchors:
            continue
        anchors = np.asarray(anchors)
        targets_ = np.asarray(targets_)
        prev_rows = np.asarray(prev_rows)
        crosses = table.track_id[targets_] != table.track_id[anchors]
        z_anchor = standardized[anchors]
        has_prev = prev_rows >= 0
        z_prev = standardized[np.where(has_prev, prev_rows, anchors)]
        dt = np.where(has_prev, table.t[anchors] - table.t[np.where(has_prev, prev_rows, anchors)], 1).astype(
            np.float32
        )
        linear = z_anchor + (z_anchor - z_prev) * (k / np.maximum(dt, 1))[:, None]
        predictors = {
            "head": predicted[:, j][np.searchsorted(test_rows, anchors)],
            "persistence": z_anchor,
            "linear_extrapolation": linear,
        }
        hits = {name: np.zeros(len(anchors), dtype=bool) for name in predictors}
        n_candidates = np.zeros(len(anchors), dtype=np.int32)
        key = pd.DataFrame({"g": table.group_token[anchors], "t": table.t[targets_]})
        for _, group_rows in key.groupby(["g", "t"], observed=True, sort=False).indices.items():
            idx = np.asarray(group_rows)
            candidates = table.frame_index[(str(key["g"].iloc[idx[0]]), int(key["t"].iloc[idx[0]]))]
            n_candidates[idx] = len(candidates)
            cand = standardized[candidates]
            cand = cand / np.clip(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12, None)
            target_pos = np.searchsorted(candidates, targets_[idx])
            for name, pred in predictors.items():
                q = pred[idx]
                q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
                best = (q @ cand.T).argmax(axis=1)
                hits[name][idx] = candidates[best] == candidates[target_pos]
        for name in predictors:
            for subset, mask in (
                ("all", np.ones(len(anchors), bool)),
                ("same_track", ~crosses),
                ("crosses_division", crosses),
            ):
                if not mask.any():
                    continue
                rows_out.append(
                    {
                        "split": split.name,
                        "split_kind": split.kind,
                        "k_frames": int(k),
                        "predictor": name,
                        "subset": subset,
                        "n_anchors": int(mask.sum()),
                        "mean_n_candidates": float(n_candidates[mask].mean()),
                        "top1": float(hits[name][mask].mean()),
                    }
                )
    return rows_out


def run_dynaformer(
    input_paths: Sequence[str | Path],
    config: DynaFormerConfig,
    output_dir: str | Path,
    *,
    embedding_key: str | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """Train per split, evaluate the ladder, and write all artifacts to ``output_dir``."""
    log = log or (lambda message: None)
    key = embedding_key or config.feature_key
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels, features, input_contract = load_forecast_data(input_paths, config, key)  # type: ignore[arg-type]
    interval_hours = infer_frame_interval_hours(labels, time_column=config.time_column)
    table = build_lineage_table(labels, parent_column=config.parent_column, time_column=config.time_column)
    log(f"rows={len(labels)} paths={table.stats['n_paths']} interval={interval_hours:.3f} h")
    targets = build_forecast_targets(
        labels,
        [config.primary],
        config.horizons_hours,
        group_columns=config.group_columns,
        time_column=config.time_column,
    )
    current_state = _current_state_target(labels, config)
    fated = np.isfinite(targets.lead_hours[:, 0])
    splits, folds = make_splits(labels, config)

    probe_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    training_logs: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    key_frame = labels[list(config.key_columns)].copy()

    for split_index, split in enumerate(splits):
        log(f"[{split.name}] training head on {int(split.train_mask.sum())} rows")
        standardizer = FoldStandardizer.fit(features, np.flatnonzero(split.train_mask))
        standardized = standardizer.apply(features)
        features_t = torch.from_numpy(standardized).to(device)
        seed = config.random_seed + 1000 * split_index
        head, delta_mlp, training_log = train_head(table, features_t, split.train_mask, config, seed, device, log=log)
        training_log.insert(0, "split", split.name)
        training_logs.append(training_log)
        torch.save(
            {"head": head.state_dict(), "delta_mlp": delta_mlp.state_dict(), "standardizer": standardizer.__dict__},
            output / f"head_{split.name}.pt",
        )

        rng = np.random.default_rng(seed)
        reps = _representations(table, standardized, features_t, head, config, device, rng)
        prospective_names = [
            *PROSPECTIVE_REPRESENTATIONS,
            *[f"z_mean_k{k}" for k in config.probe.last_k],
            *[f"z_concat_k{k}" for k in config.probe.last_k],
        ]

        for j, head_info in targets.heads.iterrows():
            metric_rows, scores = _fit_and_score(
                reps, prospective_names, targets.values[:, j], split, config.probe.C, subsets={"fated_only": fated}
            )
            for row in metric_rows:
                probe_rows.append(
                    {
                        "split": split.name,
                        "split_kind": split.kind,
                        "task": "onset_within",
                        "horizon_hours": head_info["horizon_hours"],
                        **row,
                    }
                )
            probe_rows.append(
                {
                    "split": split.name,
                    "split_kind": split.kind,
                    "task": "onset_within",
                    "horizon_hours": head_info["horizon_hours"],
                    "representation": "persistence_label",
                    "subset": "all",
                    "n_test": int((split.test_mask & np.isfinite(targets.values[:, j])).sum()),
                    "n_positive": int((targets.values[split.test_mask, j] == 1).sum()),
                    "auroc": np.nan,
                    "average_precision": np.nan,
                    "balanced_accuracy": np.nan,
                }
            )
            for name in SCORE_REPRESENTATIONS:
                if name in scores:
                    rows = np.flatnonzero(split.test_mask & np.isfinite(targets.values[:, j]))
                    part = key_frame.loc[rows].copy()
                    part["split"], part["task"], part["horizon_hours"], part["representation"] = (
                        split.name,
                        "onset_within",
                        head_info["horizon_hours"],
                        name,
                    )
                    part["target"], part["score"], part["lead_hours"] = (
                        targets.values[rows, j],
                        scores[name][rows],
                        targets.lead_hours[rows, j],
                    )
                    score_parts.append(part)

        metric_rows, scores = _fit_and_score(reps, RETROSPECTIVE_REPRESENTATIONS, current_state, split, config.probe.C)
        for row in metric_rows:
            probe_rows.append(
                {"split": split.name, "split_kind": split.kind, "task": "current_state", "horizon_hours": np.nan, **row}
            )
        timing_rows.extend(_onset_timing_and_flicker(labels, table, targets, scores, split, config))
        for name in SCORE_REPRESENTATIONS:
            if name in scores:
                rows = np.flatnonzero(split.test_mask & np.isfinite(current_state))
                part = key_frame.loc[rows].copy()
                part["split"], part["task"], part["horizon_hours"], part["representation"] = (
                    split.name,
                    "current_state",
                    np.nan,
                    name,
                )
                part["target"], part["score"], part["lead_hours"] = current_state[rows], scores[name][rows], np.nan
                score_parts.append(part)

        test_rows = np.flatnonzero(split.test_mask)
        _, predicted = encode_rows(
            head,
            features_t,
            test_rows,
            lambda rows: build_eval_windows(table, rows, EvalWindowSpec(history_budget=config.n_max_frames)),
            device=device,
            batch_size=512,
            delta_mlp=delta_mlp,
        )
        skill_rows.extend(_forecast_skill(table, standardized, predicted, test_rows, config, split))

        for budget in config.history_budget:
            h_budget, _ = encode_rows(
                head,
                features_t,
                np.arange(table.n_rows),
                lambda rows, b=budget: build_eval_windows(table, rows, EvalWindowSpec(history_budget=b)),
                device=device,
                batch_size=512,
            )
            local = {"h": h_budget}
            for j, head_info in targets.heads.iterrows():
                rows_j, _ = _fit_and_score(local, ["h"], targets.values[:, j], split, config.probe.C)
                for row in rows_j:
                    history_rows.append(
                        {
                            "split": split.name,
                            "split_kind": split.kind,
                            "history_budget": budget,
                            "task": "onset_within",
                            "horizon_hours": head_info["horizon_hours"],
                            **row,
                        }
                    )
            rows_c, _ = _fit_and_score(local, ["h"], current_state, split, config.probe.C)
            for row in rows_c:
                history_rows.append(
                    {
                        "split": split.name,
                        "split_kind": split.kind,
                        "history_budget": budget,
                        "task": "current_state",
                        "horizon_hours": np.nan,
                        **row,
                    }
                )
        for budget in config.future_budget:
            h_budget, _ = encode_rows(
                head,
                features_t,
                np.arange(table.n_rows),
                lambda rows, b=budget: build_eval_windows(
                    table,
                    rows,
                    EvalWindowSpec(
                        history_budget=config.n_max_frames,
                        future_budget=b,
                        hide_offsets=tuple(int(k) for k in config.k_frames),
                    ),
                ),
                device=device,
                batch_size=512,
            )
            rows_c, _ = _fit_and_score({"h": h_budget}, ["h"], current_state, split, config.probe.C)
            for row in rows_c:
                future_rows.append(
                    {
                        "split": split.name,
                        "split_kind": split.kind,
                        "future_budget": budget,
                        "task": "current_state",
                        **row,
                    }
                )

        geometry_z = embedding_geometry(reps["z"][test_rows])
        for name in ("z", "h_causal", "h_bidir"):
            geometry = embedding_geometry(reps[name][test_rows])
            rank_rows.append(
                {
                    "split": split.name,
                    "split_kind": split.kind,
                    "representation": name,
                    "n_dimensions": geometry["n_dimensions"],
                    "effective_rank": geometry["effective_rank"],
                    "participation_ratio": geometry["participation_ratio"],
                    "retention_vs_z": geometry["effective_rank"] / geometry_z["effective_rank"]
                    if geometry_z["effective_rank"]
                    else np.nan,
                }
            )
        del reps, features_t, head, delta_mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probe_metrics = pd.DataFrame(probe_rows)
    shuffle_rows: list[dict[str, Any]] = []
    for base, shuffled in (("h_causal", "h_causal_shuffled"), ("h_bidir", "h_bidir_shuffled")):
        merged = probe_metrics[probe_metrics.representation == base].merge(
            probe_metrics[probe_metrics.representation == shuffled],
            on=["split", "split_kind", "task", "horizon_hours", "subset"],
            suffixes=("", "_shuffled"),
        )
        for _, row in merged.iterrows():
            shuffle_rows.append(
                {
                    "split": row["split"],
                    "split_kind": row["split_kind"],
                    "task": row["task"],
                    "horizon_hours": row["horizon_hours"],
                    "subset": row["subset"],
                    "representation": base,
                    "auroc": row["auroc"],
                    "auroc_shuffled": row["auroc_shuffled"],
                    "auroc_gain_over_shuffled": row["auroc"] - row["auroc_shuffled"],
                }
            )
    fold_manifest = key_frame.copy()
    fold_manifest["fov_fold"] = folds
    fold_manifest["canonical_path"] = table.canonical_path
    fold_manifest["path_multiplicity"] = table.multiplicity
    fold_manifest["division_boundary"] = table.division_boundary
    audit = [
        {
            **info.to_dict(),
            "n_eligible": int(np.isfinite(targets.values[:, j]).sum()),
            "n_positive": int((targets.values[:, j] == 1).sum()),
            "n_fated_rows": int(fated.sum()),
        }
        for j, info in targets.heads.iterrows()
    ]
    audit.append(
        {
            "head": "current_state",
            "event": config.primary_event,
            "horizon_hours": np.nan,
            "n_eligible": int(np.isfinite(current_state).sum()),
            "n_positive": int((current_state == 1).sum()),
            "n_fated_rows": int(fated.sum()),
        }
    )

    probe_metrics.to_csv(output / "probe_metrics.csv", index=False)
    pd.DataFrame(skill_rows).to_csv(output / "forecast_skill.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output / "value_of_history.csv", index=False)
    pd.DataFrame(future_rows).to_csv(output / "value_of_future.csv", index=False)
    pd.DataFrame(shuffle_rows).to_csv(output / "time_shuffle.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(output / "rank_gate.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "onset_timing.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "target_audit.csv", index=False)
    pd.concat(training_logs, ignore_index=True).to_csv(output / "training_log.csv", index=False)
    pd.concat(score_parts, ignore_index=True).to_parquet(output / "out_of_fold_probabilities.parquet", index=False)
    fold_manifest.to_parquet(output / "fold_manifest.parquet", index=False)
    run_info = {
        "embedding_key": key,
        "marker": config.marker,
        "input_contract": input_contract,
        "n_rows": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "device": str(device),
        "frame_interval_hours": interval_hours,
        "k_frames": config.k_frames,
        "k_hours": [k * interval_hours for k in config.k_frames],
        "horizons_hours": config.horizons_hours,
        "lineage": table.stats,
        "splits": [
            {"name": s.name, "kind": s.kind, "n_train": int(s.train_mask.sum()), "n_test": int(s.test_mask.sum())}
            for s in splits
        ],
        "config": config.model_dump(),
        "forecast_reference_dir": config.forecast_reference_dir,
        "contracts": {
            "sequence": "root-to-leaf lineage path per (experiment, fov_name); division_boundary token on the first frame after a track_id change",
            "training": "self-supervised only; prediction-target frames p+k are never visible in the attention context; mixed causal/bidirectional masks",
            "prospective": "causal representations only; onset-within-k targets strictly pre-onset (build_forecast_targets); persistence_label is uninformative by construction (all pre-onset rows are annotated negative) and is listed with NaN metrics",
            "retrospective": "bidirectional representations hide frames at offsets k_frames after the anchor to match training; reported for current_state, onset timing, flicker, and value_of_future only",
            "probes": "LinearSVC (balanced) on StandardScaler, fitted after training on frozen representations; scores are decision values, threshold 0",
            "rank_gate": "centered effective rank on identical held-out rows; retention_vs_z below ~0.9 means h is a task state, not an embedding",
        },
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str) + "\n")
    (output / "_SUCCESS.json").write_text(json.dumps({"n_splits": len(splits)}) + "\n")
    return run_info


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-c", "--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--input-path",
    "input_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Per-marker embedding zarr; repeat per dataset.",
)
@click.option("--embedding-key", default=None, help="Feature matrix: X or an obsm key; defaults to the config.")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    required=True,
    help="Artifact root; the next NN_<tag> attempt directory is created inside.",
)
@click.option("--tag", default="dynaformer_ladder", show_default=True)
@click.option("--dry-run", is_flag=True)
def main(
    config: Path, input_paths: tuple[Path, ...], embedding_key: str | None, output_root: Path, tag: str, dry_run: bool
) -> None:
    """Train the DynaFormer head and evaluate the baseline ladder for one embedding identity."""
    settings = load_dynaformer_config(config)
    key = embedding_key or settings.feature_key
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "input_paths": [str(p) for p in input_paths],
                    "embedding_key": key,
                    "output_root": str(output_root),
                    "tag": tag,
                    "datasets": [d.experiment for d in settings.datasets],
                    "primary_event": settings.primary_event,
                    "k_frames": settings.k_frames,
                    "horizons_hours": settings.horizons_hours,
                    "train_steps": settings.train.steps,
                },
                indent=2,
            )
        )
        return
    output_dir = next_run_dir(output_root, tag)
    click.echo(f"output_dir: {output_dir}")
    run_dynaformer(input_paths, settings, output_dir, embedding_key=key, log=click.echo)


if __name__ == "__main__":
    main()
