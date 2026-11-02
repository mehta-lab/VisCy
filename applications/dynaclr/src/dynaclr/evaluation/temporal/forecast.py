"""Pre-onset temporal event forecasting from one embedding identity.

The evaluator joins representation-independent human state annotations to one
embedding model/checkpoint, builds history-only windows, and emits out-of-fold
probabilities that an event will begin within each configured future horizon.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import click
import numpy as np
import pandas as pd
import torch
import zarr
from anndata.io import read_elem
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from dynaclr.evaluation.temporal.config_forecast import (
    ForecastEvent,
    TemporalForecastConfig,
    load_temporal_forecast_config,
)
from dynaclr.evaluation.temporal.model import (
    BidirectionalRoPETransformer,
    CausalTCN,
)
from dynaclr.evaluation.temporal.sequences import build_causal_sequences


@dataclass(frozen=True)
class ForecastTargets:
    """Row-aligned multi-event, multi-horizon targets."""

    values: np.ndarray
    lead_hours: np.ndarray
    heads: pd.DataFrame


@dataclass
class _TemporalFit:
    model: torch.nn.Module
    center: np.ndarray
    scale: np.ndarray
    active_heads: np.ndarray
    device: torch.device

    def predict_proba(
        self,
        sequences: np.ndarray,
        rows: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        output = np.full(
            (len(rows), len(self.active_heads)),
            np.nan,
            dtype=np.float32,
        )
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                stop = min(start + batch_size, len(rows))
                batch = (sequences[rows[start:stop]] - self.center) / self.scale
                logits = self.model(torch.from_numpy(batch.astype(np.float32)).to(self.device))
                if logits.ndim == 1:
                    logits = logits[:, None]
                output[start:stop] = torch.sigmoid(logits).cpu().numpy()
        output[:, ~self.active_heads] = np.nan
        return output


def _canonical_tokens(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    """Stable string keys across zarr categorical/integer dtype differences."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing key columns: {sorted(missing)}")
    values = frame[str(columns[0])].astype("string").fillna("<NA>")
    for column in columns[1:]:
        values = values.str.cat(
            frame[str(column)].astype("string").fillna("<NA>"),
            sep="\x1f",
        )
    return values.to_numpy(dtype=str)


def _read_obs(path: Path) -> pd.DataFrame:
    root = zarr.open_group(str(path), mode="r")
    return read_elem(root["obs"]).reset_index(drop=True)


def _feature_matrix(adata: ad.AnnData, key: str) -> np.ndarray:
    if key == "X":
        values = adata.X
    else:
        if key not in adata.obsm:
            raise KeyError(f"{key!r} is missing from {adata.filename or 'embedding store'}")
        values = adata.obsm[key]
    if hasattr(values, "toarray"):
        values = values.toarray()
    output = np.asarray(values, dtype=np.float32)
    if output.ndim != 2:
        raise ValueError(f"Embedding feature matrix must be 2D, got {output.shape}")
    return output


def _collapse_annotations(
    path: Path,
    key_columns: Sequence[str],
    event_columns: Sequence[str],
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = set(key_columns) - set(raw.columns)
    if missing:
        raise KeyError(f"{path}: missing annotation keys {sorted(missing)}")
    pieces = [raw[list(key_columns)].drop_duplicates()]
    for column in event_columns:
        if column not in raw:
            collapsed = raw[list(key_columns)].drop_duplicates()
            collapsed[column] = pd.NA
            pieces.append(collapsed)
            continue
        work = raw[[*key_columns, column]].dropna(subset=[column]).copy()
        work[column] = work[column].astype("string").str.strip().str.lower()
        grouped = work.groupby(
            list(key_columns),
            observed=True,
            sort=False,
            dropna=False,
        )[column].agg(lambda values: tuple(sorted(set(values))))
        collapsed = grouped.map(lambda values: values[0] if len(values) == 1 else pd.NA).rename(column).reset_index()
        pieces.append(collapsed)
    output = pieces[0]
    for piece in pieces[1:]:
        output = output.merge(
            piece,
            on=list(key_columns),
            how="left",
            validate="one_to_one",
        )
    return output


def _cohort_intersection(
    config: TemporalForecastConfig,
) -> set[str] | None:
    if not config.cohort_intersection_paths:
        return None
    experiments = {item.experiment for item in config.datasets}
    keys_by_experiment: dict[str, list[set[str]]] = {}
    for raw_path in config.cohort_intersection_paths:
        path = Path(raw_path)
        if path.stem != config.marker:
            continue
        obs = _read_obs(path)
        if "experiment" not in obs:
            raise KeyError(f"{path}: missing experiment in obs")
        for experiment, part in obs.groupby(
            "experiment",
            observed=True,
            sort=False,
        ):
            experiment = str(experiment)
            if experiment not in experiments:
                continue
            if config.blank_column and config.blank_column in part:
                part = part[~part[config.blank_column].fillna(False).astype(bool)]
            keys_by_experiment.setdefault(experiment, []).append(set(_canonical_tokens(part, config.key_columns)))
    if not keys_by_experiment:
        raise ValueError(f"cohort_intersection_paths contained no stores matching marker={config.marker!r}")
    missing = sorted(experiments - set(keys_by_experiment))
    if missing:
        raise ValueError(f"No cohort intersection stores were supplied for {missing}")
    cohort: set[str] = set()
    for experiment, key_sets in keys_by_experiment.items():
        common = set.intersection(*key_sets)
        if not common:
            raise ValueError(f"The configured cohort intersection is empty for {experiment}")
        cohort.update(common)
    return cohort


def load_forecast_data(
    input_paths: Sequence[str | Path],
    config: TemporalForecastConfig,
    embedding_key: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load one marker across configured datasets and join human annotations."""

    specs = {item.experiment: item for item in config.datasets}
    event_columns = [event.column for event in config.events]
    annotation_keys = [column for column in config.key_columns if column != "experiment"]
    cohort = _cohort_intersection(config)
    label_parts: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    contracts: list[dict[str, Any]] = []

    for raw_path in input_paths:
        path = Path(raw_path)
        if path.stem != config.marker:
            continue
        obs_preview = _read_obs(path)
        if "experiment" not in obs_preview:
            raise KeyError(f"{path}: missing experiment in obs")
        experiments = obs_preview["experiment"].astype(str).unique()
        if len(experiments) != 1 or experiments[0] not in specs:
            continue
        experiment = str(experiments[0])
        spec = specs[experiment]

        adata = ad.read_zarr(path)
        obs = adata.obs.reset_index(drop=True).copy()
        features = _feature_matrix(adata, embedding_key)
        if len(obs) != len(features):
            raise ValueError(f"{path}: obs/features row mismatch")
        keep = np.ones(len(obs), dtype=bool)
        if config.blank_column and config.blank_column in obs:
            keep &= ~obs[config.blank_column].fillna(False).to_numpy(dtype=bool)
        if cohort is not None:
            keep &= np.isin(
                _canonical_tokens(obs, config.key_columns),
                np.asarray(sorted(cohort), dtype=str),
            )
        obs = obs.loc[keep].reset_index(drop=True)
        features = features[keep]

        annotations = _collapse_annotations(
            Path(spec.annotation),
            annotation_keys,
            event_columns,
        )
        obs["_row_order"] = np.arange(len(obs))
        obs = obs.merge(
            annotations,
            on=annotation_keys,
            how="left",
            sort=False,
            validate="one_to_one",
        ).sort_values("_row_order", kind="stable")
        obs = obs.drop(columns="_row_order").reset_index(drop=True)
        if len(obs) != len(features):
            raise RuntimeError(f"{experiment}: annotation join changed row count")

        label_parts.append(obs)
        feature_parts.append(features)
        contracts.append(
            {
                "experiment": experiment,
                "embedding_path": str(path),
                "annotation_path": str(spec.annotation),
                "n_rows": int(len(obs)),
                "feature_dim": int(features.shape[1]),
            }
        )
        del adata, obs_preview, features
        gc.collect()

    found = {str(part["experiment"].iloc[0]) for part in label_parts}
    missing = sorted(set(specs) - found)
    if missing:
        raise FileNotFoundError(f"No {config.marker} embedding store found for {missing}")
    labels = pd.concat(label_parts, ignore_index=True)
    features = np.concatenate(feature_parts, axis=0).astype(np.float32)
    if labels.duplicated(list(config.key_columns)).any():
        raise ValueError("Duplicate forecast keys after pooling datasets")
    return (
        labels,
        features,
        {
            "stores": contracts,
            "cohort_intersection_size": (None if cohort is None else len(cohort)),
        },
    )


def build_forecast_targets(
    labels: pd.DataFrame,
    events: Sequence[ForecastEvent],
    horizons_hours: Sequence[float],
    *,
    group_columns: Sequence[str] = ("experiment", "fov_name", "track_id"),
    time_column: str = "hours_post_perturbation",
) -> ForecastTargets:
    """Build strict pre-onset targets with conservative right-censoring.

    Fated tracks require an observed negative state before the first positive
    state. Rows at or after onset are excluded. Event-free tracks contribute a
    negative target only when a negative annotation is observed at least one
    full forecast horizon later.
    """

    missing = set(group_columns) | {time_column}
    missing -= set(labels.columns)
    if missing:
        raise KeyError(f"Missing temporal columns: {sorted(missing)}")
    heads = pd.DataFrame(
        [
            {
                "head": f"{event.name}__within_{horizon:g}h",
                "event": event.name,
                "horizon_hours": float(horizon),
            }
            for event in events
            for horizon in horizons_hours
        ]
    )
    values = np.full((len(labels), len(heads)), np.nan, dtype=np.float32)
    lead = np.full_like(values, np.nan)
    times = labels[time_column].to_numpy(dtype=float)
    groups = labels.groupby(
        list(group_columns),
        observed=True,
        sort=False,
    ).indices

    head_offset = 0
    for event in events:
        if event.column not in labels:
            raise KeyError(f"Missing event column: {event.column!r}")
        states = labels[event.column].astype("string").str.strip().str.lower().fillna("<NA>").to_numpy()
        negative = event.negative.strip().lower()
        positive = event.positive.strip().lower()
        for raw_rows in groups.values():
            rows = np.asarray(raw_rows, dtype=int)
            rows = rows[np.argsort(times[rows], kind="stable")]
            local_states = states[rows]
            local_times = times[rows]
            negative_times = local_times[local_states == negative]
            positive_times = local_times[local_states == positive]
            onset = None
            if len(negative_times) and len(positive_times):
                eligible_onsets = positive_times[positive_times > np.min(negative_times)]
                if len(eligible_onsets):
                    onset = float(np.min(eligible_onsets))

            for horizon_index, horizon in enumerate(horizons_hours):
                head = head_offset + horizon_index
                if onset is not None:
                    prediction_rows = rows[local_times < onset]
                    event_lead = onset - times[prediction_rows]
                    values[prediction_rows, head] = (event_lead <= horizon).astype(np.float32)
                    lead[prediction_rows, head] = event_lead
                elif len(negative_times):
                    censor_time = float(np.max(negative_times))
                    prediction_rows = rows[local_times + float(horizon) <= censor_time]
                    values[prediction_rows, head] = 0.0
        head_offset += len(horizons_hours)
    return ForecastTargets(values=values, lead_hours=lead, heads=heads)


def assign_fov_folds(
    labels: pd.DataFrame,
    n_folds: int,
    seed: int,
    *,
    split_group_columns: Sequence[str] = ("experiment", "fov_name"),
) -> np.ndarray:
    """Assign deterministic dataset-stratified FOV folds."""

    missing = set(split_group_columns) - set(labels.columns)
    if missing:
        raise KeyError(f"Missing split columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    folds = np.full(len(labels), -1, dtype=np.int16)
    experiment = labels["experiment"].astype(str).to_numpy()
    tokens = _canonical_tokens(labels, split_group_columns)
    for name in sorted(np.unique(experiment)):
        groups = np.unique(tokens[experiment == name])
        if len(groups) < n_folds:
            raise ValueError(f"{name}: {len(groups)} split groups cannot fill {n_folds} folds")
        rng.shuffle(groups)
        mapping = {group: index % n_folds for index, group in enumerate(groups)}
        local = experiment == name
        folds[local] = [mapping[token] for token in tokens[local]]
    return folds


def _make_model(
    name: str,
    in_dim: int,
    out_dim: int,
    config: TemporalForecastConfig,
) -> torch.nn.Module:
    settings = config.models
    if name == "tcn":
        return CausalTCN(
            in_dim=in_dim,
            hidden=settings.tcn_hidden,
            dilations=settings.tcn_dilations,
            out_dim=out_dim,
            dropout=settings.dropout,
        )
    if name == "rope_transformer":
        # The entire supplied window is observed history ending at prediction
        # time. Bidirectional attention within that history cannot see future
        # observations because no future token is present.
        return BidirectionalRoPETransformer(
            in_dim=in_dim,
            d_model=settings.rope_d_model,
            n_heads=settings.rope_heads,
            num_layers=settings.rope_layers,
            feedforward_multiplier=settings.rope_ff_multiplier,
            dropout=settings.dropout,
            out_dim=out_dim,
            readout_index=-1,
        )
    raise ValueError(f"Unknown temporal model: {name}")


def _fit_temporal_model(
    name: str,
    sequences: np.ndarray,
    targets: np.ndarray,
    train_rows_by_head: Sequence[np.ndarray],
    config: TemporalForecastConfig,
    seed: int,
) -> _TemporalFit:
    settings = config.models
    n_heads = targets.shape[1]
    active = np.zeros(n_heads, dtype=bool)
    selected: list[np.ndarray] = []
    for head, raw_rows in enumerate(train_rows_by_head):
        rows = np.asarray(raw_rows, dtype=int)
        classes = np.unique(targets[rows, head]) if len(rows) else np.array([])
        if len(classes) == 2:
            active[head] = True
            selected.append(rows)
        else:
            selected.append(np.array([], dtype=int))
    if not active.any():
        raise ValueError("No forecast head has both classes in the training fold")
    union = np.unique(np.concatenate([selected[i] for i in np.flatnonzero(active)]))
    flat = sequences[union].reshape(-1, sequences.shape[-1])
    center = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        flat.std(axis=0, dtype=np.float64).astype(np.float32),
        1e-6,
    )

    train_targets = targets[union].copy()
    train_targets[~np.isfinite(train_targets)] = 0.0
    mask = np.zeros((len(union), n_heads), dtype=np.float32)
    for head in np.flatnonzero(active):
        mask[:, head] = np.isin(union, selected[head])
    positive = ((train_targets == 1) * mask).sum(axis=0)
    negative = ((train_targets == 0) * mask).sum(axis=0)
    positive_weight = np.ones(n_heads, dtype=np.float32)
    positive_weight[active] = negative[active] / positive[active]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(
        name,
        sequences.shape[-1],
        n_heads,
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.from_numpy(positive_weight).to(device),
    )
    x = ((sequences[union] - center) / scale).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(train_targets),
            torch.from_numpy(mask),
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for _ in range(settings.epochs):
        for batch_x, batch_y, batch_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            if logits.ndim == 1:
                logits = logits[:, None]
            losses = loss_fn(logits, batch_y.to(device))
            active_mask = batch_mask.to(device)
            per_head = (losses * active_mask).sum(dim=0) / active_mask.sum(dim=0).clamp_min(1)
            per_head[torch.from_numpy(active).to(device)].mean().backward()
            optimizer.step()
    return _TemporalFit(
        model=model,
        center=center,
        scale=scale,
        active_heads=active,
        device=device,
    )


def _metric_row(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "average_precision": np.nan,
            "balanced_accuracy": np.nan,
            "brier": np.nan,
        }
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, probability >= 0.5)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def run_temporal_forecast(
    input_paths: Sequence[str | Path],
    config: TemporalForecastConfig,
    output_dir: str | Path,
    *,
    embedding_key: str | None = None,
) -> dict[str, Any]:
    """Run cross-validated forecasting and write per-cell probabilities."""

    key = embedding_key or config.feature_key
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels, features, input_contract = load_forecast_data(
        input_paths,
        config,
        key,
    )
    target = build_forecast_targets(
        labels,
        config.events,
        config.horizons_hours,
        group_columns=config.group_columns,
        time_column=config.time_column,
    )
    sequence_info = build_causal_sequences(
        labels,
        features,
        window=config.window_observations,
        group_columns=config.group_columns,
        time_column="t",
    )
    eligible_context = sequence_info.full_context if config.require_full_context else np.ones(len(labels), dtype=bool)
    folds = assign_fov_folds(
        labels,
        config.n_folds,
        config.random_seed,
        split_group_columns=config.split_group_columns,
    )

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    output_columns = [
        column
        for column in (
            *config.key_columns,
            "perturbation",
            config.time_column,
        )
        if column in labels
    ]
    for head, head_info in target.heads.iterrows():
        finite = np.isfinite(target.values[:, head]) & eligible_context
        audit_rows.append(
            {
                **head_info.to_dict(),
                "n_eligible": int(finite.sum()),
                "n_positive": int((target.values[finite, head] == 1).sum()),
                "n_negative": int((target.values[finite, head] == 0).sum()),
            }
        )

    for fold in range(config.n_folds):
        train_rows_by_head = [
            np.flatnonzero((folds != fold) & eligible_context & np.isfinite(target.values[:, head]))
            for head in range(target.values.shape[1])
        ]
        test_union = np.flatnonzero((folds == fold) & eligible_context & np.isfinite(target.values).any(axis=1))
        for model_index, model_name in enumerate(config.models.names):
            fit = _fit_temporal_model(
                model_name,
                sequence_info.sequences,
                target.values,
                train_rows_by_head,
                config,
                config.random_seed + fold * 100 + model_index,
            )
            probability = fit.predict_proba(
                sequence_info.sequences,
                test_union,
                config.models.batch_size,
            )
            local_lookup = np.full(len(labels), -1, dtype=int)
            local_lookup[test_union] = np.arange(len(test_union))
            for head, head_info in target.heads.iterrows():
                rows = np.flatnonzero((folds == fold) & eligible_context & np.isfinite(target.values[:, head]))
                if not len(rows) or not fit.active_heads[head]:
                    continue
                p = probability[local_lookup[rows], head]
                y = target.values[rows, head]
                metrics = _metric_row(y, p)
                metric_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        **head_info.to_dict(),
                        "n_test": int(len(rows)),
                        "n_positive": int((y == 1).sum()),
                        "n_negative": int((y == 0).sum()),
                        **metrics,
                    }
                )
                part = labels.loc[rows, output_columns].copy()
                part["fold"] = fold
                part["model"] = model_name
                part["head"] = head_info["head"]
                part["event"] = head_info["event"]
                part["horizon_hours"] = head_info["horizon_hours"]
                part["target"] = y
                part["probability"] = p
                part["lead_hours"] = target.lead_hours[rows, head]
                prediction_parts.append(part)
            del fit, probability
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No evaluable model/fold/event/horizon combinations")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary = (
        metrics.groupby(
            ["model", "event", "horizon_hours"],
            observed=True,
            sort=False,
        )
        .agg(
            n_folds=("fold", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )
    fold_manifest = labels[list(config.key_columns)].copy()
    fold_manifest["fold"] = folds
    fold_manifest["full_context"] = sequence_info.full_context
    fold_manifest["distinct_observations"] = sequence_info.distinct_observations

    predictions.to_parquet(output / "out_of_fold_probabilities.parquet", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "target_audit.csv", index=False)
    fold_manifest.to_parquet(output / "fold_manifest.parquet", index=False)
    run_info = {
        "embedding_key": key,
        "marker": config.marker,
        "input_contract": input_contract,
        "n_rows": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "window_observations": config.window_observations,
        "horizons_hours": config.horizons_hours,
        "n_folds": config.n_folds,
        "events": [event.model_dump() for event in config.events],
        "models": config.models.model_dump(),
        "split_contract": (
            "deterministic dataset-stratified FOV folds; all tracks and lineages within a FOV remain in one fold"
        ),
        "target_contract": (
            "strictly before first positive onset after an observed negative; "
            "event-free negatives require a full observed future horizon"
        ),
        "probability_contract": (
            "out-of-fold sigmoid risk for event onset within the named horizon; not post-hoc calibrated"
        ),
        "rope_contract": (
            "bidirectional RoPE attention only within the supplied "
            "history window; the window contains no future observations"
        ),
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--input-path",
    "input_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Per-marker embedding zarr; repeat for multiple datasets/markers.",
)
@click.option(
    "--embedding-key",
    default=None,
    help="Feature matrix: X or an obsm key; defaults to the config.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--dry-run", is_flag=True)
def main(
    config: Path,
    input_paths: tuple[Path, ...],
    embedding_key: str | None,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Evaluate pre-onset event forecasting for one embedding identity."""

    settings = load_temporal_forecast_config(config)
    key = embedding_key or settings.feature_key
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "input_paths": [str(path) for path in input_paths],
                    "embedding_key": key,
                    "output_dir": str(output_dir),
                    "datasets": [item.experiment for item in settings.datasets],
                    "marker": settings.marker,
                    "heads": [
                        f"{event.name}__within_{horizon:g}h"
                        for event in settings.events
                        for horizon in settings.horizons_hours
                    ],
                    "models": settings.models.names,
                },
                indent=2,
            )
        )
        return
    run_temporal_forecast(
        input_paths,
        settings,
        output_dir,
        embedding_key=key,
    )


if __name__ == "__main__":
    main()
