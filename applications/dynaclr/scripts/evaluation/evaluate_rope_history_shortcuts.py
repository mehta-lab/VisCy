"""Intervene on trained five-frame CELL-DINO RoPE histories.

This no-retraining ablation separates chronological order, same-cell context,
and current-frame-only behavior in the acquisition-held-out infection model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import evaluate_phase_classifier_ood_generalization as base
import evaluate_phase_classifier_ood_rope_context as rope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dynaclr.evaluation.temporal.model import BidirectionalRoPETransformer

DEFAULT_PARENT = Path(
    "/home/eduardo.hirata/repos/viscy/.ed_planning/dynaclr/"
    "batch_correction/output/phase-classifier-ood-generalization-v1/"
    "experiments/rope-context1-5-seed17-v1"
)
DEFAULT_OUTPUT = Path(
    "/home/eduardo.hirata/repos/viscy/.ed_planning/dynaclr/"
    "batch_correction/output/phase-classifier-ood-generalization-v1/"
    "experiments/"
    "cell-dino-rope-context5-history-shortcut-ablation-seed17-v1"
)
INTERVENTION_SEEDS = (101, 102, 103, 104, 105)


def _load_features(fold: Path, labels: pd.DataFrame, phase_x: np.ndarray) -> np.ndarray:
    normalizer_fit = np.load(fold / "preprocessing/phase_normalizer.npz")
    normalizer = base.ControlNormalizer(
        hpi_grid=normalizer_fit["hpi_grid"],
        centers=normalizer_fit["centers"],
        scale=normalizer_fit["scale"],
    )
    normalized = base._apply_source_normalizer(phase_x, labels, normalizer)
    pca = np.load(fold / "preprocessing/phase_pca.npz")
    features = ((normalized - pca["mean"]) @ pca["components"].T).astype(np.float32)
    return features


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[BidirectionalRoPETransformer, np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BidirectionalRoPETransformer(
        in_dim=int(checkpoint["in_dim"]),
        d_model=int(checkpoint["d_model"]),
        n_heads=int(checkpoint["n_heads"]),
        num_layers=int(checkpoint["num_layers"]),
        feedforward_multiplier=int(checkpoint["feedforward_multiplier"]),
        dropout=float(checkpoint["dropout"]),
        out_dim=1,
        readout_index=int(checkpoint["readout_index"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["center"], checkpoint["scale"], checkpoint


def _repeat_current(sequences: np.ndarray) -> np.ndarray:
    return np.repeat(sequences[:, -1:, :], sequences.shape[1], axis=1)


def _physical_track_keys(labels: pd.DataFrame) -> np.ndarray:
    return (
        labels["dataset"].astype(str) + "::" + labels["fov_name"].astype(str) + "::" + labels["track_id"].astype(str)
    ).to_numpy()


def _matched_other_track_donors(labels: pd.DataFrame, *, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose another track at the same acquisition, condition, and HPI."""
    rng = np.random.default_rng(seed)
    donor = np.full(len(labels), -1, dtype=np.int64)
    track_key = _physical_track_keys(labels)
    audit: list[dict[str, object]] = []
    columns = ["dataset", "perturbation", "hours_post_perturbation"]
    for group_key, raw_rows in labels.groupby(columns, observed=True, sort=False).indices.items():
        rows = np.asarray(raw_rows, dtype=np.int64)
        order = rows[rng.permutation(len(rows))]
        matched = 0
        if len(order) > 1:
            for position, target_row in enumerate(order):
                for offset in range(1, len(order)):
                    candidate = order[(position + offset) % len(order)]
                    if track_key[candidate] != track_key[target_row]:
                        donor[target_row] = candidate
                        matched += 1
                        break
        dataset, perturbation, hpi = group_key
        audit.append(
            {
                "dataset": dataset,
                "perturbation": perturbation,
                "hours_post_perturbation": hpi,
                "n_rows": len(rows),
                "n_matched": matched,
                "n_unmatched": len(rows) - matched,
            }
        )
    valid = donor >= 0
    if np.any(track_key[donor[valid]] == track_key[np.flatnonzero(valid)]):
        raise AssertionError("A donor came from the target physical track")
    return donor, pd.DataFrame(audit)


def _borrow_history(sequences: np.ndarray, donor: np.ndarray) -> np.ndarray:
    output = _repeat_current(sequences)
    valid = donor >= 0
    output[valid, :-1] = sequences[donor[valid], :-1]
    if not np.array_equal(output[:, -1], sequences[:, -1]):
        raise AssertionError("History intervention changed the current token")
    return output


def _teacher_from_parent(
    fold: Path, labels: pd.DataFrame, test_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saved = pd.read_parquet(fold / "held_out_predictions.parquet")
    saved = saved.loc[
        (saved["model_arm"] == "rope") & (saved["inference_order"] == "ordered") & (saved["context_frames"] == 5)
    ].reset_index(drop=True)
    test_rows = np.flatnonzero(test_mask)
    wanted = base._key_index(labels.iloc[test_rows])
    positions = base._key_index(saved).get_indexer(wanted)
    if np.any(positions < 0):
        raise RuntimeError("Saved teacher rows do not cover the held-out cohort")
    aligned = saved.iloc[positions]
    target = np.full(len(labels), np.nan, dtype=np.float32)
    witness = np.full(len(labels), np.nan, dtype=np.float32)
    posterior = np.full(len(labels), np.nan, dtype=np.float32)
    target[test_rows] = aligned["teacher_target"].to_numpy(np.float32)
    witness[test_rows] = aligned["teacher_witness"].to_numpy(np.float32)
    posterior[test_rows] = aligned["teacher_positive_posterior"].to_numpy(np.float32)
    return (
        target,
        witness,
        posterior,
        aligned["probability"].to_numpy(np.float32),
    )


def _score(
    model: BidirectionalRoPETransformer,
    sequences: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probability = np.full(len(sequences), np.nan, dtype=np.float32)
    probability[rows] = rope._predict_rope(
        model,
        sequences,
        center,
        scale,
        batch_size=batch_size,
        device=device,
        rows=rows,
    )
    return probability


def _result_rows(
    *,
    args: argparse.Namespace,
    intervention: str,
    intervention_seed: int,
    held_out: str,
    labels: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    test_mask: np.ndarray,
    calibration_controls: np.ndarray,
) -> pd.DataFrame:
    metrics = base._operating_metrics(
        "infection",
        held_out,
        "cell_dino",
        5,
        args.seed,
        target,
        probability,
        labels,
        test_mask,
        probability[calibration_controls],
        args.target_control_fpr,
    )
    metrics.insert(3, "intervention", intervention)
    metrics.insert(4, "intervention_seed", intervention_seed)
    return metrics


def _prediction_rows(
    *,
    intervention: str,
    intervention_seed: int,
    held_out: str,
    labels: pd.DataFrame,
    target: np.ndarray,
    witness: np.ndarray,
    posterior: np.ndarray,
    probability: np.ndarray,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "fov_name",
        "track_id",
        "t",
        "parent_track_id",
        "perturbation",
        "hours_post_perturbation",
    ]
    rows = np.flatnonzero(test_mask)
    output = labels.iloc[rows][columns].reset_index(drop=True).copy()
    output.insert(0, "held_out_dataset", held_out)
    output.insert(1, "intervention", intervention)
    output.insert(2, "intervention_seed", intervention_seed)
    output["teacher_target"] = target[rows]
    output["teacher_witness"] = witness[rows]
    output["teacher_positive_posterior"] = posterior[rows]
    output["probability"] = probability[rows]
    return output


def run_fold(args: argparse.Namespace) -> None:
    held_out = args.held_out_dataset
    if held_out not in base.ENDPOINT_DATASETS["infection"]:
        raise ValueError(f"Unknown held-out dataset: {held_out}")
    parent_fold = args.parent_dir / "folds" / "infection" / held_out / "cell_dino"
    final = args.output_dir / "folds" / held_out
    working = final.with_name(final.name + ".in_progress")
    if final.exists() or working.exists():
        raise FileExistsError(f"Refusing to overwrite {final} or {working}")
    working.mkdir(parents=True)
    try:
        labels, _, phase_x, coverage = base._load_endpoint("infection", "cell_dino", load_marker=False)
        if phase_x is None:
            raise RuntimeError("CELL-DINO Phase3D embeddings were not loaded")
        dataset = labels["dataset"].astype(str).to_numpy()
        condition = labels["perturbation"].astype(str).to_numpy()
        source_mask = dataset != held_out
        test_mask = dataset == held_out
        features = _load_features(parent_fold, labels, phase_x)
        del phase_x
        sequences, full, distinct, divisions, generation = base._track_sequences(labels, features, 5)
        del features
        target, witness, posterior, saved_ordered = _teacher_from_parent(parent_fold, labels, test_mask)
        _, calibration_mask, split_audit = base._split_source_groups(
            labels,
            source_mask,
            calibration_fraction=args.calibration_fraction,
            seed=args.random_seed + 1_000 * args.seed,
        )
        calibration_controls = calibration_mask & (condition == "uninfected")
        scored_rows = np.flatnonzero(calibration_controls | test_mask)
        device = rope._device(args.device)
        model, center, scale, checkpoint = _load_model(parent_fold / "models/rope_frames_5.pt", device)

        metric_parts: list[pd.DataFrame] = []
        prediction_parts: list[pd.DataFrame] = []

        def evaluate(name: str, local_seed: int, data: np.ndarray) -> None:
            if not np.array_equal(data[:, -1], sequences[:, -1]):
                raise AssertionError(f"{name} changed the current token")
            probability = _score(
                model,
                data,
                center,
                scale,
                scored_rows,
                batch_size=args.batch_size,
                device=device,
            )
            metric_parts.append(
                _result_rows(
                    args=args,
                    intervention=name,
                    intervention_seed=local_seed,
                    held_out=held_out,
                    labels=labels,
                    target=target,
                    probability=probability,
                    test_mask=test_mask,
                    calibration_controls=calibration_controls,
                )
            )
            prediction_parts.append(
                _prediction_rows(
                    intervention=name,
                    intervention_seed=local_seed,
                    held_out=held_out,
                    labels=labels,
                    target=target,
                    witness=witness,
                    posterior=posterior,
                    probability=probability,
                    test_mask=test_mask,
                )
            )
            if name == "ordered":
                error = np.max(np.abs(probability[test_mask] - saved_ordered))
                if error > 1e-6:
                    raise RuntimeError(f"Ordered replay differs from saved probabilities: {error}")
                (working / "ordered_replay_max_abs_error.txt").write_text(f"{error:.12g}\n")
            print(f"{held_out}: {name}/{local_seed} complete", flush=True)

        evaluate("ordered", 0, sequences)
        repeated = _repeat_current(sequences)
        evaluate("current_repeated", 0, repeated)
        del repeated

        donor_audits: list[pd.DataFrame] = []
        for local_seed in args.intervention_seeds:
            shuffled = rope._shuffle_prior_tokens(sequences, seed=args.seed * 10_000 + local_seed)
            evaluate("same_track_shuffled", local_seed, shuffled)
            del shuffled

            donor, audit = _matched_other_track_donors(labels, seed=args.seed * 20_000 + local_seed)
            audit.insert(0, "intervention_seed", local_seed)
            donor_audits.append(audit)
            borrowed = _borrow_history(sequences, donor)
            evaluate("matched_other_track", local_seed, borrowed)
            del borrowed, donor

        pd.concat(metric_parts, ignore_index=True).to_csv(working / "fold_metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_parquet(working / "held_out_predictions.parquet", index=False)
        pd.concat(donor_audits, ignore_index=True).to_csv(working / "donor_match_audit.csv", index=False)
        split_audit.to_csv(working / "source_split_audit.csv", index=False)
        coverage.to_csv(working / "input_coverage.csv", index=False)
        pd.DataFrame(
            {
                "full_context": full,
                "distinct_context_observations": distinct,
                "context_divisions_crossed": divisions,
                "generation_depth": generation,
            }
        ).describe().to_csv(working / "sequence_context_summary.csv")
        run_info = {
            "status": "complete",
            "held_out_dataset": held_out,
            "representation": "cell_dino",
            "context_frames": 5,
            "checkpoint": str(parent_fold / "models/rope_frames_5.pt"),
            "checkpoint_best_epoch": checkpoint["best_epoch"],
            "interventions": [
                "ordered",
                "current_repeated",
                "same_track_shuffled",
                "matched_other_track",
            ],
            "intervention_seeds": list(args.intervention_seeds),
            "donor_matching": (
                "same acquisition, perturbation, and exact HPI; different "
                "physical track; unmatched rows receive repeated-current history"
            ),
            "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        (working / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
        working.rename(final)
    except Exception:
        shutil.rmtree(working, ignore_errors=True)
        raise
    print(f"Wrote {final}", flush=True)


def _fold_averages(metrics: pd.DataFrame) -> pd.DataFrame:
    raw = metrics.loc[metrics["decision_policy"] == "raw_p0p5"].copy()
    values = ["within_perturbed_auroc", "target_control_fpr"]
    return raw.groupby(["held_out_dataset", "intervention"], observed=True)[values].mean().reset_index()


def _decision_table(fold: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    baseline = fold.loc[fold["intervention"] == "ordered"].set_index("held_out_dataset")
    rows: list[dict[str, object]] = []
    for intervention in (
        "same_track_shuffled",
        "current_repeated",
        "matched_other_track",
    ):
        candidate = fold.loc[fold["intervention"] == intervention].set_index("held_out_dataset")
        common = baseline.index.intersection(candidate.index)
        auc_delta = candidate.loc[common, "within_perturbed_auroc"] - baseline.loc[common, "within_perturbed_auroc"]
        fpr_delta = candidate.loc[common, "target_control_fpr"] - baseline.loc[common, "target_control_fpr"]
        close = (auc_delta.abs() <= 0.01) & (fpr_delta.abs() <= 0.02)
        rows.append(
            {
                "intervention": intervention,
                "mean_delta_auroc_vs_ordered": auc_delta.mean(),
                "mean_delta_raw_control_fpr_vs_ordered": fpr_delta.mean(),
                "n_close_folds": int(close.sum()),
                "n_folds": len(common),
                "macro_close": bool(abs(auc_delta.mean()) <= 0.01 and abs(fpr_delta.mean()) <= 0.02),
            }
        )
    table = pd.DataFrame(rows)
    close = table.set_index("intervention")["macro_close"].to_dict()
    fold_close = table.set_index("intervention")["n_close_folds"].to_dict()
    current_shortcut = bool(close.get("current_repeated", False) and fold_close.get("current_repeated", 0) >= 4)
    same_track_unordered = bool(
        close.get("same_track_shuffled", False) and fold_close.get("same_track_shuffled", 0) >= 4
    )
    cross_track_close = bool(close.get("matched_other_track", False) and fold_close.get("matched_other_track", 0) >= 4)
    if current_shortcut:
        mechanism = "current_frame_transformer_shortcut_supported"
    elif same_track_unordered and not cross_track_close:
        mechanism = "unordered_same_cell_context_supported"
    elif same_track_unordered and cross_track_close:
        mechanism = "matched_marginal_or_acquisition_context_supported"
    else:
        mechanism = "inconclusive_or_mixed"
    decision = {
        "current_frame_shortcut_supported": current_shortcut,
        "same_track_order_insensitive": same_track_unordered,
        "matched_other_track_close": cross_track_close,
        "temporal_order_learning_supported": not same_track_unordered,
        "mechanistic_conclusion": mechanism,
        "attention_attribution_allowed": not same_track_unordered,
    }
    return table, decision


def _plot_summary(fold: pd.DataFrame, path: Path) -> None:
    order = [
        "ordered",
        "same_track_shuffled",
        "current_repeated",
        "matched_other_track",
    ]
    labels = ["ordered", "same-track\nshuffled", "current\nrepeated", "other-track\nhistory"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = [
        ("within_perturbed_auroc", "Held-out within-ZIKV AUROC"),
        ("target_control_fpr", "Held-out raw control FPR"),
    ]
    colors = ["#2b6cb0", "#7b61a8", "#dd6b20", "#319795"]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for index, (name, color) in enumerate(zip(order, colors, strict=True)):
            values = fold.loc[fold["intervention"] == name, metric].to_numpy()
            axis.scatter(np.full(len(values), index), values, color=color, s=40, zorder=3)
            if len(values):
                axis.plot(
                    [index - 0.22, index + 0.22],
                    [values.mean(), values.mean()],
                    color="black",
                    linewidth=2,
                )
        axis.set_xticks(range(len(order)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed CELL-DINO five-frame RoPE: what information produces the gain?")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    donor_parts: list[pd.DataFrame] = []
    for held_out in base.ENDPOINT_DATASETS["infection"]:
        fold = args.output_dir / "folds" / held_out
        for filename, parts, reader in (
            ("fold_metrics.csv", metric_parts, pd.read_csv),
            ("held_out_predictions.parquet", prediction_parts, pd.read_parquet),
            ("donor_match_audit.csv", donor_parts, pd.read_csv),
        ):
            path = fold / filename
            if not path.exists():
                raise FileNotFoundError(path)
            frame = reader(path)
            frame.insert(0, "held_out_dataset_fold", held_out)
            parts.append(frame)
    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    donors = pd.concat(donor_parts, ignore_index=True)
    fold = _fold_averages(metrics)
    table, decision = _decision_table(fold)
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_parquet(args.output_dir / "held_out_predictions.parquet", index=False)
    donors.to_csv(args.output_dir / "donor_match_audit.csv", index=False)
    fold.to_csv(args.output_dir / "raw_fold_averages.csv", index=False)
    table.to_csv(args.output_dir / "mechanism_deltas.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    _plot_summary(fold, figures / "history_shortcut_ablation.png")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-fold", "summarize"))
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-out-dataset")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--intervention-seeds", type=int, nargs="+", default=list(INTERVENTION_SEEDS))
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--target-control-fpr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-fold":
        if not args.held_out_dataset:
            raise ValueError("--held-out-dataset is required for run-fold")
        run_fold(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
