"""Paired bootstrap intervals and landmark counts for the Leger et al. RPE1 benchmark (``configs/cellcycle_leger``).

Reads the harness's prediction files (``Andrea_mlbi/code/results/predictions``) and the 200-step check run kept in
``runs/cellcycle_leger/check_200_steps/harness``, and scores them with the harness's own metrics: per-track L1 per
channel, multivariate ΔDTW, track r (``reproduce.report.track_r``) and the Appendix A.5 landmarks. Differences between
two runs are taken over the same tracks, with a 95% percentile bootstrap interval over tracks.

Usage
-----
Run from ``Andrea_mlbi/code`` with the harness's environment, so ``reproduce`` is importable::

    PYTHONPATH=. .venv/bin/python <app>/scripts/leger_benchmark_stats.py <out_dir>

Writes ``<out_dir>/leger_benchmark_stats.md`` and ``.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from reproduce.metrics import (
    FRAME_MINUTES,
    FUCCI1,
    FUCCI2,
    f1_per_class,
    phase_labels,
    r_squared,
    track_dtw,
    track_l1,
    transitions,
)
from reproduce.report import track_r

PREDICTIONS = Path("results/predictions")
CHECK = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle_leger/check_200_steps/harness")
ENCODERS = ("morphem", "cell-dino", "dinov3", "c-radio", "vit-untrained")
SPLITS = ("test", "test_drugged")
#: References for the paired comparisons: the August UNeXt2 dynaformer and the two released sequence checkpoints.
REFERENCES = {"August UNeXt2": "dynaformer_temporal", "bf_mamba": "bf_mamba", "bf_transformer": "bf_transformer"}
N_BOOT, SEED = 10_000, 0


def load(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read one prediction file as ``{track: (prediction, truth)}``, each ``(T, 2)`` float64."""
    data = np.load(path)
    return {str(n): (data[f"pred_{n}"].astype(np.float64), data[f"true_{n}"].astype(np.float64)) for n in data["names"]}


def per_track(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-track scores in track-name order: L1 per channel and averaged, ΔDTW and track r."""
    names = sorted(results)
    l1 = np.array([track_l1(*results[n]) for n in names])
    return {
        "names": np.array(names),
        "l1_cdt1": l1[:, FUCCI1],
        "l1_geminin": l1[:, FUCCI2],
        "l1": l1.mean(axis=1),
        "dtw": np.array([track_dtw(*results[n]) for n in names]),
        "r": np.array([track_r(*results[n]) for n in names]),
    }


def paired(a: dict[str, np.ndarray], b: dict[str, np.ndarray], key: str) -> tuple[float, float, float]:
    """Mean of ``a - b`` over the same tracks and its 95% percentile bootstrap interval over tracks.

    Raises
    ------
    ValueError
        If the two runs do not cover the same tracks.
    """
    if not np.array_equal(a["names"], b["names"]):
        raise ValueError("The two runs do not cover the same tracks.")
    difference = a[key] - b[key]
    resampled = np.random.default_rng(SEED).integers(0, len(difference), size=(N_BOOT, len(difference)))
    low, high = np.percentile(difference[resampled].mean(axis=1), [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def landmarks(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Appendix A.5 landmark scores, with how many tracks each transition was found in, predicted and true."""
    d_g1s, d_sg2, pairs_g1s, pairs_sg2, true_labels, pred_labels = [], [], [], [], [], []
    for prediction, truth in results.values():
        predicted, true = transitions(prediction), transitions(truth)
        if predicted[0] is not None and true[0] is not None:
            d_g1s.append(abs(predicted[0] - true[0]) * FRAME_MINUTES)
            pairs_g1s.append((true[0], predicted[0]))
        if predicted[1] is not None and true[1] is not None:
            d_sg2.append(abs(predicted[1] - true[1]) * FRAME_MINUTES)
            pairs_sg2.append((true[1], predicted[1]))
        true_labels.append(phase_labels(truth))
        pred_labels.append(phase_labels(prediction))
    g1s, sg2 = np.array(pairs_g1s).reshape(-1, 2), np.array(pairs_sg2).reshape(-1, 2)
    return {
        "dt_g1s": float(np.mean(d_g1s)) if d_g1s else float("nan"),
        "dt_sg2": float(np.mean(d_sg2)) if d_sg2 else float("nan"),
        "n_sg2": len(d_sg2),
        "n_tracks": len(results),
        "distinct_g1s": len(np.unique(g1s[:, 1])),
        "f1": [float(f) for f in f1_per_class(np.concatenate(true_labels), np.concatenate(pred_labels))],
        "r2_g1s": r_squared(g1s[:, 0], g1s[:, 1]),
        "r2_sg2": r_squared(sg2[:, 0], sg2[:, 1]),
    }


def comparisons(runs: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, str, float, float, float]]:
    """Every paired comparison reported in the README, as ``(name, key, mean, low, high)``."""
    rows = []
    for encoder in ENCODERS:
        ours = runs[f"dynaformer_{encoder}_temporal"]
        for reference, run in REFERENCES.items():
            rows.append((f"{encoder} temporal − {reference}", "l1", *paired(ours, runs[run], "l1")))
        single = runs[f"dynaformer_{encoder}_single_frame"]
        rows.append((f"{encoder} temporal − single-frame", "l1", *paired(ours, single, "l1")))
        rows.append((f"{encoder} single-frame − bf_mlp", "l1", *paired(single, runs["bf_mlp"], "l1")))
        if encoder != "c-radio":
            rows.append(
                (
                    f"c-radio temporal − {encoder} temporal",
                    "l1",
                    *paired(runs["dynaformer_c-radio_temporal"], ours, "l1"),
                )
            )
    c_radio = runs["dynaformer_c-radio_temporal"]
    for reference, run in REFERENCES.items():
        rows.append((f"c-radio temporal − {reference}", "dtw", *paired(c_radio, runs[run], "dtw")))
    check, final = runs["check_200_steps"], runs["dynaformer_morphem_temporal"]
    for key in ("l1", "dtw", "r"):
        rows.append(("check (200 steps) − morphem temporal (15k steps)", key, *paired(check, final, key)))
    return rows


def main() -> None:
    """Score every run on both test splits and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    names = ["bf_mlp", "bf_mamba", "bf_transformer", "dynaformer_temporal", "dynaformer_single_frame"]
    names += [f"dynaformer_{e}_{v}" for e in ENCODERS for v in ("temporal", "single_frame")]
    lines, report = [], {}
    for split in SPLITS:
        results = {name: load(PREDICTIONS / f"{name}_{split}.npz") for name in names}
        results["check_200_steps"] = load(CHECK / f"dynaformer_morphem_temporal_{split}.npz")
        runs = {name: per_track(r) for name, r in results.items()}
        rows = comparisons(runs)
        report[split] = {"comparisons": rows}
        lines += [
            f"## {split}: paired differences over the same tracks, 95% bootstrap CI",
            "",
            "| comparison | score | difference | 95% CI |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {name} | {key} | {mean:+.4f} | [{low:+.4f}, {high:+.4f}] |" for name, key, mean, low, high in rows
        ]
        lines += ["", f"## {split}: check run", ""]
        check = runs["check_200_steps"]
        lines.append(
            f"L1 Cdt1 {check['l1_cdt1'].mean():.3f}, L1 Geminin {check['l1_geminin'].mean():.3f}, "
            f"ΔDTW {check['dtw'].mean():.3f}, track r {np.nanmean(check['r']):.3f}"
        )
        lines.append("")
        if split == "test":
            marks = {name: landmarks(r) for name, r in results.items()}
            report[split]["landmarks"] = marks
            lines += [
                "## test: landmarks",
                "",
                "| run | Δt G1/S | Δt S/G2 | S/G2 found | distinct G1/S | F1 G1 / S / G2 | R² G1/S | R² S/G2 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name, m in marks.items():
                f1 = " / ".join(f"{f:.2f}" for f in m["f1"])
                lines.append(
                    f"| `{name}` | {m['dt_g1s']:.1f} | {m['dt_sg2']:.1f} | {m['n_sg2']}/{m['n_tracks']} "
                    f"| {m['distinct_g1s']} | {f1} | {m['r2_g1s']:.3f} | {m['r2_sg2']:.3f} |"
                )
            lines.append("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leger_benchmark_stats.md").write_text("\n".join(lines))
    (args.out_dir / "leger_benchmark_stats.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
