"""Plot supporting diagnostics for ZIKV direct-pAL temporal alignment.

This is the ZIKV counterpart of plot_denv_direct_pal_alignment_support.py.
The shared 07_01 acquisition contains all three organelle markers, so alignment
QC curves are kept separate by dataset-marker pair. The persistent-onset audit
uses the replicate-augmented scope (07_01 plus the marker-specific replicate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plot_denv_direct_pal_alignment_support as support

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)

SERIES_ORDER = (
    "07_01_SEC61B_TOMM20_G3BP1::SEC61B",
    "03_24_SEC61B::SEC61B",
    "07_01_SEC61B_TOMM20_G3BP1::TOMM20",
    "04_10_TOMM20::TOMM20",
    "07_01_SEC61B_TOMM20_G3BP1::G3BP1",
    "04_08_G3BP1::G3BP1",
)
SERIES_LABELS = {
    "07_01_SEC61B_TOMM20_G3BP1::SEC61B": "SEC61B · 07_01",
    "03_24_SEC61B::SEC61B": "SEC61B · 03_24",
    "07_01_SEC61B_TOMM20_G3BP1::TOMM20": "TOMM20 · 07_01",
    "04_10_TOMM20::TOMM20": "TOMM20 · 04_10",
    "07_01_SEC61B_TOMM20_G3BP1::G3BP1": "G3BP1 · 07_01",
    "04_08_G3BP1::G3BP1": "G3BP1 · 04_08",
}
SERIES_COLORS = {
    "07_01_SEC61B_TOMM20_G3BP1::SEC61B": "#56B4E9",
    "03_24_SEC61B::SEC61B": "#0072B2",
    "07_01_SEC61B_TOMM20_G3BP1::TOMM20": "#65C99B",
    "04_10_TOMM20::TOMM20": "#009E73",
    "07_01_SEC61B_TOMM20_G3BP1::G3BP1": "#E4A4CF",
    "04_08_G3BP1::G3BP1": "#CC79A7",
}


def _load_lineage_time_states(input_path: Path) -> pd.DataFrame:
    parts = []
    for marker in support.MARKER_ORDER:
        frame = pd.read_parquet(
            input_path / f"{marker.lower()}_cell_manifest.parquet",
            columns=[
                "dataset",
                "marker",
                "fov_name",
                "lineage_id",
                "t",
                "hours_post_perturbation",
                "gmm_positive_posterior",
                "pal_onset_hpi",
                "tau",
                "perturbation",
            ],
        )
        frame = frame.loc[frame["perturbation"].astype(str).eq("ZIKV") & frame["pal_onset_hpi"].notna()]
        states = (
            frame.groupby(
                ["dataset", "marker", "fov_name", "lineage_id", "t"],
                sort=False,
                observed=True,
            )
            .agg(
                hours_post_perturbation=("hours_post_perturbation", "median"),
                gmm_positive_posterior=("gmm_positive_posterior", "median"),
                pal_onset_hpi=("pal_onset_hpi", "first"),
                tau=("tau", "median"),
            )
            .reset_index()
        )
        states["source_dataset"] = states["dataset"].astype(str)
        states["dataset"] = states["source_dataset"] + "::" + states["marker"]
        states["lineage_uid"] = (
            states["dataset"] + "::" + states["fov_name"].astype(str) + "::" + states["lineage_id"].astype(str)
        )
        parts.append(states)
    result = pd.concat(parts, ignore_index=True)
    key = ["dataset", "fov_name", "lineage_id", "t"]
    if result.duplicated(key).any():
        raise ValueError("Duplicate lineage-time states after branch aggregation")
    missing = sorted(set(SERIES_ORDER) - set(result["dataset"].unique()))
    if missing:
        raise ValueError(f"Missing expected dataset-marker series: {missing}")
    return result


def run(args: argparse.Namespace) -> None:
    input_path = args.input
    states = _load_lineage_time_states(input_path)
    curves = pd.concat(
        [
            support._posterior_curve(states, "hpi"),
            support._posterior_curve(states, "tau"),
        ],
        ignore_index=True,
    )
    onsets = support._onset_table(states)
    lineage_support = support._support_curve(states)

    mmd_curves = pd.read_csv(input_path / "mmd_curves.csv")
    metrics = pd.read_csv(input_path / "alignment_curve_metrics.csv")
    mmd_curves = mmd_curves.loc[mmd_curves["scope"].eq(args.scope)].copy()
    metrics = metrics.loc[metrics["scope"].eq(args.scope)].copy()
    expected = {(marker, "tau") for marker in support.MARKER_ORDER}
    observed = set(zip(metrics["marker"], metrics["axis"], strict=False))
    if not expected.issubset(observed):
        raise ValueError(f"Scope {args.scope!r} lacks one or more marker tau curves")

    curves.to_csv(input_path / "direct_pal_alignment_qc_curves.csv", index=False)
    onsets.to_csv(input_path / "direct_pal_alignment_qc_onsets.csv", index=False)
    lineage_support.to_csv(input_path / "direct_pal_alignment_qc_support.csv", index=False)

    support.DATASET_ORDER = SERIES_ORDER
    support.DATASET_LABELS = SERIES_LABELS
    support.DATASET_COLORS = SERIES_COLORS
    support._plot_alignment_qc(
        curves,
        onsets,
        lineage_support,
        input_path / "direct_pal_alignment_qc.png",
        perturbation="ZIKV",
    )
    support._plot_persistent_onsets(
        mmd_curves,
        metrics,
        input_path / "organelle_persistent_onset_audit.png",
        args.persistent_bins,
    )

    tau_zero = (
        curves.loc[curves["axis"].eq("tau")]
        .assign(distance=lambda x: x["axis_value"].abs())
        .sort_values(["dataset", "distance"])
        .groupby("dataset", observed=True)
        .first()["posterior_median"]
        .to_dict()
    )
    onset_rows = metrics.loc[
        metrics["axis"].eq("tau"),
        ["marker", "persistent_baseline_onset"],
    ]
    run_info = {
        "input": str(input_path),
        "candidate_population": "ZIKV lineages with a sustained direct-pAL onset",
        "branch_aggregation": "median posterior per lineage and frame",
        "posterior_summary": "median and interquartile range",
        "alignment_qc_series": SERIES_LABELS,
        "median_posterior_nearest_tau_zero": tau_zero,
        "persistent_onset": {
            "scope": args.scope,
            "baseline_tau_hours": [-10.0, -6.0],
            "rule": "lower stability bound exceeds baseline ceiling",
            "consecutive_bins": args.persistent_bins,
            "bin_width_hours": 0.5,
            "calls_hours": dict(
                zip(
                    onset_rows["marker"],
                    onset_rows["persistent_baseline_onset"],
                    strict=True,
                )
            ),
        },
        "interpretation_limit": (
            "Aligned pAL sharpening is expected by construction and is an alignment QC, "
            "not independent evidence of biological prediction"
        ),
        "outputs": [
            "direct_pal_alignment_qc.png",
            "organelle_persistent_onset_audit.png",
        ],
    }
    (input_path / "support_figure_run_info.json").write_text(json.dumps(run_info, indent=2))
    print(input_path / "direct_pal_alignment_qc.png")
    print(input_path / "organelle_persistent_onset_audit.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--scope", default="replicate_augmented")
    parser.add_argument("--persistent-bins", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
