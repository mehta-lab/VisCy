"""Render ZIKV lineage-family crops every three hours around pAL alignment.

Each row is one parent-linked ZIKV lineage family observed at every requested
pAL-relative time from tau=-9 h through tau=+9 h. Lineages are sampled within
each configured marker dataset. A single descendant is chosen deterministically
at branching times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from iohub import open_ome_zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)
COMMON_DATASET = "07_01_SEC61B_TOMM20_G3BP1"
MARKERS = ("SEC61B", "TOMM20", "G3BP1")
TIMEPOINTS = tuple((f"tau_{tau:+03d}", float(tau)) for tau in range(-9, 10, 3))
REPLICATES = {
    "SEC61B": "03_24_SEC61B",
    "TOMM20": "04_10_TOMM20",
    "G3BP1": "04_08_G3BP1",
}
DATASETS_BY_MARKER = {marker: (COMMON_DATASET, REPLICATES[marker]) for marker in MARKERS}
SAMPLES_BY_MARKER_DATASET: dict[tuple[str, str], int] = {}
DATASET_LABELS = {
    COMMON_DATASET: "07_01",
    "03_24_SEC61B": "03_24",
    "04_10_TOMM20": "04_10",
    "04_08_G3BP1": "04_08",
}
RAW_PATHS = {
    COMMON_DATASET: Path(
        "/hpc/projects/organelle_phenotyping/datasets/"
        "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/"
        "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV.zarr"
    ),
    "03_24_SEC61B": Path(
        "/hpc/projects/organelle_phenotyping/datasets/2026_03_24_A549_SEC61_ZIKV/2026_03_24_A549_SEC61_ZIKV.zarr"
    ),
    "04_10_TOMM20": Path(
        "/hpc/projects/organelle_phenotyping/datasets/2026_04_10_A549_TOMM20_ZIKV/2026_04_10_A549_TOMM20_ZIKV.zarr"
    ),
    "04_08_G3BP1": Path(
        "/hpc/projects/organelle_phenotyping/datasets/2026_04_08_A549_G3BP1_ZIKV/2026_04_08_A549_G3BP1_ZIKV.zarr"
    ),
}
ORGANELLE_CHANNEL = "raw GFP EX488 EM525-45"
PAL_CHANNEL = "raw mCherry EX561 EM600-37"
PHASE_CHANNEL = "Phase3D"
PERTURBATION = "ZIKV"
PAL_DISPLAY_NAME = "pAL17"
OUTPUT_PREFIX = "zikv"
CURATED_EXCLUSIONS = {
    ("SEC61B", "03_24_SEC61B", "B/3/fov5", 63),
    ("SEC61B", "03_24_SEC61B", "A/3/fov0", 15),
    ("TOMM20", COMMON_DATASET, "B/3/001000", 80),
    ("TOMM20", "04_10_TOMM20", "A/4/0008", 128),
    ("G3BP1", "04_08_G3BP1", "C/5/0000", 62),
}


def _crop_bounds(center: float, size: int, maximum: int) -> tuple[int, int]:
    start = max(0, min(int(round(center)) - size // 2, maximum - size))
    return start, start + size


def _focus_z(position, time: int, phase_channel: int) -> int:
    try:
        focus = dict(position.zattrs)["focus_slice"]["Phase3D"]
        per_timepoint = focus.get("per_timepoint", {})
        value = int(per_timepoint.get(str(time), 0))
        if value > 0:
            return value
        return int(round(focus["fov_statistics"]["z_focus_mean"]))
    except (KeyError, TypeError, ValueError):
        phase = np.asarray(position["0"][time, phase_channel], np.float32)
        score = np.nanpercentile(phase, 99, axis=(1, 2)) - np.nanmedian(phase, axis=(1, 2))
        return int(np.nanargmax(score))


def _target_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"marker": marker, "stage": stage, "target_tau": tau} for marker in MARKERS for stage, tau in TIMEPOINTS]
    )


def _prefer_distinct_fovs(
    candidates: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    shuffled = candidates.iloc[rng.permutation(len(candidates))].copy()
    distinct = shuffled.drop_duplicates("fov_name")
    chosen = distinct.head(count)
    if len(chosen) < count:
        remaining = shuffled.loc[~shuffled.index.isin(chosen.index)]
        chosen = pd.concat([chosen, remaining.head(count - len(chosen))])
    if len(chosen) != count:
        raise ValueError(f"Need {count} complete lineages, found {len(chosen)}")
    return chosen


def _choose_descendant(
    candidates: pd.DataFrame,
    previous_xy: np.ndarray | None,
) -> pd.Series:
    ordered = candidates.sort_values(["lineage_depth", "track_id", "id"], kind="stable")
    if previous_xy is None or len(ordered) == 1:
        return ordered.iloc[0]
    xy = ordered[["x", "y"]].to_numpy(float)
    distance = np.square(xy - previous_xy[None, :]).sum(axis=1)
    return ordered.iloc[int(np.argmin(distance))]


def _apply_curated_replacements(
    selected: pd.DataFrame,
    complete: pd.DataFrame,
    marker: str,
    dataset: str,
) -> pd.DataFrame:
    """Replace rejected examples without changing other seeded choices."""
    result = selected.copy().reset_index(drop=True)
    for index, row in result.iterrows():
        key = (marker, dataset, str(row["fov_name"]), int(row["lineage_id"]))
        if key not in CURATED_EXCLUSIONS:
            continue
        used = set(map(tuple, result[["fov_name", "lineage_id"]].to_numpy()))
        keep = []
        for candidate in complete.itertuples(index=False):
            candidate_key = (str(candidate.fov_name), candidate.lineage_id)
            full_key = (
                marker,
                dataset,
                str(candidate.fov_name),
                int(candidate.lineage_id),
            )
            keep.append(candidate_key not in used and full_key not in CURATED_EXCLUSIONS)
        alternatives = complete.loc[keep].copy()
        unused_fov = alternatives.loc[~alternatives["fov_name"].isin(result["fov_name"])]
        pool = unused_fov if not unused_fov.empty else alternatives
        if pool.empty:
            raise ValueError(f"No replacement lineage available for {key}")
        replacement = pool.sort_values(["fov_name", "lineage_id"], kind="stable").iloc[0]
        result.loc[index, ["fov_name", "lineage_id"]] = replacement[["fov_name", "lineage_id"]].to_numpy()
    return result


def _select_rows(
    input_path: Path,
    targets: pd.DataFrame,
    samples_per_dataset: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    sample_id = 0
    for marker in MARKERS:
        columns = [
            "dataset",
            "fov_name",
            "lineage_id",
            "lineage_depth",
            "track_id",
            "id",
            "t",
            "x",
            "y",
            "hours_post_perturbation",
            "pal_onset_hpi",
            "tau",
            "gmm_positive_posterior",
            "perturbation",
            "is_blank",
        ]
        frame = pd.read_parquet(
            input_path / f"{marker.lower()}_cell_manifest.parquet",
            columns=columns,
        )
        frame = frame.loc[
            frame["perturbation"].astype(str).eq(PERTURBATION) & ~frame["is_blank"].astype(bool) & frame["tau"].notna()
        ].copy()
        frame["tau_bin"] = np.round(frame["tau"].to_numpy(float) * 2.0) / 2.0
        marker_targets = targets.loc[targets["marker"].eq(marker)].copy()
        target_values = marker_targets["target_tau"].to_numpy(float)
        for dataset in DATASETS_BY_MARKER[marker]:
            local = frame.loc[frame["dataset"].eq(dataset)]
            complete: set[tuple[object, object]] | None = None
            for target in target_values:
                keys = set(
                    map(
                        tuple,
                        local.loc[
                            np.isclose(local["tau_bin"], target),
                            ["fov_name", "lineage_id"],
                        ]
                        .drop_duplicates()
                        .to_numpy(),
                    )
                )
                complete = keys if complete is None else complete & keys
            complete_frame = pd.DataFrame(sorted(complete or set()), columns=["fov_name", "lineage_id"])
            sample_count = SAMPLES_BY_MARKER_DATASET.get((marker, dataset), samples_per_dataset)
            selected_lineages = _prefer_distinct_fovs(complete_frame, sample_count, rng)
            selected_lineages = _apply_curated_replacements(selected_lineages, complete_frame, marker, dataset)
            for _, lineage in selected_lineages.iterrows():
                sample_id += 1
                lineage_rows = local.loc[
                    local["fov_name"].eq(lineage["fov_name"]) & local["lineage_id"].eq(lineage["lineage_id"])
                ]
                previous_xy: np.ndarray | None = None
                for _, target in marker_targets.iterrows():
                    candidates = lineage_rows.loc[np.isclose(lineage_rows["tau_bin"], target["target_tau"])]
                    if candidates.empty:
                        raise RuntimeError("Complete-lineage selection lost a target bin")
                    row = _choose_descendant(candidates, previous_xy)
                    previous_xy = row[["x", "y"]].to_numpy(float)
                    record = row.to_dict()
                    record.update(
                        {
                            "sample_id": sample_id,
                            "marker": marker,
                            "stage": target["stage"],
                            "target_tau": float(target["target_tau"]),
                            "raw_path": str(RAW_PATHS[dataset]),
                        }
                    )
                    records.append(record)
    selected = pd.DataFrame(records)
    expected = sum(
        SAMPLES_BY_MARKER_DATASET.get((marker, dataset), samples_per_dataset)
        for marker in MARKERS
        for dataset in DATASETS_BY_MARKER[marker]
    ) * len(TIMEPOINTS)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected crops, found {len(selected)}")
    return selected


def _extract_crops(
    selected: pd.DataFrame,
    crop_size: int,
    z_window: int,
) -> dict[int, np.ndarray]:
    crops: dict[int, np.ndarray] = {}
    for (raw_path, fov_name), rows in selected.groupby(["raw_path", "fov_name"], sort=False, observed=True):
        with open_ome_zarr(str(Path(raw_path) / str(fov_name)), mode="r") as position:
            array = position["0"]
            organelle_index = position.channel_names.index(ORGANELLE_CHANNEL)
            pal_index = position.channel_names.index(PAL_CHANNEL)
            phase_index = position.channel_names.index(PHASE_CHANNEL)
            for index, row in rows.iterrows():
                time = int(row["t"])
                focus = np.clip(_focus_z(position, time, phase_index), 0, int(array.shape[2]) - 1)
                below = int(round(z_window * 0.3))
                z0 = max(0, int(focus) - below)
                z1 = min(int(array.shape[2]), z0 + z_window)
                z0 = max(0, z1 - z_window)
                y0, y1 = _crop_bounds(row["y"], crop_size, int(array.shape[-2]))
                x0, x1 = _crop_bounds(row["x"], crop_size, int(array.shape[-1]))
                fluorescence = np.asarray(
                    array[
                        time,
                        [pal_index, organelle_index],
                        z0:z1,
                        y0:y1,
                        x0:x1,
                    ],
                    dtype=np.float32,
                ).max(axis=1)
                phase = np.asarray(
                    array[time, phase_index, int(focus), y0:y1, x0:x1],
                    dtype=np.float32,
                )
                crop = np.stack([fluorescence[0], fluorescence[1], phase])
                if crop.shape != (3, crop_size, crop_size):
                    raise ValueError(f"Unexpected crop shape {crop.shape} for row {index}")
                crops[int(index)] = crop
    if len(crops) != len(selected):
        raise RuntimeError("Not every selected row produced a crop")
    return crops


def _display_limits(
    selected: pd.DataFrame,
    crops: dict[int, np.ndarray],
    lower_percentile: float,
    upper_percentile: float,
    pal_lower_percentile: float,
    pal_upper_percentile: float,
    organelle_upper_percentile: float,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Fix contrast within each lineage across all seven timepoints."""
    limits: dict[tuple[int, int], tuple[float, float]] = {}
    for sample_id, rows in selected.groupby("sample_id", observed=True):
        for channel in range(3):
            values = np.concatenate([crops[int(index)][channel].ravel() for index in rows.index])
            channel_lower = pal_lower_percentile if channel == 0 else lower_percentile
            channel_upper = (
                pal_upper_percentile
                if channel == 0
                else organelle_upper_percentile
                if channel == 1
                else upper_percentile
            )
            low, high = np.nanpercentile(values, [channel_lower, channel_upper])
            limits[(int(sample_id), channel)] = (
                float(low),
                max(float(high), float(low) + 1.0),
            )
    return limits


def _render(
    selected: pd.DataFrame,
    crops: dict[int, np.ndarray],
    limits: dict[tuple[int, int], tuple[float, float]],
    output_stem: Path,
    marker_name: str,
    dpi: int,
) -> None:
    stages = tuple(stage for stage, _ in TIMEPOINTS)
    stage_labels = {stage: ("pAL alignment (tau=0 h)" if tau == 0 else f"tau={tau:+g} h") for stage, tau in TIMEPOINTS}
    samples = selected[["sample_id", "marker", "dataset", "fov_name", "lineage_id"]].drop_duplicates()
    samples["dataset_order"] = samples["dataset"].map(
        {dataset: index for index, dataset in enumerate(DATASETS_BY_MARKER[marker_name])}
    )
    samples = samples.sort_values(["dataset_order", "sample_id"], kind="stable").reset_index(drop=True)
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows,
        len(stages) * 2,
        figsize=(28.0, 1.9 * n_rows + 2.2),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("black")
    selected_lookup = selected.set_index(["sample_id", "stage"])
    for display_row, sample in samples.iterrows():
        sample_id = int(sample["sample_id"])
        marker = str(sample["marker"])
        dataset = str(sample["dataset"])
        for stage_index, stage in enumerate(stages):
            row = selected_lookup.loc[(sample_id, stage)]
            source_index = int(row["source_index"])
            pal_low, pal_high = limits[(sample_id, 0)]
            organelle_low, organelle_high = limits[(sample_id, 1)]
            pal = np.clip(
                (crops[source_index][0] - pal_low) / (pal_high - pal_low),
                0,
                1,
            )
            organelle = np.clip(
                (crops[source_index][1] - organelle_low) / (organelle_high - organelle_low),
                0,
                1,
            )
            overlay = np.stack([pal, organelle, pal], axis=-1)
            for panel in range(2):
                column = stage_index * 2 + panel
                axis = axes[display_row, column]
                if panel == 0:
                    axis.imshow(overlay, interpolation="nearest")
                else:
                    phase_low, phase_high = limits[(sample_id, 2)]
                    axis.imshow(
                        crops[source_index][2],
                        cmap="gray",
                        vmin=phase_low,
                        vmax=phase_high,
                        interpolation="nearest",
                    )
                if display_row == 0:
                    axis.set_title(
                        f"{stage_labels[stage]}\n" + (f"{PAL_DISPLAY_NAME} + organelle" if panel == 0 else "Phase3D"),
                        fontsize=9,
                    )
                if panel == 0:
                    axis.text(
                        0.03,
                        0.04,
                        f"p={float(row['gmm_positive_posterior']):.2f}",
                        color="white",
                        fontsize=6,
                        transform=axis.transAxes,
                        ha="left",
                        va="bottom",
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.45,
                            "pad": 1,
                            "edgecolor": "none",
                        },
                    )
                else:
                    axis.text(
                        0.03,
                        0.04,
                        f"tau={float(row['tau']):+.1f} h | HPI={float(row['hours_post_perturbation']):.1f}",
                        color="white",
                        fontsize=5.8,
                        transform=axis.transAxes,
                        ha="left",
                        va="bottom",
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.45,
                            "pad": 1,
                            "edgecolor": "none",
                        },
                    )
        axes[display_row, 0].set_ylabel(
            f"{marker} | {DATASET_LABELS[dataset]}\nlineage {sample['lineage_id']} | #{sample_id}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
        if display_row > 0 and samples.iloc[display_row - 1]["dataset"] != dataset:
            for axis in axes[display_row]:
                axis.spines["top"].set_color("white")
                axis.spines["top"].set_linewidth(4.0)
    fig.suptitle(
        f"{marker_name} lineage-family examples every three hours around pAL alignment\n"
        f"Matched 221x221 crops: {PAL_DISPLAY_NAME} (magenta) + organelle (green) overlay, and Phase3D",
        fontsize=15,
        y=0.998,
    )
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.018, top=0.955, wspace=0.025, hspace=0.055)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    global MARKERS
    MARKERS = tuple(args.markers)
    targets = _target_table()
    selected = (
        _select_rows(
            args.input,
            targets,
            args.samples_per_dataset,
            args.random_seed,
        )
        .reset_index(drop=False)
        .rename(columns={"index": "source_index"})
    )
    crops = _extract_crops(selected, args.crop_size, args.z_window)
    limits = _display_limits(
        selected,
        crops,
        args.lower_percentile,
        args.upper_percentile,
        args.pal_lower_percentile,
        args.pal_upper_percentile,
        args.organelle_upper_percentile,
    )
    outputs: list[str] = []
    for marker in MARKERS:
        marker_rows = selected.loc[selected["marker"].eq(marker)].copy()
        output_stem = args.input / f"{OUTPUT_PREFIX}_{marker.lower()}_pal_aligned_3h_lineage_cell_montage"
        _render(marker_rows, crops, limits, output_stem, marker, args.dpi)
        marker_rows.to_csv(
            output_stem.with_name(output_stem.name + "_manifest.csv"),
            index=False,
        )
        outputs.extend([str(output_stem.with_suffix(".png")), str(output_stem.with_suffix(".pdf"))])
        print(output_stem.with_suffix(".png"))
        print(output_stem.with_suffix(".pdf"))
    run_info = {
        "analysis": "per-marker paired raw-cell examples every three hours around pAL alignment",
        "input": str(args.input),
        "outputs": outputs,
        "selection": {
            "condition": PERTURBATION,
            "sampling_unit": "parent-linked lineage family",
            "same_lineage_at_all_bins": True,
            "time_grid": "tau=-9 to +9 hours in 3-hour increments",
            "default_samples_per_marker_dataset": args.samples_per_dataset,
            "samples_by_marker_dataset": {
                f"{marker}::{dataset}": SAMPLES_BY_MARKER_DATASET.get((marker, dataset), args.samples_per_dataset)
                for marker in MARKERS
                for dataset in DATASETS_BY_MARKER[marker]
            },
            "datasets": {marker: list(DATASETS_BY_MARKER[marker]) for marker in MARKERS},
            "target_bins": targets.to_dict(orient="records"),
            "descendant_choice": "minimum lineage depth/track at first bin, then nearest XY descendant",
            "random_seed": args.random_seed,
            "curated_excluded_lineages": sorted(CURATED_EXCLUSIONS),
        },
        "crop": {
            "size_pixels": args.crop_size,
            "fluorescence_projection": f"maximum intensity projection over {args.z_window} slices",
            "phase_plane": "Phase3D focus plane",
            "channels": {
                PAL_DISPLAY_NAME: PAL_CHANNEL,
                "organelle": ORGANELLE_CHANNEL,
                "Phase3D": PHASE_CHANNEL,
            },
            "colormaps": {
                "overlay": f"additive RGB: {PAL_DISPLAY_NAME}=magenta, organelle=green",
                "Phase3D": "grayscale",
            },
            "display_scaling": (
                "fixed within lineage/channel across all seven stages; "
                f"organelle percentiles {args.lower_percentile}-{args.organelle_upper_percentile}; "
                f"pAL percentiles {args.pal_lower_percentile}-{args.pal_upper_percentile} linearly scaled; "
                f"Phase3D percentiles {args.lower_percentile}-{args.upper_percentile}; "
                "absolute brightness is not comparable between lineage rows"
            ),
        },
        "dpi": args.dpi,
    }
    (args.input / f"{OUTPUT_PREFIX}_pal_aligned_3h_marker_montages_run_info.json").write_text(
        json.dumps(run_info, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--markers",
        nargs="+",
        choices=("SEC61B", "TOMM20", "G3BP1"),
        default=["SEC61B", "TOMM20", "G3BP1"],
    )
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=221)
    parser.add_argument("--z-window", type=int, default=16)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.7)
    parser.add_argument("--pal-lower-percentile", type=float, default=10.0)
    parser.add_argument("--pal-upper-percentile", type=float, default=99.5)
    parser.add_argument("--organelle-upper-percentile", type=float, default=99.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
