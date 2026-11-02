"""Direct-pAL alignment of organelle population divergence.

The four ZIKV datasets contain a pAL17 readout. This analysis therefore uses
the existing pooled direct-pAL witness/GMM posterior to estimate a sustained
infection onset for each tracked ZIKV cell, without a Phase surrogate.

For SEC61B, G3BP1, and TOMM20 independently:

1. align nonblank organelle embeddings to the direct-pAL cell table;
2. normalize by smoothed exact-HPI control medians and one fixed control MAD;
3. fit a pooled, marker-specific PCA retaining 80% variance;
4. compute HPI-matched MMD in physical-HPI and pAL-onset-relative bins;
5. standardize MMD by a fixed-bandwidth permutation null; and
6. estimate curve stability with repeated matched track subsampling.

The primary cross-marker plot uses 07_01, the one dataset containing all three
organelle readouts. A replicate-augmented plot balances 07_01 with each
marker-specific second dataset. MMD remains a population statistic throughout.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import evaluate_pooled_zikv_pal17_phase3d as pooled
import matplotlib.pyplot as plt

from viscy_utils.evaluation.mmd import (
    compute_mmd_unbiased,
    median_heuristic,
    mmd_permutation_test,
)

EMBEDDING_SUFFIX = Path(
    "2-phenotyping/predictions/DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/embeddings"
)
SHARED_OUTPUT_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank"
)
DEFAULT_TEACHER = (
    SHARED_OUTPUT_ROOT / "posterior_threshold_sweep_all_cells_gmm_gated" / "predictions_by_threshold.parquet"
)
DEFAULT_OUTPUT = SHARED_OUTPUT_ROOT / "direct_pal_aligned_organelle_mmd"

KEYS = ["fov_name", "track_id", "t"]
TRACK_KEYS = ["dataset", "fov_name", "track_id"]
LINEAGE_KEYS = ["dataset", "fov_name", "lineage_id"]
CONDITIONS = ("uninfected", "ZIKV")
PERTURBATION = "ZIKV"
COMMON_DATASET = "07_01_SEC61B_TOMM20_G3BP1"
MARKER_ORDER = ("TOMM20", "SEC61B", "G3BP1")
COLORS = {
    "TOMM20": "#009E73",
    "SEC61B": "#0072B2",
    "G3BP1": "#CC79A7",
}
DISPLAY = {"TOMM20": "TOMM20", "SEC61B": "SEC61B", "G3BP1": "G3BP1"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    marker: str

    @property
    def embedding(self) -> Path:
        return self.root / EMBEDDING_SUFFIX / f"{self.marker}.zarr"


DATASETS = (
    DatasetSpec(
        COMMON_DATASET,
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV"),
        "SEC61B",
    ),
    DatasetSpec(
        "03_24_SEC61B",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_24_A549_SEC61_ZIKV"),
        "SEC61B",
    ),
    DatasetSpec(
        COMMON_DATASET,
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV"),
        "G3BP1",
    ),
    DatasetSpec(
        "04_08_G3BP1",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_04_08_A549_G3BP1_ZIKV"),
        "G3BP1",
    ),
    DatasetSpec(
        COMMON_DATASET,
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV"),
        "TOMM20",
    ),
    DatasetSpec(
        "04_10_TOMM20",
        Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_04_10_A549_TOMM20_ZIKV"),
        "TOMM20",
    ),
)

SCOPE_DATASETS = {
    "common_07_01": {marker: (COMMON_DATASET,) for marker in MARKER_ORDER},
    "replicate_augmented": {
        marker: tuple(spec.name for spec in DATASETS if spec.marker == marker) for marker in MARKER_ORDER
    },
}
SCOPE_TITLES = {
    "common_07_01": "07_01 common dataset",
    "replicate_augmented": "replicate-augmented, date-balanced",
}


def _marker_specs(marker: str) -> tuple[DatasetSpec, ...]:
    return tuple(spec for spec in DATASETS if spec.marker == marker)


def _lineage_map(obs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Resolve each track to its oldest observed or latent ancestor per FOV."""
    required = {"fov_name", "track_id", "parent_track_id"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"{dataset}: missing lineage columns: {sorted(missing)}")
    nodes = obs[["fov_name", "track_id", "parent_track_id"]].drop_duplicates()
    inconsistent = nodes.groupby(["fov_name", "track_id"], observed=True)["parent_track_id"].nunique()
    if (inconsistent > 1).any():
        raise ValueError(f"{dataset}: tracks have inconsistent parent_track_id")
    nodes = nodes.drop_duplicates(["fov_name", "track_id"])
    rows: list[dict[str, object]] = []
    for fov_name, group in nodes.groupby("fov_name", sort=False, observed=True):
        parent_of = {
            int(track_id): int(parent_id) if pd.notna(parent_id) else -1
            for track_id, parent_id in zip(group["track_id"], group["parent_track_id"], strict=True)
        }
        observed_tracks = set(parent_of)
        for track_id in parent_of:
            current = int(track_id)
            depth = 0
            seen = {current}
            missing_parent = False
            while parent_of.get(current, -1) >= 0:
                parent = int(parent_of[current])
                depth += 1
                if parent in seen:
                    raise ValueError(f"{dataset}/{fov_name}: cycle in lineage at track {parent}")
                seen.add(parent)
                current = parent
                if current not in observed_tracks:
                    missing_parent = True
                    break
            rows.append(
                {
                    "dataset": dataset,
                    "fov_name": fov_name,
                    "track_id": int(track_id),
                    "lineage_id": int(current),
                    "lineage_depth": int(depth),
                    "lineage_root_observed": bool(current in observed_tracks),
                    "lineage_has_missing_parent": bool(missing_parent),
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(TRACK_KEYS).any():
        raise ValueError(f"{dataset}: duplicate track-to-lineage mappings")
    return result


def _load_teacher(path: Path) -> pd.DataFrame:
    required = [
        "dataset",
        *KEYS,
        "perturbation",
        "hours_post_perturbation",
        "witness_score",
        "gmm_positive_posterior",
    ]
    teacher = pd.read_parquet(path, columns=required)
    if teacher.duplicated(["dataset", *KEYS]).any():
        raise ValueError("Direct-pAL table has duplicate dataset/cell keys")
    expected = {spec.name for spec in DATASETS}
    missing = expected - set(teacher["dataset"].astype(str))
    if missing:
        raise ValueError(f"Direct-pAL table lacks datasets: {sorted(missing)}")
    return teacher


def _sustained_onset(
    group: pd.DataFrame,
    *,
    threshold: float,
    run_length: int,
) -> float | None:
    group = group.sort_values("t")
    t = group["t"].to_numpy(dtype=int)
    posterior = group["gmm_positive_posterior"].to_numpy(dtype=float)
    negative = posterior <= 1.0 - threshold
    positive = posterior >= threshold
    latest_negative_end = -1
    for start in range(len(group) - run_length + 1):
        stop = start + run_length
        if not np.all(np.diff(t[start:stop]) == 1):
            continue
        if negative[start:stop].all():
            latest_negative_end = stop - 1
        if positive[start:stop].all() and latest_negative_end < start:
            if latest_negative_end >= 0:
                return float(group.iloc[start]["hours_post_perturbation"])
    return None


def _onset_table(
    teacher: pd.DataFrame,
    *,
    threshold: float,
    run_length: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    perturbed = teacher.loc[teacher["perturbation"].astype(str).eq(PERTURBATION)]
    for key, group in perturbed.groupby(TRACK_KEYS, sort=False, observed=True):
        onset = _sustained_onset(
            group,
            threshold=threshold,
            run_length=run_length,
        )
        rows.append(
            {
                "dataset": str(key[0]),
                "fov_name": key[1],
                "track_id": key[2],
                "pal_onset_hpi": onset,
                "n_pal_frames": int(len(group)),
                "pal_start_hpi": float(group["hours_post_perturbation"].min()),
                "pal_end_hpi": float(group["hours_post_perturbation"].max()),
            }
        )
    return pd.DataFrame(rows)


def _onset_sensitivity(teacher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in (0.8, 0.9):
        for run_length in (2, 3):
            onset = _onset_table(
                teacher,
                threshold=threshold,
                run_length=run_length,
            )
            for dataset, group in onset.groupby("dataset", observed=True):
                finite = group["pal_onset_hpi"].notna()
                rows.append(
                    {
                        "dataset": dataset,
                        "posterior_threshold": threshold,
                        "run_length": run_length,
                        "n_tracks": int(len(group)),
                        "n_observed_onsets": int(finite.sum()),
                        "observed_onset_rate": float(finite.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _lineage_onset_table(
    teacher: pd.DataFrame,
    lineage_map: pd.DataFrame,
    *,
    threshold: float,
    run_length: int,
) -> pd.DataFrame:
    """Detect sustained onset after stitching mother/daughter track segments."""
    mapped = teacher.merge(
        lineage_map[TRACK_KEYS + ["lineage_id"]],
        on=TRACK_KEYS,
        how="inner",
        validate="many_to_one",
    )
    perturbed = mapped.loc[mapped["perturbation"].astype(str).eq(PERTURBATION)]
    # A branching lineage can contain multiple daughters at the same frame.
    # Median aggregation gives the lineage one pAL state per time point and
    # prevents a division from weighting the onset decision twice.
    per_time = (
        perturbed.groupby(LINEAGE_KEYS + ["t"], sort=False, observed=True)
        .agg(
            hours_post_perturbation=("hours_post_perturbation", "median"),
            gmm_positive_posterior=("gmm_positive_posterior", "median"),
            n_active_tracks=("track_id", "nunique"),
        )
        .reset_index()
    )
    track_counts = perturbed.groupby(LINEAGE_KEYS, observed=True)["track_id"].nunique()
    rows: list[dict[str, object]] = []
    for key, group in per_time.groupby(LINEAGE_KEYS, sort=False, observed=True):
        onset = _sustained_onset(
            group,
            threshold=threshold,
            run_length=run_length,
        )
        rows.append(
            {
                "dataset": str(key[0]),
                "fov_name": key[1],
                "lineage_id": key[2],
                "pal_onset_hpi": onset,
                "n_pal_frames": int(len(group)),
                "n_tracks_in_lineage": int(track_counts.loc[key]),
                "max_simultaneous_tracks": int(group["n_active_tracks"].max()),
                "pal_start_hpi": float(group["hours_post_perturbation"].min()),
                "pal_end_hpi": float(group["hours_post_perturbation"].max()),
            }
        )
    return pd.DataFrame(rows)


def _lineage_onset_sensitivity(
    teacher: pd.DataFrame,
    lineage_map: pd.DataFrame,
    marker: str,
) -> pd.DataFrame:
    rows = []
    for threshold in (0.8, 0.9):
        for run_length in (2, 3):
            onset = _lineage_onset_table(
                teacher,
                lineage_map,
                threshold=threshold,
                run_length=run_length,
            )
            for dataset, group in onset.groupby("dataset", observed=True):
                finite = group["pal_onset_hpi"].notna()
                rows.append(
                    {
                        "marker": marker,
                        "dataset": dataset,
                        "posterior_threshold": threshold,
                        "run_length": run_length,
                        "n_lineages": int(len(group)),
                        "n_observed_onsets": int(finite.sum()),
                        "observed_onset_rate": float(finite.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _load_marker_dataset(
    spec: DatasetSpec,
    teacher: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    data = ad.read_zarr(spec.embedding)
    obs = data.obs.reset_index(drop=True).copy()
    lineage = _lineage_map(obs, spec.name)
    if obs.duplicated(KEYS).any():
        raise ValueError(f"{spec.name}/{spec.marker}: duplicate cell keys")
    if "is_blank" not in obs:
        raise KeyError(f"{spec.name}/{spec.marker}: missing is_blank")
    keep = ~obs["is_blank"].to_numpy(dtype=bool)
    marker_rows = np.flatnonzero(keep)
    marker_obs = obs.iloc[marker_rows].reset_index(drop=True)
    marker_keys = marker_obs[KEYS].copy()
    marker_keys["_marker_row"] = marker_rows
    direct = teacher.loc[
        teacher["dataset"].astype(str).eq(spec.name),
        [*KEYS, "perturbation", "hours_post_perturbation", "witness_score", "gmm_positive_posterior"],
    ]
    aligned = marker_keys.merge(direct, on=KEYS, how="inner", validate="one_to_one")
    if not len(aligned):
        raise ValueError(f"{spec.name}/{spec.marker}: no direct-pAL-aligned cells")
    aligned_rows = aligned["_marker_row"].to_numpy(dtype=int)
    aligned_meta = obs.iloc[aligned_rows].reset_index(drop=True)
    for column in ("perturbation", "hours_post_perturbation"):
        left = aligned_meta[column].to_numpy()
        right = aligned[column].to_numpy()
        if column == "hours_post_perturbation":
            agrees = np.allclose(left.astype(float), right.astype(float), equal_nan=True)
        else:
            agrees = np.array_equal(left.astype(str), right.astype(str))
        if not agrees:
            raise ValueError(f"{spec.name}/{spec.marker}: {column} metadata disagree")
    x = np.asarray(data.X, dtype=np.float32)[aligned_rows].copy()
    normalized, grid, centers, scale = pooled._control_normalization_contract(
        x,
        aligned_meta,
        sigma_timepoints=args.smooth_sigma_timepoints,
        floor_quantile=args.scale_floor_quantile,
    )
    aligned_meta = aligned_meta.copy()
    aligned_meta["dataset"] = spec.name
    aligned_meta["marker"] = spec.marker
    aligned_meta = aligned_meta.merge(
        lineage[
            TRACK_KEYS
            + [
                "lineage_id",
                "lineage_depth",
                "lineage_root_observed",
                "lineage_has_missing_parent",
            ]
        ],
        on=TRACK_KEYS,
        how="left",
        validate="many_to_one",
    )
    aligned_meta["witness_score"] = aligned["witness_score"].to_numpy(float)
    aligned_meta["gmm_positive_posterior"] = aligned["gmm_positive_posterior"].to_numpy(float)
    return (
        normalized,
        aligned_meta,
        {
            "hpi_grid": grid,
            "control_centers": centers,
            "control_mad": scale,
        },
        lineage,
    )


def _balanced_pca_rows(
    frames: list[pd.DataFrame],
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0, *[len(frame) for frame in frames[:-1]]])
    selected: list[np.ndarray] = []
    manifests = []
    for offset, frame in zip(offsets, frames, strict=True):
        condition = frame["perturbation"].astype(str).to_numpy()
        for name in CONDITIONS:
            rows = np.flatnonzero(condition == name)
            n = min(maximum, len(rows))
            chosen = rng.choice(rows, n, replace=False)
            selected.append(offset + chosen)
            manifest = frame.iloc[chosen][["dataset", *KEYS, "perturbation"]].copy()
            manifests.append(manifest)
    return np.concatenate(selected), pd.concat(manifests, ignore_index=True)


def _fit_marker_pca(
    marker: str,
    arrays: list[np.ndarray],
    frames: list[pd.DataFrame],
    args: argparse.Namespace,
    output: Path,
) -> tuple[np.ndarray, pd.DataFrame, PCA, float]:
    combined = np.concatenate(arrays, axis=0)
    obs = pd.concat(frames, ignore_index=True)
    rows, sample_manifest = _balanced_pca_rows(
        frames,
        args.pca_max_per_dataset_condition,
        args.random_seed + 101 * (MARKER_ORDER.index(marker) + 1),
    )
    pca = PCA(
        n_components=args.pca_variance_threshold,
        svd_solver="full",
        random_state=args.random_seed,
    )
    pca.fit(combined[rows])
    scores = pca.transform(combined).astype(np.float32)
    bandwidth_rows = rows
    rng = np.random.default_rng(args.random_seed + 211 * (MARKER_ORDER.index(marker) + 1))
    if len(bandwidth_rows) > args.bandwidth_max_cells:
        bandwidth_rows = rng.choice(
            bandwidth_rows,
            args.bandwidth_max_cells,
            replace=False,
        )
    condition = obs["perturbation"].astype(str).to_numpy()
    control_rows = bandwidth_rows[condition[bandwidth_rows] == "uninfected"]
    perturbed_rows = bandwidth_rows[condition[bandwidth_rows] == PERTURBATION]
    n = min(len(control_rows), len(perturbed_rows), args.bandwidth_max_cells // 2)
    control_rows = rng.choice(control_rows, n, replace=False)
    perturbed_rows = rng.choice(perturbed_rows, n, replace=False)
    bandwidth = median_heuristic(scores[control_rows], scores[perturbed_rows])

    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, model_dir / f"{marker.lower()}_pca80.joblib")
    sample_manifest.insert(0, "marker", marker)
    sample_manifest.to_parquet(
        model_dir / f"{marker.lower()}_pca_fit_cells.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "marker": marker,
            "pc": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(model_dir / f"{marker.lower()}_pca_scree.csv", index=False)
    return scores, obs, pca, float(bandwidth)


def _nearest_per_unit(
    frame: pd.DataFrame,
    center: float,
    unit_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_distance"] = np.abs(frame["axis_value"].to_numpy(float) - center)
    frame["_tie_break"] = rng.random(len(frame))
    return (
        frame.sort_values([unit_column, "_distance", "_tie_break"])
        .drop_duplicates(unit_column)
        .drop(columns=["_distance", "_tie_break"])
    )


def _matched_rows(
    frame: pd.DataFrame,
    *,
    center: float,
    width: float,
    datasets: tuple[str, ...],
    max_per_dataset: int,
    min_per_dataset: int,
    seed: int,
    unit_column: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    rng = np.random.default_rng(seed)
    half = width / 2.0
    infected = frame.loc[
        frame["perturbation"].astype(str).eq(PERTURBATION)
        & frame["dataset"].astype(str).isin(datasets)
        & frame["axis_value"].between(center - half, center + half, inclusive="left")
    ]
    infected = _nearest_per_unit(infected, center, unit_column, rng)
    controls = frame.loc[
        frame["perturbation"].astype(str).eq("uninfected") & frame["dataset"].astype(str).isin(datasets)
    ]
    z_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    manifest_parts = []
    for dataset in datasets:
        z_dataset = infected.loc[infected["dataset"].astype(str).eq(dataset)]
        c_dataset = controls.loc[controls["dataset"].astype(str).eq(dataset)]
        pairs: list[tuple[int, int]] = []
        used_control_units: set[str] = set()
        for hpi, z_group in z_dataset.groupby("hpi_match", observed=True):
            c_group = c_dataset.loc[np.isclose(c_dataset["hpi_match"], float(hpi))]
            c_group = c_group.loc[~c_group[unit_column].isin(used_control_units)]
            c_group = c_group.assign(_tie_break=rng.random(len(c_group)))
            c_group = c_group.sort_values("_tie_break").drop_duplicates(unit_column)
            n = min(len(z_group), len(c_group))
            if n == 0:
                continue
            z_rows = rng.choice(z_group.index.to_numpy(), n, replace=False)
            c_rows = rng.choice(c_group.index.to_numpy(), n, replace=False)
            used_control_units.update(frame.loc[c_rows, unit_column].astype(str).tolist())
            pairs.extend(zip(z_rows.tolist(), c_rows.tolist(), strict=True))
        if len(pairs) < min_per_dataset:
            return None
        if len(pairs) > max_per_dataset:
            chosen = rng.choice(len(pairs), max_per_dataset, replace=False)
            pairs = [pairs[index] for index in chosen]
        z_index = np.asarray([pair[0] for pair in pairs], dtype=int)
        c_index = np.asarray([pair[1] for pair in pairs], dtype=int)
        z_parts.append(frame.loc[z_index, "pc_row"].to_numpy(dtype=int))
        c_parts.append(frame.loc[c_index, "pc_row"].to_numpy(dtype=int))
        manifest_columns = [
            "dataset",
            "marker",
            *KEYS,
            "parent_track_id",
            "lineage_id",
            "lineage_depth",
            "lineage_root_observed",
            "lineage_has_missing_parent",
            "perturbation",
            "hours_post_perturbation",
            "pal_onset_hpi",
            "tau",
            "pc_row",
        ]
        z_manifest = frame.loc[z_index, manifest_columns].copy()
        z_manifest["sample_group"] = PERTURBATION
        c_manifest = frame.loc[c_index, manifest_columns].copy()
        c_manifest["sample_group"] = "control"
        manifest_parts.extend([z_manifest, c_manifest])
    if not z_parts:
        return None
    return (
        np.concatenate(z_parts),
        np.concatenate(c_parts),
        pd.concat(manifest_parts, ignore_index=True),
    )


def _curve(
    scores: np.ndarray,
    frame: pd.DataFrame,
    *,
    marker: str,
    axis: str,
    scope: str,
    datasets: tuple[str, ...],
    centers: np.ndarray,
    width: float,
    bandwidth: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    if args.same_onset_cohort:
        same_cohort = working["perturbation"].astype(str).eq("uninfected") | working["pal_onset_hpi"].notna()
        working = working.loc[same_cohort].copy()
    if axis == "tau":
        working["axis_value"] = working["tau"]
        # Controls intentionally have no pAL-relative time. Keep them in the
        # candidate pool so each perturbed tau bin can be compared with controls
        # matched to the selected cells physical HPI and dataset mixture.
        finite_axis = np.isfinite(working["axis_value"])
        keep = finite_axis | working["perturbation"].eq("uninfected")
    elif axis == "hpi":
        working["axis_value"] = working["hours_post_perturbation"]
        keep = np.isfinite(working["axis_value"])
    else:
        raise ValueError(axis)
    working = working.loc[keep].copy()
    summaries = []
    sample_manifests = []
    for bin_index, center in enumerate(centers):
        seed = (
            args.random_seed
            + 10_000 * (MARKER_ORDER.index(marker) + 1)
            + 1_000 * (axis == "tau")
            + 100 * list(SCOPE_DATASETS).index(scope)
            + bin_index
        )
        matched = _matched_rows(
            working,
            center=float(center),
            width=width,
            datasets=datasets,
            max_per_dataset=args.mmd_max_per_dataset,
            min_per_dataset=args.mmd_min_per_dataset,
            seed=seed,
            unit_column=("lineage_uid" if args.sampling_unit == "lineage" else "track_uid"),
        )
        if matched is None:
            continue
        z_rows, c_rows, sample_manifest = matched
        z = scores[z_rows]
        c = scores[c_rows]
        observed, p_value, null = mmd_permutation_test(
            z,
            c,
            n_permutations=args.null_permutations,
            bandwidth=bandwidth,
            seed=seed + 300_000,
        )
        null_mean = float(np.mean(null))
        null_std = float(np.std(null, ddof=1))
        if not np.isfinite(null_std) or null_std <= 1e-12:
            continue
        z_score = (observed - null_mean) / null_std
        null_q95_z = (float(np.quantile(null, 0.95)) - null_mean) / null_std
        rng = np.random.default_rng(seed + 600_000)
        subset_n = max(4, int(np.floor(args.subsample_fraction * len(z_rows))))
        subsampled = []
        for _ in range(args.subsample_repeats):
            selected = rng.choice(len(z_rows), subset_n, replace=False)
            value = compute_mmd_unbiased(
                z[selected],
                c[selected],
                bandwidth=bandwidth,
            )
            subsampled.append((value - null_mean) / null_std)
        summaries.append(
            {
                "marker": marker,
                "axis": axis,
                "scope": scope,
                "axis_center": float(center),
                "bin_width": float(width),
                "n_perturbed": int(len(z_rows)),
                "n_control": int(len(c_rows)),
                "n_perturbed_tracks": int(
                    sample_manifest.loc[
                        sample_manifest["sample_group"].eq(PERTURBATION),
                        ["dataset", "fov_name", "track_id"],
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_perturbed_lineages": int(
                    sample_manifest.loc[
                        sample_manifest["sample_group"].eq(PERTURBATION),
                        ["dataset", "fov_name", "lineage_id"],
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "sampling_unit": args.sampling_unit,
                "mmd2_unbiased": float(observed),
                "null_mean": null_mean,
                "null_std": null_std,
                "null_q95_z": null_q95_z,
                "null_standardized_mmd": float(z_score),
                "subsample_q025": float(np.quantile(subsampled, 0.025)),
                "subsample_median": float(np.median(subsampled)),
                "subsample_q975": float(np.quantile(subsampled, 0.975)),
                "permutation_p_value": float(p_value),
                "bandwidth": float(bandwidth),
            }
        )
        sample_manifest.insert(0, "axis_center", float(center))
        sample_manifest.insert(0, "scope", scope)
        sample_manifest.insert(0, "axis", axis)
        sample_manifests.append(sample_manifest)
    return (
        pd.DataFrame(summaries),
        pd.concat(sample_manifests, ignore_index=True) if sample_manifests else pd.DataFrame(),
    )


def _response_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in curves.groupby(["marker", "axis", "scope"], observed=True):
        group = group.sort_values("axis_center").reset_index(drop=True)
        centers = group["axis_center"].to_numpy(float)
        z = group["null_standardized_mmd"].to_numpy(float)
        low = group["subsample_q025"].to_numpy(float)
        critical = group["null_q95_z"].to_numpy(float)
        above = low > critical
        onset_index = None
        for index in range(len(group) - 1):
            expected_step = np.isclose(
                centers[index + 1] - centers[index],
                group.iloc[index]["bin_width"],
            )
            if above[index] and above[index + 1] and expected_step:
                onset_index = index
                break
        peak_index = int(np.nanargmax(z))
        slopes = np.gradient(z, centers) if len(group) >= 2 else np.array([np.nan])
        rows.append(
            {
                "marker": key[0],
                "axis": key[1],
                "scope": key[2],
                "divergence_onset": (float(centers[onset_index]) if onset_index is not None else np.nan),
                "onset_left_censored": bool(onset_index == 0),
                "peak_axis_value": float(centers[peak_index]),
                "peak_null_standardized_mmd": float(z[peak_index]),
                "max_slope": float(np.nanmax(slopes)),
                "area_under_z_curve": float(np.trapezoid(z, centers)),
                "n_bins": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _plot_overlay(
    curves: pd.DataFrame,
    response: pd.DataFrame,
    *,
    axis: str,
    scope: str,
    output: Path,
) -> None:
    subset = curves.loc[curves["axis"].eq(axis) & curves["scope"].eq(scope)]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for marker in MARKER_ORDER:
        group = subset.loc[subset["marker"].eq(marker)].sort_values("axis_center")
        if group.empty:
            continue
        x = group["axis_center"].to_numpy(float)
        y = group["null_standardized_mmd"].to_numpy(float)
        lo = group["subsample_q025"].to_numpy(float)
        hi = group["subsample_q975"].to_numpy(float)
        ax.plot(x, y, color=COLORS[marker], lw=2.5, marker="o", ms=3.5, label=DISPLAY[marker])
        ax.fill_between(x, lo, hi, color=COLORS[marker], alpha=0.16, linewidth=0)
        summary = response.loc[response["marker"].eq(marker) & response["axis"].eq(axis) & response["scope"].eq(scope)]
        if not summary.empty and np.isfinite(summary.iloc[0]["divergence_onset"]):
            onset = float(summary.iloc[0]["divergence_onset"])
            onset_y = float(np.interp(onset, x, y))
            ax.scatter([onset], [onset_y], marker="v", s=75, color=COLORS[marker], edgecolor="white", zorder=5)
            ax.annotate(
                f"{marker}: {onset:g} h",
                (onset, onset_y),
                xytext=(4, 7),
                textcoords="offset points",
                color=COLORS[marker],
                fontsize=9,
            )
    ax.axhline(0, color="0.35", lw=1)
    if axis == "tau":
        ax.axvline(0, color="black", ls="--", lw=1.2, label="pAL onset")
        xlabel = "Hours relative to sustained direct-pAL onset (τ)"
        title_axis = "direct-pAL-aligned response time"
    else:
        xlabel = "Hours post infection (HPI)"
        title_axis = "physical HPI"
    scope_title = SCOPE_TITLES[scope]
    ax.set(
        xlabel=xlabel,
        ylabel="Null-standardized MMD² vs HPI/date-matched controls",
        title=f"Organelle population divergence over {title_axis}\n{scope_title}",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_hpi_vs_tau(
    curves: pd.DataFrame,
    response: pd.DataFrame,
    *,
    scope: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for ax, axis in zip(axes, ("hpi", "tau"), strict=True):
        subset = curves.loc[curves["axis"].eq(axis) & curves["scope"].eq(scope)]
        for marker in MARKER_ORDER:
            group = subset.loc[subset["marker"].eq(marker)].sort_values("axis_center")
            if group.empty:
                continue
            x = group["axis_center"].to_numpy(float)
            y = group["null_standardized_mmd"].to_numpy(float)
            ax.plot(x, y, color=COLORS[marker], lw=2.2, label=marker)
            ax.fill_between(
                x,
                group["subsample_q025"].to_numpy(float),
                group["subsample_q975"].to_numpy(float),
                color=COLORS[marker],
                alpha=0.14,
            )
        ax.axhline(0, color="0.35", lw=1)
        if axis == "tau":
            ax.axvline(0, color="black", ls="--", lw=1.2)
            ax.set_xlabel("Hours relative to sustained direct-pAL onset (τ)")
            ax.set_title("Aligned by direct pAL")
        else:
            ax.set_xlabel("Hours post infection (HPI)")
            ax.set_title("Unaligned physical time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Null-standardized MMD² vs matched controls")
    axes[1].legend(frameon=False)
    scope_title = SCOPE_TITLES[scope]
    fig.suptitle(f"Does direct-pAL alignment sharpen organelle divergence? — {scope_title}")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _save_contract(
    output: Path,
    marker: str,
    dataset: str,
    contract: dict[str, np.ndarray],
) -> None:
    target = output / "normalization_contracts"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / f"{marker.lower()}__{dataset}.npz",
        **contract,
    )


def run(args: argparse.Namespace) -> None:
    output = args.output
    in_progress = output.with_name(output.name + ".in_progress")
    if in_progress.exists():
        raise FileExistsError(f"Incomplete output already exists: {in_progress}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    in_progress.mkdir(parents=True)
    try:
        teacher = _load_teacher(args.teacher)
        track_onset = _onset_table(
            teacher,
            threshold=args.pal_posterior_threshold,
            run_length=args.pal_run_length,
        )
        track_onset.to_parquet(in_progress / "direct_pal_track_onsets.parquet", index=False)
        _onset_sensitivity(teacher).to_csv(
            in_progress / "direct_pal_onset_sensitivity.csv",
            index=False,
        )

        all_curves = []
        all_samples = []
        pca_rows = []
        cell_funnels = []
        onset_funnels = []
        lineage_sensitivity = []
        tau_centers = np.arange(
            args.tau_min,
            args.tau_max + args.tau_bin_hours / 2,
            args.tau_bin_hours,
        )
        hpi_centers = np.arange(
            args.hpi_min,
            args.hpi_max + args.hpi_bin_hours / 2,
            args.hpi_bin_hours,
        )
        for marker in MARKER_ORDER:
            print(f"Load and normalize {marker}", flush=True)
            arrays: list[np.ndarray] = []
            frames: list[pd.DataFrame] = []
            lineage_maps: list[pd.DataFrame] = []
            for spec in _marker_specs(marker):
                x, obs, contract, lineage = _load_marker_dataset(spec, teacher, args)
                arrays.append(x)
                frames.append(obs)
                lineage_maps.append(lineage)
                _save_contract(in_progress, marker, spec.name, contract)
                cell_funnels.append(
                    {
                        "marker": marker,
                        "dataset": spec.name,
                        "n_direct_pal_aligned_nonblank": int(len(obs)),
                        "n_control": int(obs["perturbation"].astype(str).eq("uninfected").sum()),
                        "n_perturbed": int(obs["perturbation"].astype(str).eq(PERTURBATION).sum()),
                        "n_tracks": int(obs[["fov_name", "track_id"]].drop_duplicates().shape[0]),
                        "n_lineages": int(obs[["fov_name", "lineage_id"]].drop_duplicates().shape[0]),
                    }
                )
            lineage_map = pd.concat(lineage_maps, ignore_index=True)
            conflicts = lineage_map.groupby(TRACK_KEYS, observed=True)["lineage_id"].nunique()
            if (conflicts > 1).any():
                raise ValueError(f"{marker}: conflicting lineage roots across stores")
            lineage_map = lineage_map.drop_duplicates(TRACK_KEYS)
            lineage_map.to_parquet(in_progress / f"{marker.lower()}_lineage_map.parquet", index=False)
            if args.sampling_unit == "lineage":
                onset = _lineage_onset_table(
                    teacher,
                    lineage_map,
                    threshold=args.pal_posterior_threshold,
                    run_length=args.pal_run_length,
                )
                onset.to_parquet(
                    in_progress / f"{marker.lower()}_direct_pal_lineage_onsets.parquet",
                    index=False,
                )
                lineage_sensitivity.append(_lineage_onset_sensitivity(teacher, lineage_map, marker))
                onset_keys = LINEAGE_KEYS
            else:
                onset = track_onset
                onset_keys = TRACK_KEYS
            print(f"Fit pooled PCA80 for {marker}", flush=True)
            scores, frame, pca, bandwidth = _fit_marker_pca(
                marker,
                arrays,
                frames,
                args,
                in_progress,
            )
            pca_rows.append(
                {
                    "marker": marker,
                    "n_components": int(pca.n_components_),
                    "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                    "bandwidth": bandwidth,
                    "n_cells": int(len(frame)),
                }
            )
            frame = frame.merge(onset, on=onset_keys, how="left", validate="many_to_one")
            frame["tau"] = frame["hours_post_perturbation"].to_numpy(float) - frame["pal_onset_hpi"].to_numpy(float)
            frame.loc[
                ~frame["perturbation"].astype(str).eq(PERTURBATION),
                ["pal_onset_hpi", "tau"],
            ] = np.nan
            frame["track_uid"] = (
                frame["dataset"].astype(str)
                + "::"
                + frame["fov_name"].astype(str)
                + "::"
                + frame["track_id"].astype(str)
            )
            frame["lineage_uid"] = (
                frame["dataset"].astype(str)
                + "::"
                + frame["fov_name"].astype(str)
                + "::"
                + frame["lineage_id"].astype(str)
            )
            frame["hpi_match"] = np.round(frame["hours_post_perturbation"].to_numpy(float) * 2.0) / 2.0
            frame["pc_row"] = np.arange(len(frame), dtype=int)
            perturbed_frame = frame.loc[frame["perturbation"].astype(str).eq(PERTURBATION)]
            for dataset, dataset_frame in perturbed_frame.groupby("dataset", observed=True):
                unit_column = "lineage_uid" if args.sampling_unit == "lineage" else "track_uid"
                onset_funnels.append(
                    {
                        "marker": marker,
                        "dataset": dataset,
                        "sampling_unit": args.sampling_unit,
                        "n_perturbed_cells": int(len(dataset_frame)),
                        "n_perturbed_cells_with_onset": int(dataset_frame["pal_onset_hpi"].notna().sum()),
                        "n_perturbed_units": int(dataset_frame[unit_column].nunique()),
                        "n_perturbed_units_with_onset": int(
                            dataset_frame.loc[dataset_frame["pal_onset_hpi"].notna(), unit_column].nunique()
                        ),
                    }
                )
            np.save(in_progress / f"{marker.lower()}_pca80_scores.npy", scores)
            frame.to_parquet(
                in_progress / f"{marker.lower()}_cell_manifest.parquet",
                index=False,
            )
            for scope, marker_datasets in SCOPE_DATASETS.items():
                datasets = marker_datasets[marker]
                for axis, centers, width in (
                    ("hpi", hpi_centers, args.hpi_bin_hours),
                    ("tau", tau_centers, args.tau_bin_hours),
                ):
                    print(f"MMD {marker} {scope} {axis}", flush=True)
                    curve, samples = _curve(
                        scores,
                        frame,
                        marker=marker,
                        axis=axis,
                        scope=scope,
                        datasets=datasets,
                        centers=centers,
                        width=width,
                        bandwidth=bandwidth,
                        args=args,
                    )
                    all_curves.append(curve)
                    if not samples.empty:
                        all_samples.append(samples)
            del arrays, frames, scores, frame
            gc.collect()

        curves = pd.concat(all_curves, ignore_index=True)
        samples = pd.concat(all_samples, ignore_index=True)
        response = _response_summary(curves)
        curves.to_csv(in_progress / "mmd_curves.csv", index=False)
        samples.to_parquet(in_progress / "mmd_point_sample_manifest.parquet", index=False)
        response.to_csv(in_progress / "marker_response_summary.csv", index=False)
        pd.DataFrame(pca_rows).to_csv(in_progress / "marker_pca_contracts.csv", index=False)
        pd.DataFrame(cell_funnels).to_csv(in_progress / "cell_funnels.csv", index=False)
        pd.DataFrame(onset_funnels).to_csv(in_progress / "onset_coverage_funnels.csv", index=False)
        if lineage_sensitivity:
            pd.concat(lineage_sensitivity, ignore_index=True).to_csv(
                in_progress / "direct_pal_lineage_onset_sensitivity.csv",
                index=False,
            )
        for scope in SCOPE_DATASETS:
            _plot_overlay(
                curves,
                response,
                axis="tau",
                scope=scope,
                output=in_progress / f"direct_pal_aligned_mmd_all_markers__{scope}.png",
            )
            _plot_overlay(
                curves,
                response,
                axis="hpi",
                scope=scope,
                output=in_progress / f"mmd_vs_hpi_all_markers__{scope}.png",
            )
            _plot_hpi_vs_tau(
                curves,
                response,
                scope=scope,
                output=in_progress / f"mmd_hpi_vs_pal_alignment__{scope}.png",
            )

        run_info = {
            "analysis": "direct pAL-aligned organelle null-standardized MMD",
            "teacher": str(args.teacher),
            "output": str(output),
            "perturbation": PERTURBATION,
            "scopes": SCOPE_DATASETS,
            "scope_titles": SCOPE_TITLES,
            "markers": list(MARKER_ORDER),
            "sampling_unit": args.sampling_unit,
            "lineage": {
                "edge": "parent_track_id within dataset and FOV",
                "branch_time_aggregation": "median pAL posterior",
                "one_observation_per_unit_per_bin": True,
                "missing_parents": "retained as shared latent roots",
            },
            "onset": {
                "posterior_threshold": args.pal_posterior_threshold,
                "run_length": args.pal_run_length,
                "requires_prior_sustained_negative": True,
            },
            "normalization": {
                "center": "smoothed exact-HPI control median per dataset/marker",
                "scale": "one fixed normal-consistent control MAD after centering",
                "smooth_sigma_timepoints": args.smooth_sigma_timepoints,
                "scale_floor_quantile": args.scale_floor_quantile,
            },
            "pca": {
                "variance_threshold": args.pca_variance_threshold,
                "max_per_dataset_condition": args.pca_max_per_dataset_condition,
            },
            "mmd": {
                "fixed_marker_bandwidth": True,
                "max_per_dataset": args.mmd_max_per_dataset,
                "min_per_dataset": args.mmd_min_per_dataset,
                "null_permutations": args.null_permutations,
                "subsample_repeats": args.subsample_repeats,
                "subsample_fraction": args.subsample_fraction,
                "control_matching": "exact dataset and 0.5-h HPI, no control pseudotime",
            },
            "limitations": [
                "MMD is a population displacement magnitude, not a direction or single-cell score",
                "cross-marker magnitudes are comparable only within acquisition scopes that share datasets",
                "subsample bands are stability intervals, not biological-replicate confidence intervals",
                "onset applies only to sampling units with a sustained negative-to-positive pAL transition",
            ],
            "arguments": vars(args) | {"teacher": str(args.teacher), "output": str(args.output)},
        }
        (in_progress / "run_info.json").write_text(json.dumps(run_info, indent=2))
        in_progress.rename(output)
        print(f"Wrote {output}", flush=True)
    except Exception:
        print(f"Incomplete output retained at {in_progress}", flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pal-posterior-threshold", type=float, default=0.8)
    parser.add_argument("--pal-run-length", type=int, default=3)
    parser.add_argument("--sampling-unit", choices=("track", "lineage"), default="track")
    parser.add_argument(
        "--same-onset-cohort",
        action="store_true",
        help="Use only onset-bearing perturbed units on both HPI and tau axes.",
    )
    parser.add_argument("--smooth-sigma-timepoints", type=float, default=1.0)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.01)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.8)
    parser.add_argument("--pca-max-per-dataset-condition", type=int, default=5_000)
    parser.add_argument("--bandwidth-max-cells", type=int, default=4_000)
    parser.add_argument("--tau-min", type=float, default=-10.0)
    parser.add_argument("--tau-max", type=float, default=12.0)
    parser.add_argument("--tau-bin-hours", type=float, default=1.0)
    parser.add_argument("--hpi-min", type=float, default=3.0)
    parser.add_argument("--hpi-max", type=float, default=36.0)
    parser.add_argument("--hpi-bin-hours", type=float, default=1.0)
    parser.add_argument("--mmd-max-per-dataset", type=int, default=180)
    parser.add_argument("--mmd-min-per-dataset", type=int, default=20)
    parser.add_argument("--null-permutations", type=int, default=200)
    parser.add_argument("--subsample-repeats", type=int, default=50)
    parser.add_argument("--subsample-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
