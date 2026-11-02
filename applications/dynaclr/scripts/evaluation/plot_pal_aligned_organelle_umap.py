"""Plot organelle PCA80 UMAPs by dataset, HPI, pAL time, and remodeling rate.

Each virus/marker gets one UMAP fit on lineage-family observations from the
onset-bearing perturbed cohort. The exact coordinates are reused across every
coloring. Remodeling color is the dataset-balanced fraction of confident
witness/GMM-positive lineage families in the corresponding 30-minute tau bin;
ambiguous or unlabeled families remain in the denominator as non-positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

MARKERS = ("SEC61B", "TOMM20", "G3BP1")
MARKER_STEMS = {marker: marker.lower() for marker in MARKERS}
DATASET_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
)


@dataclass(frozen=True)
class VirusSpec:
    virus: str
    condition: str
    input_dir: Path
    teacher_paths: dict[str, Path]


ZIKV_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800"
)
DENV_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_04_28_A549_SEC61B_DENV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800"
)


SPECS = {
    "zikv": VirusSpec(
        virus="ZIKV",
        condition="ZIKV",
        input_dir=ZIKV_ROOT / "classifier_exploration/outputs/"
        "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
        "direct_pal_lineage_native30_same_cohort_alignment",
        teacher_paths={
            "SEC61B": ZIKV_ROOT / "__zikv_sec61b_external_labels__",
            "TOMM20": Path(
                "/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_04_10_A549_TOMM20_ZIKV/2-phenotyping/predictions/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/epoch105-step84800/outputs/00-tomm20_teacher_joint_tied_07_01_04_10/tomm20_teacher_manifest.parquet"
            ),
            "G3BP1": ZIKV_ROOT / "__zikv_g3bp1_external_labels__",
        },
    ),
    "denv": VirusSpec(
        virus="DENV",
        condition="DENV",
        input_dir=DENV_ROOT / "classifier_exploration/outputs/"
        "pooled_denv_pal40_phase3d_joint_tied_qc_nonblank_5dataset/"
        "direct_pal_lineage_native30_same_cohort_alignment",
        teacher_paths={
            "SEC61B": DENV_ROOT / "classifier_exploration/outputs/"
            "pooled_denv_pal40_phase3d_joint_tied_qc_nonblank_5dataset/"
            "phase_sec61b_concat_04_14_04_28_08_11/teacher_manifest.parquet",
        },
    ),
}


def _zikv_external_labels(marker: str) -> pd.DataFrame:
    roots = {
        "07_01_SEC61B_TOMM20_G3BP1": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
            "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/2-phenotyping/predictions/"
            "DynaCLR-2D-MIP-BagOfChannels/"
            "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
            "epoch105-step84800/gmm_pooled/labels"
        ),
        "03_24_SEC61B": ZIKV_ROOT / "labels",
        "04_08_G3BP1": Path(
            "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
            "2026_04_08_A549_G3BP1_ZIKV/2-phenotyping/predictions/"
            "DynaCLR-2D-MIP-BagOfChannels/"
            "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
            "epoch105-step84800/labels"
        ),
    }
    datasets = (
        (
            ("07_01_SEC61B_TOMM20_G3BP1", roots["07_01_SEC61B_TOMM20_G3BP1"]),
            ("03_24_SEC61B", roots["03_24_SEC61B"]),
        )
        if marker == "SEC61B"
        else (
            ("07_01_SEC61B_TOMM20_G3BP1", roots["07_01_SEC61B_TOMM20_G3BP1"]),
            ("04_08_G3BP1", roots["04_08_G3BP1"]),
        )
    )
    parts = []
    for dataset, root in datasets:
        frame = pd.read_csv(
            root / f"{marker}_organelle_remodeling_state.csv",
            usecols=["fov_name", "track_id", "t", "organelle_remodeling_state"],
        )
        frame.insert(0, "dataset", dataset)
        frame["teacher_state"] = frame["organelle_remodeling_state"].eq("remodel").astype(float)
        parts.append(frame[["dataset", "fov_name", "track_id", "t", "teacher_state"]])
    return pd.concat(parts, ignore_index=True)


def _load_teacher(spec: VirusSpec, marker: str) -> tuple[pd.DataFrame | None, str]:
    path = spec.teacher_paths.get(marker)
    if path is None:
        return None, "no frozen remodeling teacher"
    if spec.virus == "ZIKV" and marker in {"SEC61B", "G3BP1"}:
        return _zikv_external_labels(marker), "frozen organelle remodeling CSV"
    frame = pd.read_parquet(path)
    if spec.virus == "ZIKV" and marker == "TOMM20":
        frame["dataset"] = frame["dataset"].replace({"07_01_TOMM20": "07_01_SEC61B_TOMM20_G3BP1"})
        column = "teacher_state"
        source = "exploratory pooled TOMM20 witness/GMM teacher"
    else:
        column = "teacher__remodeling"
        source = "frozen pooled SEC61B witness/GMM teacher"
    return (
        frame[["dataset", "fov_name", "track_id", "t", column]].rename(columns={column: "teacher_state"}),
        source,
    )


def _lineage_rows(
    manifest: pd.DataFrame,
    condition: str,
    tau_min: float,
    tau_max: float,
) -> np.ndarray:
    frame = manifest.loc[
        manifest["perturbation"].astype(str).eq(condition)
        & manifest["tau"].notna()
        & manifest["tau"].between(tau_min, tau_max)
    ].copy()
    frame["tau_bin"] = np.round(frame["tau"].to_numpy(float) * 2.0) / 2.0
    frame = frame.sort_values(
        ["dataset", "fov_name", "lineage_id", "tau_bin", "lineage_depth", "track_id"],
        kind="stable",
    )
    return frame.drop_duplicates(["dataset", "fov_name", "lineage_id", "tau_bin"], keep="first").index.to_numpy(int)


def _balanced_fit_rows(table: pd.DataFrame, per_dataset_tau: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse_tau = np.floor(table["tau"].to_numpy(float) / 2.0) * 2.0
    selected = []
    for (_, _), indices in (
        table.assign(_coarse_tau=coarse_tau)
        .groupby(["dataset", "_coarse_tau"], observed=True, sort=True)
        .indices.items()
    ):
        rows = np.asarray(indices, dtype=int)
        selected.append(rng.choice(rows, min(per_dataset_tau, len(rows)), replace=False))
    return np.concatenate(selected)


def _fit_umap(features: np.ndarray, fit_rows: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    model = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_seed,
        transform_seed=args.random_seed,
        n_jobs=1,
    )
    model.fit(features[fit_rows])
    return np.asarray(model.transform(features), dtype=np.float32)


def _remodeling_summary(table: pd.DataFrame) -> pd.DataFrame:
    per_dataset = (
        table.assign(
            is_remodeled=table["teacher_state"].eq(1.0),
            is_confident=table["teacher_state"].notna(),
        )
        .groupby(["dataset", "tau_bin"], observed=True)
        .agg(
            n_lineages=("lineage_uid", "size"),
            n_confident=("is_confident", "sum"),
            n_remodeled=("is_remodeled", "sum"),
            fraction_remodeled=("is_remodeled", "mean"),
        )
        .reset_index()
    )
    summary = (
        per_dataset.groupby("tau_bin", observed=True)
        .agg(
            fraction_remodeled=("fraction_remodeled", "mean"),
            n_datasets=("dataset", "nunique"),
            n_lineages=("n_lineages", "sum"),
            n_confident=("n_confident", "sum"),
            n_remodeled=("n_remodeled", "sum"),
        )
        .reset_index()
    )
    summary["confident_coverage"] = summary["n_confident"] / summary["n_lineages"]
    return summary


def _scatter(axis: plt.Axes, table: pd.DataFrame, color, **kwargs) -> object:
    return axis.scatter(
        table["umap_1"],
        table["umap_2"],
        c=color,
        s=2.2,
        alpha=0.52,
        linewidths=0,
        rasterized=True,
        **kwargs,
    )


def _finish_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")


def run(args: argparse.Namespace) -> None:
    spec = SPECS[args.virus]
    input_dir = args.input_dir or spec.input_dir
    output_dir = args.output_dir or input_dir / "organelle_umap_hpi_tau_remodeling"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    tables: dict[str, pd.DataFrame] = {}
    summaries = []
    contracts = []
    for marker_index, marker in enumerate(MARKERS):
        stem = MARKER_STEMS[marker]
        manifest = pd.read_parquet(input_dir / f"{stem}_cell_manifest.parquet")
        scores = np.load(input_dir / f"{stem}_pca80_scores.npy", mmap_mode="r")
        rows = _lineage_rows(
            manifest,
            spec.condition,
            args.tau_min,
            args.tau_max,
        )
        table = manifest.loc[rows].copy().reset_index(drop=True)
        features = np.asarray(scores[manifest.loc[rows, "pc_row"].to_numpy(int)], dtype=np.float32)
        table["tau_bin"] = np.round(table["tau"].to_numpy(float) * 2.0) / 2.0
        fit_rows = _balanced_fit_rows(
            table,
            args.fit_per_dataset_tau,
            args.random_seed + marker_index,
        )
        coordinates = _fit_umap(features, fit_rows, args)
        table["umap_1"] = coordinates[:, 0]
        table["umap_2"] = coordinates[:, 1]
        table["umap_fit_row"] = False
        table.loc[fit_rows, "umap_fit_row"] = True

        teacher, source = _load_teacher(spec, marker)
        if teacher is not None:
            keys = ["dataset", "fov_name", "track_id", "t"]
            if teacher.duplicated(keys).any():
                raise ValueError(f"{marker}: duplicate teacher keys")
            table = table.merge(teacher, on=keys, how="left", validate="one_to_one")
            summary = _remodeling_summary(table)
            summary.insert(0, "marker", marker)
            summaries.append(summary)
            table = table.merge(
                summary[["tau_bin", "fraction_remodeled"]],
                on="tau_bin",
                how="left",
                validate="many_to_one",
            )
        else:
            table["teacher_state"] = np.nan
            table["fraction_remodeled"] = np.nan
        table.insert(0, "virus", spec.virus)
        table.to_parquet(output_dir / f"{stem}_umap_cells.parquet", index=False)
        tables[marker] = table
        contracts.append(
            {
                "marker": marker,
                "n_pca_components": int(features.shape[1]),
                "n_lineage_timepoints": int(len(table)),
                "n_lineages": int(table["lineage_uid"].nunique()),
                "n_datasets": int(table["dataset"].nunique()),
                "n_umap_fit_rows": int(len(fit_rows)),
                "teacher_source": source,
                "teacher_coverage": float(table["teacher_state"].notna().mean()),
            }
        )

    summary_table = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary_table.to_csv(output_dir / "remodeling_frequency_by_tau.csv", index=False)
    pd.DataFrame(contracts).to_csv(output_dir / "umap_contracts.csv", index=False)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12.5), constrained_layout=True)
    hpi_values = np.concatenate([table["hours_post_perturbation"].to_numpy(float) for table in tables.values()])
    tau_values = np.concatenate([table["tau"].to_numpy(float) for table in tables.values()])
    hpi_norm = Normalize(vmin=float(np.nanmin(hpi_values)), vmax=float(np.nanmax(hpi_values)))
    tau_limit = float(np.nanmax(np.abs(np.quantile(tau_values, [0.01, 0.99]))))
    tau_norm = TwoSlopeNorm(vmin=-tau_limit, vcenter=0.0, vmax=tau_limit)

    for row, marker in enumerate(MARKERS):
        table = tables[marker]
        datasets = sorted(table["dataset"].astype(str).unique())
        color_map = {dataset: DATASET_COLORS[i] for i, dataset in enumerate(datasets)}
        order = np.random.default_rng(args.random_seed + row).permutation(len(table))
        plotted = table.iloc[order]
        _scatter(axes[row, 0], plotted, plotted["dataset"].map(color_map))
        axes[row, 0].legend(
            handles=[Line2D([0], [0], marker="o", ls="", color=color_map[d], label=d, ms=4) for d in datasets],
            frameon=False,
            fontsize=7,
            loc="best",
        )
        _finish_axis(axes[row, 0], f"{marker} · dataset")

        hpi_artist = _scatter(axes[row, 1], plotted, plotted["hours_post_perturbation"], cmap="viridis", norm=hpi_norm)
        fig.colorbar(hpi_artist, ax=axes[row, 1], label="HPI", shrink=0.78)
        _finish_axis(axes[row, 1], f"{marker} · physical time")

        tau_artist = _scatter(axes[row, 2], plotted, plotted["tau"], cmap="coolwarm", norm=tau_norm)
        fig.colorbar(tau_artist, ax=axes[row, 2], label="τ (hours)", shrink=0.78)
        _finish_axis(axes[row, 2], f"{marker} · pAL-aligned pseudotime")

        if plotted["fraction_remodeled"].notna().any():
            rem_artist = _scatter(
                axes[row, 3], plotted, 100 * plotted["fraction_remodeled"], cmap="magma", vmin=0, vmax=100
            )
            fig.colorbar(rem_artist, ax=axes[row, 3], label="Remodeled families (%)", shrink=0.78)
            source = next(item["teacher_source"] for item in contracts if item["marker"] == marker)
            _finish_axis(axes[row, 3], f"{marker} · τ-bin remodeling frequency\n{source}")
        else:
            axes[row, 3].text(
                0.5,
                0.5,
                "No frozen remodeling teacher\nHPI and τ coordinates remain available",
                ha="center",
                va="center",
                transform=axes[row, 3].transAxes,
                fontsize=11,
                color="0.3",
            )
            _finish_axis(axes[row, 3], f"{marker} · remodeling frequency unavailable")

    fig.suptitle(
        f"{spec.virus} organelle PCA80 UMAPs on the same onset-bearing lineage-family cohort\n"
        "One shared UMAP per organelle; coordinates reused for dataset, HPI, τ, and remodeling color",
        fontsize=15,
    )
    stem = output_dir / f"{args.virus}_organelle_umap_hpi_tau_remodeling"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    info = {
        "virus": spec.virus,
        "input_dir": str(input_dir),
        "cohort": (
            "perturbed, finite direct-pAL tau, one lineage family per 30-minute tau bin, "
            f"tau in [{args.tau_min}, {args.tau_max}] hours"
        ),
        "representation": "marker-specific control-centered, MAD-scaled PCA80 scores",
        "shared_umap_definition": "one pooled-across-datasets UMAP per marker, reused across colorings",
        "frequency_definition": "dataset-balanced mean of lineage-family teacher-positive fractions; ambiguous/unlabeled are non-positive",
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "fit_per_dataset_2h_tau_bin": args.fit_per_dataset_tau,
            "random_seed": args.random_seed,
        },
        "contracts": contracts,
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(stem.with_suffix(".pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--virus", choices=tuple(SPECS), required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fit-per-dataset-tau", type=int, default=500)
    parser.add_argument("--tau-min", type=float, default=-10.0)
    parser.add_argument("--tau-max", type=float, default=12.0)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
