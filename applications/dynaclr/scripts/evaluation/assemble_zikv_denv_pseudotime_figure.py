"""Assemble the ZIKV/DENV direct-pAL-alignment paper figure.

The figure is composed from frozen analysis tables and six curated lineage
families. Raw images are read only to render the already-selected 221x221
lineage crops; no embedding, normalization, PCA, witness, MMD, or classifier
calculation is repeated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_zikv_pal_aligned_3h_marker_pdfs as montage

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


ZIKV_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_03_24_A549_SEC61_ZIKV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)
DENV_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_04_28_A549_SEC61B_DENV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_denv_pal40_phase3d_joint_tied_qc_nonblank_5dataset/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)
DENV_CURATED_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2026_04_28_A549_SEC61B_DENV/2-phenotyping/predictions/"
    "DynaCLR-2D-MIP-BagOfChannels/"
    "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/"
    "epoch105-step84800/classifier_exploration/outputs/"
    "pooled_denv_pal40_phase3d_joint_tied_qc_nonblank/"
    "direct_pal_lineage_native30_same_cohort_alignment"
)

COLORS = {"SEC61B": "#0072B2", "TOMM20": "#009E73", "G3BP1": "#CC79A7"}
FINAL_WIDTH_IN = 7.1
FINAL_HEIGHT_IN = 7.5
CURATED = {
    ("ZIKV", "SEC61B"): 8,
    ("ZIKV", "TOMM20"): 10,
    ("ZIKV", "G3BP1"): 10,
    ("DENV", "SEC61B"): 8,
    ("DENV", "TOMM20"): 18,
    ("DENV", "G3BP1"): 27,
}
PHASE_CONTRAST_OVERRIDES = {
    # Curated DENV SEC61B #8 has a narrower Phase3D intensity distribution
    # than the other selected lineages. Keep one fixed scale over all seven
    # frames, but tighten its display percentiles to reveal cell texture.
    ("DENV", "SEC61B", 8): (5.0, 99.2),
}
STAGES = tuple(f"tau_{tau:+03d}" for tau in range(-9, 10, 3))
STAGE_LABELS = tuple(f"{tau:+d} h" if tau else "0 h" for tau in range(-9, 10, 3))


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(labelsize=6, width=0.6, length=2.5, pad=1.5)
    ax.grid(alpha=0.15, linewidth=0.5)


def _alignment_cartoon(ax: plt.Axes) -> None:
    ax.set_axis_off()
    x = np.linspace(0, 1, 180)
    centers = (0.31, 0.55, 0.77)
    offsets = (0.70, 0.43, 0.16)
    for center, offset in zip(centers, offsets, strict=True):
        y = offset + 0.18 * np.exp(-0.5 * ((x - center) / 0.11) ** 2)
        ax.plot(0.08 + 0.80 * x, 0.55 + 0.40 * y, color="black", lw=1.0)
    ax.annotate("", (0.91, 0.57), (0.06, 0.57), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.text(0.48, 0.53, "physical time", ha="center", va="top", fontsize=6.5)
    ax.text(0.01, 0.79, "cell-specific\nonset times", rotation=90, ha="center", va="center", fontsize=6.5)
    ax.annotate("", (0.50, 0.42), (0.50, 0.54), arrowprops={"arrowstyle": "->", "lw": 1.0})
    for offset in offsets:
        y = offset + 0.18 * np.exp(-0.5 * ((x - 0.50) / 0.11) ** 2)
        ax.plot(0.08 + 0.80 * x, 0.02 + 0.40 * y, color="black", lw=1.0)
    ax.axvline(0.48, ymin=0.03, ymax=0.38, color="#D55E00", ls="--", lw=1.0)
    ax.annotate("", (0.91, 0.04), (0.06, 0.04), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.text(0.48, 0.00, "pAL-aligned pseudotime", ha="center", va="top", fontsize=6.5)
    ax.text(0.51, 0.25, "pAL onset", color="#D55E00", fontsize=6.5, va="center")


def _alignment_well_curves(root: Path) -> pd.DataFrame:
    """Recompute display-only pAL summaries for one line per physical well."""
    parts = []
    for marker in COLORS:
        frame = pd.read_parquet(
            root / f"{marker.lower()}_cell_manifest.parquet",
            columns=[
                "dataset",
                "marker",
                "fov_name",
                "lineage_uid",
                "t",
                "hours_post_perturbation",
                "tau",
                "gmm_positive_posterior",
                "pal_onset_hpi",
                "perturbation",
            ],
        )
        frame = frame.loc[frame["perturbation"].astype(str).eq("ZIKV") & frame["pal_onset_hpi"].notna()].copy()
        frame["well"] = frame["fov_name"].astype(str).str.split("/").str[:2].str.join("/")
        # A lineage family contributes once per time point even after division.
        states = (
            frame.groupby(
                ["dataset", "marker", "well", "lineage_uid", "t"],
                sort=False,
                observed=True,
            )
            .agg(
                hpi=("hours_post_perturbation", "median"),
                tau=("tau", "median"),
                posterior=("gmm_positive_posterior", "median"),
            )
            .reset_index()
        )
        parts.append(states)
    states = pd.concat(parts, ignore_index=True)
    summaries = []
    for axis_name in ("hpi", "tau"):
        working = states.loc[np.isfinite(states[axis_name])].copy()
        working["axis_value"] = np.round(working[axis_name].to_numpy(float) * 2.0) / 2.0
        summary = (
            working.groupby(
                ["dataset", "marker", "well", "axis_value"],
                sort=True,
                observed=True,
            )["posterior"]
            .agg(
                posterior_q25=lambda x: float(np.quantile(x, 0.25)),
                posterior_median="median",
                posterior_q75=lambda x: float(np.quantile(x, 0.75)),
                n_lineages="size",
            )
            .reset_index()
        )
        summary["axis"] = axis_name
        summary["well_uid"] = summary["dataset"].astype(str) + "::" + summary["well"].astype(str)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def _alignment_qc(
    ax: plt.Axes,
    curves: pd.DataFrame,
    well_colors: dict[str, tuple[float, float, float, float]],
    axis_name: str,
) -> None:
    curves = curves.loc[curves["axis"].eq(axis_name)]
    for well_uid, group in curves.groupby("well_uid", sort=True):
        group = group.sort_values("axis_value")
        color = well_colors[str(well_uid)]
        x = group["axis_value"].to_numpy(float)
        med = group["posterior_median"].to_numpy(float)
        lo = group["posterior_q25"].to_numpy(float)
        hi = group["posterior_q75"].to_numpy(float)
        ax.plot(x, med, color=color, lw=0.9, alpha=0.85)
        ax.fill_between(x, lo, hi, color=color, alpha=0.07, linewidth=0)
    ax.axhline(0.8, color="0.45", ls=":", lw=0.7)
    if axis_name == "tau":
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_xlim(-10.5, 12.5)
        ax.set_xlabel("τ (h)", fontsize=6.5, labelpad=1)
        ax.set_title("After pAL alignment · same wells", fontsize=7.5, pad=2)
    else:
        ax.set_xlabel("HPI (h)", fontsize=6.5, labelpad=1)
        ax.set_title("Before alignment · one color per well", fontsize=7.5, pad=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("direct-pAL\nposterior", fontsize=6.5, labelpad=2)
    _style_axis(ax)


def _mmd_panel(ax: plt.Axes, virus: str, root: Path, scope: str) -> None:
    curves = pd.read_csv(root / "mmd_curves.csv")
    curves = curves.loc[curves["axis"].eq("tau") & curves["scope"].eq(scope)]
    for marker, group in curves.groupby("marker", sort=False):
        group = group.sort_values("axis_center")
        x = group["axis_center"].to_numpy(float)
        y = group["subsample_median"].to_numpy(float)
        lo = group["subsample_q025"].to_numpy(float)
        hi = group["subsample_q975"].to_numpy(float)
        ax.plot(x, y, color=COLORS[str(marker)], lw=1.0, label=str(marker))
        ax.fill_between(x, lo, hi, color=COLORS[str(marker)], alpha=0.16, linewidth=0)
    ax.axhline(0, color="0.45", lw=0.7)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlim(-10.5, 12.5)
    ax.set_xlabel("")
    ax.set_ylabel("null-standardized\nMMD²" if virus == "ZIKV" else "", fontsize=6.5, labelpad=2)
    ax.set_title(f"{virus} · MMD", fontsize=8, weight="bold", pad=3)
    ax.set_box_aspect(1)
    _style_axis(ax)
    pal = pd.read_csv(root / f"{virus.lower()}_unaligned_vs_pal_aligned_pal_overlay.csv")
    pal = pal.loc[pal["axis"].eq("tau") & pal["scope"].eq(scope)]
    pal = pal.groupby("axis_value", as_index=False)["posterior_median"].median().sort_values("axis_value")
    twin = ax.twinx()
    twin.plot(pal["axis_value"], pal["posterior_median"], color="0.2", ls=":", lw=0.9)
    twin.set_ylim(-0.03, 1.03)
    if virus == "DENV":
        twin.set_ylabel("direct-pAL\nposterior", fontsize=6, color="0.3", labelpad=2)
        twin.tick_params(axis="y", labelsize=6, colors="0.3", width=0.5, length=2, pad=1.5)
        twin.spines["right"].set_linewidth(0.75)
    else:
        twin.set_yticklabels([])
        twin.tick_params(axis="y", width=0, length=0)
        twin.spines["right"].set_visible(False)
    twin.spines["top"].set_visible(False)


def _frequency_panel(ax: plt.Axes, virus: str, root: Path) -> None:
    if virus == "ZIKV":
        frame = pd.read_csv(root / "sec61b_g3bp1_teacher_remodeled_frequency_hpi_tau.csv")
        frame = frame.loc[frame["axis"].eq("tau") & frame["condition"].eq("ZIKV")]
        for marker, group in frame.groupby("marker", sort=False):
            group = group.sort_values("axis_center")
            x = group["axis_center"].to_numpy(float)
            y = 100 * group["fraction_remodeled"].to_numpy(float)
            lo = 100 * group["bootstrap_q025"].to_numpy(float)
            hi = 100 * group["bootstrap_q975"].to_numpy(float)
            ax.plot(x, y, color=COLORS[str(marker)], lw=1.0, label=str(marker))
            ax.fill_between(x, lo, hi, color=COLORS[str(marker)], alpha=0.16, linewidth=0)
    else:
        frame = pd.read_csv(root / "denv_sec61b_g3bp1_teacher_remodeled_frequency_hpi_tau.csv")
        frame = frame.loc[frame["axis"].eq("tau") & frame["condition"].eq("DENV")]
        for marker, group in frame.groupby("marker", sort=False):
            group = group.sort_values("axis_center")
            x = group["axis_center"].to_numpy(float)
            y = 100 * group["fraction_remodeled"].to_numpy(float)
            lo = 100 * group["bootstrap_q025"].to_numpy(float)
            hi = 100 * group["bootstrap_q975"].to_numpy(float)
            ax.plot(x, y, color=COLORS[str(marker)], lw=1.0, label=str(marker))
            ax.fill_between(x, lo, hi, color=COLORS[str(marker)], alpha=0.16, linewidth=0)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlim(-10.5, 12.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("τ (h)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("remodeled\nfamilies (%)" if virus == "ZIKV" else "", fontsize=6.5, labelpad=2)
    ax.set_title(f"{virus} · remodeled frequency", fontsize=7.5, pad=2)
    ax.set_box_aspect(1)
    _style_axis(ax)


def _curated_rows() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (virus, marker), sample_id in CURATED.items():
        root = ZIKV_ROOT if virus == "ZIKV" else DENV_CURATED_ROOT
        path = root / f"{virus.lower()}_{marker.lower()}_pal_aligned_3h_lineage_cell_montage_manifest.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["sample_id"].eq(sample_id)].copy()
        if set(frame["stage"]) != set(STAGES):
            raise ValueError(f"Curated lineage {virus} {marker} #{sample_id} lacks a complete seven-stage strip")
        frame["virus"] = virus
        frame["curated_number"] = sample_id
        parts.append(frame)
    selected = pd.concat(parts, ignore_index=True)
    # Candidate numbers are sheet-local (for example, ZIKV TOMM20 #10 and
    # G3BP1 #10 are different lineages). Give each curated lineage a unique
    # internal key so contrast limits never mix across sheets or viruses.
    render_keys = selected[["virus", "marker", "curated_number"]].drop_duplicates()
    render_keys["render_sample_id"] = np.arange(1, len(render_keys) + 1, dtype=int)
    selected = selected.merge(
        render_keys,
        on=["virus", "marker", "curated_number"],
        how="left",
        validate="many_to_one",
    )
    selected["manifest_sample_id"] = selected["sample_id"]
    selected["sample_id"] = selected["render_sample_id"]
    selected["source_index"] = np.arange(len(selected), dtype=int)
    return selected.set_index("source_index", drop=False)


def _render_cell_strips(
    fig: plt.Figure,
    grid,
    selected: pd.DataFrame,
    crops: dict[int, np.ndarray],
    limits: dict[tuple[int, int], tuple[float, float]],
) -> None:
    row_order = [
        ("ZIKV", "SEC61B"),
        ("ZIKV", "TOMM20"),
        ("ZIKV", "G3BP1"),
        ("DENV", "SEC61B"),
        ("DENV", "TOMM20"),
        ("DENV", "G3BP1"),
    ]
    for row_index, (virus, marker) in enumerate(row_order):
        local = selected.loc[selected["virus"].eq(virus) & selected["marker"].eq(marker)]
        sample_id = int(local["sample_id"].iloc[0])
        curated_number = int(local["curated_number"].iloc[0])
        lookup = local.set_index("stage")
        for stage_index, stage in enumerate(STAGES):
            row = lookup.loc[stage]
            crop = crops[int(row["source_index"])]
            pal_low, pal_high = limits[(sample_id, 0)]
            org_low, org_high = limits[(sample_id, 1)]
            phase_low, phase_high = limits[(sample_id, 2)]
            pal = np.clip((crop[0] - pal_low) / (pal_high - pal_low), 0, 1)
            org = np.clip((crop[1] - org_low) / (org_high - org_low), 0, 1)
            overlay = np.stack([pal, org, pal], axis=-1)
            for view in range(2):
                ax = fig.add_subplot(grid[row_index, stage_index * 2 + view])
                if view == 0:
                    ax.imshow(overlay, interpolation="nearest")
                else:
                    ax.imshow(crop[2], cmap="gray", vmin=phase_low, vmax=phase_high, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("0.35")
                if row_index == 0:
                    ax.set_title(
                        (STAGE_LABELS[stage_index] + "\noverlay") if view == 0 else "Phase3D", fontsize=6, pad=2
                    )
                if stage_index == 0 and view == 0:
                    dataset = str(row["dataset"]).split("_")[0:2]
                    ax.set_ylabel(
                        f"{virus}\n{marker}\n#{curated_number} · {'_'.join(dataset)}",
                        fontsize=6.2,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=22,
                    )
        if row_index == 2:
            for col in range(14):
                ax = fig.axes[-14 + col]
                ax.spines["bottom"].set_linewidth(1.0)
                ax.spines["bottom"].set_color("white")


def run(args: argparse.Namespace) -> None:
    selected = _curated_rows()
    crops = montage._extract_crops(selected, crop_size=221, z_window=16)
    limits = montage._display_limits(
        selected,
        crops,
        lower_percentile=1.0,
        upper_percentile=99.7,
        pal_lower_percentile=10.0,
        pal_upper_percentile=99.5,
        organelle_upper_percentile=99.2,
    )
    for key, (lower_percentile, upper_percentile) in PHASE_CONTRAST_OVERRIDES.items():
        virus, marker, curated_number = key
        rows = selected.loc[
            selected["virus"].eq(virus) & selected["marker"].eq(marker) & selected["curated_number"].eq(curated_number)
        ]
        if rows.empty:
            raise ValueError(f"Missing curated Phase3D contrast target: {key}")
        sample_id = int(rows["sample_id"].iloc[0])
        phase_values = np.concatenate([crops[int(index)][2].ravel() for index in rows.index])
        low, high = np.nanpercentile(phase_values, [lower_percentile, upper_percentile])
        limits[(sample_id, 2)] = (
            float(low),
            max(float(high), float(low) + 1e-6),
        )

    # Render at the intended two-column manuscript width. Font and line sizes
    # are therefore the sizes seen after placement; no downstream scaling from
    # the earlier 17.8-inch composition canvas is required.
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, FINAL_HEIGHT_IN), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.52, 0.48], hspace=0.20)
    top = outer[0].subgridspec(2, 4, width_ratios=[0.82, 1.22, 1.0, 1.0], hspace=0.52, wspace=0.48)

    cartoon = fig.add_subplot(top[:, 0])
    _alignment_cartoon(cartoon)
    cartoon.text(-0.10, 1.04, "a", transform=cartoon.transAxes, fontsize=8, weight="bold", va="top")
    before = fig.add_subplot(top[0, 1])
    after = fig.add_subplot(top[1, 1])
    alignment_curves = _alignment_well_curves(ZIKV_ROOT)
    well_ids = sorted(alignment_curves["well_uid"].astype(str).unique())
    palette = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(well_ids), endpoint=True))
    well_colors = dict(zip(well_ids, palette, strict=True))
    _alignment_qc(before, alignment_curves, well_colors, "hpi")
    _alignment_qc(after, alignment_curves, well_colors, "tau")

    zikv_mmd = fig.add_subplot(top[0, 2])
    denv_mmd = fig.add_subplot(top[0, 3])
    zikv_freq = fig.add_subplot(top[1, 2])
    denv_freq = fig.add_subplot(top[1, 3])
    _mmd_panel(zikv_mmd, "ZIKV", ZIKV_ROOT, "replicate_augmented")
    _mmd_panel(denv_mmd, "DENV", DENV_ROOT, "marker_available_dates")
    _frequency_panel(zikv_freq, "ZIKV", ZIKV_ROOT)
    _frequency_panel(denv_freq, "DENV", DENV_ROOT)
    zikv_mmd.text(-0.22, 1.23, "b", transform=zikv_mmd.transAxes, fontsize=8, weight="bold", va="top")
    zikv_mmd.legend(
        frameon=False,
        fontsize=6,
        ncol=1,
        loc="upper left",
        handlelength=1.4,
    )

    bottom = outer[1].subgridspec(6, 14, hspace=0.06, wspace=0.025)
    _render_cell_strips(fig, bottom, selected, crops, limits)
    fig.text(0.006, 0.565, "c", fontsize=8, weight="bold", va="top")
    fig.text(
        0.51,
        0.025,
        "pAL alignment (τ = 0); paired 221×221 crops every 3 h · pAL magenta + organelle green; Phase3D grayscale",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.115, right=0.975, top=0.975, bottom=0.055)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    manifest_columns = [
        "virus",
        "marker",
        "curated_number",
        "manifest_sample_id",
        "render_sample_id",
        "dataset",
        "fov_name",
        "lineage_id",
        "stage",
        "target_tau",
        "t",
        "hours_post_perturbation",
        "tau",
        "gmm_positive_posterior",
        "raw_path",
        "source_index",
    ]
    selected.reset_index(drop=True)[manifest_columns].to_csv(
        args.output.with_name(args.output.name + "_curated_lineages.csv"), index=False
    )
    provenance = {
        "figure": "ZIKV/DENV direct-pAL-aligned organelle dynamics",
        "output_stem": str(args.output),
        "curated": {f"{virus}::{marker}": sample for (virus, marker), sample in CURATED.items()},
        "analysis_roots": {"ZIKV": str(ZIKV_ROOT), "DENV": str(DENV_ROOT)},
        "curated_montage_roots": {
            "ZIKV": str(ZIKV_ROOT),
            "DENV": str(DENV_CURATED_ROOT),
        },
        "panel_a": (
            "ZIKV marker cell manifests; display-only lineage-family summaries; "
            "one uniquely colored trajectory per dataset-well before and after alignment"
        ),
        "panel_b_mmd": {
            "ZIKV": "mmd_curves.csv scope=replicate_augmented axis=tau",
            "DENV": "mmd_curves.csv scope=marker_available_dates axis=tau",
        },
        "panel_b_frequency": {
            "ZIKV": "SEC61B and G3BP1 frozen witness/GMM teachers",
            "DENV": "SEC61B and G3BP1 frozen witness/GMM teachers",
        },
        "panel_c": {
            "crop_pixels": 221,
            "fluorescence": "16-slice maximum-intensity projection",
            "phase": "single Phase3D focus plane",
            "timepoints_hours": list(range(-9, 10, 3)),
            "contrast": "fixed within each lineage/channel over all seven timepoints",
            "phase_contrast_overrides": {"DENV::SEC61B::8": "fixed seven-frame 5.0–99.2 percentile scale"},
        },
        "typography": {
            "font": "Arial",
            "text_size_points": "6–8",
            "primary_stroke_width_points": 1.0,
            "supporting_stroke_width_points": "0.5–0.9",
            "physical_canvas_inches": [FINAL_WIDTH_IN, FINAL_HEIGHT_IN],
            "placement_scale_required": 1.0,
            "pdf_fonttype": 42,
            "svg_text": "editable",
        },
        "dpi": args.dpi,
    }
    args.output.with_name(args.output.name + "_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".svg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ZIKV_ROOT
        / "paper_figure_pseudotime"
        / "zikv_denv_pseudotime_figure_v8_well_colors_square_panels_denv5",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
