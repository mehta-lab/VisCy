"""Figure 6 attempt 12: discrete Phase-head labels with best-validation PCNA RoPE4/64.

Starts from attempt-09 scores and UMAP coordinates. Binary heads are thresholded
at 0.5; PCNA is the argmax of four states. TOMM20 neighbour enrichment, crops and
morphology are recomputed with the new cycle labels. No fitting.
"""

import json
import shutil
from pathlib import Path

import build_fig6_phase_prototype as prototype
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_fig6_phase_prototype import (
    ATLAS,
    COLORS,
    DATES,
    KEY,
    MITO,
    OBJ,
    STATES,
    crops,
    enrichment,
    sha,
    write_json,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

FIG = Path("/home/eduardo.hirata/repos/dynaclr-organelle-paper/organelle_remodeling_paper/figures/labelfree_infection")
SOURCE = FIG / "09_sec61b_phase_inference"
OUTPUT = FIG / "12_phase_labels_best_pcna"
BUNDLE = Path(
    "/hpc/projects/organelle_phenotyping/datasets/annotations/pcna_cell_cycle/checkpoints/"
    "PCNA_to_Phase/DynaCLR/2026-10-14_current_candidate_v1"
)
RELEASE = Path(
    "/hpc/projects/organelle_phenotyping/datasets/annotations/pcna_cell_cycle/evals/"
    "2026-10-14_sookie_partial_v2/population_transfer/tomm20_retry2"
)
ITERATIONS = Path("/hpc/projects/organelle_phenotyping/comparisons/pcna_cell_cycle/iterations")
ITER41 = ITERATIONS / "41_matched_hpi_and_tomm20_transfer/tomm20_retry2"
ITER49 = ITERATIONS / "49_fig6_cohort_best_phase_transfer"
PCNA_FILES = {
    DATES["04_10_TOMM20"]: RELEASE / f"{DATES['04_10_TOMM20']}_predictions.parquet",
    DATES["07_01_SEC61B_TOMM20_G3BP1"]: RELEASE / f"{DATES['07_01_SEC61B_TOMM20_G3BP1']}_predictions.parquet",
    DATES["04_30_TOMM20"]: ITER49 / f"{DATES['04_30_TOMM20']}_predictions.parquet",
}
THRESHOLD = 0.5
HEADS = {"pal": "pal_phase_score", "sec61b": "sec61b_phase_score", "g3bp1": "g3bp1_phase_score"}
EXP_OBJ = ["experiment"] + OBJ
CYCLE_PROBABILITIES = ["cycle_probability_" + s for s in STATES]
UMAPS = ["Phase3D_umap1", "Phase3D_umap2", "TOMM20_umap1", "TOMM20_umap2"]
CROP_COLUMNS = ["example_id", "selection_label", "cycle_state", "cycle_confidence", "fov_name", "track_id", "t"]
BINARY = {"positive": "#d55e00", "negative": "#a6bddb"}
UNAVAILABLE = "#e6e6e6"
SOURCES = {}


def record(p):
    """Hash an input file into ``SOURCES`` and return its path."""
    SOURCES[str(p)] = sha(p)
    return p


def read(p):
    """Read a recorded parquet input."""
    return pd.read_parquet(record(p))


def assemble():
    """Join best-model PCNA predictions onto the attempt-09 table and derive labels.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Joined cohort table and PCNA prediction-file provenance.
    """
    d = read(SOURCE / "source_data/observation_predictions.parquet")
    assert len(d) == 184514 and not d.duplicated(EXP_OBJ).any() and not d.duplicated(KEY + ["id"]).any()
    assert d.dataset.map(DATES).eq(d.experiment).all()
    frames = []
    files = {}
    for experiment, path in PCNA_FILES.items():
        files[experiment] = dict(path=str(path), sha256=sha(path))
        if path.parent == RELEASE:
            original = ITER41 / path.name
            assert sha(path) == sha(original), f"Release copy differs from iteration 41: {path.name}"
            files[experiment]["iteration41_original"] = str(original)
        p = read(path)
        assert p.experiment.eq(experiment).all()
        frames.append(p[EXP_OBJ + ["prediction", "real_context"] + ["probability_" + s for s in STATES]])
    p = pd.concat(frames, ignore_index=True)
    p["fov_name"] = p.fov_name.astype(str)
    d["fov_name"] = d.fov_name.astype(str)
    assert len(p) == len(d)
    d = d.merge(p, on=EXP_OBJ, how="left", validate="one_to_one")
    assert d.prediction.notna().all()
    d["cycle_state_tcn40_previous"] = d.cycle_state
    d["cycle_state"] = d.prediction
    for s in STATES:
        d["cycle_probability_" + s] = d["probability_" + s]
    d["cycle_confidence"] = d[CYCLE_PROBABILITIES].max(axis=1)
    assert (np.array(STATES)[d[CYCLE_PROBABILITIES].to_numpy().argmax(1)] == d.cycle_state.to_numpy()).all()
    d = d.drop(columns=["prediction", "distinct_context"] + ["probability_" + s for s in STATES])
    for head, col in HEADS.items():
        label = pd.Series(np.where(d[col] >= THRESHOLD, "positive", "negative"), index=d.index)
        d[head + "_label"] = label.where(d[col].notna())
    assert d.pal_label.notna().all() and d.sec61b_label.notna().all()
    assert d.g3bp1_label.notna().eq(d.dataset.eq("04_30_TOMM20")).all()
    return d, files


def label_summary(d, out):
    """Write per-dataset x perturbation label counts and fractions.

    Parameters
    ----------
    d : pandas.DataFrame
        Joined cohort table with label columns.
    out : pathlib.Path
        Attempt directory.

    Returns
    -------
    pandas.DataFrame
        The written summary, including ``all`` rows.
    """
    groups = list(d.groupby(["dataset", "perturbation"], observed=True))
    groups += [((k, "all"), g) for k, g in d.groupby("dataset")] + [(("all", "all"), d)]
    rows = []
    for (dataset, perturbation), g in groups:
        row = dict(dataset=dataset, perturbation=perturbation, observations=len(g))
        for head in HEADS:
            label = g[head + "_label"]
            available = int(label.notna().sum())
            positive = int(label.eq("positive").sum())
            row[head + "_available"] = available
            row[head + "_positive"] = positive
            row[head + "_positive_fraction"] = positive / available if available else np.nan
        for s in STATES:
            row[f"pcna_{s}_fraction"] = float(g.cycle_state.eq(s).mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "source_data/label_summary.csv", index=False)
    return summary


def render(d, neighbors, selection, images, out):
    """Render the main figure and supplement with discrete top-row labels.

    Parameters
    ----------
    d : pandas.DataFrame
        Joined cohort table.
    neighbors : pandas.DataFrame
        TOMM20 neighbour-enrichment summary.
    selection : pandas.DataFrame
        Selected crop rows.
    images : dict
        Paired Phase and TOMM20 crops keyed by example ID.
    out : pathlib.Path
        Attempt directory.

    Returns
    -------
    dict
        Display and morphology counts for verification.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    display = d.sample(min(18000, len(d)), random_state=17).copy()
    display = pd.concat([display, d.merge(selection[KEY + ["id"]], on=KEY + ["id"], validate="one_to_one")])
    display = display.drop_duplicates(KEY + ["id"])
    display.to_parquet(out / "source_data/display_rows.parquet", index=False)
    fig = plt.figure(figsize=(12, 12))
    fig.text(0.04, 0.98, "FIGURE 6 | Fluorescence-supervised readouts from label-free Phase", size=15, weight="bold")
    fig.text(
        0.04,
        0.959,
        "Exploratory prototype · 184,514 paired observations · transferred predictions are not biological ground truth",
        size=9,
        color="#555",
    )
    fig.text(
        0.04,
        0.923,
        "A   Fluorescence labels → paired Phase training → Phase-only inference → biological readouts",
        size=11,
        weight="bold",
    )
    fig.text(
        0.04,
        0.902,
        "pAL / SEC61B / G3BP1 / PCNA are supervision sources—not inference channels. No TOMM20 enters the Phase map.",
        size=9,
    )
    palettes = dict(zip(STATES, COLORS))
    limits = []
    maps = [
        (
            "pal_label",
            "pAL → Phase",
            "Saved leave-acquisition-out\nstatic heads · score ≥ 0.5",
            [("positive", "pAL+ (infected)"), ("negative", "pAL−")],
        ),
        (
            "sec61b_label",
            "SEC61B → Phase",
            "Saved causal TCN5 · score ≥ 0.5\nDENV: cross-virus extrapolation",
            [("positive", "Remodeled"), ("negative", "Not remodeled")],
        ),
        (
            "g3bp1_label",
            "G3BP1 → Phase",
            "DENV-only static head · score ≥ 0.5\nZIKV unavailable",
            [("positive", "Remodeled"), ("negative", "Not remodeled"), (None, "Unavailable")],
        ),
        (
            "cycle_state",
            "PCNA → Phase",
            "Best-validation RoPE4/64 · argmax\nunvalidated transfer",
            [(s, s) for s in STATES],
        ),
    ]
    xy = display[["Phase3D_umap1", "Phase3D_umap2"]].to_numpy()
    for j, (col, title, detail, entries) in enumerate(maps):
        palette = palettes if col == "cycle_state" else BINARY
        ax = fig.add_axes([0.045 + j * 0.24, 0.653, 0.205, 0.204])
        ok = display[col].notna().to_numpy()
        ax.scatter(*xy[~ok].T, c=UNAVAILABLE, s=1, rasterized=True)
        ax.scatter(
            *xy[ok].T,
            c=display.loc[ok, col].map(palette).to_numpy(),
            s=1,
            alpha=0.65 if col == "cycle_state" else 0.8,
            rasterized=True,
        )
        handles = [
            Line2D([], [], marker="o", ls="", color=palette[key] if key else UNAVAILABLE, label=label, markersize=4)
            for key, label in entries
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(entries),
            fontsize=7,
            frameon=False,
            handletextpad=0.1,
            columnspacing=0.4,
        )
        for _, r in selection.iterrows():
            ax.scatter(r.Phase3D_umap1, r.Phase3D_umap2, s=20, facecolors="none", edgecolors="black", linewidths=0.6)
            ax.annotate(
                r.example_id, (r.Phase3D_umap1, r.Phase3D_umap2), fontsize=5, xytext=(3, 3), textcoords="offset points"
            )
        ax.set(xticks=[], yticks=[], title=title)
        limits.append((ax.get_xlim(), ax.get_ylim()))
        fig.text(0.045 + j * 0.24, 0.882, ("B   " if j == 0 else "") + detail, size=7, va="bottom")
        if col == "cycle_state":
            line = f"Labeled: {d[col].notna().sum():,}/{len(d):,}"
        else:
            available = d[col].notna()
            line = f"Positive: {100 * d.loc[available, col].eq('positive').mean():.1f}% of {available.sum():,}"
        fig.text(0.045 + j * 0.24, 0.599, line, size=7.5)
    assert all(np.allclose(limits[0], v) for v in limits)
    fig.text(
        0.045,
        0.578,
        "Same Phase map and cells. Labels: binary heads at uncalibrated score ≥ 0.5; "
        "PCNA = most probable of four states. Gray = unavailable.",
        size=9,
    )
    display_limits = draw_panel_c(fig, selection, images)
    _, counts = draw_panel_d(fig, display, neighbors, d, out, ["uninfected", "ZIKV", "DENV"])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"main_phase_prototype.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    write_json(out / "source_data/display_limits.json", display_limits)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    supplement = [
        ("dataset", "Acquisition (same Phase coordinates)"),
        ("cycle_confidence", "Cycle maximum probability (uncalibrated)"),
        ("real_context", "Real context frames (RoPE4/64, ≤64)"),
    ]
    for ax, (col, title) in zip(axes, supplement):
        if col == "dataset":
            for name, g in display.groupby(col):
                ax.scatter(g.Phase3D_umap1, g.Phase3D_umap2, s=1, label=name, alpha=0.5, rasterized=True)
            ax.legend(fontsize=6)
        else:
            sc = ax.scatter(display.Phase3D_umap1, display.Phase3D_umap2, c=display[col], s=1, rasterized=True)
            fig.colorbar(sc, ax=ax)
        ax.set(title=title, xticks=[], yticks=[])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"supplement_phase_diagnostics.{ext}", dpi=220)
    plt.close(fig)
    return dict(display_rows=len(display), **counts)


def draw_panel_c(fig, selection, images):
    """Draw panel C: paired Phase / TOMM20 crops of the selected examples with fixed contrast.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Main figure.
    selection : pandas.DataFrame
        Selected crop rows (``example_id``, ``selection_label``, ``cycle_confidence``, ``pixel_um``).
    images : dict
        Paired Phase and TOMM20 crops keyed ``{example_id}_{phase|tomm20}``.

    Returns
    -------
    dict
        Display limits (1st and 99.5th percentile) per modality.
    """
    fig.text(0.045, 0.549, "C   Paired microscopy: Phase input and observed TOMM20 morphology", size=11, weight="bold")
    display_limits = {}
    for modality in ["phase", "tomm20"]:
        vals = np.concatenate([v.ravel() for k, v in images.items() if k.endswith("_" + modality)])
        display_limits[modality] = np.percentile(vals, [1, 99.5]).tolist()
    for j, (_, r) in enumerate(selection.iterrows()):
        for k, mod in enumerate(["phase", "tomm20"]):
            ax = fig.add_axes([0.065 + j * 0.182, 0.405 - k * 0.112, 0.14, 0.105])
            arr = images[r.example_id + "_" + mod]
            cmap = "gray" if mod == "phase" else LinearSegmentedColormap.from_list("mito", ["black", "#00ff66"])
            ax.imshow(arr, cmap=cmap, vmin=display_limits[mod][0], vmax=display_limits[mod][1])
            ax.axis("off")
            ax.plot([12, 12 + 5 / r.pixel_um], [210, 210], c="white", lw=2)
            if k == 0:
                ax.set_title(f"{r.example_id}: {r.selection_label}\nmax P={r.cycle_confidence:.2f}", size=8)
            else:
                ax.text(12, 194, "5 µm", color="white", size=6)
        if j == 0:
            fig.text(0.027, 0.44, "Phase", rotation=90, size=8)
            fig.text(0.027, 0.327, "TOMM20", rotation=90, size=8)
    fig.text(
        0.045,
        0.277,
        "Same cell/time per pair; controls from Apr10. Phase: focus slice. TOMM20: 16-slice MIP. Fixed contrast per channel.",
        size=8,
    )
    return display_limits


def draw_panel_d(fig, display, neighbors, d, out, conditions, enrichment_legend="best"):
    """Draw panel D: TOMM20 UMAP, TOMM20 neighbour enrichment of controls and FOV-median morphology.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Main figure.
    display : pandas.DataFrame
        Rows shown on the TOMM20 UMAP (``TOMM20_umap1/2``, ``cycle_state``).
    neighbors : pandas.DataFrame
        TOMM20 neighbour-enrichment summary.
    d : pandas.DataFrame
        Cohort table whose ``cycle_state`` replaces the morphology export's stored state.
    out : pathlib.Path
        Attempt directory; morphology tables are written to ``source_data``.
    conditions : list of str
        Morphology conditions, in plotting order.
    enrichment_legend : str, optional
        Legend location of the enrichment axes (matplotlib default ``"best"``).

    Returns
    -------
    tuple of (matplotlib.axes.Axes, dict)
        The TOMM20 UMAP axes and morphology counts.
    """
    palettes = dict(zip(STATES, COLORS))
    fig.text(
        0.045,
        0.249,
        "D   Fluorescence application: does TOMM20 structure associate with Phase-predicted cycle?",
        size=11,
        weight="bold",
    )
    umap_ax = fig.add_axes([0.045, 0.065, 0.25, 0.158])
    umap_ax.scatter(
        display.TOMM20_umap1, display.TOMM20_umap2, c=display.cycle_state.map(palettes), s=1, alpha=0.6, rasterized=True
    )
    umap_ax.set(xticks=[], yticks=[], title="TOMM20 UMAP (not label-free)")
    ax = fig.add_axes([0.36, 0.078, 0.26, 0.14])
    ax.axhline(0, c="#aaa", lw=1)
    for i, (ds, g) in enumerate(neighbors.groupby("dataset")):
        control = g[g.perturbation.eq("uninfected")].set_index("cycle_state").reindex(STATES)
        v = control.estimate.to_numpy()
        ok = control.supported.eq(True).to_numpy()
        ax.errorbar(
            np.arange(4) + (i - 1) * 0.12,
            np.where(ok, v, np.nan),
            yerr=np.vstack([v - control.low, control.high - v]),
            fmt="o",
            ms=3,
            label=ds[:5],
        )
    ax.set(xticks=range(4), xticklabels=STATES, ylabel="30-NN same-state excess", title="TOMM20 PCs · controls")
    ax.legend(fontsize=6, frameon=False, loc=enrichment_legend)
    morph = read(MITO / "03-tomm20_features_cycle_export_fix/all_joined_features.parquet")
    morph_rows = len(morph)
    # The stored export carries the superseded TCN40 state; use the new labels.
    morph = morph.drop(columns=["cycle_state", "cycle_confidence"]).merge(
        d[KEY + ["id", "cycle_state"]], on=KEY + ["id"], validate="one_to_one"
    )
    selected = morph[morph.segmentation_valid & np.isfinite(morph.length)]
    fov = (
        selected.groupby(["dataset", "perturbation", "fov_name", "cycle_state"], observed=True)
        .length.median()
        .reset_index()
    )
    fov.to_csv(out / "source_data/morphology_fov_medians.csv", index=False)
    ax = fig.add_axes([0.72, 0.078, 0.24, 0.14])
    rng = np.random.default_rng(17)
    mrows = []
    for i, condition in enumerate(conditions):
        yy = []
        lo = []
        hi = []
        for state in STATES:
            g = fov[fov.perturbation.eq(condition) & fov.cycle_state.eq(state)]
            values = g.length.to_numpy()
            if len(values) < 3:
                yy.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            val = np.median(values)
            boot = np.median(rng.choice(values, (500, len(values)), replace=True), axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            yy.append(val)
            lo.append(val - low)
            hi.append(high - val)
            mrows.append(dict(condition=condition, state=state, median=val, low=low, high=high, n_fovs=len(values)))
        ax.errorbar(np.arange(4) + (i - 1) * 0.15, yy, yerr=[lo, hi], fmt="o", ms=3, label=condition)
    pd.DataFrame(mrows).to_csv(out / "source_data/morphology_summary.csv", index=False)
    ax.set(xticks=range(4), xticklabels=STATES, ylabel="Network length (µm/crop)", title="Measured TOMM20 morphology")
    ax.legend(fontsize=6, frameon=False)
    fig.text(
        0.045,
        0.025,
        "Exploratory associations, not causal effects. FOV-bootstrap intervals are not biological-replicate uncertainty. "
        "Cycle transfer remains unvalidated.",
        size=8,
    )
    return umap_ax, dict(
        morphology_export_rows=morph_rows,
        morphology_rows_joined=len(morph),
        morphology_rows_selected=len(selected),
        morphology_fov_groups=len(fov),
        morphology_summary_rows=len(mrows),
    )


def verify(d, counts, selection, out):
    """Check the joined table and recomputed outputs against attempt 09.

    Parameters
    ----------
    d : pandas.DataFrame
        Joined cohort table.
    counts : dict
        Counts returned by ``render``.
    selection : pandas.DataFrame
        Selected crop rows.
    out : pathlib.Path
        Attempt directory.

    Returns
    -------
    dict
        Verification record.
    """
    source = pd.read_parquet(SOURCE / "source_data/observation_predictions.parquet")
    source["fov_name"] = source.fov_name.astype(str)
    scores = list(HEADS.values())
    m = source[EXP_OBJ + UMAPS + scores].merge(
        d[EXP_OBJ + UMAPS + scores], on=EXP_OBJ, suffixes=("_09", "_12"), validate="one_to_one"
    )
    same_umap = len(m) == len(d) and all(np.array_equal(m[c + "_09"], m[c + "_12"]) for c in UMAPS)
    same_scores = all(np.array_equal(m[c + "_09"], m[c + "_12"], equal_nan=True) for c in scores)
    same_order = source[EXP_OBJ].equals(d[EXP_OBJ])
    display = pd.read_parquet(out / "source_data/display_rows.parquet")
    old_display = pd.read_parquet(SOURCE / "source_data/display_rows.parquet")
    n = min(18000, len(d))
    cols = KEY + ["id", "Phase3D_umap1", "Phase3D_umap2"]
    same_sample = display[cols].head(n).reset_index(drop=True).equals(old_display[cols].head(n).reset_index(drop=True))
    rows = pd.read_parquet(out / "source_data/tomm20_neighbor_rows.parquet")
    check = rows[KEY + ["cycle_state"]].merge(
        d[KEY + ["cycle_state"]], on=KEY, suffixes=("", "_new"), validate="one_to_one"
    )
    agree = d.cycle_state.eq(d.cycle_state_tcn40_previous)
    zikv = d.dataset.ne("04_30_TOMM20")
    result = dict(
        rows=len(d),
        unique_keys=bool(not d.duplicated(EXP_OBJ).any() and not d.duplicated(KEY + ["id"]).any()),
        pcna_predictions_present=int(d.cycle_state.notna().sum()),
        all_pcna_predictions_present=bool(d.cycle_state.notna().all()),
        same_umap_coordinates_as_09=bool(same_umap),
        same_row_order_as_09=bool(same_order),
        pal_sec61b_g3bp1_scores_unchanged=bool(same_scores),
        same_display_sample_as_09=bool(same_sample),
        display_rows=counts["display_rows"],
        label_counts={h + "_label": d[h + "_label"].fillna("unavailable").value_counts().to_dict() for h in HEADS},
        pcna_state_counts=d.cycle_state.value_counts().to_dict(),
        enrichment_recomputed_with_new_labels=bool(
            len(check) == len(rows) and check.cycle_state.eq(check.cycle_state_new).all()
        ),
        enrichment_rows=len(rows),
        morphology_recomputed_with_new_labels=True,
        **{k: v for k, v in counts.items() if k.startswith("morphology")},
        paired_crop_count=len(selection),
        crops=json.loads(selection[CROP_COLUMNS].to_json(orient="records")),
        tcn40_vs_rope4_agreement=dict(
            all_rows=float(agree.mean()),
            apr10_jul01=float(agree[zikv].mean()),
            per_dataset=agree.groupby(d.dataset).mean().to_dict(),
        ),
    )
    assert result["all_pcna_predictions_present"] and result["same_umap_coordinates_as_09"]
    assert result["pal_sec61b_g3bp1_scores_unchanged"] and result["enrichment_recomputed_with_new_labels"]
    return result


def main():
    """Claim the attempt directory, assemble labels, recompute panels and write records."""
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source_data").mkdir()
    (OUTPUT / "code").mkdir()
    d, files = assemble()
    d.to_parquet(OUTPUT / "source_data/observation_predictions.parquet", index=False)
    summary = label_summary(d, OUTPUT)
    print(summary.to_string(index=False), flush=True)
    record(ATLAS / "run_metadata.json")
    neighbors = enrichment(d, OUTPUT)
    print("TOMM20 enrichment complete", flush=True)
    selection, images = crops(d, OUTPUT)
    print(selection[CROP_COLUMNS].to_string(index=False), flush=True)
    counts = render(d, neighbors, selection, images, OUTPUT)
    checks = verify(d, counts, selection, OUTPUT)
    write_json(OUTPUT / "verification.json", checks)
    scripts = [
        Path(__file__),
        Path(__file__).with_name("infer_fig6_pcna_best_phase.py"),
        Path(prototype.__file__),
    ]
    for p in scripts:
        shutil.copyfile(p, OUTPUT / "code" / p.name)
    heads = json.loads(record(SOURCE / "head_provenance.json").read_text())
    selection_json = json.loads(record(BUNDLE / "selection.json").read_text())
    winner = next(w for w in selection_json["winners"] if w["modality"] == "Phase3D")
    stage1 = json.loads(record(ITER49 / "manifest.json").read_text())
    replay = json.loads(record(ITER49 / "replay_apr10.json").read_text())
    pcna = dict(
        status="best_validation_PCNA_to_Phase_exploratory_transfer",
        input="Phase3D",
        bundle=str(BUNDLE),
        model=str(BUNDLE / "Phase3D_best.pt"),
        model_sha256=sha(BUNDLE / "Phase3D_best.pt"),
        preprocessing=str(BUNDLE / "Phase3D_preprocessing.joblib"),
        preprocessing_sha256=sha(BUNDLE / "Phase3D_preprocessing.joblib"),
        selection=winner,
        selection_rule=selection_json["rule"],
        prediction_files=files,
        apr30_inference=dict(
            iteration=str(ITER49), inputs=stage1["inputs"], script_sha256=stage1["script_sha256"], gpu=stage1["gpu"]
        ),
        apr10_replay=replay,
        labels="argmax of four softmax probabilities; no abstention",
        calibration="uncalibrated; unvalidated transfer to TOMM20 acquisitions",
    )
    write_json(
        OUTPUT / "manifest.json",
        dict(
            sources=SOURCES,
            script_sha256={p.name: sha(p) for p in scripts},
            source_attempt=str(SOURCE),
            cohort_rows=len(d),
            display_seed=17,
            display_rows=counts["display_rows"],
            thresholds=dict(
                pal_phase_score=THRESHOLD,
                sec61b_phase_score=THRESHOLD,
                g3bp1_phase_score=THRESHOLD,
                rule="positive if score >= 0.5, negative if < 0.5, NaN if score is NaN",
                pcna="argmax over G1/S/G2/M",
            ),
            heads=dict(pAL=heads["pAL"], SEC61B=heads["SEC61B"], G3BP1=heads["G3BP1"], PCNA=pcna),
            superseded_pcna_tcn40=heads["PCNA"],
            tcn40_vs_rope4_agreement=checks["tcn40_vs_rope4_agreement"],
            neighbor_method=(
                "TOMM20 49 PCs; 30NN within acquisition/condition/6h; one midpoint per track/bin; "
                "same-state fraction minus stratum frequency; 500 FOV bootstraps"
            ),
            morphology_method=(
                "Stored export cycle_state/cycle_confidence dropped; new cycle_state merged one-to-one; "
                "segmentation_valid & finite length; FOV medians; 500 FOV bootstraps"
            ),
            limitations=[
                "Binary thresholds at 0.5 on uncalibrated scores",
                "PCNA RoPE4/64 transfer to TOMM20 acquisitions has no target-domain cycle ground truth",
                "Apr30 DENV was not in PCNA training",
                "G3BP1 head is DENV-only; ZIKV rows unavailable",
                "SEC61B head is ZIKV-trained; DENV is cross-virus extrapolation",
                "Morphology and neighbour panels are descriptive associations, not causal effects",
            ],
        ),
    )
    print(json.dumps(checks, indent=1, default=str), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
