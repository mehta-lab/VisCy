"""Phase-only Figure 6 prototype with explicit transfer gaps; never train heads."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BASE = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
SUF = Path(
    "2-phenotyping/predictions/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/epoch105-step84800"
)
MITO = BASE / "2026_04_10_A549_TOMM20_ZIKV" / SUF / "outputs"
ATLAS = MITO / "01-tomm20_pal_cell_cycle_umap"
Z = (
    BASE
    / "2026_03_24_A549_SEC61_ZIKV"
    / SUF
    / "classifier_exploration/outputs/pooled_zikv_pal17_phase3d_joint_tied_qc_nonblank"
)
D = (
    BASE
    / "2026_04_28_A549_SEC61B_DENV"
    / SUF
    / "classifier_exploration/outputs/pooled_denv_pal40_phase3d_joint_tied_qc_nonblank_5dataset"
)
DATES = {
    "04_10_TOMM20": "2026_04_10_A549_TOMM20_ZIKV",
    "07_01_SEC61B_TOMM20_G3BP1": "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV",
    "04_30_TOMM20": "2026_04_30_A549_TOMM20_DENV",
}
KEY = ["dataset", "fov_name", "track_id", "t"]
OBJ = ["fov_name", "track_id", "t", "id"]
STATES = ["G1", "S", "G2", "M"]
COLORS = ["#0072b2", "#e69f00", "#9564a7", "#00846b"]
SOURCES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record(p):
    SOURCES[str(p)] = sha(p)
    return p


def read(p):
    return pd.read_parquet(record(p))


def write_json(p, v):
    p.write_text(json.dumps(v, indent=2, default=str))


def read_features(dataset, marker, atlas):
    p = BASE / DATES[dataset] / SUF / "embeddings" / f"{marker}.zarr"
    g = zarr.open_group(str(p), mode="r")
    o = read_elem(g["obs"]).reset_index(drop=True)
    assert not o.duplicated(OBJ).any()
    q = atlas[OBJ].merge(
        o[OBJ + ["x", "y"]].assign(feature_row=np.arange(len(o))), on=OBJ, how="left", validate="one_to_one"
    )
    assert q.feature_row.notna().all() and np.allclose(q.x, atlas.x) and np.allclose(q.y, atlas.y)
    meta = read_elem(g["uns/X_normalized_pca80"])
    expected = json.loads((ATLAS / "run_metadata.json").read_text())["pca_checksums"][marker]
    assert meta["pca_model_checksums"][marker] == expected
    return np.asarray(g["obsm/X_normalized_pca80"][:], np.float32)[q.feature_row.to_numpy(int)]


def predict_g3(d, out):
    """Reconstruct the saved DENV preprocessing; verify against saved pAL student."""
    ds = "04_30_TOMM20"
    root = BASE / DATES[ds] / SUF / "embeddings"
    gp = zarr.open_group(str(root / "Phase3D.zarr"), mode="r")
    op = read_elem(gp["obs"]).reset_index(drop=True)
    og = read_elem(zarr.open_group(str(root / "pAL40.zarr"), mode="r")["obs"]).reset_index(drop=True)
    og = og.loc[~og.is_blank.astype(bool)].reset_index(drop=True)
    assert not op.duplicated(OBJ).any() and not og.duplicated(OBJ).any()
    lookup = pd.MultiIndex.from_frame(op[OBJ]).get_indexer(pd.MultiIndex.from_frame(og[OBJ]))
    assert (lookup >= 0).all()
    assert not op.iloc[lookup].is_blank.astype(bool).any()
    x = np.asarray(gp["X"][:], np.float32)[lookup]
    source = Path(__file__).with_name("evaluate_pooled_denv_pal40_phase3d.py")
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ["_normal_consistent_mad", "_control_normalization_contract"]
    ]
    env = dict(np=np, pd=pd, gaussian_filter1d=gaussian_filter1d)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), env)
    json.loads(record(D / "run_info.json").read_text())
    # Historical producer defaults; replay below must agree before exporting any score.
    normalized, grid, centers, scale = env["_control_normalization_contract"](
        x, og, sigma_timepoints=2.0, floor_quantile=0.05
    )
    expected = np.load(record(D / "models/per_dataset_control_mad_scales.npz"))[ds + "__Phase3D"]
    assert np.allclose(scale, expected, rtol=1e-4, atol=1e-5), "Normalization scale differs from saved model"
    pca = joblib.load(record(D / "models/pooled_Phase3D_pca80.joblib"))
    pc = pca.transform(normalized)
    pal = joblib.load(record(D / f"models/lodo_{ds}__Phase3D.joblib"))
    pred = pal.predict_proba(pc)[:, list(pal.classes_).index(1)]
    saved = read(D / "leave_one_dataset_out_predictions.parquet")
    saved = saved[saved.dataset.eq(ds)]
    replay = (
        og[OBJ]
        .assign(replay=pred)
        .merge(saved[["fov_name", "track_id", "t", "pal_to_phase_probability"]], on=OBJ[:3], validate="one_to_one")
    )
    error = float(np.max(np.abs(replay.replay - replay.pal_to_phase_probability)))
    assert error < 1e-4, ("pAL replay mismatch", error)
    modelpath = D / "g3bp1_remodeling_teacher_student_joint_tied_qc_nonblank/models/Phase3D__repeat0.joblib"
    head = joblib.load(record(modelpath))
    assert head.n_features_in_ == pc.shape[1]
    score = head.predict_proba(pc)[:, list(head.classes_).index(1)]
    preds = og[OBJ].assign(dataset=ds, g3bp1_phase_score=score)
    joined = d[KEY + ["id"]].merge(preds, on=KEY + ["id"], how="left", validate="one_to_one")
    np.savez_compressed(out / "source_data/g3bp1_target_normalization.npz", grid=grid, centers=centers, scale=scale)
    shutil.copyfile(source, out / "code/denv_normalization_source.py")
    return joined.g3bp1_phase_score.to_numpy(), dict(
        status="DENV_only_exploratory_transfer",
        model=str(modelpath),
        pca=str(D / "models/pooled_Phase3D_pca80.joblib"),
        pAL_replay_max_error=error,
        normalization_source_sha256=sha(source),
        selection="fixed repeat0, not selected on TOMM20 appearance",
        calibration="uncalibrated",
    )


def enrichment(d, out):
    """30-neighbor enrichment in TOMM20 PCs; one midpoint per track/6h bin."""
    x = np.empty((len(d), 49), np.float32)
    for ds, g in d.groupby("dataset", sort=False):
        x[g.index] = read_features(ds, "TOMM20", g)
    a = d.copy()
    a["hpi_bin"] = np.floor(a.hours_post_perturbation / 6) * 6
    a["distance_bin_center"] = abs(a.hours_post_perturbation - a.hpi_bin - 3)
    a = a.sort_values(["distance_bin_center", "t"]).drop_duplicates(["dataset", "fov_name", "track_id", "hpi_bin"])
    records = []
    support = []
    for key, g in a.groupby(["dataset", "perturbation", "hpi_bin"], observed=True):
        support.append(dict(dataset=key[0], perturbation=key[1], hpi_bin=key[2], rows=len(g), supported=len(g) >= 31))
        if len(g) < 31:
            continue
        idx = g.index.to_numpy()
        labels = g.cycle_state.to_numpy()
        nn = (
            NearestNeighbors(n_neighbors=31, algorithm="brute", n_jobs=4)
            .fit(x[idx])
            .kneighbors(x[idx], return_distance=False)
        )
        neighbors = np.array([row[row != i][:30] for i, row in enumerate(nn)])
        match = (labels[neighbors] == labels[:, None]).mean(1)
        freq = g.cycle_state.value_counts()
        chance = np.array([(freq[s] - 1) / (len(g) - 1) for s in labels])
        q = g[KEY + ["perturbation", "hpi_bin", "cycle_state", "cycle_confidence"]].copy()
        q["neighbor_same_state"] = match
        q["chance_same_state"] = chance
        q["excess"] = match - chance
        records.append(q)
    rows = pd.concat(records)
    rows.to_parquet(out / "source_data/tomm20_neighbor_rows.parquet", index=False)
    pd.DataFrame(support).to_csv(out / "source_data/neighbor_strata_support.csv", index=False)
    # FOV resampling preserves temporal/track dependence within each FOV.
    summary = []
    rng = np.random.default_rng(17)
    for key, g in rows.groupby(["dataset", "perturbation", "cycle_state"], observed=True):
        values = g.groupby("fov_name").excess.mean().to_numpy()
        supported = len(values) >= 3
        boot = rng.choice(values, (500, len(values)), replace=True).mean(1) if supported else np.array([np.nan])
        summary.append(
            dict(
                dataset=key[0],
                perturbation=key[1],
                cycle_state=key[2],
                estimate=values.mean(),
                low=np.nanquantile(boot, 0.025) if supported else np.nan,
                high=np.nanquantile(boot, 0.975) if supported else np.nan,
                fovs=len(values),
                rows=len(g),
                supported=supported,
            )
        )
    result = pd.DataFrame(summary)
    result.to_csv(out / "source_data/tomm20_neighbor_summary.csv", index=False)
    return result


def crops(d, out):
    """Coverage-first deterministic examples: class median confidence, plus uncertain."""
    pool = d[d.dataset.eq("04_10_TOMM20") & d.perturbation.eq("uninfected")].copy()
    chosen = []
    images = {}
    image_root = (
        Path("/hpc/projects/organelle_phenotyping/datasets") / DATES["04_10_TOMM20"] / (DATES["04_10_TOMM20"] + ".zarr")
    )
    root = zarr.open_group(str(image_root), mode="r")
    for label in STATES + ["uncertain"]:
        g = pool if label == "uncertain" else pool[pool.cycle_state.eq(label)]
        if len(g) == 0:
            continue
        target = g.cycle_confidence.quantile(0.1) if label == "uncertain" else g.cycle_confidence.median()
        g = g.assign(selection_distance=abs(g.cycle_confidence - target)).sort_values(
            ["selection_distance", "fov_name", "track_id", "t"]
        )
        for _, r in g.iterrows():
            pos = root[r.fov_name]
            ar = pos["0"]
            t = int(r.t)
            y = int(round(r.y))
            x = int(round(r.x))
            attrs = dict(pos.attrs)
            if min(y, x) < 112 or y + 112 > ar.shape[-2] or x + 112 > ar.shape[-1]:
                continue
            focus = attrs.get("focus_slice", {}).get("Phase3D", {}).get("per_timepoint", {}).get(str(t))
            if focus is None or focus < 5 or focus + 11 > ar.shape[2]:
                continue
            channels = [v["label"] for v in attrs["ome"]["omero"]["channels"]]
            phase = np.asarray(ar[t, channels.index("Phase3D"), int(focus), y - 112 : y + 112, x - 112 : x + 112])
            mito = np.asarray(
                ar[
                    t,
                    channels.index("raw GFP EX488 EM525-45"),
                    int(focus) - 5 : int(focus) + 11,
                    y - 112 : y + 112,
                    x - 112 : x + 112,
                ]
            ).max(0)
            eid = f"E{len(chosen) + 1}"
            images[eid + "_phase"] = phase
            images[eid + "_tomm20"] = mito
            row = r.to_dict()
            row.update(
                example_id=eid,
                selection_label=label,
                z_focus=int(focus),
                image_path=str(image_root),
                pixel_um=attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][-1],
            )
            chosen.append(row)
            pool = pool.drop(index=r.name)
            break
    c = pd.DataFrame(chosen)
    c.to_csv(out / "source_data/crop_selection.csv", index=False)
    np.savez_compressed(out / "source_data/paired_raw_crops.npz", **images)
    return c, images


def render(d, neighbors, selection, images, out, head_details=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    display = d.sample(min(18000, len(d)), random_state=17).copy()
    display = pd.concat(
        [display, d.merge(selection[KEY + ["id"]], on=KEY + ["id"], validate="one_to_one")]
    ).drop_duplicates(KEY + ["id"])
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
    cols = ["pal_phase_score", "sec61b_phase_score", "g3bp1_phase_score", "cycle_state"]
    titles = ["pAL → Phase", "SEC61B → Phase", "G3BP1 → Phase", "PCNA → Phase"]
    details = [
        "Saved leave-acquisition-out\nstatic heads",
        "PENDING: fitted head\nunavailable",
        "DENV-only static head\nZIKV unavailable",
        "Saved causal TCN40\nunvalidated transfer",
    ]
    if head_details is not None:
        details = [head_details.get(col, detail) for col, detail in zip(cols, details)]
    for j, (col, title, detail) in enumerate(zip(cols, titles, details)):
        ax = fig.add_axes([0.045 + j * 0.24, 0.653, 0.205, 0.204])
        xy = display[["Phase3D_umap1", "Phase3D_umap2"]].to_numpy()
        ax.scatter(*xy.T, c="#ddd", s=1, rasterized=True)
        if col == "cycle_state":
            ax.scatter(*xy.T, c=display[col].map(palettes), s=1, alpha=0.65, rasterized=True)
            ax.legend(
                handles=[
                    Line2D([], [], marker="o", ls="", color=c, label=s, markersize=4) for s, c in palettes.items()
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=4,
                fontsize=7,
                frameon=False,
                handletextpad=0.1,
                columnspacing=0.4,
            )
        else:
            ok = display[col].notna()
            sc = ax.scatter(
                xy[ok, 0],
                xy[ok, 1],
                c=display.loc[ok, col],
                vmin=0,
                vmax=1,
                cmap="viridis",
                s=1,
                alpha=0.8,
                rasterized=True,
            )
            if ok.any():
                fig.colorbar(
                    sc,
                    cax=fig.add_axes([0.075 + j * 0.24, 0.633, 0.145, 0.006]),
                    orientation="horizontal",
                    ticks=[0, 0.5, 1],
                )
        for _, r in selection.iterrows():
            ax.scatter(r.Phase3D_umap1, r.Phase3D_umap2, s=20, facecolors="none", edgecolors="black", linewidths=0.6)
            ax.annotate(
                r.example_id, (r.Phase3D_umap1, r.Phase3D_umap2), fontsize=5, xytext=(3, 3), textcoords="offset points"
            )
        ax.set(xticks=[], yticks=[], title=title)
        limits.append((ax.get_xlim(), ax.get_ylim()))
        fig.text(0.045 + j * 0.24, 0.882, ("B   " if j == 0 else "") + detail, size=7, va="bottom")
        fig.text(0.045 + j * 0.24, 0.599, f"Available: {d[col].notna().sum():,}/{len(d):,}", size=7.5)
    assert all(np.allclose(limits[0], v) for v in limits)
    fig.text(
        0.045,
        0.578,
        "Same Phase map and cells. Continuous colors = uncalibrated scores; gray = unavailable. No fluorescence scores substituted.",
        size=9,
    )
    fig.text(0.045, 0.549, "C   Paired microscopy: Phase input and observed TOMM20 morphology", size=11, weight="bold")
    display_limits = {}
    for modality, cmap in [("phase", "gray"), ("tomm20", "Greens")]:
        vals = np.concatenate([v.ravel() for k, v in images.items() if k.endswith("_" + modality)])
        display_limits[modality] = np.percentile(vals, [1, 99.5]).tolist()
    for j, (_, r) in enumerate(selection.iterrows()):
        for k, mod in enumerate(["phase", "tomm20"]):
            ax = fig.add_axes([0.065 + j * 0.182, 0.405 - k * 0.112, 0.14, 0.105])
            arr = images[r.example_id + "_" + mod]
            ax.imshow(
                arr,
                cmap="gray" if mod == "phase" else LinearSegmentedColormap.from_list("mito", ["black", "#00ff66"]),
                vmin=display_limits[mod][0],
                vmax=display_limits[mod][1],
            )
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
    fig.text(
        0.045,
        0.249,
        "D   Fluorescence application: does TOMM20 structure associate with Phase-predicted cycle?",
        size=11,
        weight="bold",
    )
    ax = fig.add_axes([0.045, 0.065, 0.25, 0.158])
    ax.scatter(
        display.TOMM20_umap1, display.TOMM20_umap2, c=display.cycle_state.map(palettes), s=1, alpha=0.6, rasterized=True
    )
    ax.set(xticks=[], yticks=[], title="TOMM20 UMAP (not label-free)")
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
    ax.legend(fontsize=6, frameon=False)
    morph = read(MITO / "03-tomm20_features_cycle_export_fix/all_joined_features.parquet")
    morph = morph.merge(d[KEY + ["id", "cycle_state"]], on=KEY + ["id"], validate="one_to_one", suffixes=("", "_check"))
    assert morph.cycle_state.eq(morph.cycle_state_check).all()
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
    for i, condition in enumerate(["uninfected", "ZIKV", "DENV"]):
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
            ci_lo, h = np.quantile(boot, [0.025, 0.975])
            yy.append(val)
            lo.append(val - ci_lo)
            hi.append(h - val)
            mrows.append(dict(condition=condition, state=state, median=val, low=ci_lo, high=h, n_fovs=len(values)))
        ax.errorbar(np.arange(4) + (i - 1) * 0.15, yy, yerr=[lo, hi], fmt="o", ms=3, label=condition)
    pd.DataFrame(mrows).to_csv(out / "source_data/morphology_summary.csv", index=False)
    ax.set(xticks=range(4), xticklabels=STATES, ylabel="Network length (µm/crop)", title="Measured TOMM20 morphology")
    ax.legend(fontsize=6, frameon=False)
    fig.text(
        0.045,
        0.025,
        "Exploratory associations, not causal effects. FOV-bootstrap intervals are not biological-replicate uncertainty. Cycle transfer remains unvalidated.",
        size=8,
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"main_phase_prototype.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    write_json(out / "source_data/display_limits.json", display_limits)
    # Supplement: no new timing claims; diagnostics for the shared Phase map.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, col, title in zip(
        axes,
        ["dataset", "cycle_confidence", "distinct_context"],
        [
            "Acquisition (same Phase coordinates)",
            "Cycle maximum probability (uncalibrated)",
            "Distinct causal history frames",
        ],
    ):
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


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_data").mkdir()
    (out / "code").mkdir()
    d = read(ATLAS / "paired_umap_predictions.parquet")
    assert len(d) == 184514 and not d.duplicated(KEY + ["id"]).any()
    tables = []
    for source in [Z, D]:
        p = read(source / "leave_one_dataset_out_predictions.parquet")
        p = p[p.dataset.isin(DATES)]
        tables.append(p[KEY + ["pal_to_phase_probability"]])
    p = pd.concat(tables)
    assert not p.duplicated(KEY).any()
    d = d.merge(p, on=KEY, how="left", validate="one_to_one").rename(
        columns={"pal_to_phase_probability": "pal_phase_score"}
    )
    assert d.pal_phase_score.notna().all()
    # Legacy table lacks object ID: certify all target keys against current atlas Phase metadata.
    for ds, g in d.groupby("dataset", sort=False):
        read_features(ds, "Phase3D", g)
    heads = {
        "pAL": dict(
            status="saved_static_LODO_predictions",
            calibration="uncalibrated; pooled upstream PCA exposure",
            input="Phase3D",
        ),
        "SEC61B": dict(
            status="unavailable",
            reason="Temporal study did not persist fitted heads; no compatible TOMM20 predictions located",
        ),
        "PCNA": dict(
            status="saved_causal_TCN40",
            provenance=json.loads(record(ATLAS / "run_metadata.json").read_text()),
            input="Phase3D",
            calibration="unvalidated transfer; no abstention threshold",
        ),
    }
    d["sec61b_phase_score"] = np.nan
    try:
        d["g3bp1_phase_score"], heads["G3BP1"] = predict_g3(d, out)
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        d["g3bp1_phase_score"] = np.nan
        heads["G3BP1"] = dict(status="unavailable", reason=str(error))
        print("G3BP1 unavailable:", error, flush=True)
    d.to_parquet(out / "source_data/observation_predictions.parquet", index=False)
    write_json(out / "head_provenance.json", heads)
    print("Predictions assembled", flush=True)
    neighbors = enrichment(d, out)
    print("TOMM20 enrichment complete", flush=True)
    selection, images = crops(d, out)
    render(d, neighbors, selection, images, out)
    shutil.copyfile(__file__, out / "code/build_fig6_phase_prototype.py")
    write_json(
        out / "manifest.json",
        dict(
            sources=SOURCES,
            script_sha256=sha(__file__),
            cohort_rows=len(d),
            display_seed=17,
            display_rows=min(18000, len(d)),
            heads=heads,
            neighbor_method="TOMM2049 PCs;30NN within acquisition/condition/6h; one midpoint per track/bin; same-state fraction minus stratum frequency;500 FOV bootstraps",
            limitations=[
                "Exploratory only; transferred cycle labels unvalidated",
                "No inference fluorescence input; original paired QC uses fluorescence availability",
                "No continuous pseudotime or cell-level delay estimated",
                "Morphology panel is descriptive, not acquisition/time-adjusted causal effect",
            ],
        ),
    )
    write_json(
        out / "verification.json",
        dict(
            unique_cell_keys=True,
            shared_phase_coordinates=True,
            no_teacher_scores_used_as_student_scores=True,
            paired_crop_count=len(selection),
            source_tables_written=True,
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-source", type=Path)
    args = p.parse_args()
    if args.render_source:
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(args.render_source / "source_data", args.output / "source_data")
        (args.output / "code").mkdir()
        shutil.copyfile(args.render_source / "head_provenance.json", args.output / "head_provenance.json")
        d = read(args.render_source / "source_data/observation_predictions.parquet")
        nn = pd.read_csv(args.render_source / "source_data/tomm20_neighbor_summary.csv")
        sel = pd.read_csv(args.render_source / "source_data/crop_selection.csv")
        imgs = np.load(args.render_source / "source_data/paired_raw_crops.npz")
        render(d, nn, sel, imgs, args.output)
        meta = json.loads((args.render_source / "manifest.json").read_text())
        meta.update(
            render_source=str(args.render_source),
            renderer_sha256=sha(__file__),
            render_sources=SOURCES,
            display_rows=len(pd.read_parquet(args.output / "source_data/display_rows.parquet")),
        )
        write_json(args.output / "manifest.json", meta)
        shutil.copyfile(__file__, args.output / "code/build_fig6_phase_prototype.py")
        shutil.copyfile(args.render_source / "verification.json", args.output / "verification.json")
    else:
        main(args.output)
