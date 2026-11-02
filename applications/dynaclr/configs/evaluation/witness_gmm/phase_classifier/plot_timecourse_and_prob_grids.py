"""Per-timepoint AUROC overlay + probability-split sample grids for the ZIKV plate.

Two diagnostics for the witness-GMM phase-classifier run on
2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV, both on OUR plate:

D1  timecourse_auroc_sec61b_vs_phase3d.{csv,png}
    Per-hpi AUROC of the SEC61B organelle witness AND the Phase3D classifier on the
    held-out val split, graded against the condition label (ZIKV vs uninfected) -- the
    two-line style of Soorya's 03_metrics_over_time.svg.

D2  prob_samples_{SEC61B,Phase3D}.png
    Like sample_labels_by_hpi.py (rows = HPI bins), but columns are 5 P(remodel) buckets
    (0/25/50/75/100 %) instead of classes, so the morphology sweep with predicted
    probability is legible per timepoint.

Both plots need a clean per-cell P(remodel) for EVERY cell, which the Stage-A CSV does not
persist, so we recompute:
  SEC61B  -- re-score the SEC61B embeddings with the MMD witness (control=uninfected,
             perturbed=ZIKV) + fit_gmm_labels (seed 42) -> per-cell remodel posterior.
  Phase3D -- apply the Stage-B pipeline joblib -> predict_proba[:, remodel].

P(remodel) and crop coords BOTH come from the canonical-tree embeddings (row order differs
between trees); crops come from DATA_ZARR keyed by obs (fov_name, t, y, x).

Run:
    /hpc/mydata/eduardo.hirata/repos/viscy/.venv/bin/python plot_timecourse_and_prob_grids.py
"""

from pathlib import Path

import anndata as ad
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmap import Colormap
from iohub import open_ome_zarr
from sklearn.metrics import roc_auc_score

from dynaclr.evaluation.evaluate_config import WitnessGmmLabelsConfig
from dynaclr.evaluation.linear_classifiers.witness_gmm_labels import compute_marker_scores
from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.linear_classifier import group_ids_from_obs, group_val_split
from viscy_utils.evaluation.witness_gmm import fit_gmm_labels

# ------------------------------------------------------------------ config ----
# Canonical prediction tree (5-marker embeddings + existing labels/plots).
EMB_DIR = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics"
    "/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/2-phenotyping/predictions"
    "/DynaCLR-2D-MIP-BagOfChannels"
    "/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler"
    "/epoch105-step84800"
)
SEC61B_EMB = EMB_DIR / "embeddings" / "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV_SEC61B.zarr"
PHASE_EMB = EMB_DIR / "embeddings" / "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV_Phase3D.zarr"
OUT_DIR = EMB_DIR / "labels" / "plots"

# Stage-A config — reused verbatim so the SEC61B witness+GMM scoring here is the SAME
# reference construction / bandwidth / seed as the Stage-A labels (no re-implementation).
STAGE_A_CONFIG = Path(__file__).with_name("01_stage_a_sec61b_labels.yml")

# Stage-B trained pipelines (from my run's tree). Phase3D = teacher/student (labels from
# SEC61B, trained on phase); SEC61B = same-modality (trained on the channel the labels came
# from) — used only for the % -remodeled plot so both markers are the SAME model type at the
# SAME threshold. NOTE the SEC61B same-modality LR is near-ceiling by construction (its labels
# were derived from these embeddings), so its curve tracks the witness gate; Phase3D is the
# informative cross-modality test.
PHASE_PIPELINE = Path(
    "/hpc/projects/organelle_phenotyping/datasets"
    "/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/2-phenotyping/witness_gmm_phase_classifier"
    "/stage_b_phase/pipelines/organelle_remodeling_state_Phase3D.joblib"
)
SEC61B_PIPELINE = Path(
    "/hpc/projects/organelle_phenotyping/datasets"
    "/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/2-phenotyping/witness_gmm_phase_classifier"
    "/stage_b_sec61b/pipelines/organelle_remodeling_state_SEC61B.joblib"
)

# Raw data zarr for crops (has focus_slice zattrs + both channels).
DATA_ZARR = Path(
    "/hpc/projects/organelle_phenotyping/datasets"
    "/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV"
    "/2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV.zarr"
)

# Soorya's reference timecourse (her plate) — overlaid as a faint band for context.
SOORYA_CSV = Path(
    "/hpc/mydata/soorya.pradeep/scratch/label-free_classifier"
    "/witness_score_testing/sec61b_timelapse_zikv/timecourse_metrics.csv"
)

SEED = 42
TRAIN_SIZE = 0.8
GMM_POS_THRESHOLD = 0.8  # SEC61B witness-GMM display threshold (matches the Stage-A gate)
SPLIT_GROUPS = ("experiment", "fov_name", "track_id")

# HPI bins for the grids: right-open 2-hour windows across 0-36h.
HPI_BINS = list(range(0, 38, 2))
# P(remodel) buckets by nearest-25% (label -> [lo, hi)).
PROB_BUCKETS = [
    ("0%", 0.0, 0.125),
    ("25%", 0.125, 0.375),
    ("50%", 0.375, 0.625),
    ("75%", 0.625, 0.875),
    ("100%", 0.875, 1.0001),
]

# Crop / render (from sample_labels_by_hpi.py).
Z_RANGE = (15, 45)
PATCH = 160
N_PER_CELL = 20
CMAP_GREEN = Colormap("green").to_mpl()
DISPLAY_PCTL = (1.0, 99.5)
SCALE_SAMPLE = 60
# --------------------------------------------------------------------------- #


# ---- crop helpers (reused verbatim from sample_labels_by_hpi.py) ----------- #
def focus_z(pos, marker: str, t: int) -> int:
    """In-focus z-slice for (marker, timepoint) from the FOV's ``focus_slice`` metadata."""
    fs = dict(pos.zattrs)["focus_slice"][marker]
    z = fs["per_timepoint"].get(str(t), 0)
    if not z:
        z = int(round(fs["fov_statistics"]["z_focus_mean"]))
    return int(z)


def crop(
    pos_img, t: int, ch_idx: int, y: int, x: int, projection: str = "mip", z_slice: int | None = None
) -> np.ndarray:
    """Crop one cell: PATCH box at (y,x) of channel ch at time t (MIP over Z_RANGE or focus slice)."""
    half = PATCH // 2
    _T, _C, Z, Y, X = pos_img.shape
    y0, y1 = max(0, y - half), min(Y, y + half)
    x0, x1 = max(0, x - half), min(X, x + half)
    if projection == "focus":
        z = min(max(0, z_slice if z_slice is not None else Z // 2), Z - 1)
        return pos_img[t, ch_idx, z, y0:y1, x0:x1]
    z0, z1 = max(0, Z_RANGE[0]), min(Z, Z_RANGE[1])
    return pos_img[t, ch_idx, z0:z1, y0:y1, x0:x1].max(axis=0)


# ---- per-cell P(remodel) recomputation ------------------------------------- #
def sec61b_p_remodel(adata: ad.AnnData) -> np.ndarray:
    """Per-cell P(remodel) + witness score for ALL SEC61B cells, via the Stage-A path.

    Runs the real Stage-A scorer (``compute_marker_scores``) with the actual Stage-A config
    (``01_stage_a_sec61b_labels.yml``) so the references, bandwidth, and seed are identical to
    the Stage-A labels — no re-implementation. Then fits the 2-component GMM on the perturbed
    cells' scores (as ``label_marker`` does) and applies its remodel-component posterior to
    every cell, giving a continuous P(remodel) for all cells (the CSV only persists confident
    positives + controls).
    """
    config = WitnessGmmLabelsConfig(**load_config(STAGE_A_CONFIG)["witness_gmm_labels"])
    ms = compute_marker_scores(adata, config.experiments, config)
    if ms is None:
        raise RuntimeError("compute_marker_scores found no control/perturbed reference cells")
    scores = ms.scores
    res = fit_gmm_labels(
        scores[ms.perturbed_mask], pos_threshold=config.gmm_pos_threshold, random_state=config.random_seed
    )
    posterior = res.gmm.predict_proba(scores.reshape(-1, 1))[:, res.remod_component]
    print(
        f"  SEC61B (Stage-A path) GMM means={res.gmm.means_.ravel().round(4)} "
        f"weights={res.gmm.weights_.round(3)}  separated={res.separated}"
    )
    return posterior, scores


def lr_p_remodel(adata: ad.AnnData, pipeline_path: Path) -> np.ndarray:
    """Per-cell P(remodel) for ALL cells via a trained Stage-B logistic pipeline."""
    pipe = joblib.load(pipeline_path)
    pos_idx = list(pipe.classifier.classes_).index("remodel")
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    return pipe.predict_proba(X)[:, pos_idx]


# ---- val split (the shared LC-backend split — identical to Stage B) --------- #
def val_mask(obs: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Boolean mask of the held-out val cells via the shared LC split helper.

    Uses ``group_ids_from_obs`` + ``group_val_split`` — the SAME functions
    ``run-linear-classifiers`` uses — so this val set is exactly Stage B's, not a
    look-alike. ``y`` is the per-cell label array (used only for the non-grouped
    fallback; grouped splits ignore it).
    """
    groups = group_ids_from_obs(obs, list(SPLIT_GROUPS))
    _, idx_val = group_val_split(len(obs), y, groups, TRAIN_SIZE, SEED)
    m = np.zeros(len(obs), dtype=bool)
    m[idx_val] = True
    return m


def per_hpi_auroc(hpi: np.ndarray, y_cond: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Per-hpi AUROC of ``score`` vs the binary condition label; skips single-class bins."""
    df = pd.DataFrame({"hpi": np.round(hpi, 1), "y": y_cond, "s": score})
    rows = []
    for h, g in df.groupby("hpi"):
        yt = g["y"].to_numpy()
        if yt.sum() == 0 or (1 - yt).sum() == 0:
            continue
        rows.append({"hpi": h, "auroc": roc_auc_score(yt, g["s"].to_numpy()), "n_cells": len(g)})
    return pd.DataFrame(rows).sort_values("hpi")


# ---- D1: AUROC overlay ----------------------------------------------------- #
def plot_timecourse(sec_obs, sec_scores, phase_obs, phase_p) -> None:
    """Overlay per-hpi AUROC of SEC61B witness and Phase3D classifier (val, vs condition)."""
    sec_y = (sec_obs["perturbation"] == "ZIKV").astype(int).to_numpy()
    sm = val_mask(sec_obs, sec_y)
    # SEC61B: lower witness = more ZIKV-like, so grade on -witness_score.
    sec_tc = per_hpi_auroc(
        sec_obs.loc[sm, "hours_post_perturbation"].to_numpy(),
        (sec_obs.loc[sm, "perturbation"] == "ZIKV").astype(int).to_numpy(),
        -sec_scores[sm],
    ).rename(columns={"auroc": "org_auroc", "n_cells": "n_org"})

    phase_y = (phase_obs["perturbation"] == "ZIKV").astype(int).to_numpy()
    pm = val_mask(phase_obs, phase_y)
    phase_tc = per_hpi_auroc(
        phase_obs.loc[pm, "hours_post_perturbation"].to_numpy(),
        (phase_obs.loc[pm, "perturbation"] == "ZIKV").astype(int).to_numpy(),
        phase_p[pm],
    ).rename(columns={"auroc": "phase_auroc", "n_cells": "n_phase"})

    merged = phase_tc.merge(sec_tc, on="hpi", how="outer").sort_values("hpi")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "timecourse_auroc_sec61b_vs_phase3d.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        merged["hpi"], merged["phase_auroc"], color="mediumorchid", lw=2, marker="o", ms=3, label="Phase3D classifier"
    )
    ax.plot(
        merged["hpi"],
        merged["org_auroc"],
        color="darkorange",
        lw=2,
        ls="--",
        marker="o",
        ms=3,
        label="SEC61B organelle witness",
    )
    if SOORYA_CSV.exists():
        s = pd.read_csv(SOORYA_CSV)
        ax.plot(s["hpi"], s["phase_auroc"], color="mediumorchid", lw=1, alpha=0.35, label="Phase3D (Soorya ref)")
        ax.plot(s["hpi"], s["org_auroc"], color="darkorange", lw=1, ls="--", alpha=0.35, label="SEC61B (Soorya ref)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Time post-infection (hpi)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Per-timepoint AUROC — SEC61B organelle vs Phase3D classifier\nheld-out val · ground truth = ZIKV vs uninfected"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "timecourse_auroc_sec61b_vs_phase3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  D1: wrote timecourse_auroc_sec61b_vs_phase3d.{{csv,png}} "
        f"(phase mean {merged['phase_auroc'].mean():.3f}, org mean {merged['org_auroc'].mean():.3f})"
    )


# ---- D2: probability-split grid -------------------------------------------- #
def plot_prob_grid(
    marker: str, obs: pd.DataFrame, p_remodel: np.ndarray, channel: str, projection: str, cmap, plate, rng
) -> None:
    """One grid: rows = HPI bins, columns = 5 P(remodel) buckets, N cells each."""
    df = obs.copy().reset_index(drop=True)
    df["p_remodel"] = p_remodel
    df["fov_name"] = df["fov_name"].astype(object).str.strip("/")
    df["bin"] = pd.cut(df["hours_post_perturbation"], bins=HPI_BINS, right=False)
    bins = [b for b in df["bin"].cat.categories if (df["bin"] == b).any()]

    ch_idx = plate.channel_names.index(channel)

    def crop_row(row) -> np.ndarray:
        pos = plate[str(row["fov_name"]).strip("/")]
        t = int(row["t"])
        z = focus_z(pos, marker, t) if projection == "focus" else None
        return crop(pos["0"], t, ch_idx, int(row["y"]), int(row["x"]), projection, z)

    scale_rows = df.sample(min(SCALE_SAMPLE, len(df)), random_state=SEED)
    pooled = np.concatenate([crop_row(r).ravel() for _, r in scale_rows.iterrows()])
    vmin, vmax = np.percentile(pooled, DISPLAY_PCTL)
    vmax = max(vmax, vmin + 1)

    ncols = N_PER_CELL * len(PROB_BUCKETS)
    fig, axes = plt.subplots(len(bins), ncols, figsize=(0.9 * ncols, 1.2 * len(bins)), squeeze=False)
    for r, b in enumerate(bins):
        for c_i, (blabel, lo, hi) in enumerate(PROB_BUCKETS):
            sub = df[(df["bin"] == b) & (df["p_remodel"] >= lo) & (df["p_remodel"] < hi)]
            picks = sub.sample(min(N_PER_CELL, len(sub)), random_state=int(rng.integers(1e9))) if len(sub) else sub
            for k in range(N_PER_CELL):
                ax = axes[r][c_i * N_PER_CELL + k]
                if k < len(picks):
                    ax.imshow(crop_row(picks.iloc[k]), cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")
        axes[r][0].set_ylabel(f"{int(b.left)}-{int(b.right)}h", fontsize=8, color="white")

    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.suptitle(f"{marker} — {N_PER_CELL} cells per HPI bin x P(remodel) bucket", fontsize=12, color="white", y=0.995)
    for c_i, (blabel, _lo, _hi) in enumerate(PROB_BUCKETS):
        left = axes[0][c_i * N_PER_CELL].get_position().x0
        right = axes[0][c_i * N_PER_CELL + N_PER_CELL - 1].get_position().x1
        fig.text((left + right) / 2, 0.97, f"P(remodel)={blabel}", ha="center", va="bottom", fontsize=11, color="white")
    for c_i in range(1, len(PROB_BUCKETS)):
        x_prev = axes[0][c_i * N_PER_CELL - 1].get_position().x1
        x_next = axes[0][c_i * N_PER_CELL].get_position().x0
        xdiv = (x_prev + x_next) / 2
        fig.add_artist(plt.Line2D([xdiv, xdiv], [0.02, 0.96], color="white", lw=1.2, alpha=0.6))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"prob_samples_{marker}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  D2: wrote {out}  ({N_PER_CELL} cells x {len(bins)} bins x {len(PROB_BUCKETS)} buckets)")


# ---- D3: % remodeled vs time (both markers, ZIKV + control) ---------------- #
def _pct_pos_by_hpi(hpi: np.ndarray, is_pos: np.ndarray) -> pd.DataFrame:
    """Per-hpi % positive ± Bernoulli SEM over a cell population (paper convention)."""
    df = pd.DataFrame({"t": np.round(np.asarray(hpi), 1), "is_pos": np.asarray(is_pos, dtype=float)})
    rows = []
    for t, g in df.groupby("t"):
        n = len(g)
        p = float(g["is_pos"].mean())
        rows.append({"t": t, "pct": 100.0 * p, "sem": 100.0 * np.sqrt(p * (1 - p) / n), "n": n})
    return pd.DataFrame(rows).sort_values("t")


def plot_percent_remodeled(series: list[dict]) -> None:
    """% cells 'remodel' (P>=thr) vs hpi, ZIKV (solid) vs control FP (dashed), for N series.

    Each ``series`` entry: ``{name, color, obs, prob, thr}``. The % is over the WHOLE well
    population per hpi (positives / all cells), ± Bernoulli SEM — the reference
    ``plot_remodeling_vs_time`` convention. Plots all series on one axis so the SEC61B
    witness-GMM gate (label source), the SEC61B same-modality LR, and the Phase3D
    cross-modality LR are directly comparable.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    csv_rows = []
    for spec in series:
        name, color, obs, prob, thr = spec["name"], spec["color"], spec["obs"], spec["prob"], spec["thr"]
        obs = obs.reset_index(drop=True)
        is_pos = np.asarray(prob) >= thr
        hpi = obs["hours_post_perturbation"].to_numpy()
        pert = obs["perturbation"].to_numpy()
        for cond, style, alpha in [("ZIKV", "-", 1.0), ("uninfected", "--", 0.6)]:
            m = pert == cond
            if not m.any():
                continue
            stats = _pct_pos_by_hpi(hpi[m], is_pos[m])
            rate = 100.0 * float(is_pos[m].mean())
            if cond == "ZIKV":
                label = f"{name} ZIKV (n={int(stats['n'].sum())})"
            else:
                label = f"{name} control FP ({rate:.1f}%)"
            ax.errorbar(
                stats["t"],
                stats["pct"],
                yerr=stats["sem"],
                marker="o",
                markersize=3,
                capsize=2,
                linewidth=1.7 if cond == "ZIKV" else 1.0,
                linestyle=style,
                alpha=alpha,
                color=color,
                label=label,
            )
            s = stats.copy()
            s["series"] = name
            s["condition"] = cond
            csv_rows.append(s)

    ax.set_ylim(-5, 105)
    ax.set_title(
        "% cells 'remodel' vs time — witness-GMM gate + same/cross-modality classifiers", fontsize=11, fontweight="bold"
    )
    ax.text(
        0.5,
        -0.15,
        "SEC61B witness-GMM = label source (thr 0.8) · SEC61B LR = same-modality, near-ceiling (thr 0.5) · "
        "Phase3D LR = cross-modality teacher/student (thr 0.5)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="0.4",
    )
    ax.set_xlabel("hours post perturbation", fontsize=11)
    ax.set_ylabel("% cells 'remodel'", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=7.5, ncol=3, loc="upper left")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(csv_rows, ignore_index=True).to_csv(OUT_DIR / "percent_remodeled_vs_time.csv", index=False)
    fig.savefig(OUT_DIR / "percent_remodeled_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  D3: wrote percent_remodeled_vs_time.{png,csv}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("Loading embeddings ...")
    sec = ad.read_zarr(SEC61B_EMB)
    sec.obs_names_make_unique()
    phase = ad.read_zarr(PHASE_EMB)
    phase.obs_names_make_unique()

    # The SEC61B classifier is trained/valid only on the SEC61B wells (A/2 uninfected,
    # B/2 ZIKV). The Phase3D zarr spans all 6 wells (A/3-4, B/3-4 carry TOMM20/G3BP1, unseen
    # by the classifier). Restrict Phase3D to the SEC61B wells so both models are graded on
    # the same in-domain cell population and the overlay is apples-to-apples.
    sec_wells = sorted({"/".join(str(f).strip("/").split("/")[:2]) for f in sec.obs["fov_name"]})
    phase_well = phase.obs["fov_name"].astype(object).map(lambda f: "/".join(str(f).strip("/").split("/")[:2]))
    phase = phase[phase_well.isin(sec_wells).to_numpy()].copy()
    phase.obs_names_make_unique()
    print(f"Restricted Phase3D to SEC61B wells {sec_wells}: {phase.n_obs} cells")

    print("Recomputing per-cell P(remodel) ...")
    # GMM posterior (label source) — drives D1 SEC61B witness AUROC + D2 SEC61B grid.
    sec_p_gmm, sec_scores = sec61b_p_remodel(sec)
    # Same-modality LR — drives D3 so both markers are the SAME model type at the SAME thr.
    sec_p_lr = lr_p_remodel(sec, SEC61B_PIPELINE)
    phase_p = lr_p_remodel(phase, PHASE_PIPELINE)

    print("D1 — AUROC overlay ...")
    plot_timecourse(sec.obs, sec_scores, phase.obs, phase_p)

    print("D3 — % remodeled vs time (witness gate + SEC61B LR + Phase3D LR) ...")
    plot_percent_remodeled(
        [
            {
                "name": "SEC61B witness-GMM",
                "color": "tab:green",
                "obs": sec.obs,
                "prob": sec_p_gmm,
                "thr": GMM_POS_THRESHOLD,
            },
            {"name": "SEC61B LR", "color": "tab:olive", "obs": sec.obs, "prob": sec_p_lr, "thr": 0.5},
            {"name": "Phase3D LR", "color": "tab:blue", "obs": phase.obs, "prob": phase_p, "thr": 0.5},
        ]
    )

    print("D2 — probability-split grids ...")
    with open_ome_zarr(DATA_ZARR, mode="r") as plate:
        plot_prob_grid("SEC61B", sec.obs, sec_p_gmm, "raw GFP EX488 EM525-45", "mip", CMAP_GREEN, plate, rng)
        plot_prob_grid("Phase3D", phase.obs, phase_p, "Phase3D", "focus", "gray", plate, rng)
    print("Done.")


if __name__ == "__main__":
    main()
