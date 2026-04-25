r"""Per-cell timing metrics for organelle remodeling (Stage 3 analysis).

Given a sensor alignment parquet and one organelle channel, computes per-cell
timing scalars on each cell's cosine-distance-from-pre-baseline curve, then
pools them into a per-organelle distribution. Cross-organelle comparisons are
population-level (disjoint FOVs share only the sensor-aligned t_rel axis).

Metrics per cell (computed on the aligned region only):

- ``t_onset_abs``  : first t_rel where (distance − pre_median) crosses
                     an absolute threshold (default 0.10 cosine-distance
                     units). SNR-robust: small Δpeak cells can't fake an
                     onset by having their noise floor crossed.
- ``t50``          : first t_rel where distance crosses pre_median + 0.5 × Δpeak,
                     restricted to the pre-endpoint window so DTW endpoint-pinning
                     doesn't saturate the metric.
- ``t_peak``       : t_rel of argmax distance within the *interior* of the
                     aligned region (last 2 frames excluded — they're where
                     DTW endpoint-pinning crowds cells onto ``tc[-1]``).
- ``rise_rate_per_hour`` : slope of distance vs t_rel over the aligned region,
                           in Δcos per hour (not per minute).
- ``delta_peak``   : max(aligned distance) − median(pre distance).

Outputs:

- ``<stem>_per_cell.parquet`` : one row per cell with all metrics + dataset_id,
  fov_name, track_id, organelle_channel, length_normalized_cost.
- ``<stem>_summary.md``       : markdown summary — per-well medians, pooled
  median ± 95% bootstrap CI, rank-sum vs a reference organelle (optional).
- ``<stem>_strips.png``       : per-metric strip/violin comparing organelles
  (only meaningful when called twice with different organelles then merged).

Example::

    cd applications/dynaclr/scripts/pseudotime/3-organelle-remodeling
    uv run python compute_timing_metrics.py \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/align_cells.yaml \
        --template infection_nondividing_sensor --flavor raw \
        --query-set sensor_all_07_24 \
        --organelle-channel organelle_sec61 --top-n 30

Run twice (once per organelle) then pass both per-cell parquets to
``--compare`` to emit cross-organelle plots and stats.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dynaclr.pseudotime import (
    date_prefix_from_dataset_id,
    find_embedding_zarr,
    read_time_calibration,
)

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR.parent / "1-build_template" / "templates"
ALIGNMENTS_DIR = SCRIPT_DIR.parent / "2-align_cells" / "alignments"
OUT_DIR = SCRIPT_DIR / "timing"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

_date_prefix_from_dataset_id = date_prefix_from_dataset_id
_find_zarr = find_embedding_zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _top_n_cells(alignments: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select rows for top-N cells ranked by length-normalized DTW cost."""
    costs = alignments.groupby(["dataset_id", "fov_name", "track_id"])["length_normalized_cost"].first()
    top_keys = set(costs.sort_values().head(top_n).index)
    mask = [
        (ds, fov, tid) in top_keys
        for ds, fov, tid in zip(alignments["dataset_id"], alignments["fov_name"], alignments["track_id"])
    ]
    return alignments[mask].reset_index(drop=True)


def _join_organelle_embeddings(
    selected: pd.DataFrame,
    dataset_cfgs: dict[str, dict],
    organelle_pattern: str,
) -> pd.DataFrame:
    """Attach organelle embedding vectors via ``(fov, track, t)`` lookup."""
    parts = []
    for dataset_id, ds_align in selected.groupby("dataset_id"):
        ds_cfg = dataset_cfgs[dataset_id]
        prefix = _date_prefix_from_dataset_id(dataset_id)
        zarr_path = _find_zarr(ds_cfg["pred_dir"], prefix + organelle_pattern)
        adata = ad.read_zarr(zarr_path)
        adata.obs_names_make_unique()
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        obs = adata.obs.reset_index(drop=True)
        lookup = {(str(row["fov_name"]), int(row["track_id"]), int(row["t"])): i for i, row in obs.iterrows()}
        aligned_rows = ds_align.reset_index(drop=True).copy()
        embeddings = []
        for _, row in aligned_rows.iterrows():
            key = (str(row["fov_name"]), int(row["track_id"]), int(row["t"]))
            idx = lookup.get(key)
            embeddings.append(X[idx] if idx is not None else None)
        aligned_rows["embedding"] = embeddings
        aligned_rows = aligned_rows[aligned_rows["embedding"].notna()].reset_index(drop=True)
        parts.append(aligned_rows)
    return pd.concat(parts, ignore_index=True)


def _cosine_distance_from_baseline(joined: pd.DataFrame) -> np.ndarray:
    """Per-frame cosine distance to that cell's mean pre-event embedding."""
    distances = np.full(len(joined), np.nan, dtype=np.float64)
    for (_, _, _), group in joined.groupby(["dataset_id", "fov_name", "track_id"], sort=False):
        idx = group.index.to_numpy()
        emb = np.stack(group["embedding"].to_list())
        pre_mask = group["alignment_region"].to_numpy() == "pre"
        if pre_mask.any():
            baseline = emb[pre_mask].mean(axis=0)
        else:
            aligned_mask = group["alignment_region"].to_numpy() == "aligned"
            if not aligned_mask.any():
                continue
            earliest = aligned_mask.nonzero()[0][: max(1, aligned_mask.sum() // 4)]
            baseline = emb[earliest].mean(axis=0)
        bn = np.linalg.norm(baseline)
        en = np.linalg.norm(emb, axis=1)
        denom = bn * en
        cos_sim = np.where(denom > 0, (emb @ baseline) / np.where(denom > 0, denom, 1.0), 0.0)
        distances[idx] = 1.0 - cos_sim
    return distances


def _compute_per_cell_metrics(
    joined: pd.DataFrame,
    distances: np.ndarray,
    t_rel: np.ndarray,
) -> pd.DataFrame:
    """Return one row per (dataset_id, fov_name, track_id) with timing scalars."""
    joined = joined.copy()
    joined["distance"] = distances
    joined["t_rel"] = t_rel

    rows = []
    for (ds, fov, tid), grp in joined.groupby(["dataset_id", "fov_name", "track_id"], sort=False):
        aligned = grp[grp["alignment_region"] == "aligned"].sort_values("t_rel")
        pre = grp[grp["alignment_region"] == "pre"]
        if len(aligned) < 3:
            continue
        a_t = aligned["t_rel"].to_numpy(dtype=float)
        a_d = aligned["distance"].to_numpy(dtype=float)
        mask = np.isfinite(a_t) & np.isfinite(a_d)
        if mask.sum() < 3:
            continue
        a_t = a_t[mask]
        a_d = a_d[mask]

        pre_median = float(np.nanmedian(pre["distance"])) if len(pre) else float(a_d.min())

        # Drop the last 2 aligned frames for peak/t_peak/t50 — DTW endpoint
        # constraints pin many cells' warp paths onto tc[-1], crowding frames
        # at the last template position. The INTERIOR peak is what reflects
        # true remodeling amplitude; the endpoint pile-up is a warp-path artifact.
        interior_n = max(3, len(a_t) - 2)
        i_t = a_t[:interior_n]
        i_d = a_d[:interior_n]
        peak = float(i_d.max())
        delta_peak = peak - pre_median

        # t50 on the interior (half-rise in absolute units, not normalized).
        if delta_peak <= 1e-6:
            t50 = np.nan
        else:
            t50 = _first_crossing(i_t, i_d, pre_median + 0.5 * delta_peak)

        # Absolute-threshold onset (SNR-robust across cells with different Δpeak).
        t_onset_abs = _first_crossing(a_t, a_d, pre_median + 0.10)

        t_peak = float(i_t[int(np.argmax(i_d))])

        # Rise-rate in Δcos per hour (multiply per-minute slope by 60).
        if len(a_t) >= 2 and (a_t.max() - a_t.min()) > 1e-6:
            slope, _intercept, _r, _p, _se = stats.linregress(a_t, a_d)
            rise_rate_per_hour = float(slope) * 60.0
        else:
            rise_rate_per_hour = np.nan

        rows.append(
            {
                "dataset_id": ds,
                "fov_name": fov,
                "track_id": int(tid),
                "cell_uid": f"{ds}/{fov}/{tid}",
                "well": _extract_well(fov),
                "length_normalized_cost": float(grp["length_normalized_cost"].iloc[0]),
                "n_aligned_frames": int(len(a_t)),
                "pre_median_distance": pre_median,
                "peak_distance": peak,
                "delta_peak": delta_peak,
                "t_onset_abs": t_onset_abs,
                "t50": t50,
                "t_peak": t_peak,
                "rise_rate_per_hour": rise_rate_per_hour,
            }
        )
    return pd.DataFrame(rows)


def _first_crossing(t: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """First ``t`` value where the signal crosses ``threshold`` upward, linearly interpolated."""
    above = y >= threshold
    if not above.any():
        return float("nan")
    first_above = int(np.argmax(above))
    if first_above == 0:
        return float(t[0])
    t_before, t_after = t[first_above - 1], t[first_above]
    y_before, y_after = y[first_above - 1], y[first_above]
    if y_after == y_before:
        return float(t_after)
    frac = (threshold - y_before) / (y_after - y_before)
    return float(t_before + frac * (t_after - t_before))


def _extract_well(fov_name: str) -> str:
    """Return ``'A/2'`` from ``'A/2/000000'`` style FOV names, else full FOV."""
    parts = fov_name.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return fov_name


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (median, lo, hi) with a percentile bootstrap on the median."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.median(rng.choice(values, size=len(values), replace=True))
    med = float(np.median(values))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return med, lo, hi


def _summary_markdown(per_cell: pd.DataFrame, organelle_channel: str) -> str:
    """Render per-well + pooled median ± CI as markdown for copy to Confluence."""
    lines = []
    lines.append(f"# Timing metrics — {organelle_channel}")
    lines.append("")
    lines.append(f"**n cells**: {len(per_cell)}")
    lines.append("")

    lines.append("## Per-well medians")
    lines.append("")
    lines.append("| well | n | t_onset_abs (min) | t50 (min) | t_peak (min) | delta_peak | rise_rate (Δcos/hr) |")
    lines.append("|---|---|---|---|---|---|---|")
    for well, grp in per_cell.groupby("well"):
        lines.append(
            f"| {well} | {len(grp)} | "
            f"{grp['t_onset_abs'].median():.0f} | {grp['t50'].median():.0f} | "
            f"{grp['t_peak'].median():.0f} | {grp['delta_peak'].median():.3f} | "
            f"{grp['rise_rate_per_hour'].median():.3f} |"
        )
    lines.append("")

    lines.append("## Pooled median ± 95% bootstrap CI")
    lines.append("")
    lines.append("| metric | median | 95% CI |")
    lines.append("|---|---|---|")
    for metric in ["t_onset_abs", "t50", "t_peak", "delta_peak", "rise_rate_per_hour"]:
        med, lo, hi = _bootstrap_ci(per_cell[metric].to_numpy(dtype=float))
        lines.append(f"| {metric} | {med:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    return "\n".join(lines)


def _compare_organelles(per_cell_files: list[Path], out_stem: Path) -> None:
    """Merge per-cell parquets from multiple organelles, emit comparison plots + stats."""
    dfs = []
    for p in per_cell_files:
        df = pd.read_parquet(p)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    metrics = ["t_onset_abs", "t50", "t_peak", "delta_peak", "rise_rate_per_hour"]
    organelles = sorted(merged["organelle_channel"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 4.2), squeeze=False)
    axes = axes[0]
    colors = plt.get_cmap("tab10").colors
    for ax, metric in zip(axes, metrics):
        positions = np.arange(len(organelles))
        for i, org in enumerate(organelles):
            vals = merged.loc[merged["organelle_channel"] == org, metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full_like(vals, i, dtype=float) + jitter,
                vals,
                s=22,
                color=colors[i % len(colors)],
                alpha=0.7,
                edgecolor="none",
            )
            med, lo, hi = _bootstrap_ci(vals)
            ax.hlines(med, i - 0.25, i + 0.25, color="black", linewidth=2, zorder=5)
            ax.vlines(i, lo, hi, color="black", linewidth=1.2, zorder=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(organelles, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.axhline(
            0 if metric in {"t_onset_abs", "t50", "t_peak"} else ax.get_ylim()[0],
            color="red",
            linestyle=":",
            alpha=0.3,
            linewidth=0.8,
        )
        ax.set_title(metric)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _logger.info(f"Wrote {png_path}")

    lines = ["# Cross-organelle timing comparison", ""]
    lines.append(f"**Organelles**: {', '.join(organelles)}")
    lines.append("")
    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| organelle | n | median | 95% CI |")
        lines.append("|---|---|---|---|")
        per_org = {}
        for org in organelles:
            vals = merged.loc[merged["organelle_channel"] == org, metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            per_org[org] = vals
            med, lo, hi = _bootstrap_ci(vals)
            lines.append(f"| {org} | {len(vals)} | {med:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")
        if len(organelles) >= 2:
            lines.append("**Pairwise rank-sum tests (Mann-Whitney U, two-sided)**")
            lines.append("")
            lines.append("| a | b | median(a) − median(b) | U | p |")
            lines.append("|---|---|---|---|---|")
            for i in range(len(organelles)):
                for j in range(i + 1, len(organelles)):
                    a, b = per_org[organelles[i]], per_org[organelles[j]]
                    if len(a) >= 2 and len(b) >= 2:
                        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                        diff = float(np.median(a) - np.median(b))
                        lines.append(f"| {organelles[i]} | {organelles[j]} | {diff:.3f} | {u:.1f} | {p:.3g} |")
            lines.append("")

    md_path = out_stem.with_suffix(".md")
    md_path.write_text("\n".join(lines))
    _logger.info(f"Wrote {md_path}")


def main() -> None:
    """Compute per-cell timing metrics OR merge existing per-cell parquets for comparison."""
    parser = argparse.ArgumentParser(description="Per-cell timing metrics for organelle remodeling.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="Compute per-cell metrics for one organelle.")
    p_compute.add_argument("--datasets", required=True)
    p_compute.add_argument("--config", required=True)
    p_compute.add_argument("--template", required=True)
    p_compute.add_argument("--flavor", choices=["raw", "pca"], default="raw")
    p_compute.add_argument("--query-set", required=True)
    p_compute.add_argument("--organelle-channel", required=True)
    p_compute.add_argument("--top-n", type=int, default=30)

    p_compare = sub.add_parser("compare", help="Merge per-cell parquets across organelles.")
    p_compare.add_argument(
        "--per-cell", nargs="+", required=True, help="Paths to per-cell parquets from prior `compute` runs."
    )
    p_compare.add_argument("--out-stem", required=True, help="Output path stem (no extension).")

    args = parser.parse_args()

    if args.cmd == "compute":
        config = load_stage_config(args.datasets, args.config)
        dataset_cfgs = {d["dataset_id"]: d for d in config["datasets"]}
        if args.organelle_channel not in config["embeddings"]:
            raise ValueError(f"organelle-channel {args.organelle_channel!r} not found in embeddings")
        organelle_pattern = config["embeddings"][args.organelle_channel]

        alignment_path = ALIGNMENTS_DIR / f"{args.template}_{args.flavor}_on_{args.query_set}.parquet"
        if not alignment_path.exists():
            raise FileNotFoundError(f"Sensor alignment parquet not found: {alignment_path}")

        _logger.info(f"Reading sensor alignment {alignment_path}")
        alignments = pd.read_parquet(alignment_path)

        selected = _top_n_cells(alignments, args.top_n)
        frame_interval_by_ds = {d["dataset_id"]: float(d["frame_interval_minutes"]) for d in config["datasets"]}
        selected = selected.copy()
        selected["frame_interval"] = selected["dataset_id"].map(frame_interval_by_ds)

        template_path = TEMPLATES_DIR / f"template_{args.template}.zarr"
        try:
            tc = read_time_calibration(template_path, args.flavor)
        except KeyError:
            tc = None

        def _extrapolate_minutes(row: pd.Series) -> float:
            if row["alignment_region"] == "aligned":
                return float(row["estimated_t_rel_minutes"])
            fi = row["frame_interval"]
            if tc is None:
                return float("nan")
            if row["alignment_region"] == "pre":
                return float(tc[0] + (row["t"] - row["match_q_start"]) * fi)
            return float(tc[-1] + (row["t"] - row["match_q_end"]) * fi)

        selected["t_rel_minutes_extrap"] = selected.apply(_extrapolate_minutes, axis=1)

        joined = _join_organelle_embeddings(selected, dataset_cfgs, organelle_pattern)
        _logger.info(f"  {joined['cell_uid'].nunique()} cells after organelle join")

        distances = _cosine_distance_from_baseline(joined)
        t_rel = joined["t_rel_minutes_extrap"].to_numpy(dtype=float)

        per_cell = _compute_per_cell_metrics(joined, distances, t_rel)
        per_cell["organelle_channel"] = args.organelle_channel
        per_cell["template"] = args.template
        per_cell["flavor"] = args.flavor
        per_cell["query_set"] = args.query_set

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUT_DIR / f"timing_{args.template}_{args.flavor}_{args.organelle_channel}_{args.query_set}"
        per_cell_path = stem.with_name(stem.name + "_per_cell.parquet")
        per_cell.to_parquet(per_cell_path, index=False)
        _logger.info(f"Wrote {per_cell_path}  ({len(per_cell)} cells)")

        md = _summary_markdown(per_cell, args.organelle_channel)
        md_path = stem.with_name(stem.name + "_summary.md")
        md_path.write_text(md)
        _logger.info(f"Wrote {md_path}")

    elif args.cmd == "compare":
        _compare_organelles([Path(p) for p in args.per_cell], Path(args.out_stem))


if __name__ == "__main__":
    main()
