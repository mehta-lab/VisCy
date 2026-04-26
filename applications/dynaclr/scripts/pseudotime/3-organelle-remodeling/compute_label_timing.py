r"""Per-cell label-timing metrics from linear classifier predictions (Stage 3).

Embedding-based timing (``compute_timing_metrics.py``) measures cosine
distance from each cell's pre-baseline. This script is the label-side
complement: it reduces each cell's *predicted label* trajectory to timing
scalars. Both scripts share the sensor-aligned ``t_rel`` axis, so their
outputs are directly comparable.

Label taxonomy (~/memory/project_label_taxonomy.md):

- ``{state}``            : human annotation (sparse).
- ``predicted_{state}``  : linear classifier output (dense). **Used here.**
- ``dtw_{state}``        : DTW-propagated template label (aligned-only).

Per-cell metrics on the binarized predicted-label trajectory (1 = positive):

- ``t_first_pos``     : first t_rel where the cell is predicted positive.
- ``t_run_start``     : first t_rel where the cell enters a run of
                        ``min_run`` consecutive positive predictions
                        (default 3). Robust to single-frame flicker.
- ``t_run_end``       : last t_rel where the cell is in a positive run.
- ``pos_duration``    : ``t_run_end − t_run_start`` (minutes).
- ``pos_fraction``    : fraction of aligned frames predicted positive.
- ``flips``           : number of 0→1 or 1→0 transitions over the full track.

Outputs:

- ``<stem>_per_cell.parquet`` : one row per cell.
- ``<stem>_summary.md``        : per-well + pooled median ± bootstrap CI.

Example::

    cd applications/dynaclr/scripts/pseudotime/3-organelle-remodeling
    uv run python compute_label_timing.py compute \
        --datasets ../../../configs/pseudotime/datasets.yaml \
        --config ../../../configs/pseudotime/align_cells.yaml \
        --template infection_nondividing_sensor --flavor raw \
        --query-set sensor_all_07_24 \
        --organelle-channel organelle_sec61 \
        --state-column organelle_state --state-positive remodel \
        --top-n 30

Pair a SEC61 and G3BP1 run then::

    uv run python compute_label_timing.py compare \
        --per-cell timing_labels/..._sec61_..._per_cell.parquet \
                   timing_labels/..._g3bp1_..._per_cell.parquet \
        --out-stem timing_labels/compare_sec61_vs_g3bp1
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
OUT_DIR = SCRIPT_DIR / "timing_labels"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils import load_stage_config  # noqa: E402

_date_prefix_from_dataset_id = date_prefix_from_dataset_id
_find_zarr = find_embedding_zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _top_n_cells(alignments: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select rows belonging to the top-N cells by length-normalized DTW cost."""
    costs = alignments.groupby(["dataset_id", "fov_name", "track_id"])["length_normalized_cost"].first()
    top_keys = set(costs.sort_values().head(top_n).index)
    mask = [
        (ds, fov, tid) in top_keys
        for ds, fov, tid in zip(alignments["dataset_id"], alignments["fov_name"], alignments["track_id"])
    ]
    return alignments[mask].reset_index(drop=True)


def _lookup_predicted_labels(
    selected: pd.DataFrame,
    dataset_cfgs: dict[str, dict],
    organelle_pattern: str,
    predicted_column: str,
    positive_value: str,
) -> np.ndarray:
    """Per-row binarized predicted-label value (1.0, 0.0, or NaN if missing)."""
    labels = np.full(len(selected), np.nan, dtype=np.float64)
    for dataset_id, ds_rows in selected.groupby("dataset_id"):
        ds_cfg = dataset_cfgs[dataset_id]
        prefix = _date_prefix_from_dataset_id(dataset_id)
        zarr_path = _find_zarr(ds_cfg["pred_dir"], prefix + organelle_pattern)
        adata = ad.read_zarr(zarr_path)
        adata.obs_names_make_unique()
        if predicted_column not in adata.obs.columns:
            _logger.warning(f"  [{dataset_id}] obs has no {predicted_column!r} column — skipping")
            continue
        lookup = {
            (str(row["fov_name"]), int(row["track_id"]), int(row["t"])): str(row[predicted_column])
            for _, row in adata.obs.iterrows()
        }
        for idx_local, row in enumerate(ds_rows.itertuples(index=False)):
            key = (str(row.fov_name), int(row.track_id), int(row.t))
            val = lookup.get(key)
            if val is None or val == "nan":
                continue
            global_idx = ds_rows.index[idx_local]
            labels[global_idx] = 1.0 if val == positive_value else 0.0
    return labels


def _longest_positive_run(is_pos: np.ndarray, min_run: int) -> tuple[int, int] | None:
    """Return (start_idx, end_idx) of the earliest run of ≥``min_run`` consecutive True values."""
    in_run = False
    run_start = -1
    for i, v in enumerate(is_pos):
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            if i - run_start >= min_run:
                return run_start, i - 1
            in_run = False
    if in_run and len(is_pos) - run_start >= min_run:
        return run_start, len(is_pos) - 1
    return None


def _compute_per_cell(
    selected: pd.DataFrame,
    labels: np.ndarray,
    t_rel: np.ndarray,
    min_run: int,
) -> pd.DataFrame:
    """Return one row per (dataset_id, fov, track_id) with label-timing scalars."""
    df = selected.copy()
    df["predicted_pos"] = labels
    df["t_rel"] = t_rel

    rows = []
    for (ds, fov, tid), grp in df.groupby(["dataset_id", "fov_name", "track_id"], sort=False):
        grp = grp.sort_values("t_rel")
        y = grp["predicted_pos"].to_numpy(dtype=float)
        t = grp["t_rel"].to_numpy(dtype=float)
        aligned_mask = grp["alignment_region"].to_numpy() == "aligned"
        mask = np.isfinite(y) & np.isfinite(t)
        if mask.sum() < 3:
            continue
        y = y[mask]
        t = t[mask]
        aligned_mask = aligned_mask[mask]

        is_pos = y == 1.0
        flips = int(np.abs(np.diff(y)).sum())

        if is_pos.any():
            t_first_pos = float(t[int(np.argmax(is_pos))])
        else:
            t_first_pos = np.nan

        run = _longest_positive_run(is_pos, min_run=min_run)
        if run is not None:
            t_run_start = float(t[run[0]])
            t_run_end = float(t[run[1]])
            pos_duration = t_run_end - t_run_start
        else:
            t_run_start = np.nan
            t_run_end = np.nan
            pos_duration = np.nan

        if aligned_mask.any():
            pos_fraction = float(is_pos[aligned_mask].mean())
        else:
            pos_fraction = float(is_pos.mean())

        rows.append(
            {
                "dataset_id": ds,
                "fov_name": fov,
                "track_id": int(tid),
                "cell_uid": f"{ds}/{fov}/{tid}",
                "well": _extract_well(fov),
                "length_normalized_cost": float(grp["length_normalized_cost"].iloc[0]),
                "n_frames_labeled": int(mask.sum()),
                "t_first_pos": t_first_pos,
                "t_run_start": t_run_start,
                "t_run_end": t_run_end,
                "pos_duration": pos_duration,
                "pos_fraction": pos_fraction,
                "flips": flips,
            }
        )
    return pd.DataFrame(rows)


def _extract_well(fov_name: str) -> str:
    """Return ``'A/2'`` from ``'A/2/000000'`` style FOV names."""
    parts = fov_name.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return fov_name


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap on the median."""
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
    return float(np.median(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _summary_markdown(per_cell: pd.DataFrame, state_column: str, organelle_channel: str) -> str:
    """Per-well + pooled markdown summary."""
    lines = [f"# Label-timing metrics — predicted_{state_column} ({organelle_channel})", ""]
    lines.append(f"**n cells**: {len(per_cell)}")
    lines.append("")
    lines.append("## Per-well medians")
    lines.append("")
    header = (
        "| well | n | t_first_pos (min) | t_run_start (min) | t_run_end (min) | "
        "pos_duration (min) | pos_fraction | flips |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for well, grp in per_cell.groupby("well"):
        lines.append(
            f"| {well} | {len(grp)} | "
            f"{grp['t_first_pos'].median():.0f} | {grp['t_run_start'].median():.0f} | "
            f"{grp['t_run_end'].median():.0f} | {grp['pos_duration'].median():.0f} | "
            f"{grp['pos_fraction'].median():.3f} | {grp['flips'].median():.0f} |"
        )
    lines.append("")
    lines.append("## Pooled median ± 95% bootstrap CI")
    lines.append("")
    lines.append("| metric | median | 95% CI |")
    lines.append("|---|---|---|")
    for metric in ["t_first_pos", "t_run_start", "t_run_end", "pos_duration", "pos_fraction", "flips"]:
        med, lo, hi = _bootstrap_ci(per_cell[metric].to_numpy(dtype=float))
        lines.append(f"| {metric} | {med:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    return "\n".join(lines)


def _compare(per_cell_files: list[Path], out_stem: Path, group_by: str | None = None) -> None:
    """Merge per-cell parquets, emit strips + stats grouped by a column.

    Parameters
    ----------
    per_cell_files : list[Path]
        Per-cell parquets written by ``compute``.
    out_stem : Path
        Output path stem (no extension).
    group_by : str or None
        Column to group cells by in the comparison plot/stats. If ``None``
        (default), auto-select: use ``organelle_channel`` when multiple
        organelle values are present, otherwise fall back to ``query_set``
        so cross-virus pools (same organelle, different query sets) split
        correctly.
    """
    dfs = [pd.read_parquet(p) for p in per_cell_files]
    merged = pd.concat(dfs, ignore_index=True)

    metrics = ["t_first_pos", "t_run_start", "t_run_end", "pos_duration", "pos_fraction", "flips"]
    if group_by is None:
        n_organelles = len(merged["organelle_channel"].unique())
        group_by = "organelle_channel" if n_organelles > 1 else "query_set"
    organelles = sorted(merged[group_by].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 4.2), squeeze=False)
    axes = axes[0]
    colors = plt.get_cmap("tab10").colors
    for ax, metric in zip(axes, metrics):
        for i, org in enumerate(organelles):
            vals = merged.loc[merged[group_by] == org, metric].to_numpy(dtype=float)
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
        ax.set_xticks(np.arange(len(organelles)))
        ax.set_xticklabels(organelles, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _logger.info(f"Wrote {png}")

    lines = [
        "# Label-timing comparison",
        "",
        f"**Grouped by**: `{group_by}`",
        f"**Groups**: {', '.join(organelles)}",
        "",
    ]
    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append(f"| {group_by} | n | median | 95% CI |")
        lines.append("|---|---|---|---|")
        per_org = {}
        for org in organelles:
            vals = merged.loc[merged[group_by] == org, metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            per_org[org] = vals
            med, lo, hi = _bootstrap_ci(vals)
            lines.append(f"| {org} | {len(vals)} | {med:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")
        if len(organelles) >= 2:
            lines.append("**Pairwise rank-sum tests**")
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

    md = out_stem.with_suffix(".md")
    md.write_text("\n".join(lines))
    _logger.info(f"Wrote {md}")


def main() -> None:
    """Compute per-cell label timing OR merge across organelles."""
    parser = argparse.ArgumentParser(description="Per-cell label-timing from LC predictions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("compute")
    p_c.add_argument("--datasets", required=True)
    p_c.add_argument("--config", required=True)
    p_c.add_argument("--template", required=True)
    p_c.add_argument("--flavor", choices=["raw", "pca"], default="raw")
    p_c.add_argument("--query-set", required=True)
    p_c.add_argument("--organelle-channel", required=True)
    p_c.add_argument(
        "--state-column", required=True, help="Base state column; the script looks up 'predicted_{state_column}'."
    )
    p_c.add_argument("--state-positive", required=True)
    p_c.add_argument("--top-n", type=int, default=30)
    p_c.add_argument(
        "--min-run", type=int, default=3, help="Minimum consecutive positive frames for t_run_start (flicker filter)."
    )

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--per-cell", nargs="+", required=True)
    p_cmp.add_argument("--out-stem", required=True)
    p_cmp.add_argument(
        "--group-by",
        default=None,
        help=(
            "Column to split cells by. Default auto-picks organelle_channel "
            "if multiple organelles are present, else query_set (so cross-virus "
            "pools with the same organelle split correctly)."
        ),
    )

    args = parser.parse_args()

    if args.cmd == "compute":
        config = load_stage_config(args.datasets, args.config)
        dataset_cfgs = {d["dataset_id"]: d for d in config["datasets"]}
        if args.organelle_channel not in config["embeddings"]:
            raise ValueError(f"organelle-channel {args.organelle_channel!r} not in embeddings")
        organelle_pattern = config["embeddings"][args.organelle_channel]

        alignment_path = ALIGNMENTS_DIR / f"{args.template}_{args.flavor}_on_{args.query_set}.parquet"
        if not alignment_path.exists():
            raise FileNotFoundError(alignment_path)
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

        def _extrapolate(row):
            if row["alignment_region"] == "aligned":
                return float(row["estimated_t_rel_minutes"])
            fi = row["frame_interval"]
            if tc is None:
                return float("nan")
            if row["alignment_region"] == "pre":
                return float(tc[0] + (row["t"] - row["match_q_start"]) * fi)
            return float(tc[-1] + (row["t"] - row["match_q_end"]) * fi)

        selected["t_rel_minutes_extrap"] = selected.apply(_extrapolate, axis=1)

        predicted_col = f"predicted_{args.state_column}"
        _logger.info(f"Looking up {predicted_col!r} (positive={args.state_positive!r}) from {organelle_pattern}")
        labels = _lookup_predicted_labels(selected, dataset_cfgs, organelle_pattern, predicted_col, args.state_positive)
        n_labeled = int(np.isfinite(labels).sum())
        _logger.info(f"  {n_labeled}/{len(labels)} rows labeled")
        if n_labeled == 0:
            raise RuntimeError(
                f"No rows had {predicted_col!r}. Has the linear classifier been run for this dataset/organelle?"
            )

        t_rel = selected["t_rel_minutes_extrap"].to_numpy(dtype=float)
        per_cell = _compute_per_cell(selected, labels, t_rel, min_run=args.min_run)
        per_cell["organelle_channel"] = args.organelle_channel
        per_cell["state_column"] = args.state_column
        per_cell["template"] = args.template
        per_cell["flavor"] = args.flavor
        per_cell["query_set"] = args.query_set

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUT_DIR / (
            f"label_timing_{args.template}_{args.flavor}_{args.organelle_channel}_{args.state_column}_{args.query_set}"
        )
        parquet = stem.with_name(stem.name + "_per_cell.parquet")
        per_cell.to_parquet(parquet, index=False)
        _logger.info(f"Wrote {parquet}  ({len(per_cell)} cells)")

        md = _summary_markdown(per_cell, args.state_column, args.organelle_channel)
        md_path = stem.with_name(stem.name + "_summary.md")
        md_path.write_text(md)
        _logger.info(f"Wrote {md_path}")

    elif args.cmd == "compare":
        _compare([Path(p) for p in args.per_cell], Path(args.out_stem), group_by=args.group_by)


if __name__ == "__main__":
    main()
