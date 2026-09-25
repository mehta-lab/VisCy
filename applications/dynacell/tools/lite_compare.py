r"""Check F: compare lite eval outputs against the full benchmark restricted to the same (FOV, frame).

1. Per-row metrics (``pixel_metrics.csv``, ``mask_metrics.csv``): a lite row
   ``(FOV, t_lite)`` maps to the full row ``(FOV, t_source)`` via the lite GT store's
   per-position ``temporal_subset.source_timepoints`` (written by
   ``build_temporal_subset_zarr.py``). Same checkpoint, same GT voxels, so differences
   measure predict/eval nondeterminism only.
2. Dataset-level deep KID / cosine: the lite value vs the SAME statistic recomputed with
   the pipeline's own ``_kid`` / ``_median_cosine_similarity`` on the full benchmark's
   per-cell embeddings restricted to the lite ``(FOV, t_source)`` cells. Differences =
   re-predicted embeddings + (A549) focus planes recomputed.
3. Rankings: per metric, Spearman of lite vs full-restricted across systems in each
   (bucket, train_set).

Nucleus only: the lite GT stores below carry the nucleus target.

Run::

    uv run --no-sync python applications/dynacell/tools/lite_compare.py --out /path/to/compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from scipy.stats import spearmanr

from dynacell.evaluation.feature_metrics import _kid, _median_cosine_similarity
from dynacell.evaluation.paths import DATA_ROOT, LITE_DATA_ROOT

ORG = "nucleus"
MODELS = ("fcmae_vscyto3d_pretrained", "fcmae_vscyto3d_scratch", "fnet3d_paper", "unetvit3d")
TRAINS = ("ipsc", "a549")
ROW_METRICS = {"pixel_metrics.csv": ["SI_SSIM", "Spectral_PCC", "PCC"], "mask_metrics.csv": ["instance_dice", "dice"]}
DEEP = {"celldino": "CellDINO", "dinov3": "DINOv3", "dynaclr": "DynaCLR", "morphem": "MorphEm"}
LITE_GT = {
    "ipsc": "ipsc/dataset_v4/test_cropped/cell.zarr",
    "a549__mock": "a549/mantis/test/dual_nucl_memb_mock.zarr",
    "a549__denv": "a549/mantis/test/dual_nucl_memb_DENV.zarr",
    "a549__zikv": "a549/mantis/test/dual_nucl_memb_ZIKV.zarr",
}


def frame_map(gt_store: Path) -> dict[tuple[str, int], int]:
    """Map ``(FOV, t_lite)`` to ``t_source`` from a lite GT store's per-position provenance.

    Parameters
    ----------
    gt_store : Path
        A store written by ``build_temporal_subset_zarr``.

    Returns
    -------
    dict
        ``(row/col/fov, t_lite)`` to the source store's timepoint index.

    Raises
    ------
    KeyError
        If a position carries no ``temporal_subset`` provenance.
    """
    out = {}
    with open_ome_zarr(gt_store, mode="r") as plate:
        for name, pos in plate.positions():
            for t_lite, t_src in enumerate(pos.zattrs["temporal_subset"]["source_timepoints"]):
                out[(name, t_lite)] = t_src
    return out


def per_row(lite_dir: Path, full_dir: Path, fmap: dict[tuple[str, int], int]) -> list[dict]:
    """Compare each lite per-row metric with the full row at the mapped source frame.

    Parameters
    ----------
    lite_dir, full_dir : Path
        Lite and full eval dirs of the same (model, train_set, bucket).
    fmap : dict
        Output of :func:`frame_map` for the bucket.

    Returns
    -------
    list of dict
        One record per metric present in both: row count, max / median absolute
        difference, lite mean and full-restricted mean.
    """
    rows = []
    for fname, metrics in ROW_METRICS.items():
        lite = pd.read_csv(lite_dir / fname)
        full = pd.read_csv(full_dir / fname).set_index(["FOV", "Timepoint"])
        lite["t_src"] = [fmap[(f, int(t))] for f, t in zip(lite.FOV, lite.Timepoint)]
        for m in metrics:
            if m not in lite.columns or m not in full.columns:
                continue
            ref = full.loc[list(zip(lite.FOV, lite.t_src)), m].to_numpy()
            d = np.abs(lite[m].to_numpy() - ref)
            rows.append(
                {
                    "metric": m,
                    "n_rows": len(d),
                    "max_abs_diff": float(np.nanmax(d)),
                    "median_abs_diff": float(np.nanmedian(d)),
                    "lite_mean": float(lite[m].mean()),
                    "full_restricted_mean": float(np.nanmean(ref)),
                }
            )
    return rows


def dataset_level(lite_dir: Path, full_dir: Path, fmap: dict[tuple[str, int], int]) -> list[dict]:
    """Compare lite dataset-level deep KID / cosine with the full embeddings restricted to the lite frames.

    Parameters
    ----------
    lite_dir, full_dir : Path
        Lite and full eval dirs of the same (model, train_set, bucket).
    fmap : dict
        Output of :func:`frame_map` for the bucket.

    Returns
    -------
    list of dict
        One record per (extractor, statistic): lite value, full-restricted value and
        the restricted GT cell count.
    """
    keep = {(f, t) for (f, _), t in fmap.items()}
    lite_row = pd.read_csv(lite_dir / "feature_metrics.csv").iloc[0]
    rows = []
    for key, prefix in DEEP.items():
        sides = {}
        for side in ("gt", "pred"):
            with np.load(full_dir / f"embeddings/{side}_{key}_single_cell_embeddings.npz") as npz:
                sel = np.array([(str(f), int(t)) in keep for f, t in zip(npz["fov"], npz["timepoint"])])
                sides[side] = npz["embeddings"][sel].astype(np.float32)
        kid, _ = _kid(sides["pred"], sides["gt"], 100, 1000, 2020)
        cos = _median_cosine_similarity(sides["pred"], sides["gt"])
        for stat, ref in (("KID", kid), ("Median_Cosine_Similarity", cos)):
            col = f"Dataset_{prefix}_{stat}"
            rows.append(
                {
                    "metric": f"{prefix}_{stat}",
                    "lite": float(lite_row[col]),
                    "full_restricted": float(ref),
                    "n_gt_cells": len(sides["gt"]),
                }
            )
    return rows


def compare(out: Path, lite_root: Path = LITE_DATA_ROOT, data_root: Path = DATA_ROOT) -> None:
    """Write ``per_row.csv``, ``dataset_level.csv``, ``ranks.csv`` and ``meta.json`` to ``out``.

    Every (model, train_set, bucket) with a lite ``feature_metrics.csv`` is compared.

    Parameters
    ----------
    out : Path
        Output directory (created).
    lite_root : Path
        Lite root holding the lite GT stores and eval dirs.
    data_root : Path
        Full benchmark root.
    """
    out.mkdir(parents=True, exist_ok=True)
    row_recs, ds_recs = [], []
    for bucket, gt_rel in LITE_GT.items():
        fmap = frame_map(lite_root / gt_rel)
        for model in MODELS:
            for train in TRAINS:
                lite_dir = lite_root / ORG / model / train / bucket
                full_dir = data_root / ORG / model / train / bucket
                if not (lite_dir / "feature_metrics.csv").exists():
                    continue
                tag = {"bucket": bucket, "model": model, "train": train}
                row_recs += [{**tag, **r} for r in per_row(lite_dir, full_dir, fmap)]
                ds_recs += [{**tag, **r} for r in dataset_level(lite_dir, full_dir, fmap)]
    rows, ds = pd.DataFrame(row_recs), pd.DataFrame(ds_recs)
    rows.to_csv(out / "per_row.csv", index=False)
    ds["rel_diff"] = (ds.lite - ds.full_restricted) / ds.full_restricted.abs()
    ds.to_csv(out / "dataset_level.csv", index=False)
    ranks = []
    for (bucket, train), g in rows.groupby(["bucket", "train"]):
        for m, gm in g.groupby("metric"):
            if len(gm) >= 3:
                ranks.append(
                    {
                        "bucket": bucket,
                        "train": train,
                        "metric": m,
                        "n_sys": len(gm),
                        "spearman": spearmanr(gm.lite_mean, gm.full_restricted_mean).statistic,
                    }
                )
        for m, gm in ds[(ds.bucket == bucket) & (ds.train == train)].groupby("metric"):
            if len(gm) >= 3:
                ranks.append(
                    {
                        "bucket": bucket,
                        "train": train,
                        "metric": m,
                        "n_sys": len(gm),
                        "spearman": spearmanr(gm.lite, gm.full_restricted).statistic,
                    }
                )
    pd.DataFrame(ranks).to_csv(out / "ranks.csv", index=False)
    print(rows.groupby("metric")[["max_abs_diff", "median_abs_diff"]].max().to_string())
    print(ds.groupby("metric").rel_diff.agg(["min", "median", "max"]).to_string())
    print(pd.DataFrame(ranks).groupby("metric").spearman.min().to_string())
    with open(out / "meta.json", "w") as f:
        json.dump({"n_dirs": int(rows[["bucket", "model", "train"]].drop_duplicates().shape[0])}, f)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run :func:`compare`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lite-root", type=Path, default=LITE_DATA_ROOT)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args(argv)
    compare(args.out, args.lite_root, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
