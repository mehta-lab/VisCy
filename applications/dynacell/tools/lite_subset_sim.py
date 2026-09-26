r"""Offline subset simulator for DynaCell-lite (block-sum accelerated).

How well do random FOV / timepoint subsets of a DynaCell test bucket reproduce the
full-bucket metric values and the cross-model ranking? Everything here is computed
offline from a finished full benchmark's eval dirs (``pixel_metrics.csv``,
``mask_metrics.csv``, ``feature_metrics.csv`` and the per-cell embedding ``.npz``),
with no re-prediction.

Speed: every (FOV, Timepoint) is a *block* of cells. The three poly-kernel matrices
are reduced once per system to block-sum matrices ``B[a, b]`` = sum of K over cells
in blocks a x b (plus per-block traces), so KID for any block multiset S is
O(|S|^2) instead of O(n^2 d).

KID estimator: a single unbiased MMD^2 over all selected cells (poly kernel degree 3,
gamma 1/d, coef0 1) -- identical to torch-fidelity's per-subset estimator, so it
equals the pipeline's ``100 x min(1000, n)`` subset mean exactly when n <= 1000 and
in expectation above that. GLCM+ (``cp``) is scored in the pipeline's CP space:
the target's CP reference mask plus the eval dataset's GT scaler (the parent's for a
lite dataset), clipped to the reference's z_clip, both read via the run's ``cp_selected_feature_mask.json`` sidecar.
That transform is fixed, so it does not depend on the cell subset; CP is still
scored directly through an exact degree-3 feature map (22-dim: cheap).

Run (on a compute node; the login node has one CPU)::

    uv run --no-sync python applications/dynacell/tools/lite_subset_sim.py \
        --org nucleus --bucket ipsc --out /path/to/out
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from collections.abc import Callable, Iterator
from math import comb, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from build_temporal_subset_zarr import spread_timepoints
from scipy.stats import kendalltau, spearmanr

from dynacell.evaluation.cp_reference import sidecar_cp_space
from dynacell.evaluation.paths import DATA_ROOT

EXTRACTORS = {"cp": "CP", "dinov3": "DINOv3", "dynaclr": "DynaCLR", "celldino": "CellDINO", "morphem": "MorphEm"}
PIXEL_METRICS = ["SI_SSIM", "Spectral_PCC", "PCC"]
MASK_METRICS = ["Dice", "instance_dice", "AP_0.50"]
MODELS_3D = [
    "fcmae_vscyto3d_scratch",
    "fcmae_vscyto3d_pretrained",
    "unetvit3d",
    "pix2pix3d_unetvit",
    "fnet3d_paper",
    "celldiff_r2",
    "celldiff_r2_iterative",
    "fnet3d_bigpatch",
    "fnet3d_vscyto3daug",
    "fnet3d_t01",
    "fnet3d_tspread",
]
MODELS_2D = ["fcmae_vscyto2d_scratch", "fcmae_vscyto2d_pretrained", "fnet2d", "celldiff_2d", "pix2pix2d_unetvit"]
TRAIN_SETS = ["ipsc", "a549", "joint"]
KID_MIN = 16
_EVAL_CSVS = ("pixel_metrics.csv", "mask_metrics.csv", "feature_metrics.csv")


def poly_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return torch-fidelity's KID kernel ``(x.y / d + 1)^3`` between the rows of X and Y.

    Parameters
    ----------
    X, Y : np.ndarray
        ``(n, d)`` and ``(m, d)`` feature arrays.

    Returns
    -------
    np.ndarray
        ``(n, m)`` kernel matrix.
    """
    return (X @ Y.T / X.shape[1] + 1.0) ** 3


def poly3_features(X: np.ndarray) -> np.ndarray:
    """Exact finite feature map for ``k(x, y) = (x.y/d + 1)^3``, so ``K = Phi Phi^T``.

    ``(u.v + 1)^3 = sum_m C(3,m) (u.v)^m`` with ``u = x/sqrt(d)``; ``(u.v)^m`` expands
    over multi-indices ``|alpha| = m`` with multinomial weights ``m!/prod(alpha!)``.
    The dimension for d = 12 is 455, so kernel sums over any cell subset cost
    O(n * 455) instead of O(n^2 d).

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` feature array.

    Returns
    -------
    np.ndarray
        ``(n, C(d + 3, 3))`` feature map.
    """
    n, d = X.shape
    U = X / np.sqrt(d)
    cols = [np.ones(n)]
    for m in (1, 2, 3):
        for combo in itertools.combinations_with_replacement(range(d), m):
            cnt = Counter(combo)
            w = comb(3, m) * factorial(m) / np.prod([factorial(c) for c in cnt.values()])
            f = np.sqrt(w) * np.ones(n)
            for k in combo:
                f = f * U[:, k]
            cols.append(f)
    return np.stack(cols, axis=1)


def mmd2_from_features(PX: np.ndarray, PY: np.ndarray) -> float:
    """Unbiased MMD^2 from explicit feature maps (rows = paired samples, equal counts).

    Parameters
    ----------
    PX, PY : np.ndarray
        ``(m, D)`` feature maps of the predicted and target samples.

    Returns
    -------
    float
        Unbiased MMD^2 with the diagonal removed from the within-side sums.
    """
    m = PX.shape[0]
    sx, sy = PX.sum(0), PY.sum(0)
    sum_xx, tr_xx = sx @ sx, np.einsum("ij,ij->", PX, PX)
    sum_yy, tr_yy = sy @ sy, np.einsum("ij,ij->", PY, PY)
    sum_xy = sx @ sy
    return (sum_xx - tr_xx + sum_yy - tr_yy) / (m * (m - 1)) - 2 * sum_xy / (m * m)


class System:
    """One (model, train_set) evaluated on one bucket, reduced to per-(FOV, T) blocks.

    Parameters
    ----------
    path : Path
        Eval dir ``<root>/<org>/<model>/<train_set>/<bucket>``.
    blocks : list of tuple of (str, int)
        The shared ``(FOV, Timepoint)`` universe, in block-index order.

    Raises
    ------
    ValueError
        If an extractor's pred and GT embeddings are not cell-aligned.
    """

    def __init__(self, path: Path, blocks: list[tuple[str, int]]):
        self.path = path
        self.name = f"{path.parts[-3]}/{path.parts[-2]}"
        self.block_index = {b: i for i, b in enumerate(blocks)}
        nb = len(blocks)
        px = pd.read_csv(path / "pixel_metrics.csv")
        mk = pd.read_csv(path / "mask_metrics.csv")
        rows = px.merge(mk, on=["FOV", "Timepoint"], how="outer", suffixes=("", "_mask"))
        rows["FOV"] = rows["FOV"].astype(str)
        self.row_vals = {}
        for c in PIXEL_METRICS + MASK_METRICS:
            v = np.full(nb, np.nan)
            if c in rows:
                for f, t, x in zip(rows.FOV, rows.Timepoint, rows[c]):
                    if (f, t) in self.block_index:
                        v[self.block_index[(f, t)]] = x
            self.row_vals[c] = v
        ft = pd.read_csv(path / "feature_metrics.csv")
        self.csv_dataset = {
            c: float(ft[c].iloc[0])
            for c in ft.columns
            if c.startswith("Dataset_") and (c.endswith("_KID") or c.endswith("_Median_Cosine_Similarity"))
        }
        self.emb = {}
        self.cp = None
        for tok, prefix in EXTRACTORS.items():
            g = path / "embeddings" / f"gt_{tok}_single_cell_embeddings.npz"
            p = path / "embeddings" / f"pred_{tok}_single_cell_embeddings.npz"
            if not (g.exists() and p.exists()):
                continue
            G, P = np.load(g, allow_pickle=True), np.load(p, allow_pickle=True)
            X = np.asarray(P["embeddings"], dtype=np.float64)
            Y = np.asarray(G["embeddings"], dtype=np.float64)
            if X.shape != Y.shape or not np.array_equal(P["fov"], G["fov"]):
                raise ValueError(f"{self.name} {tok}: pred/gt cells not aligned")
            keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
            X, Y = X[keep], Y[keep]
            fov = P["fov"].astype(str)[keep]
            tp = P["timepoint"].astype(int)[keep]
            cell_block = np.array([self.block_index.get((f, t), -1) for f, t in zip(fov, tp)])
            ok = cell_block >= 0
            X, Y, cell_block = X[ok], Y[ok], cell_block[ok]
            if tok == "cp":
                # pipeline: the CP reference mask + this dataset's GT scaler, clipped, on both sides.
                space = sidecar_cp_space(path)
                self.cp = dict(X=space.transform_clipped(X), Y=space.transform_clipped(Y), cell_block=cell_block)
                continue
            # one-hot block membership (n x nb) -> block sums via M^T K M
            M = np.zeros((len(cell_block), nb))
            M[np.arange(len(cell_block)), cell_block] = 1.0
            K_XX, K_YY, K_XY = poly_kernel(X, X), poly_kernel(Y, Y), poly_kernel(X, Y)
            cos = np.einsum("ij,ij->i", X, Y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
            self.emb[prefix] = dict(
                B_XX=M.T @ K_XX @ M,
                B_YY=M.T @ K_YY @ M,
                B_XY=M.T @ K_XY @ M,
                tr_XX=M.T @ np.diag(K_XX),
                tr_YY=M.T @ np.diag(K_YY),
                n_block=M.sum(0),
                cos=cos,
                cell_block=cell_block,
            )

    def kid(self, prefix: str, w: np.ndarray) -> float:
        """KID over the multiset of blocks with multiplicities ``w``.

        Exact for a multiset: duplicated cells count as distinct samples, and only one
        self-pair per copy is removed from the within-side sums.

        Parameters
        ----------
        prefix : str
            Deep extractor display prefix (a value of :data:`EXTRACTORS`, not ``CP``).
        w : np.ndarray
            ``(n_blocks,)`` block multiplicities.

        Returns
        -------
        float
            Unbiased MMD^2, or NaN below :data:`KID_MIN` cells.
        """
        e = self.emb[prefix]
        m = (w * e["n_block"]).sum()
        if m < KID_MIN:
            return np.nan
        sxx = w @ e["B_XX"] @ w - (w * e["tr_XX"]).sum()
        syy = w @ e["B_YY"] @ w - (w * e["tr_YY"]).sum()
        sxy = w @ e["B_XY"] @ w
        return (sxx + syy) / (m * (m - 1)) - 2 * sxy / (m * m)

    def metrics(self, sel: np.ndarray) -> dict[str, float]:
        """All metrics over the (multi)set of block indices ``sel``.

        Parameters
        ----------
        sel : np.ndarray
            Block indices; a duplicated index weights that block twice.

        Returns
        -------
        dict of str to float
            Row-mean pixel/mask metrics, ``<prefix>_KID`` / ``<prefix>_MedCos`` per
            extractor, and ``n_cells``.
        """
        out = {c: float(np.nanmean(v[sel])) if np.isfinite(v[sel]).any() else np.nan for c, v in self.row_vals.items()}
        w = np.bincount(sel, minlength=len(self.block_index)).astype(float)
        out["n_cells"] = np.nan
        for prefix, e in self.emb.items():
            out[f"{prefix}_KID"] = self.kid(prefix, w)
            reps = w[e["cell_block"]].astype(int)
            cos = np.repeat(e["cos"], reps)
            out[f"{prefix}_MedCos"] = float(np.nanmedian(cos)) if cos.size else np.nan
            out["n_cells"] = float(reps.sum())
        if self.cp is not None:
            reps = w[self.cp["cell_block"]].astype(int)
            X, Y = np.repeat(self.cp["X"], reps, axis=0), np.repeat(self.cp["Y"], reps, axis=0)
            m = X.shape[0]
            if m >= KID_MIN:
                out["CP_KID"] = mmd2_from_features(poly3_features(X), poly3_features(Y))
                cos = np.einsum("ij,ij->i", X, Y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
                out["CP_MedCos"] = float(np.nanmedian(cos))
            else:
                out["CP_KID"] = out["CP_MedCos"] = np.nan
        return out


def load_systems(
    org: str,
    bucket: str,
    models: list[str],
    data_root: Path = DATA_ROOT,
) -> tuple[list[System], list[tuple[str, int]]]:
    """Load every (model, train_set) eval dir of one bucket that has all three eval CSVs.

    Parameters
    ----------
    org : str
        Organelle token (``nucleus``, ``membrane``, ``er``, ``mito``).
    bucket : str
        Test bucket (``ipsc``, ``a549__mock``, ...).
    models : list of str
        Model code names to look for, crossed with :data:`TRAIN_SETS`.
    data_root : Path
        Benchmark root laid out as ``<org>/<model>/<train_set>/<bucket>``.

    Returns
    -------
    systems : list of System
        One per eval dir found.
    blocks : list of tuple of (str, int)
        Sorted ``(FOV, Timepoint)`` rows common to every system's ``pixel_metrics.csv``.
    """
    paths = [data_root / org / m / ts / bucket for m in models for ts in TRAIN_SETS]
    paths = [p for p in paths if all((p / f).exists() for f in _EVAL_CSVS)]
    key_sets = []
    for p in paths:
        px = pd.read_csv(p / "pixel_metrics.csv")
        key_sets.append(set(zip(px.FOV.astype(str), px.Timepoint.astype(int))))
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    if common != union:
        print(
            f"  {org}/{bucket}: row universe differs across systems (common {len(common)}, union {len(union)}); "
            "using common"
        )
    blocks = sorted(common)
    return [System(p, blocks) for p in paths], blocks


def t_rules() -> dict[str, Callable[[int], list[int]] | None]:
    """Return the per-FOV timepoint rules by design suffix.

    ``None`` entries depend on the FOV's ordinal and are resolved in :func:`subset_keys`:
    ``t_ord`` is ``ordinal % T`` and ``t<n>tool`` is exactly what
    ``build_temporal_subset_zarr.py --mode spread -n <n>`` selects.

    Returns
    -------
    dict
        Rule name to ``T -> timepoints`` callable, or ``None``.
    """
    return {
        "t0": lambda T: [0],
        "tlast": lambda T: [T - 1],
        "tmid": lambda T: [T // 2],
        "t2spread": lambda T: sorted({0, T - 1}),
        "t3spread": lambda T: sorted(set(np.linspace(0, T - 1, 3).round().astype(int))),
        "t5spread": lambda T: sorted(set(np.linspace(0, T - 1, 5).round().astype(int))),
        "tall": lambda T: list(range(T)),
        "t_ord": None,
        "t2tool": None,
        "t3tool": None,
        "t5tool": None,
    }


def designs_for(bucket: str, n_fov: int) -> Iterator[tuple[str, int, str]]:
    """Yield the ``(design_name, k_fov, t_rule)`` grid for one bucket.

    iPSC (one timepoint per FOV) varies only the FOV count; A549 crosses FOV counts
    with every :func:`t_rules` rule.

    Parameters
    ----------
    bucket : str
        Test bucket.
    n_fov : int
        FOVs in the bucket's full set.

    Yields
    ------
    tuple of (str, int, str)
        Design name, FOVs to draw, timepoint rule name.
    """
    if bucket == "ipsc":
        for k in [5, 10, 15, 20, 30, 50, 75]:
            if k < n_fov:
                yield f"fov{k:03d}", k, "tall"
    else:
        for k in [3, 6, 9, 12]:
            if k > n_fov:
                continue
            for tn in t_rules():
                if k == n_fov and tn == "tall":
                    continue
                yield f"fov{k:02d}_{tn}", k, tn


def subset_keys(
    fovs: list[str],
    t_of: dict[str, list[int]],
    bidx: dict[tuple[str, int], int],
    chosen: list[str],
    tn: str,
) -> list[tuple[str, int]]:
    """Return the ``(FOV, Timepoint)`` rows a design keeps for the chosen FOVs.

    Parameters
    ----------
    fovs : list of str
        All FOVs of the bucket, sorted; a FOV's ordinal is its index here.
    t_of : dict of str to list of int
        Timepoints present per FOV.
    bidx : dict
        ``(FOV, Timepoint)`` to block index; rows outside it are dropped.
    chosen : list of str
        FOVs in the subset.
    tn : str
        Timepoint rule name (a key of :func:`t_rules`).

    Returns
    -------
    list of tuple of (str, int)
        Kept rows.
    """
    if tn == "t_ord":
        return [(f, fovs.index(f) % len(t_of[f])) for f in chosen if (f, fovs.index(f) % len(t_of[f])) in bidx]
    if tn.endswith("tool"):
        n_keep = int(tn[1])
        return [
            (f, t) for f in chosen for t in spread_timepoints(fovs.index(f), len(t_of[f]), n_keep) if (f, t) in bidx
        ]
    return [(f, t) for f in chosen for t in t_rules()[tn](len(t_of[f])) if (f, t) in bidx]


def run(
    org: str,
    bucket: str,
    reps: int,
    include_2d: bool,
    out: Path,
    designs: list[str] | None = None,
    data_root: Path = DATA_ROOT,
) -> None:
    """Score every design of one bucket over ``reps`` random FOV draws and write the records.

    Writes ``kidcheck_<org>_<bucket>.csv`` (block-sum full-set KID vs the CSV's),
    ``records_<org>_<bucket>[_<designs>].csv`` and ``full_<org>_<bucket>.csv`` to ``out``.

    Parameters
    ----------
    org, bucket : str
        Organelle and test bucket.
    reps : int
        Random FOV draws per design (seeded, ``default_rng(0)``).
    include_2d : bool
        Add :data:`MODELS_2D` to the roster.
    out : Path
        Output directory.
    designs : list of str, optional
        Restrict to these design names.
    data_root : Path
        Benchmark root.
    """
    models = MODELS_3D + (MODELS_2D if include_2d else [])
    systems, blocks = load_systems(org, bucket, models, data_root)
    if len(systems) < 3:
        print(f"{org}/{bucket}: only {len(systems)} systems, skipping")
        return
    fovs = sorted({f for f, _ in blocks})
    t_of = {f: sorted(t for ff, t in blocks if ff == f) for f in fovs}
    bidx = {b: i for i, b in enumerate(blocks)}
    all_sel = np.arange(len(blocks))
    full = {s.name: s.metrics(all_sel) for s in systems}
    xc = []
    for s in systems:
        for prefix in EXTRACTORS.values():
            k = f"Dataset_{prefix}_KID"
            kc = f"Dataset_{prefix}_Median_Cosine_Similarity"
            if k in s.csv_dataset and f"{prefix}_KID" in full[s.name]:
                xc.append(
                    dict(
                        org=org,
                        bucket=bucket,
                        system=s.name,
                        extractor=prefix,
                        ours=full[s.name][f"{prefix}_KID"],
                        csv=s.csv_dataset[k],
                        ours_cos=full[s.name].get(f"{prefix}_MedCos"),
                        csv_cos=s.csv_dataset.get(kc),
                    )
                )
    pd.DataFrame(xc).to_csv(out / f"kidcheck_{org}_{bucket}.csv", index=False)
    records = []
    rng = np.random.default_rng(0)
    for dname, k, tn in designs_for(bucket, len(fovs)):
        if designs and dname not in designs:
            continue
        for rep in range(reps):
            chosen = rng.choice(fovs, k, replace=False) if k < len(fovs) else fovs
            keys = subset_keys(fovs, t_of, bidx, chosen, tn)
            sel = np.array(sorted(bidx[kk] for kk in keys))
            for s in systems:
                sub = s.metrics(sel)
                for metric, v in sub.items():
                    records.append(
                        dict(
                            org=org,
                            bucket=bucket,
                            design=dname,
                            k_fov=k,
                            n_rows=len(sel),
                            rep=rep,
                            system=s.name,
                            metric=metric,
                            full=full[s.name].get(metric, np.nan),
                            subset=v,
                        )
                    )
            if k == len(fovs):
                break  # FOV choice is deterministic; T rule is deterministic
    records_df = pd.DataFrame.from_records(records)
    suffix = "" if not designs else "_" + "-".join(designs)
    records_df.to_csv(out / f"records_{org}_{bucket}{suffix}.csv", index=False)
    pd.DataFrame(full).T.to_csv(out / f"full_{org}_{bucket}.csv")
    print(
        f"{org}/{bucket}: {len(systems)} systems, {len(fovs)} FOVs, {len(blocks)} rows -> {len(records_df)} records",
        flush=True,
    )


def summarize(out: Path) -> pd.DataFrame:
    """Aggregate every ``records_*.csv`` in ``out`` into per-design ranking agreement.

    Parameters
    ----------
    out : Path
        Directory holding the :func:`run` records; ``summary.csv`` is written here.

    Returns
    -------
    pd.DataFrame
        One row per (org, bucket, design, metric): Spearman / Kendall of subset vs full
        across systems, MAE (absolute and over the full spread), pair flip rate, top-1
        agreement.
    """
    files = sorted(out.glob("records_*.csv"))
    records = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    records = records[records.metric != "n_cells"]
    rows = []
    for (org, bucket, design, metric), g in records.groupby(["org", "bucket", "design", "metric"]):
        sp, kt, mae, flips, nrm, top = [], [], [], [], [], []
        for _, gr in g.groupby("rep"):
            gr = gr.dropna(subset=["full", "subset"])
            if len(gr) < 3:
                continue
            sp.append(spearmanr(gr.full, gr.subset).correlation)
            kt.append(kendalltau(gr.full, gr.subset).correlation)
            mae.append(np.mean(np.abs(gr.subset - gr.full)))
            spread = gr.full.max() - gr.full.min()
            nrm.append(np.mean(np.abs(gr.subset - gr.full)) / spread if spread > 0 else np.nan)
            f, s = gr.full.to_numpy(), gr.subset.to_numpy()
            pairs = [(i, j) for i, j in itertools.combinations(range(len(f)), 2) if f[i] != f[j]]
            flips.append(np.mean([np.sign(f[i] - f[j]) != np.sign(s[i] - s[j]) for i, j in pairs]) if pairs else np.nan)
            top.append(np.argmax(f) == np.argmax(s) if "KID" not in metric else np.argmin(f) == np.argmin(s))
        if not sp:
            continue
        rows.append(
            dict(
                org=org,
                bucket=bucket,
                design=design,
                metric=metric,
                n_sys=g.system.nunique(),
                n_rows=int(g.n_rows.iloc[0]),
                reps=len(sp),
                spearman_mean=np.mean(sp),
                spearman_p10=np.percentile(sp, 10),
                kendall_mean=np.mean(kt),
                mae=np.mean(mae),
                mae_over_spread=np.nanmean(nrm),
                pair_flip_rate=np.nanmean(flips),
                top1_agree=np.mean(top),
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the subset grid for the requested organelles and buckets, then summarize."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", nargs="+", default=["nucleus", "membrane", "er", "mito"])
    ap.add_argument("--bucket", nargs="+", default=["ipsc", "a549__mock", "a549__denv", "a549__zikv"])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--include-2d", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--designs", nargs="*", default=None, help="restrict to these design names, e.g. fov12_t_ord")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(exist_ok=True)
    if not a.summarize_only:
        for org in a.org:
            for bucket in a.bucket:
                run(org, bucket, a.reps, a.include_2d, a.out, a.designs, a.data_root)
    summary = summarize(a.out)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
