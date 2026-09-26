r"""Checks B, B2 and E of the DynaCell-lite plan, on top of :mod:`lite_subset_sim`.

B. Resolvable-pair gate. Bootstrap the FULL set over FOVs (all T of a resampled FOV
   kept, the same draw for every system) to get each pair's paired SE per metric. A
   pair is *resolvable* when ``|full_i - full_j| > 2 * SE_pair``. Then, from the
   :func:`lite_subset_sim.run` grid records, report per design the fraction of
   resolvable pairs whose ordering the subset preserves (overall and within the top
   half), and the flip rate among non-resolvable pairs.

B2. Resolution retention. For each design, bootstrap FOVs *within the lite subset*
   (paired across systems) to get the lite pair SE, and count full-resolvable pairs
   that stay resolvable at the lite's own noise with the same sign. ``fixed_fovs``
   scores one exact FOV list (a built lite) instead of random draws.

E. Small-n KID floor. KID on random paired cell subsets (or whole-FOV subsets) of
   size n drawn from the full pool, 30 draws: mean and SD vs n, per extractor.

Run (on a compute node; the login node has one CPU)::

    uv run --no-sync python applications/dynacell/tools/lite_checks.py --org nucleus --bucket ipsc \
        --b2 fov050 --fixed-fovs ipsc_cell_positions.txt --out-suffix _built --out /path/to/out
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from lite_subset_sim import (
    EXTRACTORS,
    MODELS_3D,
    designs_for,
    load_systems,
    mmd2_from_features,
    poly3_features,
    poly_kernel,
    subset_keys,
)

from dynacell.evaluation.cp_reference import sidecar_cp_space
from dynacell.evaluation.paths import DATA_ROOT

E_SYSTEMS = [
    "fcmae_vscyto3d_pretrained/ipsc",
    "fnet3d_paper/ipsc",
    "fcmae_vscyto3d_pretrained/a549",
    "celldiff_r2/ipsc",
]


def check_b(
    org: str, bucket: str, out: Path, n_boot: int = 200, seed: int = 0, data_root: Path = DATA_ROOT
) -> pd.DataFrame:
    """Run check B for one bucket against the grid records already in ``out``.

    Writes ``fullse_<org>_<bucket>_{mean,se}.csv`` and ``checkB_<org>_<bucket>.csv``.

    Parameters
    ----------
    org, bucket : str
        Organelle and test bucket.
    out : Path
        Directory holding ``records_<org>_<bucket>*.csv``; outputs are written here.
    n_boot : int
        Full-set FOV bootstrap draws.
    seed : int
        Bootstrap seed.
    data_root : Path
        Benchmark root.

    Returns
    -------
    pd.DataFrame
        One row per (metric, design).
    """
    systems, blocks = load_systems(org, bucket, MODELS_3D, data_root)
    fovs = sorted({f for f, _ in blocks})
    fov_blocks = {f: np.array([i for i, (ff, _) in enumerate(blocks) if ff == f]) for f in fovs}
    rng = np.random.default_rng(seed)
    full = {s.name: s.metrics(np.arange(len(blocks))) for s in systems}
    boots = {s.name: [] for s in systems}
    for _ in range(n_boot):
        draw = rng.choice(fovs, len(fovs), replace=True)
        sel = np.concatenate([fov_blocks[f] for f in draw])  # duplicates keep their weight
        for s in systems:
            boots[s.name].append(s.metrics(sel))
    bootdf = {name: pd.DataFrame(v) for name, v in boots.items()}  # rows = draws (same FOV draw for every system)
    fulldf = pd.DataFrame(full).T
    sedf = pd.DataFrame({name: b.std() for name, b in bootdf.items()}).T
    fulldf.to_csv(out / f"fullse_{org}_{bucket}_mean.csv")
    sedf.to_csv(out / f"fullse_{org}_{bucket}_se.csv")
    # PAIRED SE: the same FOV draw is applied to both systems, so the null for a pair is the
    # bootstrap SD of the DIFFERENCE, not the independent sum of the two SEs.
    rec = pd.concat([pd.read_csv(f) for f in sorted(out.glob(f"records_{org}_{bucket}*.csv"))], ignore_index=True)
    rec = rec[rec.metric != "n_cells"]
    rows = []
    for metric in sedf.columns:
        if metric not in fulldf.columns:
            continue
        names = [n for n in fulldf.index if np.isfinite(fulldf.loc[n, metric]) and np.isfinite(sedf.loc[n, metric])]
        pairs = list(itertools.combinations(names, 2))
        higher = "KID" not in metric
        order = fulldf.loc[names, metric].sort_values(ascending=not higher).index.tolist()
        top_half = set(order[: max(2, len(order) // 2)])
        resolv = {}
        for i, j in pairs:
            gap = abs(fulldf.loc[i, metric] - fulldf.loc[j, metric])
            se_pair = float((bootdf[i][metric] - bootdf[j][metric]).std())
            resolv[(i, j)] = gap > 2 * se_pair
        n_res = sum(resolv.values())
        n_res_top = sum(v for (i, j), v in resolv.items() if i in top_half and j in top_half)
        n_pairs_top = sum(1 for i, j in pairs if i in top_half and j in top_half)
        g = rec[rec.metric == metric]
        for design, gd in g.groupby("design"):
            kept, flipped_nonres, n_nonres_eval, n_res_eval = 0, 0, 0, 0
            kept_top, n_res_top_eval = 0, 0
            for _, gr in gd.groupby("rep"):
                sub = gr.set_index("system")["subset"]
                for i, j in pairs:
                    if i not in sub or j not in sub or not (np.isfinite(sub[i]) and np.isfinite(sub[j])):
                        continue
                    same = np.sign(sub[i] - sub[j]) == np.sign(fulldf.loc[i, metric] - fulldf.loc[j, metric])
                    if resolv[(i, j)]:
                        n_res_eval += 1
                        kept += same
                        if i in top_half and j in top_half:
                            n_res_top_eval += 1
                            kept_top += same
                    else:
                        n_nonres_eval += 1
                        flipped_nonres += not same
            rows.append(
                dict(
                    org=org,
                    bucket=bucket,
                    metric=metric,
                    design=design,
                    n_sys=len(names),
                    n_pairs=len(pairs),
                    n_resolvable=n_res,
                    frac_resolvable=n_res / len(pairs) if pairs else np.nan,
                    resolvable_preserved=kept / n_res_eval if n_res_eval else np.nan,
                    nonresolvable_flip_rate=flipped_nonres / n_nonres_eval if n_nonres_eval else np.nan,
                    n_pairs_top=n_pairs_top,
                    n_resolvable_top=n_res_top,
                    resolvable_preserved_top=kept_top / max(n_res_top_eval, 1) if n_res_top_eval else np.nan,
                    reps=gd.rep.nunique(),
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / f"checkB_{org}_{bucket}.csv", index=False)
    print(f"[B] {org}/{bucket}: {len(systems)} systems, {len(fovs)} FOVs, {n_boot} boots", flush=True)
    return result


def check_e(
    org: str,
    bucket: str,
    systems_wanted: list[str],
    out: Path,
    sizes: tuple[int, ...] = (25, 50, 100, 200, 400, 800),
    draws: int = 30,
    seed: int = 0,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Run check E: KID mean / SD vs subset size, per system and extractor.

    Writes ``checkE_<org>_<bucket>.csv``.

    Parameters
    ----------
    org, bucket : str
        Organelle and test bucket.
    systems_wanted : list of str
        ``<model>/<train_set>`` eval dirs to probe.
    out : Path
        Output directory.
    sizes : tuple of int
        Subset sizes n (sizes >= the pool are skipped).
    draws : int
        Draws per (n, mode).
    seed : int
        Draw seed.
    data_root : Path
        Benchmark root.

    Returns
    -------
    pd.DataFrame
        One row per (system, extractor, n, mode) with ``mode`` in ``cells`` / ``fovs``.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for sysname in systems_wanted:
        path = data_root / org / sysname / bucket
        for tok, prefix in EXTRACTORS.items():
            g = path / "embeddings" / f"gt_{tok}_single_cell_embeddings.npz"
            p = path / "embeddings" / f"pred_{tok}_single_cell_embeddings.npz"
            if not (g.exists() and p.exists()):
                continue
            X = np.asarray(np.load(p, allow_pickle=True)["embeddings"], dtype=np.float64)
            Y = np.asarray(np.load(g, allow_pickle=True)["embeddings"], dtype=np.float64)
            keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
            X, Y = X[keep], Y[keep]
            if tok == "cp":  # the pipeline's CP space: reference mask + this dataset's GT scaler
                space = sidecar_cp_space(path)
                X, Y = space.transform(X), space.transform(Y)
            n_all = X.shape[0]
            if tok != "cp":  # deep extractors: precompute cell-level kernels once, slice per draw
                K_XX, K_YY, K_XY = poly_kernel(X, X), poly_kernel(Y, Y), poly_kernel(X, Y)

            def kid(idx: np.ndarray) -> float:
                m = len(idx)
                if tok == "cp":
                    return mmd2_from_features(poly3_features(X[idx]), poly3_features(Y[idx]))
                kxx, kyy, kxy = K_XX[np.ix_(idx, idx)], K_YY[np.ix_(idx, idx)], K_XY[np.ix_(idx, idx)]
                return (kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.sum() / (m * m)

            full = kid(np.arange(n_all))
            fov_all = np.load(p, allow_pickle=True)["fov"].astype(str)[keep]
            fovs = np.unique(fov_all)
            for n in [s for s in sizes if s < n_all]:
                for mode in ("cells", "fovs"):
                    vals = []
                    for _ in range(draws):
                        if mode == "cells":
                            idx = rng.choice(n_all, n, replace=False)
                        else:  # whole FOVs (all their cells) until >= n cells -- what a lite draw does
                            order = rng.permutation(fovs)
                            idx, k = np.array([], int), 0
                            while len(idx) < n and k < len(order):
                                idx = np.concatenate([idx, np.flatnonzero(fov_all == order[k])])
                                k += 1
                        vals.append(kid(idx))
                    rows.append(
                        dict(
                            org=org,
                            bucket=bucket,
                            system=sysname,
                            extractor=prefix,
                            n=n,
                            n_all=n_all,
                            full=full,
                            mode=mode,
                            mean=np.mean(vals),
                            sd=np.std(vals),
                            bias=np.mean(vals) - full,
                            rel_sd=np.std(vals) / abs(full) if full else np.nan,
                        )
                    )
        print(f"[E] {org}/{bucket}/{sysname} done", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / f"checkE_{org}_{bucket}.csv", index=False)
    return result


def check_b2(
    org: str,
    bucket: str,
    designs: list[str],
    out: Path,
    reps: int = 5,
    n_boot_full: int = 200,
    n_boot_lite: int = 100,
    seed: int = 1,
    fixed_fovs: list[str] | None = None,
    out_suffix: str = "",
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Run check B2 (paired resolution retention) for the named designs of one bucket.

    Writes ``checkB2_<org>_<bucket><out_suffix>.csv``.

    Parameters
    ----------
    org, bucket : str
        Organelle and test bucket.
    designs : list of str
        Design names from :func:`lite_subset_sim.designs_for` to score.
    out : Path
        Output directory.
    reps : int
        Random FOV draws per design (ignored with ``fixed_fovs``).
    n_boot_full, n_boot_lite : int
        Paired FOV bootstrap draws over the full set and within each lite subset.
    seed : int
        Seed of the single generator shared by every draw, in call order.
    fixed_fovs : list of str, optional
        Score this exact FOV list (one rep) instead of random draws; its length must
        equal the design's FOV count.
    out_suffix : str
        Appended to the output file stem.
    data_root : Path
        Benchmark root; systems are :data:`lite_subset_sim.MODELS_3D` x ``TRAIN_SETS``.

    Returns
    -------
    pd.DataFrame
        One row per (design, rep, metric): ``n_full_resolvable``, ``retention`` (fraction of
        those the lite also resolves with the same sign), ``lite_only_resolvable`` and
        ``confidently_wrong`` (both resolve, opposite signs).

    Raises
    ------
    ValueError
        If ``fixed_fovs`` names FOVs outside the bucket or has the wrong length.
    """
    systems, blocks = load_systems(org, bucket, MODELS_3D, data_root)
    fovs = sorted({f for f, _ in blocks})
    t_of = {f: sorted(t for ff, t in blocks if ff == f) for f in fovs}
    bidx = {b: i for i, b in enumerate(blocks)}
    fov_blocks = {f: np.array([i for i, (ff, _) in enumerate(blocks) if ff == f]) for f in fovs}
    rng = np.random.default_rng(seed)
    names = [s.name for s in systems]
    full = pd.DataFrame({s.name: s.metrics(np.arange(len(blocks))) for s in systems}).T
    fb = {n: [] for n in names}
    for _ in range(n_boot_full):
        draw = rng.choice(fovs, len(fovs), replace=True)
        sel = np.concatenate([fov_blocks[f] for f in draw])
        for s in systems:
            fb[s.name].append(s.metrics(sel))
    fb = {n: pd.DataFrame(v) for n, v in fb.items()}
    metrics = [m for m in full.columns if m != "n_cells"]
    rows = []
    for dname, k, tn in designs_for(bucket, len(fovs)):
        if dname not in designs:
            continue
        for rep in range(reps):
            if fixed_fovs is not None:
                missing = set(fixed_fovs) - set(fovs)
                if missing or len(fixed_fovs) != k:
                    raise ValueError(
                        f"fixed_fovs must be {k} FOVs of {org}/{bucket}; got {len(fixed_fovs)}, "
                        f"{len(missing)} not in the bucket: {sorted(missing)[:5]}"
                    )
                chosen = sorted(fixed_fovs)
            else:
                chosen = list(rng.choice(fovs, k, replace=False)) if k < len(fovs) else fovs
            keys = subset_keys(fovs, t_of, bidx, chosen, tn)
            lite_blocks = {f: np.array([bidx[(ff, t)] for ff, t in keys if ff == f]) for f in chosen}
            lite_sel = np.array(sorted(bidx[kk] for kk in keys))
            lite_val = pd.DataFrame({s.name: s.metrics(lite_sel) for s in systems}).T
            lb = {n: [] for n in names}
            for _ in range(n_boot_lite):
                draw = rng.choice(chosen, len(chosen), replace=True)
                sel = np.concatenate([lite_blocks[f] for f in draw])
                for s in systems:
                    lb[s.name].append(s.metrics(sel))
            lb = {n: pd.DataFrame(v) for n, v in lb.items()}
            for metric in metrics:
                ok = [n for n in names if np.isfinite(full.loc[n, metric]) and np.isfinite(lite_val.loc[n, metric])]
                n_full_res = n_both = n_lite_only = n_full_only_wrong = 0
                for i, j in itertools.combinations(ok, 2):
                    gap_full = full.loc[i, metric] - full.loc[j, metric]
                    se_full = float((fb[i][metric] - fb[j][metric]).std())
                    gap_lite = lite_val.loc[i, metric] - lite_val.loc[j, metric]
                    se_lite = float((lb[i][metric] - lb[j][metric]).std())
                    full_res = abs(gap_full) > 2 * se_full
                    lite_res = abs(gap_lite) > 2 * se_lite
                    same = np.sign(gap_full) == np.sign(gap_lite)
                    n_full_res += full_res
                    n_both += full_res and lite_res and same
                    n_lite_only += (not full_res) and lite_res
                    n_full_only_wrong += full_res and lite_res and (not same)
                rows.append(
                    dict(
                        org=org,
                        bucket=bucket,
                        design=dname,
                        rep=rep,
                        metric=metric,
                        n_sys=len(ok),
                        n_full_resolvable=n_full_res,
                        retention=n_both / max(n_full_res, 1) if n_full_res else np.nan,
                        lite_only_resolvable=n_lite_only,
                        confidently_wrong=n_full_only_wrong,
                        n_rows=len(lite_sel),
                    )
                )
            if k == len(fovs) or fixed_fovs is not None:
                break
        print(f"[B2] {org}/{bucket} {dname} done", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / f"checkB2_{org}_{bucket}{out_suffix}.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run checks B + E (default) or B2 (``--b2``) over the requested buckets."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", nargs="+", default=["nucleus", "membrane", "er", "mito"])
    ap.add_argument("--bucket", nargs="+", default=["ipsc", "a549__mock", "a549__denv", "a549__zikv"])
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-e", action="store_true")
    ap.add_argument("--b2", nargs="*", default=None, help="run check B2 for these designs only (skips B and E)")
    ap.add_argument("--fixed-fovs", type=Path, help="B2 on this exact FOV list (one name per line)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True, help="output dir; check B reads the grid records from here")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for org in a.org:
        for bucket in a.bucket:
            if a.b2 is not None:
                fixed = a.fixed_fovs.read_text().split() if a.fixed_fovs else None
                check_b2(org, bucket, a.b2, a.out, fixed_fovs=fixed, out_suffix=a.out_suffix, data_root=a.data_root)
                continue
            if not a.skip_b and list(a.out.glob(f"records_{org}_{bucket}*.csv")):
                check_b(org, bucket, a.out, data_root=a.data_root)
            if not a.skip_e:
                check_e(org, bucket, E_SYSTEMS, a.out, data_root=a.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
