"""CLI and analysis logic for MMD-based perturbation effect evaluation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import click
import numpy as np
import pandas as pd

from dynaclr.evaluation.mmd.config import (
    ComparisonSpec,
    MMDCombinedConfig,
    MMDEvalConfig,
    MMDPooledConfig,
    MMDSettings,
    _resolve_bin_edges,
)
from viscy_utils.compose import load_composed_config
from viscy_utils.evaluation.mmd import median_heuristic, mmd_permutation_test


def _extract_embeddings(adata: ad.AnnData, embedding_key: str | None) -> np.ndarray:
    """Extract embedding matrix from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData store with ``.X`` or ``.obsm``.
    embedding_key : str or None
        obsm key, or None to use ``.X``.

    Returns
    -------
    np.ndarray
        Embedding matrix, shape (n_cells, n_features).
    """
    if embedding_key is None:
        X = adata.X
    else:
        X = adata.obsm[embedding_key]
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _subsample(X: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or len(X) <= max_n:
        return X
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx]


def _run_one_comparison(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    settings: MMDSettings,
    bandwidth: float | None = None,
) -> tuple[float, float, float, float, float, int, int]:
    """Run MMD permutation test for one (cond_a, cond_b) pair.

    Parameters
    ----------
    emb_a : np.ndarray
        Embeddings for group A.
    emb_b : np.ndarray
        Embeddings for group B.
    settings : MMDSettings
        Algorithm settings.
    bandwidth : float or None
        Pre-computed bandwidth to use. If None, computed via median heuristic.
        Pass a value to share bandwidth across comparisons within the same group.

    Returns
    -------
    mmd2 : float
    p_value : float
    bandwidth : float
    effect_size : float
        mmd2 / bandwidth
    activity_zscore : float
        (mmd2 - null_mean) / null_std — normalizes observed MMD relative to
        the permutation null, comparable across markers and datasets.
    n_a_used : int
        Actual number of cells used from group A after subsampling/balancing.
    n_b_used : int
        Actual number of cells used from group B after subsampling/balancing.
    All metric floats are NaN if fewer than min_cells cells in either group.
    """
    rng = np.random.default_rng(settings.seed)
    emb_a = _subsample(emb_a, settings.max_cells, rng)
    emb_b = _subsample(emb_b, settings.max_cells, rng)
    if settings.balance_samples:
        min_n = min(len(emb_a), len(emb_b))
        emb_a = _subsample(emb_a, min_n, rng)
        emb_b = _subsample(emb_b, min_n, rng)
    n_a_used = len(emb_a)
    n_b_used = len(emb_b)
    if n_a_used < settings.min_cells or n_b_used < settings.min_cells:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n_a_used, n_b_used
    if bandwidth is None:
        bandwidth = median_heuristic(emb_a, emb_b)
    mmd2, p_value, null_dist = mmd_permutation_test(
        emb_a, emb_b, n_permutations=settings.n_permutations, bandwidth=bandwidth, seed=settings.seed
    )
    effect_size = mmd2 / bandwidth if bandwidth > 0 else float("nan")
    activity_zscore = float((mmd2 - null_dist.mean()) / (null_dist.std() + 1e-12))
    return mmd2, p_value, bandwidth, effect_size, activity_zscore, n_a_used, n_b_used


def _run_map_comparison(
    meta: pd.DataFrame,
    features: np.ndarray,
    comp: ComparisonSpec,
    group_by: str,
    marker: str,
    map_settings,
) -> tuple[float, float]:
    """Run copairs mAP for one comparison.

    Returns
    -------
    map_value : float
    map_p_value : float
    Both NaN on failure or if copairs is unavailable.
    """
    try:
        from viscy_utils.evaluation.embedding_map import compute_embedding_map
    except ImportError:
        return float("nan"), float("nan")
    result = compute_embedding_map(
        meta=meta,
        features=features,
        reference_condition=comp.cond_a,
        target_condition=comp.cond_b,
        condition_col=group_by,
        group_col="marker",
        distance=map_settings.distance,
        null_size=map_settings.null_size,
        seed=map_settings.seed,
    )
    if result is None:
        return float("nan"), float("nan")
    return result["mean_average_precision"], result["p_value"]


def run_mmd_analysis(adata: ad.AnnData, config: MMDEvalConfig) -> pd.DataFrame:
    """Run per-experiment MMD analysis for explicit comparison pairs across all markers.

    Each comparison is an explicit ``(cond_a, cond_b)`` pair with a label.
    The analysis is always faceted by ``obs["marker"]`` and ``obs["experiment"]``.
    Each experiment is processed independently to avoid cross-experiment pooling.

    Parameters
    ----------
    adata : AnnData
        AnnData (single- or multi-experiment) after split-embeddings step.
    config : MMDEvalConfig
        Analysis configuration.

    Returns
    -------
    pd.DataFrame
        Results with columns: experiment, marker, cond_a, cond_b, label,
        hours_bin_start, hours_bin_end, n_a, n_b, mmd2, p_value, bandwidth,
        effect_size, activity_zscore, embedding_key, and optionally map_value,
        map_p_value.
    """
    if config.obs_filter:
        mask = pd.Series([True] * len(adata), index=adata.obs.index)
        for col, val in config.obs_filter.items():
            if col not in adata.obs.columns:
                raise KeyError(f"obs_filter column '{col}' not found. Available: {list(adata.obs.columns)}")
            mask &= adata.obs[col] == val
        adata = adata[mask].copy()

    obs = adata.obs
    if config.group_by not in obs.columns:
        raise KeyError(f"obs column '{config.group_by}' not found. Available: {list(obs.columns)}")

    emb_key_label = config.embedding_key if config.embedding_key is not None else "X"
    all_emb = _extract_embeddings(adata, config.embedding_key)
    experiments = obs["experiment"].unique() if "experiment" in obs.columns else ["unknown"]

    records: list[dict] = []
    for experiment in experiments:
        exp_mask = (
            obs["experiment"] == experiment
            if "experiment" in obs.columns
            else pd.Series([True] * len(obs), index=obs.index)
        )
        for marker in sorted(obs["marker"].unique()):
            marker_mask = exp_mask & (obs["marker"] == marker)

            if config.temporal_bin_size is None and config.temporal_bins is None:
                # Aggregate mode
                shared_bw = _compute_shared_bandwidth(
                    all_emb, obs, marker_mask, config.comparisons, config.mmd, config.group_by
                )
                for comp in config.comparisons:
                    mask_a = marker_mask & (obs[config.group_by] == comp.cond_a)
                    mask_b = marker_mask & (obs[config.group_by] == comp.cond_b)
                    emb_a = all_emb[mask_a.values]
                    emb_b = all_emb[mask_b.values]
                    bw = shared_bw if shared_bw is not None else None
                    mmd2, p_value, bw_out, es, az, na, nb = _run_one_comparison(emb_a, emb_b, config.mmd, bandwidth=bw)
                    map_val, map_pval = _maybe_map(
                        obs[marker_mask.values],
                        all_emb[marker_mask.values],
                        comp,
                        config.group_by,
                        marker,
                        config.map_settings,
                    )
                    records.append(
                        _record(
                            experiment,
                            marker,
                            comp,
                            float("nan"),
                            float("nan"),
                            na,
                            nb,
                            mmd2,
                            p_value,
                            bw_out,
                            es,
                            az,
                            map_val,
                            map_pval,
                            emb_key_label,
                        )
                    )
            else:
                if "hours_post_perturbation" not in obs.columns:
                    raise KeyError("temporal binning requires obs column 'hours_post_perturbation'")
                max_hours = obs["hours_post_perturbation"].max()
                bin_pairs = _resolve_bin_edges(config.temporal_bin_size, config.temporal_bins, max_hours)
                for b_start, b_end in bin_pairs:
                    shared_bw = _compute_shared_bandwidth_temporal(
                        all_emb, obs, marker_mask, config.comparisons, config.mmd, config.group_by, b_start, b_end
                    )
                    for comp in config.comparisons:
                        mask_a = marker_mask & (obs[config.group_by] == comp.cond_a)
                        bin_mask_b = (
                            marker_mask
                            & (obs[config.group_by] == comp.cond_b)
                            & (obs["hours_post_perturbation"] >= b_start)
                            & (obs["hours_post_perturbation"] < b_end)
                        )
                        emb_a = all_emb[mask_a.values]
                        emb_b = all_emb[bin_mask_b.values]
                        bw = shared_bw if shared_bw is not None else None
                        mmd2, p_value, bw_out, es, az, na, nb = _run_one_comparison(
                            emb_a, emb_b, config.mmd, bandwidth=bw
                        )
                        map_val, map_pval = _maybe_map(
                            obs[marker_mask.values],
                            all_emb[marker_mask.values],
                            comp,
                            config.group_by,
                            marker,
                            config.map_settings,
                        )
                        records.append(
                            _record(
                                experiment,
                                marker,
                                comp,
                                b_start,
                                b_end,
                                na,
                                nb,
                                mmd2,
                                p_value,
                                bw_out,
                                es,
                                az,
                                map_val,
                                map_pval,
                                emb_key_label,
                            )
                        )
    return pd.DataFrame(records)


def _compute_shared_bandwidth(
    all_emb: np.ndarray,
    obs: pd.DataFrame,
    marker_mask: pd.Series,
    comparisons: list[ComparisonSpec],
    settings: MMDSettings,
    group_by: str,
) -> float | None:
    """Compute bandwidth from the share_bandwidth_from comparison, if configured."""
    if settings.share_bandwidth_from is None:
        return None
    for comp in comparisons:
        if comp.label == settings.share_bandwidth_from:
            mask_a = marker_mask & (obs[group_by] == comp.cond_a)
            mask_b = marker_mask & (obs[group_by] == comp.cond_b)
            emb_a = all_emb[mask_a.values]
            emb_b = all_emb[mask_b.values]
            if len(emb_a) >= settings.min_cells and len(emb_b) >= settings.min_cells:
                return median_heuristic(emb_a, emb_b)
            return None
    return None


def _compute_shared_bandwidth_temporal(
    all_emb: np.ndarray,
    obs: pd.DataFrame,
    marker_mask: pd.Series,
    comparisons: list[ComparisonSpec],
    settings: MMDSettings,
    group_by: str,
    b_start: float,
    b_end: float,
) -> float | None:
    """Compute shared bandwidth from the share_bandwidth_from comparison for a temporal bin."""
    if settings.share_bandwidth_from is None:
        return None
    for comp in comparisons:
        if comp.label == settings.share_bandwidth_from:
            mask_a = (
                marker_mask
                & (obs[group_by] == comp.cond_a)
                & (obs["hours_post_perturbation"] >= b_start)
                & (obs["hours_post_perturbation"] < b_end)
            )
            mask_b = (
                marker_mask
                & (obs[group_by] == comp.cond_b)
                & (obs["hours_post_perturbation"] >= b_start)
                & (obs["hours_post_perturbation"] < b_end)
            )
            emb_a = all_emb[mask_a.values]
            emb_b = all_emb[mask_b.values]
            if len(emb_a) >= settings.min_cells and len(emb_b) >= settings.min_cells:
                return median_heuristic(emb_a, emb_b)
            return None
    return None


def _maybe_map(
    obs_sub: pd.DataFrame,
    emb_sub: np.ndarray,
    comp: ComparisonSpec,
    group_by: str,
    marker: str,
    map_settings,
) -> tuple[float, float]:
    """Run mAP if enabled, otherwise return NaN pair."""
    if not map_settings.enabled:
        return float("nan"), float("nan")
    return _run_map_comparison(obs_sub, emb_sub, comp, group_by, marker, map_settings)


def _record(
    experiment: str,
    marker: str,
    comp: ComparisonSpec,
    hours_bin_start: float,
    hours_bin_end: float,
    n_a: int,
    n_b: int,
    mmd2: float,
    p_value: float,
    bandwidth: float,
    effect_size: float,
    activity_zscore: float,
    map_value: float,
    map_p_value: float,
    embedding_key: str,
) -> dict:
    return {
        "experiment": experiment,
        "marker": marker,
        "cond_a": comp.cond_a,
        "cond_b": comp.cond_b,
        "label": comp.label,
        "hours_bin_start": hours_bin_start,
        "hours_bin_end": hours_bin_end,
        "n_a": n_a,
        "n_b": n_b,
        "mmd2": mmd2,
        "p_value": p_value,
        "bandwidth": bandwidth,
        "effect_size": effect_size,
        "activity_zscore": activity_zscore,
        "map_value": map_value,
        "map_p_value": map_p_value,
        "embedding_key": embedding_key,
    }


def run_mmd_combined(config: MMDCombinedConfig) -> pd.DataFrame:
    """Run pairwise cross-experiment MMD, faceted by marker and condition+time bin.

    For each marker, finds all experiments that share it, then for each pair
    of those experiments runs MMD per (condition, time_bin) after centering
    within that pair only. This measures batch effects between experiments
    at matched biological states.

    Parameters
    ----------
    config : MMDCombinedConfig
        Combined analysis configuration.

    Returns
    -------
    pd.DataFrame
        Results with columns: marker, exp_a, exp_b, condition, hours_bin_start,
        hours_bin_end, n_a, n_b, mmd2, p_value, bandwidth, effect_size,
        activity_zscore, embedding_key.
    """
    from itertools import combinations

    adatas = {ad.read_zarr(p).obs["experiment"].iloc[0]: ad.read_zarr(p) for p in config.input_paths}

    if config.obs_filter:
        filtered = {}
        for exp_name, adata in adatas.items():
            mask = pd.Series([True] * len(adata), index=adata.obs.index)
            for col, val in config.obs_filter.items():
                if col not in adata.obs.columns:
                    raise KeyError(
                        f"obs_filter column '{col}' not found in {exp_name}. Available: {list(adata.obs.columns)}"
                    )
                mask &= adata.obs[col] == val
            filtered[exp_name] = adata[mask].copy()
        adatas = filtered

    marker_to_exps: dict[str, list[str]] = {}
    for exp_name, adata in adatas.items():
        for marker in adata.obs["marker"].unique():
            marker_to_exps.setdefault(marker, []).append(exp_name)

    emb_key_label = config.embedding_key if config.embedding_key is not None else "X"
    records: list[dict] = []

    for marker, exp_names in sorted(marker_to_exps.items()):
        if len(exp_names) < 2:
            continue
        for exp_a, exp_b in combinations(exp_names, 2):
            adata_a = adatas[exp_a][adatas[exp_a].obs["marker"] == marker]
            adata_b = adatas[exp_b][adatas[exp_b].obs["marker"] == marker]
            emb_a_full = _extract_embeddings(adata_a, config.embedding_key).astype(np.float32)
            emb_b_full = _extract_embeddings(adata_b, config.embedding_key).astype(np.float32)
            obs_a = adata_a.obs
            obs_b = adata_b.obs

            emb_a_full = emb_a_full - emb_a_full.mean(axis=0)
            emb_b_full = emb_b_full - emb_b_full.mean(axis=0)

            conditions = sorted(set(obs_a[config.group_by].unique()) & set(obs_b[config.group_by].unique()))
            for condition in conditions:
                cond_mask_a = obs_a[config.group_by] == condition
                cond_mask_b = obs_b[config.group_by] == condition
                emb_ca = emb_a_full[cond_mask_a.values]
                emb_cb = emb_b_full[cond_mask_b.values]

                if config.temporal_bin_size is None and config.temporal_bins is None:
                    mmd2, p_value, bw, es, az, na, nb = _run_one_comparison(emb_ca, emb_cb, config.mmd)
                    records.append(
                        _combined_record(
                            marker,
                            exp_a,
                            exp_b,
                            condition,
                            float("nan"),
                            float("nan"),
                            na,
                            nb,
                            mmd2,
                            p_value,
                            bw,
                            es,
                            az,
                            emb_key_label,
                        )
                    )
                else:
                    if "hours_post_perturbation" not in obs_a.columns:
                        raise KeyError("temporal binning requires obs column 'hours_post_perturbation'")
                    max_hours = min(obs_a["hours_post_perturbation"].max(), obs_b["hours_post_perturbation"].max())
                    bin_pairs = _resolve_bin_edges(config.temporal_bin_size, config.temporal_bins, max_hours)
                    for b_start, b_end in bin_pairs:
                        bin_mask_a = (
                            cond_mask_a
                            & (obs_a["hours_post_perturbation"] >= b_start)
                            & (obs_a["hours_post_perturbation"] < b_end)
                        )
                        bin_mask_b = (
                            cond_mask_b
                            & (obs_b["hours_post_perturbation"] >= b_start)
                            & (obs_b["hours_post_perturbation"] < b_end)
                        )
                        bin_emb_a = emb_a_full[bin_mask_a.values]
                        bin_emb_b = emb_b_full[bin_mask_b.values]
                        mmd2, p_value, bw, es, az, na, nb = _run_one_comparison(bin_emb_a, bin_emb_b, config.mmd)
                        records.append(
                            _combined_record(
                                marker,
                                exp_a,
                                exp_b,
                                condition,
                                b_start,
                                b_end,
                                na,
                                nb,
                                mmd2,
                                p_value,
                                bw,
                                es,
                                az,
                                emb_key_label,
                            )
                        )

    return pd.DataFrame(records)


def _combined_record(
    marker: str,
    exp_a: str,
    exp_b: str,
    condition: str,
    hours_bin_start: float,
    hours_bin_end: float,
    n_a: int,
    n_b: int,
    mmd2: float,
    p_value: float,
    bandwidth: float,
    effect_size: float,
    activity_zscore: float,
    embedding_key: str,
) -> dict:
    return {
        "marker": marker,
        "exp_a": exp_a,
        "exp_b": exp_b,
        "condition": condition,
        "hours_bin_start": hours_bin_start,
        "hours_bin_end": hours_bin_end,
        "n_a": n_a,
        "n_b": n_b,
        "mmd2": mmd2,
        "p_value": p_value,
        "bandwidth": bandwidth,
        "effect_size": effect_size,
        "activity_zscore": activity_zscore,
        "embedding_key": embedding_key,
    }


def run_mmd_pooled(config: MMDPooledConfig) -> pd.DataFrame:
    """Run pooled multi-experiment MMD/mAP analysis.

    Concatenates cells from all input experiments into a single pool, then
    computes MMD (and optionally mAP) per (marker, time_bin, comparison).
    Unlike the combined mode (pairwise batch-effect detection), this pools all
    experiments together for phenotypic profiling.

    Parameters
    ----------
    config : MMDPooledConfig
        Pooled analysis configuration.

    Returns
    -------
    pd.DataFrame
        Results with columns: marker, cond_a, cond_b, label, hours_bin_start,
        hours_bin_end, n_a, n_b, mmd2, p_value, bandwidth, effect_size,
        activity_zscore, map_value, map_p_value, embedding_key.
        FDR-corrected q_value column is also included.
    """
    from statsmodels.stats.multitest import multipletests

    adatas = [ad.read_zarr(p) for p in config.input_paths]
    combined = ad.concat(adatas, join="outer", label="source_experiment")
    combined.obs_names_make_unique()

    if config.obs_filter:
        mask = pd.Series([True] * len(combined), index=combined.obs.index)
        for col, val in config.obs_filter.items():
            if col not in combined.obs.columns:
                raise KeyError(f"obs_filter column '{col}' not found. Available: {list(combined.obs.columns)}")
            mask &= combined.obs[col] == val
        combined = combined[mask].copy()

    if config.condition_aliases:
        alias_map: dict[str, str] = {}
        for canonical, variants in config.condition_aliases.items():
            for v in variants:
                alias_map[v] = canonical
        combined.obs[config.group_by] = combined.obs[config.group_by].map(lambda x: alias_map.get(x, x))

    obs = combined.obs
    if config.group_by not in obs.columns:
        raise KeyError(f"obs column '{config.group_by}' not found. Available: {list(obs.columns)}")

    emb_key_label = config.embedding_key if config.embedding_key is not None else "X"
    all_emb = _extract_embeddings(combined, config.embedding_key)

    records: list[dict] = []
    for marker in sorted(obs["marker"].unique()):
        marker_mask = obs["marker"] == marker

        if config.temporal_bin_size is None and config.temporal_bins is None:
            shared_bw = _compute_shared_bandwidth(
                all_emb, obs, marker_mask, config.comparisons, config.mmd, config.group_by
            )
            for comp in config.comparisons:
                mask_a = marker_mask & (obs[config.group_by] == comp.cond_a)
                mask_b = marker_mask & (obs[config.group_by] == comp.cond_b)
                emb_a = all_emb[mask_a.values]
                emb_b = all_emb[mask_b.values]
                bw = shared_bw if shared_bw is not None else None
                mmd2, p_value, bw_out, es, az, na, nb = _run_one_comparison(emb_a, emb_b, config.mmd, bandwidth=bw)
                map_val, map_pval = _maybe_map(
                    obs[marker_mask.values],
                    all_emb[marker_mask.values],
                    comp,
                    config.group_by,
                    marker,
                    config.map_settings,
                )
                records.append(
                    _pooled_record(
                        marker,
                        comp,
                        float("nan"),
                        float("nan"),
                        na,
                        nb,
                        mmd2,
                        p_value,
                        bw_out,
                        es,
                        az,
                        map_val,
                        map_pval,
                        emb_key_label,
                    )
                )
        else:
            if "hours_post_perturbation" not in obs.columns:
                raise KeyError("temporal binning requires obs column 'hours_post_perturbation'")
            max_hours = obs["hours_post_perturbation"].max()
            bin_pairs = _resolve_bin_edges(config.temporal_bin_size, config.temporal_bins, max_hours)
            for b_start, b_end in bin_pairs:
                shared_bw = _compute_shared_bandwidth_temporal(
                    all_emb, obs, marker_mask, config.comparisons, config.mmd, config.group_by, b_start, b_end
                )
                for comp in config.comparisons:
                    mask_a = marker_mask & (obs[config.group_by] == comp.cond_a)
                    bin_mask_b = (
                        marker_mask
                        & (obs[config.group_by] == comp.cond_b)
                        & (obs["hours_post_perturbation"] >= b_start)
                        & (obs["hours_post_perturbation"] < b_end)
                    )
                    emb_a = all_emb[mask_a.values]
                    emb_b = all_emb[bin_mask_b.values]
                    bw = shared_bw if shared_bw is not None else None
                    mmd2, p_value, bw_out, es, az, na, nb = _run_one_comparison(emb_a, emb_b, config.mmd, bandwidth=bw)
                    map_val, map_pval = _maybe_map(
                        obs[marker_mask.values],
                        all_emb[marker_mask.values],
                        comp,
                        config.group_by,
                        marker,
                        config.map_settings,
                    )
                    records.append(
                        _pooled_record(
                            marker,
                            comp,
                            b_start,
                            b_end,
                            na,
                            nb,
                            mmd2,
                            p_value,
                            bw_out,
                            es,
                            az,
                            map_val,
                            map_pval,
                            emb_key_label,
                        )
                    )

    df = pd.DataFrame(records)
    if not df.empty:
        valid_p = df["p_value"].dropna()
        if len(valid_p) > 0:
            _, q_values, _, _ = multipletests(df["p_value"].fillna(1.0), alpha=0.05, method="fdr_bh")
            df["q_value"] = q_values
            df.loc[df["p_value"].isna(), "q_value"] = float("nan")
        else:
            df["q_value"] = float("nan")
    return df


def _pooled_record(
    marker: str,
    comp: ComparisonSpec,
    hours_bin_start: float,
    hours_bin_end: float,
    n_a: int,
    n_b: int,
    mmd2: float,
    p_value: float,
    bandwidth: float,
    effect_size: float,
    activity_zscore: float,
    map_value: float,
    map_p_value: float,
    embedding_key: str,
) -> dict:
    return {
        "marker": marker,
        "cond_a": comp.cond_a,
        "cond_b": comp.cond_b,
        "label": comp.label,
        "hours_bin_start": hours_bin_start,
        "hours_bin_end": hours_bin_end,
        "n_a": n_a,
        "n_b": n_b,
        "mmd2": mmd2,
        "p_value": p_value,
        "bandwidth": bandwidth,
        "effect_size": effect_size,
        "activity_zscore": activity_zscore,
        "map_value": map_value,
        "map_p_value": map_p_value,
        "embedding_key": embedding_key,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("mmd_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory. Default: same as mmd_dir."
)
def plot_mmd_heatmap_cmd(mmd_dir: Path, output_dir: Path | None) -> None:
    """Plot a combined MMD heatmap (all markers) from per-experiment CSVs in MMD_DIR."""
    from dynaclr.evaluation.mmd.plotting import plot_mmd_heatmap

    csvs = sorted(mmd_dir.glob("*_mmd_results.csv"))
    if not csvs:
        raise click.ClickException(f"No *_mmd_results.csv files found in {mmd_dir}")

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    click.echo(f"Loaded {len(df)} rows from {len(csvs)} CSV(s)")

    out = output_dir or mmd_dir
    out.mkdir(parents=True, exist_ok=True)

    for comp_label in df["label"].unique():
        sub = df[df["label"] == comp_label]
        safe = comp_label.replace(" ", "_").replace("/", "-")
        for fmt in ("pdf", "png"):
            plot_mmd_heatmap(sub, out / f"all_markers_{safe}_heatmap.{fmt}")
        click.echo(f"Saved heatmap for: {comp_label}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to MMD evaluation YAML config",
)
@click.option(
    "--combined",
    is_flag=True,
    default=False,
    help="Run cross-experiment combined mode (config must have input_paths list)",
)
@click.option(
    "--pooled",
    is_flag=True,
    default=False,
    help="Run pooled multi-experiment phenotypic analysis (config must have input_paths list)",
)
def main(config: Path, combined: bool, pooled: bool) -> None:
    """Compute MMD between explicit condition pairs in cell embeddings.

    Comparisons are defined as explicit (cond_a, cond_b, label) pairs.
    The analysis is always faceted by obs["marker"].
    """
    if combined and pooled:
        raise click.UsageError("--combined and --pooled are mutually exclusive")
    raw = load_composed_config(config)
    output_dir = Path(raw["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if combined:
        cfg = MMDCombinedConfig(**raw)
        df = run_mmd_combined(cfg)
        out_csv = output_dir / "combined_mmd_results.csv"
        df.to_csv(out_csv, index=False)
        click.echo(f"Saved: {out_csv}")
        if cfg.save_plots:
            _save_plots_combined(df, output_dir, cfg.temporal_bin_size)
        _print_summary(df, mode="combined")
    elif pooled:
        cfg = MMDPooledConfig(**raw)
        df = run_mmd_pooled(cfg)
        out_csv = output_dir / "pooled_mmd_results.csv"
        df.to_csv(out_csv, index=False)
        click.echo(f"Saved: {out_csv}")
        if cfg.save_plots and len(df):
            _save_plots_pooled(df, output_dir)
        _print_summary(df, mode="pooled")
    else:
        cfg = MMDEvalConfig(**raw)
        adata = ad.read_zarr(cfg.input_path)
        df = run_mmd_analysis(adata, cfg)
        experiment = df["experiment"].iloc[0] if len(df) else "unknown"
        out_csv = output_dir / f"{experiment}_mmd_results.csv"
        df.to_csv(out_csv, index=False)
        click.echo(f"Saved: {out_csv}")
        if cfg.save_plots and len(df):
            _save_plots(df, output_dir, experiment, cfg.temporal_bin_size or cfg.temporal_bins)
        _print_summary(df, mode="per_experiment")


def _save_plots(df: pd.DataFrame, output_dir: Path, label: str, temporal_config) -> None:
    from dynaclr.evaluation.mmd.plotting import plot_mmd_kinetics, plot_mmd_multi_panel_kinetics

    has_bins = temporal_config is not None and len(df) and not df["hours_bin_start"].isna().all()
    if not has_bins:
        return
    for comp_label in df["label"].unique():
        sub = df[df["label"] == comp_label]
        safe = comp_label.replace(" ", "_").replace("/", "-")
        for fmt in ("pdf", "png"):
            plot_mmd_kinetics(sub, output_dir / f"{label}_{safe}_kinetics.{fmt}")
    for fmt in ("pdf", "png"):
        plot_mmd_multi_panel_kinetics(df, output_dir / f"{label}_multi_panel_kinetics.{fmt}")
    if "activity_zscore" in df.columns and not df["activity_zscore"].isna().all():
        from dynaclr.evaluation.mmd.plotting import plot_activity_heatmap, plot_paired_heatmaps

        for fmt in ("pdf", "png"):
            plot_activity_heatmap(df, output_dir / f"{label}_activity_heatmap.{fmt}")
        labels = [c for c in df["label"].unique() if c]
        if len(labels) >= 2:
            for fmt in ("pdf", "png"):
                plot_paired_heatmaps(df, labels[:2], "activity_zscore", output_dir / f"{label}_paired_activity.{fmt}")


def _save_plots_combined(df: pd.DataFrame, output_dir: Path, temporal_bin_size: float | None) -> None:
    from dynaclr.evaluation.mmd.plotting import plot_mmd_combined_heatmap, plot_mmd_kinetics

    has_bins = temporal_bin_size is not None and len(df) and not df["hours_bin_start"].isna().all()
    for fmt in ("pdf", "png"):
        if has_bins:
            for marker in df["marker"].unique():
                sub = df[df["marker"] == marker]
                safe = marker.replace(" ", "_").replace("/", "-")
                plot_mmd_kinetics(sub, output_dir / f"combined_{safe}_kinetics.{fmt}")
        plot_mmd_combined_heatmap(df, output_dir / f"combined_heatmap.{fmt}")


def _save_plots_pooled(df: pd.DataFrame, output_dir: Path) -> None:
    from dynaclr.evaluation.mmd.plotting import (
        plot_activity_heatmap,
        plot_mmd_heatmap,
        plot_mmd_multi_panel_kinetics,
        plot_paired_heatmaps,
    )

    has_bins = not df["hours_bin_start"].isna().all()
    for fmt in ("pdf", "png"):
        for comp_label in df["label"].unique():
            sub = df[df["label"] == comp_label]
            safe = comp_label.replace(" ", "_").replace("/", "-")
            plot_mmd_heatmap(sub, output_dir / f"pooled_{safe}_heatmap.{fmt}")
        if has_bins:
            plot_mmd_multi_panel_kinetics(df, output_dir / f"pooled_multi_panel_kinetics.{fmt}")
        if "activity_zscore" in df.columns and not df["activity_zscore"].isna().all():
            plot_activity_heatmap(df, output_dir / f"pooled_activity_heatmap.{fmt}")
            labels = [c for c in df["label"].unique() if c]
            if len(labels) >= 2:
                plot_paired_heatmaps(df, labels[:2], "activity_zscore", output_dir / f"pooled_paired_activity.{fmt}")


def _print_summary(df: pd.DataFrame, mode: str = "per_experiment") -> None:
    if df.empty:
        click.echo("No results.")
        return
    click.echo("\n## MMD Results Summary\n")
    if mode == "combined":
        summary = (
            df.dropna(subset=["mmd2"])
            .groupby(["marker", "condition"])[["mmd2", "p_value", "effect_size"]]
            .agg({"mmd2": "mean", "p_value": "min", "effect_size": "mean"})
            .round(4)
            .reset_index()
        )
    elif mode == "pooled":
        summary = (
            df.dropna(subset=["mmd2"])
            .groupby(["marker", "label"])[["mmd2", "p_value", "effect_size", "activity_zscore"]]
            .agg({"mmd2": "mean", "p_value": "min", "effect_size": "mean", "activity_zscore": "mean"})
            .round(4)
            .reset_index()
        )
    else:
        summary = (
            df.dropna(subset=["mmd2"])
            .groupby(["marker", "label"])[["mmd2", "p_value", "effect_size"]]
            .agg({"mmd2": "mean", "p_value": "min", "effect_size": "mean"})
            .round(4)
            .reset_index()
        )
    click.echo(summary.to_string(index=False))
