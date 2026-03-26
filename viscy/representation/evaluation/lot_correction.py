"""Core functions for LOT (Linear Optimal Transport) batch correction.

Pipeline
--------
1. Load source and target embedding zarrs (AnnData format).
2. Filter cells to the uninfected reference population in each dataset.
3. Fit a shared StandardScaler + PCA on the combined source + target cells.
4. Fit a LinearTransport (LOT) map in PCA space using uninfected cells only,
   mapping source → target distribution.
5. Save the fitted pipeline (scaler, PCA, LOT) to disk with joblib.

The saved pipeline can then be applied to any source zarr to produce a new
zarr whose embeddings are in the target's PCA coordinate system, corrected
for cross-platform batch effects.
"""

import logging
from pathlib import Path
from typing import Union

import anndata as ad
import joblib
import numpy as np
import ot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_np(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 numpy array."""
    return np.array(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def _apply_filter(obs, filter_spec: dict) -> np.ndarray:
    """Return a boolean mask for rows of *obs* matching *filter_spec*.

    Parameters
    ----------
    obs : pd.DataFrame
        AnnData ``.obs`` table.
    filter_spec : dict
        Must contain ``"column"`` plus one of:

        * ``"startswith"`` – str or list[str]: keep rows where the column
          value starts with any of the given prefixes.
        * ``"equals"`` – str: keep rows where the column value equals the
          given string.

    Returns
    -------
    np.ndarray of bool
        Boolean mask with the same length as *obs*.
    """
    col = filter_spec["column"]
    values = obs[col].astype(str)

    if "startswith" in filter_spec:
        prefixes = filter_spec["startswith"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        mask = np.zeros(len(obs), dtype=bool)
        for p in prefixes:
            mask |= values.str.startswith(p).values
        return mask

    if "equals" in filter_spec:
        return (values == str(filter_spec["equals"])).values

    raise ValueError(
        "filter_spec must contain either 'startswith' or 'equals'. "
        f"Got: {list(filter_spec.keys())}"
    )


# ── public API ────────────────────────────────────────────────────────────────

def fit_lot_correction(
    source_zarr: Union[str, Path],
    target_zarr: Union[str, Path],
    source_uninf_filter: dict,
    target_uninf_filter: dict,
    n_pca: int = 50,
    ns_lot: int = 3000,
    random_seed: int = 42,
) -> dict:
    """Fit a shared PCA + LOT batch-correction pipeline.

    Loads source (e.g. light-sheet) and target (e.g. confocal) embedding
    zarrs, identifies uninfected reference cells in each, and fits:

    * A shared ``StandardScaler`` on the combined source + target cells.
    * A shared ``PCA`` (``n_pca`` components) on the combined scaled cells.
    * A ``LinearTransport`` (POT library) that maps the source uninfected
      distribution to the target uninfected distribution in PCA space.

    Parameters
    ----------
    source_zarr : str or Path
        Path to the source AnnData zarr (e.g. light-sheet embeddings).
    target_zarr : str or Path
        Path to the target AnnData zarr (e.g. confocal embeddings).
    source_uninf_filter : dict
        Filter spec (see :func:`_apply_filter`) selecting uninfected source
        cells used to fit LOT.
    target_uninf_filter : dict
        Filter spec selecting uninfected target cells used to fit LOT.
    n_pca : int, optional
        Number of PCA components, by default 50.
    ns_lot : int, optional
        Maximum number of cells subsampled per dataset for LOT fitting,
        by default 3000.
    random_seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    dict with keys:
        ``"scaler"`` : fitted StandardScaler
        ``"pca"``    : fitted PCA
        ``"lot"``    : fitted ot.da.LinearTransport
        ``"n_pca"``  : int
        ``"ns_lot"`` : int
        ``"random_seed"`` : int
    """
    rng = np.random.default_rng(random_seed)

    _logger.info("Loading source zarr: %s", source_zarr)
    adata_src = ad.read_zarr(source_zarr)
    adata_src.obs_names_make_unique()

    _logger.info("Loading target zarr: %s", target_zarr)
    adata_tgt = ad.read_zarr(target_zarr)
    adata_tgt.obs_names_make_unique()

    _logger.info(
        "Source shape: %s  Target shape: %s", adata_src.shape, adata_tgt.shape
    )

    X_src = _to_np(adata_src.X)
    X_tgt = _to_np(adata_tgt.X)

    src_uninf_mask = _apply_filter(adata_src.obs, source_uninf_filter)
    tgt_uninf_mask = _apply_filter(adata_tgt.obs, target_uninf_filter)

    _logger.info(
        "Uninfected cells — source: %d / %d,  target: %d / %d",
        src_uninf_mask.sum(), len(X_src),
        tgt_uninf_mask.sum(), len(X_tgt),
    )

    if src_uninf_mask.sum() < 5 or tgt_uninf_mask.sum() < 5:
        raise ValueError(
            "Too few uninfected cells to fit LOT "
            f"(source={src_uninf_mask.sum()}, target={tgt_uninf_mask.sum()}). "
            "Check your filter specifications."
        )

    # Fit shared PCA on ALL source + target cells
    _logger.info("Fitting shared StandardScaler + PCA-%d ...", n_pca)
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(np.vstack([X_src, X_tgt]))
    pca = PCA(n_components=n_pca, random_state=random_seed)
    Z_all = pca.fit_transform(X_combined_scaled)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    _logger.info("PCA explained variance: %.1f%%", var_exp)

    n_src = len(X_src)
    Z_src_uninf = Z_all[:n_src][src_uninf_mask]
    Z_tgt_uninf = Z_all[n_src:][tgt_uninf_mask]

    # Subsample for LOT fitting
    ns_src = min(len(Z_src_uninf), ns_lot)
    ns_tgt = min(len(Z_tgt_uninf), ns_lot)
    idx_src = rng.choice(len(Z_src_uninf), ns_src, replace=False)
    idx_tgt = rng.choice(len(Z_tgt_uninf), ns_tgt, replace=False)

    _logger.info(
        "Fitting LOT (source subsample=%d, target subsample=%d) ...", ns_src, ns_tgt
    )
    lot = ot.da.LinearTransport(reg=1e-3)
    lot.fit(Xs=Z_src_uninf[idx_src], Xt=Z_tgt_uninf[idx_tgt])
    _logger.info("LOT fitted.")

    return {
        "scaler": scaler,
        "pca": pca,
        "lot": lot,
        "n_pca": n_pca,
        "ns_lot": ns_lot,
        "random_seed": random_seed,
        "pca_variance_explained": float(var_exp),
    }


def apply_lot_correction(
    input_zarr: Union[str, Path],
    pipeline: dict,
    output_zarr: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """Apply a fitted LOT pipeline to an embedding zarr.

    Transforms all cells in *input_zarr* through the pipeline
    (StandardScaler → PCA → LOT) and writes an AnnData zarr whose ``.X``
    contains the corrected embeddings in the target's PCA space
    (shape ``n_cells × n_pca``).  All ``.obs`` metadata is preserved.

    Parameters
    ----------
    input_zarr : str or Path
        Path to the source AnnData zarr to correct.
    pipeline : dict
        Fitted pipeline as returned by :func:`fit_lot_correction`.
    output_zarr : str or Path
        Path to write the corrected AnnData zarr.
    overwrite : bool, optional
        If ``False`` (default) and *output_zarr* already exists, raise.
    """
    output_zarr = Path(output_zarr)
    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_zarr}. "
                "Set overwrite=true to overwrite."
            )
        import shutil
        shutil.rmtree(output_zarr)

    _logger.info("Loading input zarr: %s", input_zarr)
    adata_in = ad.read_zarr(input_zarr)
    adata_in.obs_names_make_unique()

    X = _to_np(adata_in.X)
    _logger.info("Input shape: %s", adata_in.shape)

    scaler = pipeline["scaler"]
    pca    = pipeline["pca"]
    lot    = pipeline["lot"]

    _logger.info("Applying StandardScaler → PCA → LOT ...")
    Z = pca.transform(scaler.transform(X))
    Z_corrected = lot.transform(Z)
    _logger.info(
        "Corrected embeddings shape: %s  (n_pca=%d)", Z_corrected.shape, pipeline["n_pca"]
    )

    obs = adata_in.obs.copy()
    # Convert StringDtype columns (and categoricals with StringDtype categories)
    # to object dtype for broad anndata / zarr compatibility.
    import pandas as pd
    for col in obs.columns:
        dtype = obs[col].dtype
        if isinstance(dtype, pd.StringDtype):
            obs[col] = obs[col].astype(object)
        elif isinstance(dtype, pd.CategoricalDtype) and isinstance(
            dtype.categories.dtype, pd.StringDtype
        ):
            obs[col] = obs[col].astype(object).astype("category")

    # Also enable anndata's opt-in for writing nullable string arrays in case
    # any StringArray-backed column is still present after the conversion above.
    try:
        ad.settings.allow_write_nullable_strings = True
    except AttributeError:
        pass  # older anndata versions don't have this setting

    adata_out = ad.AnnData(X=Z_corrected.astype(np.float32), obs=obs)
    adata_out.uns["lot_correction"] = {
        "source_zarr": str(input_zarr),
        "n_pca": pipeline["n_pca"],
        "pca_variance_explained": pipeline.get("pca_variance_explained"),
    }

    _logger.info("Writing corrected zarr: %s", output_zarr)
    adata_out.write_zarr(output_zarr)
    _logger.info("Done.")


def save_lot_pipeline(pipeline: dict, path: Union[str, Path]) -> None:
    """Save a fitted LOT pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : dict
        Fitted pipeline as returned by :func:`fit_lot_correction`.
    path : str or Path
        Output path (e.g. ``lot_pipeline.pkl``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    _logger.info("Pipeline saved to %s", path)


def load_lot_pipeline(path: Union[str, Path]) -> dict:
    """Load a fitted LOT pipeline from disk.

    Parameters
    ----------
    path : str or Path
        Path to the saved pipeline file.

    Returns
    -------
    dict
        Pipeline with keys ``"scaler"``, ``"pca"``, ``"lot"``.
    """
    pipeline = joblib.load(path)
    _logger.info(
        "Pipeline loaded from %s  (n_pca=%d, pca_var=%.1f%%)",
        path, pipeline["n_pca"], pipeline.get("pca_variance_explained", float("nan")),
    )
    return pipeline
