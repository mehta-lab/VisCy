"""Core functions for LOT (Linear Optimal Transport) batch correction.

Pipeline
--------
1. Pool one or more pre-filtered source AnnData objects and one or more
   pre-filtered target AnnData objects (each already reduced to the
   reference population, e.g. uninfected cells).
2. Fit a shared StandardScaler on the combined source + target cells.
3. Optionally fit a shared PCA on the scaled combined cells.
4. Fit a LinearTransport (LOT) map mapping the pooled source distribution to
   the pooled target distribution, in PCA space when PCA is enabled or in the
   scaled embedding space otherwise.
5. Save the fitted pipeline (scaler, optional PCA, LOT) to disk with joblib.

The saved pipeline can then be applied to any source zarr to produce a new
zarr whose embeddings are corrected for cross-platform batch effects (in the
target's PCA coordinate system when PCA is enabled).

Callers are responsible for loading zarrs and filtering to the reference
population before calling :func:`fit_lot_correction`; see
``fit_lot_correction.py`` for the CLI that does this from a YAML config.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import joblib
import numpy as np
import ot
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


def _to_np(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 numpy array."""
    return np.array(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def _is_string_dtype(dtype) -> bool:
    """Return True for pandas string extension dtypes that zarr cannot write.

    Covers both ``pd.StringDtype`` (nullable strings) and ``pd.ArrowDtype``
    backed by an Arrow string type, which pandas 3 uses by default and anndata
    cannot serialize to zarr.
    """
    if isinstance(dtype, pd.StringDtype):
        return True
    if isinstance(dtype, pd.ArrowDtype):
        return pd.api.types.is_string_dtype(dtype)
    return False


def _coerce_obs_for_zarr(obs: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of *obs* with string/Arrow dtypes coerced for zarr writing.

    Under pandas 3 / anndata 0.12, ``.obs`` columns are often string extension
    dtypes or Categoricals whose categories are string-extension-backed. Zarr
    cannot serialize either, so string columns become plain object and
    string-backed Categoricals are dropped to plain object (recategorizing keeps
    the string-extension-backed categories, which still fail to write). The obs
    index is also coerced to object.
    """
    obs = obs.copy()
    obs.index = obs.index.astype(object)
    for col in obs.columns:
        dtype = obs[col].dtype
        if _is_string_dtype(dtype):
            obs[col] = obs[col].astype(object)
        elif isinstance(dtype, pd.CategoricalDtype) and _is_string_dtype(dtype.categories.dtype):
            obs[col] = obs[col].astype(str).astype(object)
    return obs


def _pool_embeddings(adatas: list[ad.AnnData]) -> np.ndarray:
    """Stack the ``.X`` matrices of several AnnData objects into one array.

    Parameters
    ----------
    adatas : list of AnnData
        AnnData objects to pool. Must be non-empty and share the same number
        of features (``.n_vars``).

    Returns
    -------
    np.ndarray
        Row-wise concatenation of every ``.X`` as a float32 array of shape
        ``(sum_of_n_obs, n_vars)``.
    """
    if not adatas:
        raise ValueError("Expected at least one AnnData object to pool, got an empty list.")

    n_vars = {a.n_vars for a in adatas}
    if len(n_vars) != 1:
        raise ValueError(f"All AnnData objects must share the same feature dimension, got n_vars={sorted(n_vars)}.")

    return np.vstack([_to_np(a.X) for a in adatas])


def fit_lot_correction(
    source_adatas: list[ad.AnnData],
    target_adatas: list[ad.AnnData],
    channel: Optional[str] = None,
    n_pca: Optional[int] = 50,
    ns_lot: Optional[int] = 3000,
    random_seed: int = 42,
) -> dict:
    """Fit a shared (optional PCA) + LOT batch-correction pipeline.

    Both ``source_adatas`` and ``target_adatas`` are expected to be already
    filtered to the reference population (e.g. uninfected cells). Multiple
    datasets on each side are pooled (row-wise concatenated) before fitting,
    so batch effects are estimated against the combined distributions.

    Parameters
    ----------
    source_adatas : list of AnnData
        Pre-filtered source datasets (e.g. light-sheet embeddings). Pooled
        into a single source distribution.
    target_adatas : list of AnnData
        Pre-filtered target datasets (e.g. confocal embeddings). Pooled into
        a single target distribution.
    channel : str or None, optional
        The bag-of-channels channel/marker these embeddings were computed for
        (e.g. ``"Phase3D"``). Recorded in the returned pipeline for provenance
        so the fitted map is not blindly applied to a different channel. This
        is a label only — the caller is responsible for filtering the input
        AnnData to this channel. By default ``None``.
    n_pca : int or None, optional
        Number of PCA components for the shared PCA. When ``None``, PCA is
        skipped and LOT is fit in the scaled embedding space. By default 50.
    ns_lot : int or None, optional
        Maximum number of cells subsampled per side for LOT fitting. This is a
        compute cap on covariance estimation, not a source/target balancer:
        ``LinearTransport`` only needs each side's mean and covariance, so the
        two sides need not have equal counts. When ``None``, every pooled cell
        is used. By default 3000.
    random_seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    dict with keys ``"scaler"``, ``"pca"`` (``None`` when disabled), ``"lot"``,
    ``"channel"``, ``"n_pca"``, ``"ns_lot"``, ``"random_seed"``,
    ``"pca_variance_explained"`` (``None`` when PCA is disabled).
    """
    rng = np.random.default_rng(random_seed)

    _logger.info("Fitting LOT for channel: %s", channel if channel is not None else "(unspecified)")

    X_src = _pool_embeddings(source_adatas)
    X_tgt = _pool_embeddings(target_adatas)

    _logger.info(
        "Pooled source: %d cells from %d dataset(s);  target: %d cells from %d dataset(s)",
        len(X_src),
        len(source_adatas),
        len(X_tgt),
        len(target_adatas),
    )

    if len(X_src) < 5 or len(X_tgt) < 5:
        raise ValueError(
            "Too few reference cells to fit LOT "
            f"(source={len(X_src)}, target={len(X_tgt)}). "
            "Check the datasets and their filtering."
        )

    _logger.info("Fitting shared StandardScaler ...")
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(np.vstack([X_src, X_tgt]))

    n_src = len(X_src)
    if n_pca is not None:
        _logger.info("Fitting shared PCA-%d ...", n_pca)
        pca = PCA(n_components=n_pca, random_state=random_seed)
        Z_all = pca.fit_transform(X_combined_scaled)
        var_exp = float(pca.explained_variance_ratio_.sum() * 100)
        _logger.info("PCA explained variance: %.1f%%", var_exp)
    else:
        _logger.info("PCA disabled — fitting LOT in scaled embedding space.")
        pca = None
        Z_all = X_combined_scaled
        var_exp = None

    Z_src = Z_all[:n_src]
    Z_tgt = Z_all[n_src:]

    ns_src = len(Z_src) if ns_lot is None else min(len(Z_src), ns_lot)
    ns_tgt = len(Z_tgt) if ns_lot is None else min(len(Z_tgt), ns_lot)
    idx_src = rng.choice(len(Z_src), ns_src, replace=False)
    idx_tgt = rng.choice(len(Z_tgt), ns_tgt, replace=False)

    _logger.info("Fitting LOT (source subsample=%d, target subsample=%d) ...", ns_src, ns_tgt)
    lot = ot.da.LinearTransport(reg=1e-3)
    lot.fit(Xs=Z_src[idx_src], Xt=Z_tgt[idx_tgt])
    _logger.info("LOT fitted.")

    return {
        "scaler": scaler,
        "pca": pca,
        "lot": lot,
        "channel": channel,
        "n_pca": n_pca,
        "ns_lot": ns_lot,
        "random_seed": random_seed,
        "pca_variance_explained": var_exp,
    }


def apply_lot_correction(
    input_zarr: Union[str, Path],
    pipeline: dict,
    output_zarr: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """Apply a fitted LOT pipeline to an embedding zarr.

    Transforms all cells through StandardScaler → (optional PCA) → LOT and writes
    an AnnData zarr whose ``.X`` contains the corrected embeddings. The matching
    pre-LOT coordinates are stored in ``obsm["X_pre_lot"]`` for correction QC.
    Input metadata are preserved; other arrays derived from the uncorrected space
    are dropped.

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
        Existing output is replaced only after the new store is fully written.
    """
    import shutil
    import tempfile

    input_zarr = Path(input_zarr)
    output_zarr = Path(output_zarr)
    if input_zarr.resolve() == output_zarr.resolve():
        raise ValueError("input_zarr and output_zarr must be different paths.")
    if output_zarr.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {output_zarr}. Set overwrite=true to overwrite.")

    _logger.info("Loading input zarr: %s", input_zarr)
    adata_in = ad.read_zarr(input_zarr)
    adata_in.obs_names_make_unique()

    X = _to_np(adata_in.X)
    _logger.info("Input shape: %s", adata_in.shape)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    lot = pipeline["lot"]

    X_scaled = scaler.transform(X)
    if pca is not None:
        _logger.info("Applying StandardScaler → PCA → LOT ...")
        Z = pca.transform(X_scaled)
    else:
        _logger.info("Applying StandardScaler → LOT (PCA disabled) ...")
        Z = X_scaled
    Z_corrected = lot.transform(Z)
    _logger.info("Corrected embeddings shape: %s  (n_pca=%s)", Z_corrected.shape, pipeline["n_pca"])

    obs = _coerce_obs_for_zarr(adata_in.obs)

    try:
        ad.settings.allow_write_nullable_strings = True
    except AttributeError:
        pass

    adata_out = ad.AnnData(X=Z_corrected.astype(np.float32), obs=obs, uns=dict(adata_in.uns))
    adata_out.obsm["X_pre_lot"] = np.asarray(Z, dtype=np.float32)
    adata_out.var.index = adata_out.var.index.astype(object)
    adata_out.uns["lot_correction"] = {
        "source_zarr": str(input_zarr),
        "channel": pipeline.get("channel"),
        "n_pca": pipeline["n_pca"],
        "pca_variance_explained": pipeline.get("pca_variance_explained"),
    }

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_zarr.name}.", dir=output_zarr.parent))
    temp_output = temp_root / "new.zarr"
    backup_output = temp_root / "previous.zarr"
    keep_temp = False
    try:
        _logger.info("Writing corrected zarr: %s", temp_output)
        adata_out.write_zarr(temp_output, convert_strings_to_categoricals=False)
        if output_zarr.exists():
            output_zarr.rename(backup_output)
        try:
            temp_output.rename(output_zarr)
        except Exception:
            if backup_output.exists():
                try:
                    backup_output.rename(output_zarr)
                except Exception:
                    keep_temp = True
                    _logger.exception("Could not restore previous output; backup retained at %s", backup_output)
            raise
    finally:
        if not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    _logger.info("Done: %s", output_zarr)


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
    var_exp = pipeline.get("pca_variance_explained")
    _logger.info(
        "Pipeline loaded from %s  (channel=%s, n_pca=%s, pca_var=%s)",
        path,
        pipeline.get("channel") or "(unspecified)",
        pipeline["n_pca"],
        "disabled" if var_exp is None else f"{var_exp:.1f}%",
    )
    return pipeline
