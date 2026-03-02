"""Utilities for selectively updating AnnData zarr stores."""

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import zarr
from anndata.io import write_elem


def append_to_anndata_zarr(
    zarr_path: str | Path,
    *,
    obsm: dict[str, Any] | None = None,
    obs: pd.DataFrame | None = None,
    uns: dict | None = None,
) -> None:
    """Selectively write obs, obsm, or uns into an existing AnnData zarr store.

    Unlike ``adata.write_zarr()``, this only updates the specified slots
    without overwriting unrelated data (X, var, layers, etc.).

    Parameters
    ----------
    zarr_path : str | Path
        Path to an existing AnnData zarr store.
    obsm : dict[str, Any], optional
        Mapping of obsm keys to arrays. Each key is written to ``obsm/{key}``,
        replacing any existing entry.
    obs : pd.DataFrame, optional
        Observation metadata. Replaces the entire ``obs`` group.
    uns : dict, optional
        Unstructured annotation. Replaces the entire ``uns`` group.
    """
    store = zarr.open(str(zarr_path), mode="a", use_consolidated=False)
    ad.settings.allow_write_nullable_strings = True

    if obs is not None:
        if "obs" in store:
            del store["obs"]
        write_elem(store, "obs", obs)

    if obsm is not None:
        for key, value in obsm.items():
            obsm_path = f"obsm/{key}"
            if obsm_path in store:
                del store[obsm_path]
            write_elem(store, obsm_path, value)

    if uns is not None:
        if "uns" in store:
            del store["uns"]
        write_elem(store, "uns", uns)

    zarr.consolidate_metadata(str(zarr_path))
