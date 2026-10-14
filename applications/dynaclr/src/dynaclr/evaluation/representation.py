"""Helpers for selecting an AnnData representation as evaluator input."""

from __future__ import annotations

import anndata as ad
import numpy as np


def representation_as_x(adata: ad.AnnData, embedding_key: str | None) -> ad.AnnData:
    """Return an evaluator view whose ``X`` is raw or a selected ``obsm`` array.

    A new AnnData is required because marker-specific PCA representations can
    have a different width from the backbone ``X`` and AnnData does not permit
    assigning an array with a different number of variables in place.
    """
    if embedding_key is None:
        return adata
    if embedding_key not in adata.obsm:
        raise KeyError(f"obsm[{embedding_key!r}] not found")
    return ad.AnnData(
        X=np.asarray(adata.obsm[embedding_key]),
        obs=adata.obs.copy(),
    )
