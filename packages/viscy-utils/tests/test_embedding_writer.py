"""Tests for embedding writer provenance stamping."""

import anndata as ad
import numpy as np
import pandas as pd

from viscy_utils.callbacks.embedding_writer import write_embedding_dataset


def _index_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fov_name": [f"A/1/{i}" for i in range(n)],
            "track_id": list(range(n)),
            "t": [0] * n,
        }
    )


def test_uns_metadata_round_trips(tmp_path):
    """Provenance passed as uns_metadata lands in adata.uns on disk."""
    out = tmp_path / "emb.zarr"
    provenance = {
        "model_family": "DynaCLR-2D-MIP-BagOfChannels",
        "run": "2d-mip-fix-shuffler",
        "ckpt_name": "epoch105-step84800",
        "collection_path": "/configs/collections/foo.yml",
        "marker": "SEC61B",
    }
    write_embedding_dataset(
        output_path=out,
        features=np.random.default_rng(0).random((3, 4), dtype=np.float32),
        index_df=_index_df(3),
        uns_metadata=provenance,
    )
    uns = ad.read_zarr(out).uns
    for key, value in provenance.items():
        assert uns[key] == value


def test_no_uns_metadata_is_fine(tmp_path):
    """Writing without provenance still works (backward compatible)."""
    out = tmp_path / "emb.zarr"
    write_embedding_dataset(
        output_path=out,
        features=np.random.default_rng(1).random((2, 4), dtype=np.float32),
        index_df=_index_df(2),
    )
    assert ad.read_zarr(out).n_obs == 2
