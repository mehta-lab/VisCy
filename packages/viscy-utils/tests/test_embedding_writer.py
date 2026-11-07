import anndata as ad
import numpy as np
import pandas as pd

from viscy_utils.callbacks.embedding_writer import write_embedding_dataset


def _make_index_df(n: int, fov: str) -> pd.DataFrame:
    # Mimic EmbeddingWriter.write_on_epoch_end: per-batch frames concatenated, so the
    # positional index restarts each batch and the result has duplicate labels.
    half = n // 2
    batches = [
        pd.DataFrame({"fov_name": [fov] * half, "track_id": range(half), "t": range(half)}),
        pd.DataFrame(
            {"fov_name": [fov] * (n - half), "track_id": range(n - half), "t": range(n - half)}
        ),
    ]
    df = pd.concat(batches)
    assert df.index.has_duplicates  # precondition: the input that used to leak into obs_names
    return df


def test_obs_names_unique_within_store(tmp_path):
    n, d = 6, 4
    features = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    out = tmp_path / "store.zarr"

    write_embedding_dataset(output_path=out, features=features, index_df=_make_index_df(n, "A/1"))

    adata = ad.read_zarr(out)
    assert adata.obs_names.is_unique


def test_obs_names_unique_across_concatenated_stores(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 6, 4
    paths = []
    for i, fov in enumerate(["A/1", "B/2"]):
        out = tmp_path / f"store_{i}.zarr"
        features = rng.standard_normal((n, d)).astype(np.float32)
        write_embedding_dataset(output_path=out, features=features, index_df=_make_index_df(n, fov))
        paths.append(out)

    combined = ad.concat([ad.read_zarr(p) for p in paths])
    assert combined.obs_names.is_unique
