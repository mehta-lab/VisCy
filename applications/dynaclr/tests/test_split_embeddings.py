"""Integration tests for split-embeddings layouts.

Builds a small combined AnnData, writes it to zarr, and runs the real
:func:`split_embeddings` in both the flat and dataset-centric layouts. No GPU or
model required.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.paths import embedding_store
from dynaclr.evaluation.split_embeddings import split_embeddings

MF = "DynaCLR-2D-MIP-BagOfChannels"
RUN = "2d-mip-fix-shuffler"
CKPT = "epoch105-step84800"


def _combined_zarr(tmp_path):
    """Two datasets; ds_a has SEC61B+TOMM20, ds_b has SEC61B (6 cells total)."""
    rows = [
        ("ds_a", "SEC61B"),
        ("ds_a", "SEC61B"),
        ("ds_a", "TOMM20"),
        ("ds_b", "SEC61B"),
        ("ds_b", "SEC61B"),
        ("ds_b", "SEC61B"),
    ]
    obs = pd.DataFrame(
        {"experiment": [r[0] for r in rows], "marker": [r[1] for r in rows]},
        index=[str(i) for i in range(len(rows))],
    )
    adata = ad.AnnData(X=np.random.default_rng(0).random((len(rows), 4), dtype=np.float32), obs=obs)
    path = tmp_path / "embeddings.zarr"
    adata.write_zarr(path)
    return path


def test_route_by_dataset_writes_canonical_tree(tmp_path):
    combined = _combined_zarr(tmp_path)
    written = split_embeddings(
        combined,
        route_by_dataset=True,
        model_family=MF,
        run=RUN,
        ckpt_name=CKPT,
        datasets_root=tmp_path,
        keep_combined=True,
    )
    # one zarr per (experiment, marker) pair
    assert len(written) == 3
    assert embedding_store("ds_a", MF, RUN, CKPT, "SEC61B", datasets_root=tmp_path) in written
    assert embedding_store("ds_a", MF, RUN, CKPT, "TOMM20", datasets_root=tmp_path) in written
    assert embedding_store("ds_b", MF, RUN, CKPT, "SEC61B", datasets_root=tmp_path) in written
    # markers of one dataset co-locate
    a_sec = embedding_store("ds_a", MF, RUN, CKPT, "SEC61B", datasets_root=tmp_path)
    a_tom = embedding_store("ds_a", MF, RUN, CKPT, "TOMM20", datasets_root=tmp_path)
    assert a_sec.parent == a_tom.parent
    # cells partitioned correctly
    assert ad.read_zarr(a_sec).n_obs == 2
    assert ad.read_zarr(a_tom).n_obs == 1


def test_keep_combined_preserves_input(tmp_path):
    combined = _combined_zarr(tmp_path)
    split_embeddings(
        combined,
        route_by_dataset=True,
        model_family=MF,
        run=RUN,
        ckpt_name=CKPT,
        datasets_root=tmp_path,
        keep_combined=True,
    )
    assert combined.exists()


def test_default_deletes_combined(tmp_path):
    combined = _combined_zarr(tmp_path)
    split_embeddings(combined, output_dir=tmp_path / "out", group_by="experiment")
    assert not combined.exists()


def test_flat_layout_still_works(tmp_path):
    combined = _combined_zarr(tmp_path)
    out = tmp_path / "out"
    written = split_embeddings(combined, output_dir=out, group_by="experiment", keep_combined=True)
    assert {p.name for p in written} == {"ds_a.zarr", "ds_b.zarr"}


def test_route_by_dataset_requires_provenance(tmp_path):
    combined = _combined_zarr(tmp_path)
    with pytest.raises(ValueError, match="requires --model-family"):
        split_embeddings(combined, route_by_dataset=True, datasets_root=tmp_path, keep_combined=True)


def test_flat_requires_output_dir(tmp_path):
    combined = _combined_zarr(tmp_path)
    with pytest.raises(ValueError, match="output-dir is required"):
        split_embeddings(combined, keep_combined=True)
