"""Unit tests for the canonical embedding path grammar.

Covers :mod:`dynaclr.evaluation.paths` — pure path construction and the pooling
glob. No real zarr required; :func:`iter_embeddings` is exercised against a
temporary directory tree.
"""

from pathlib import Path

from dynaclr.evaluation.paths import (
    DATASETS_ROOT,
    dataset_name_from_data_path,
    dataset_root_from_data_path,
    embedding_store,
    iter_embeddings,
    prediction_dir,
    predictions_root,
)

MF = "DynaCLR-2D-MIP-BagOfChannels"
RUN = "2d-mip-fix-shuffler"
CKPT = "epoch105-step84800"


def test_dataset_name_from_data_path():
    assert dataset_name_from_data_path("/data/2026_07_01_ZIKV/2026_07_01_ZIKV.zarr") == "2026_07_01_ZIKV"


def test_dataset_root_from_data_path():
    assert dataset_root_from_data_path("/data/2026_07_01_ZIKV/2026_07_01_ZIKV.zarr") == Path("/data/2026_07_01_ZIKV")


def test_predictions_root_default_and_override():
    assert predictions_root("ds") == DATASETS_ROOT / "ds" / "2-phenotyping" / "predictions"
    assert predictions_root("ds", datasets_root="/base") == Path("/base/ds/2-phenotyping/predictions")


def test_prediction_dir():
    assert prediction_dir("ds", MF, RUN, CKPT, datasets_root="/base") == Path(
        f"/base/ds/2-phenotyping/predictions/{MF}/{RUN}/{CKPT}"
    )


def test_embedding_store():
    assert embedding_store("ds", MF, RUN, CKPT, "SEC61B", datasets_root="/base") == Path(
        f"/base/ds/2-phenotyping/predictions/{MF}/{RUN}/{CKPT}/embeddings/SEC61B.zarr"
    )


def _make_tree(root: Path, dataset: str, markers: list[str]) -> None:
    d = root / dataset / "2-phenotyping" / "predictions" / MF / RUN / CKPT / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    for m in markers:
        (d / f"{m}.zarr").mkdir()


def test_iter_embeddings_pools_across_datasets(tmp_path):
    """One marker across multiple datasets is collected by a single glob."""
    _make_tree(tmp_path, "ds_a", ["SEC61B", "TOMM20"])
    _make_tree(tmp_path, "ds_b", ["SEC61B"])
    found = iter_embeddings(MF, RUN, CKPT, marker="SEC61B", datasets_root=tmp_path)
    assert [p.name for p in found] == ["SEC61B.zarr", "SEC61B.zarr"]
    assert {p.parents[6].name for p in found} == {"ds_a", "ds_b"}


def test_iter_embeddings_all_markers(tmp_path):
    """marker=None matches every marker."""
    _make_tree(tmp_path, "ds_a", ["SEC61B", "TOMM20"])
    found = iter_embeddings(MF, RUN, CKPT, datasets_root=tmp_path)
    assert sorted(p.name for p in found) == ["SEC61B.zarr", "TOMM20.zarr"]


def test_iter_embeddings_respects_run_scope(tmp_path):
    """A different run is not matched."""
    _make_tree(tmp_path, "ds_a", ["SEC61B"])
    assert iter_embeddings(MF, "other-run", CKPT, datasets_root=tmp_path) == []
