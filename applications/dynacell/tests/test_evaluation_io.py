"""Regression tests for evaluation I/O dispatch."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _import_io_with_stubs(monkeypatch):
    """Import the I/O module with lightweight optional-dependency stubs."""
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.DictConfig = dict

    cubic_module = types.ModuleType("cubic")
    cubic_cuda_module = types.ModuleType("cubic.cuda")
    cubic_cuda_module.ascupy = lambda x: x
    cubic_cuda_module.asnumpy = lambda x: x
    cubic_skimage_module = types.ModuleType("cubic.skimage")
    cubic_skimage_module.transform = types.SimpleNamespace(resize=lambda *args, **kwargs: None)

    iohub_module = types.ModuleType("iohub")
    iohub_module.read_images = lambda *args, **kwargs: None
    iohub_ngff_module = types.ModuleType("iohub.ngff")
    iohub_ngff_module.open_ome_zarr = lambda *args, **kwargs: None

    skimage_module = types.ModuleType("skimage")
    skimage_io_module = types.ModuleType("skimage.io")
    skimage_io_module.imsave = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_module)
    monkeypatch.setitem(sys.modules, "cubic", cubic_module)
    monkeypatch.setitem(sys.modules, "cubic.cuda", cubic_cuda_module)
    monkeypatch.setitem(sys.modules, "cubic.skimage", cubic_skimage_module)
    monkeypatch.setitem(sys.modules, "iohub", iohub_module)
    monkeypatch.setitem(sys.modules, "iohub.ngff", iohub_ngff_module)
    monkeypatch.setitem(sys.modules, "skimage", skimage_module)
    monkeypatch.setitem(sys.modules, "skimage.io", skimage_io_module)
    sys.modules.pop("dynacell.evaluation.io", None)

    return importlib.import_module("dynacell.evaluation.io")


def test_is_zarr_path_checks_final_suffix(monkeypatch) -> None:
    """Only the final suffix should determine Zarr-path classification."""
    io = _import_io_with_stubs(monkeypatch)
    assert io._is_zarr_path(Path("plate.zarr"))
    assert not io._is_zarr_path(Path("plate.zarr.tiff"))
    assert not io._is_zarr_path(Path("plate.ome.tif"))


def test_imread_dispatches_by_path_type(monkeypatch) -> None:
    """Imread should route Zarr and TIFF-like paths to different backends."""
    io = _import_io_with_stubs(monkeypatch)
    calls: list[tuple[str, Path]] = []

    def fake_read_ome_zarr(path: Path) -> np.ndarray:
        calls.append(("zarr", path))
        return np.array([1], dtype=np.uint8)

    def fake_read_with_iohub(path: Path) -> np.ndarray:
        calls.append(("iohub", path))
        return np.array([2], dtype=np.uint8)

    monkeypatch.setattr(io, "_read_ome_zarr", fake_read_ome_zarr)
    monkeypatch.setattr(io, "_read_with_iohub", fake_read_with_iohub)

    assert np.array_equal(io.imread("sample.zarr"), np.array([1], dtype=np.uint8))
    assert np.array_equal(io.imread("sample.ome.tif"), np.array([2], dtype=np.uint8))
    assert np.array_equal(io.imread("sample.zarr.tiff"), np.array([2], dtype=np.uint8))
    assert calls == [
        ("zarr", Path("sample.zarr")),
        ("iohub", Path("sample.ome.tif")),
        ("iohub", Path("sample.zarr.tiff")),
    ]


def test_imsave_dispatches_by_path_type(monkeypatch) -> None:
    """Imsave should preserve TIFF-like outputs while supporting OME-Zarr."""
    io = _import_io_with_stubs(monkeypatch)
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    calls: list[tuple[str, Path, np.ndarray]] = []

    def fake_save_ome_zarr(path: Path, data: np.ndarray) -> None:
        calls.append(("zarr", path, data.copy()))

    def fake_save_with_skimage(path: Path, data: np.ndarray) -> None:
        calls.append(("tiff", path, data.copy()))

    monkeypatch.setattr(io, "_save_ome_zarr", fake_save_ome_zarr)
    monkeypatch.setattr(io, "_save_with_skimage", fake_save_with_skimage)

    io.imsave("sample.zarr", image)
    io.imsave("sample.ome.tif", image)

    assert calls[0][0] == "zarr"
    assert calls[0][1] == Path("sample.zarr")
    assert np.array_equal(calls[0][2], image)
    assert calls[1][0] == "tiff"
    assert calls[1][1] == Path("sample.ome.tif")
    assert np.array_equal(calls[1][2], image)
