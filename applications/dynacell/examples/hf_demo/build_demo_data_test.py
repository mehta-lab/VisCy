r"""Integration tests for ``build_demo_data.py``.

Each test packs a tiny synthetic release-style ``{marker}_mock.ozx`` whose voxel values
are their own flat index, then runs the real builder with only the source dir and crop
sizes patched down. Because the crop is a pure copy, assertions are exact equality.

Run::

    uv run --no-sync pytest applications/dynacell/examples/hf_demo/build_demo_data_test.py -q
"""

import sys
import zipfile
from pathlib import Path

import build_demo_data
import numpy as np
import pytest
from iohub.core.ozx import pack_ozx
from iohub.ngff import TransformationMeta, open_ome_zarr

_SHAPE = (10, 3, 12, 20, 24)
_HPI = [5.0 + 2 * t for t in range(_SHAPE[0])]


def _write_release_ozx(src_dir: Path, marker: str, target: str, negative: bool = False) -> np.ndarray:
    """Pack a release-like test store with fov0006; return that FOV's array."""
    data = np.arange(np.prod(_SHAPE), dtype=np.float32).reshape(_SHAPE)
    if negative:
        data[2, 2, 6, 10, 12] = -1.0  # inside the crop: t=2, target channel
    store = src_dir / f"{marker}_mock.zarr"
    with open_ome_zarr(
        store, layout="hcs", mode="w-", channel_names=["Phase3D", "Brightfield", target], version="0.5"
    ) as plate:
        for fov in ("fov0000", "fov0006"):
            pos = plate.create_position("0", "0", fov)
            pos.create_image(
                "0",
                data if fov == "fov0006" else np.zeros(_SHAPE, np.float32),
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 0.174, 0.1494, 0.1494])],
            )
            pos.zattrs["hpi_values"] = _HPI
    pack_ozx(store, src_dir / f"{marker}_mock.ozx")
    return data


@pytest.fixture
def small_crop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the builder at ``tmp_path/src`` with a 4 x 8 x 8 crop."""
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(build_demo_data, "RELEASE_TEST", src)
    monkeypatch.setattr(build_demo_data, "Z_SIZE", 4)
    monkeypatch.setattr(build_demo_data, "YX_SIZE", 8)
    return src


def test_main_crops_fov0006_and_zips_store_at_root(
    small_crop: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _write_release_ozx(small_crop, "CAAX", "Membrane")
    monkeypatch.setattr(build_demo_data, "MARKERS", ("CAAX",))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["build_demo_data.py", str(out)])

    build_demo_data.main()

    zip_path = out / "CAAX_mock.zarr.zip"
    assert sorted(p.name for p in out.iterdir()) == ["CAAX_mock.zarr.zip"]
    with zipfile.ZipFile(zip_path) as zf:
        assert {n.split("/")[0] for n in zf.namelist()} == {"CAAX_mock.zarr"}
        zf.extractall(tmp_path / "unzipped")
    with open_ome_zarr(tmp_path / "unzipped" / "CAAX_mock.zarr", mode="r") as plate:
        names = [name for name, _ in plate.positions()]
        assert names == ["0/0/fov0006"]
        pos = plate["0/0/fov0006"]
        assert pos.channel_names == ["Phase3D", "Brightfield", "Membrane"]
        assert pos.scale == [1.0, 1.0, 0.174, 0.1494, 0.1494]
        # Central (12-4)//2, (20-8)//2, (24-8)//2 = (4, 6, 8), timepoints 0,2,4,6,8.
        expected = data[[0, 2, 4, 6, 8], :, 4:8, 6:14, 8:16]
        np.testing.assert_array_equal(np.asarray(pos.data), expected)
        assert pos.zattrs["dynacell_demo"] == {
            "source": "s3://dynacell/v1/data/biohub-a549/test/CAAX_mock.ozx",
            "position": "0/0/fov0006",
            "timepoints": [0, 2, 4, 6, 8],
            "hpi_values": [5.0, 9.0, 13.0, 17.0, 21.0],
            "zyx_offset": [4, 6, 8],
            "zyx_size": [4, 8, 8],
        }


def test_build_store_rejects_negative_target(small_crop: Path, tmp_path: Path) -> None:
    _write_release_ozx(small_crop, "SEC61B", "Structure", negative=True)
    with pytest.raises(ValueError, match="negative target pixels"):
        build_demo_data.build_store("SEC61B", tmp_path)
    assert not (tmp_path / "SEC61B_mock.zarr").exists()
