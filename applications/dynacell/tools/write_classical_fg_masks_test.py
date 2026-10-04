"""Integration tests for ``write_classical_fg_masks.py``.

Each test builds a tiny synthetic HCS OME-Zarr in ``tmp_path`` and runs the tool
for real. ``segment`` is replaced by a fixed threshold: the eval binarizer is
tested with the evaluation code, and this tool's contract is the array it writes.

Run (worktree)::

    uv run --no-sync pytest applications/dynacell/tools/write_classical_fg_masks_test.py -q
"""

from __future__ import annotations

import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import write_classical_fg_masks
from iohub.ngff import open_ome_zarr
from write_classical_fg_masks import main

_CHANNELS = ["Phase3D", "Structure", "Nuclei"]
_SHAPE = (2, len(_CHANNELS), 4, 16, 16)
_POSITIONS = ("A/1/0", "A/2/0")


def _threshold(img: np.ndarray, target_name: str) -> np.ndarray:
    assert target_name == "er"
    return img > 0.5


@pytest.fixture(autouse=True)
def _fixed_binarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_classical_fg_masks, "segment", _threshold)


def _make_store(path: Path) -> dict[str, np.ndarray]:
    """Write a tiny random store; return each position's image."""
    rng = np.random.default_rng(0)
    images = {}
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=_CHANNELS) as plate:
        for name in _POSITIONS:
            data = rng.random(_SHAPE, dtype=np.float32)
            pos = plate.create_position(*name.split("/"))
            pos.create_image("0", data, chunks=(1, 1, 4, 16, 16))
            images[name] = data
    return images


def test_writes_uint8_tczyx_mask_with_ones_off_target(tmp_path: Path) -> None:
    """The target channel holds the binarizer output; every other channel is 1."""
    store = tmp_path / "tiny.zarr"
    images = _make_store(store)
    assert main([str(store), "--target-name", "er"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        for name, pos in plate.positions():
            mask = pos["fg_mask"]
            assert mask.dtype == np.uint8
            assert mask.shape == _SHAPE
            got = mask[:]
            want_target = (images[name][:, 1] > 0.5).astype(np.uint8)
            np.testing.assert_array_equal(got[:, 1], want_target)
            assert 0 < want_target.mean() < 1
            np.testing.assert_array_equal(got[:, [0, 2]], 1)


def test_records_provenance_on_every_array(tmp_path: Path) -> None:
    """Every written array names its writer, binarizer, target, channel and date."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    assert main([str(store), "--target-name", "er", "--channel", "Structure"]) == 0
    want = {
        "writer": "applications/dynacell/tools/write_classical_fg_masks.py",
        "binarizer": "dynacell.evaluation.segmentation.segment",
        "target_name": "er",
        "channel": "Structure",
        "cubic_version": version("cubic"),
        "date": datetime.date.today().isoformat(),
    }
    with open_ome_zarr(store, mode="r") as plate:
        provenance = [dict(pos["fg_mask"].native.attrs["provenance"]) for _, pos in plate.positions()]
    assert provenance == [want] * len(_POSITIONS)


@pytest.mark.parametrize("shard", ["2/2", "-1/2", "0/0"])
def test_rejects_an_out_of_range_shard(tmp_path: Path, shard: str) -> None:
    """A shard outside ``0 <= i < n`` raises instead of silently writing a partial store."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    with pytest.raises(ValueError, match="need 0 <= i < n"):
        main([str(store), "--target-name", "er", f"--shard={shard}"])
    with open_ome_zarr(store, mode="r") as plate:
        assert not any("fg_mask" in pos for _, pos in plate.positions())


def test_dry_run_reports_on_a_store_that_already_has_masks(tmp_path: Path) -> None:
    """A dry run segments and reports without tripping the overwrite guard or writing."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    assert main([str(store), "--target-name", "er"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        before = {name: pos["fg_mask"][:] for name, pos in plate.positions()}
    assert main([str(store), "--target-name", "er", "--dry-run"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        for name, pos in plate.positions():
            np.testing.assert_array_equal(pos["fg_mask"][:], before[name])


def test_refuses_to_overwrite_an_existing_mask(tmp_path: Path) -> None:
    """A second run raises before touching the masks the first run wrote."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    assert main([str(store), "--target-name", "er"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        before = {name: pos["fg_mask"][:] for name, pos in plate.positions()}
    with pytest.raises(FileExistsError, match="'fg_mask' already exists at A/1/0"):
        main([str(store), "--target-name", "er"])
    with open_ome_zarr(store, mode="r") as plate:
        for name, pos in plate.positions():
            np.testing.assert_array_equal(pos["fg_mask"][:], before[name])
