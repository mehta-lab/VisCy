r"""Integration tests for ``write_otsu_thresholds.py``.

Each test builds a tiny synthetic HCS OME-Zarr in ``tmp_path`` carrying a
deliberately WRONG ``normalization`` block -- statistics that do not describe the
pixels, exactly as ``cell_focus.zarr`` carries its full-Z parent's values. The tool
is then run for real and the store reopened, so the tests pin the one property the
whole design rests on: every pre-existing statistic survives byte-for-byte.

Run (worktree)::

    uv run --no-sync pytest applications/dynacell/tools/write_otsu_thresholds_test.py -q
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from write_otsu_thresholds import main  # noqa: E402

_CHANNELS = ["Phase3D", "Nuclei", "Membrane"]
# Deliberately not the pixels' real statistics: the point is that they survive.
_SENTINEL_STATS = {
    "mean": 1234.5,
    "std": 67.5,
    "median": 1200.0,
    "iqr": 42.0,
    "min": 0.0,
    "max": 9999.0,
    "p1": 1.0,
    "p5": 5.0,
    "p95": 95.0,
    "p99": 99.0,
    "p95_p5": 90.0,
    "p99_p1": 98.0,
}


def _norm_block() -> dict:
    """Build a three-level normalization block for every channel, all sentinel-valued."""
    return {
        channel: {
            "fov_statistics": dict(_SENTINEL_STATS),
            "dataset_statistics": dict(_SENTINEL_STATS),
            "timepoint_statistics": {"0": dict(_SENTINEL_STATS)},
        }
        for channel in _CHANNELS
    }


def _make_store(path: Path, positions: tuple[str, ...] = ("A/1/0", "A/2/0")) -> None:
    """Write a tiny bimodal HCS store whose stored statistics describe nothing."""
    rng = np.random.default_rng(0)
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=_CHANNELS) as plate:
        plate.zattrs.update({"normalization": _norm_block()})
        for name in positions:
            row, col, fov = name.split("/")
            # Bimodal so Otsu has a real split to find: dim background, bright blob.
            data = rng.normal(100.0, 5.0, size=(1, len(_CHANNELS), 4, 32, 32)).astype(np.float32)
            data[:, :, :, 8:24, 8:24] += 400.0
            pos = plate.create_position(row, col, fov)
            pos.create_image("0", data, chunks=(1, 1, 4, 32, 32))
            pos.zattrs.update({"normalization": _norm_block()})


def _read_norm(path: Path) -> dict:
    """Return {position_name: normalization block} plus the plate block under ''."""
    out = {}
    with open_ome_zarr(path, mode="r") as plate:
        out[""] = copy.deepcopy(plate.zattrs["normalization"])
        for name, pos in plate.positions():
            out[name] = copy.deepcopy(pos.zattrs["normalization"])
    return out


def test_only_adds_otsu_threshold_and_changes_nothing_else(tmp_path: Path) -> None:
    """Every pre-existing statistic survives byte-for-byte; one key is added."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    before = _read_norm(store)

    assert main([str(store), "--channel", "Nuclei", "--channel", "Membrane"]) == 0

    after = _read_norm(store)
    assert set(after) == set(before)
    for node, channels in after.items():
        for channel, levels in channels.items():
            expected_new = {"otsu_threshold"} if (node and channel in ("Nuclei", "Membrane")) else set()
            for level, stats in levels.items():
                prior = before[node][channel][level]
                added = set(stats) - set(prior)
                assert added == (expected_new if level == "fov_statistics" else set()), (
                    f"{node}/{channel}/{level}: unexpected added keys {added}"
                )
                for key, value in prior.items():
                    assert stats[key] == value, f"{node}/{channel}/{level}.{key} was rewritten"


def test_threshold_separates_the_two_modes(tmp_path: Path) -> None:
    """The written threshold is a real Otsu split, not a degenerate constant."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    assert main([str(store), "--channel", "Nuclei"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        for _, pos in plate.positions():
            threshold = pos.zattrs["normalization"]["Nuclei"]["fov_statistics"]["otsu_threshold"]
            assert 100.0 < threshold < 500.0, threshold


def test_all_or_nothing_across_positions(tmp_path: Path) -> None:
    """Either every (position, channel) carries the key or none does.

    A partially-written store is the dangerous state: ``_collate_norm_meta`` takes
    its stat-key set from sample 0 of the batch, so a mixed store raises KeyError
    only on batches whose sample 0 happens to carry the key -- intermittently, and
    on baseline fits too.
    """
    store = tmp_path / "tiny.zarr"
    _make_store(store, positions=("A/1/0", "A/2/0", "B/1/0"))
    assert main([str(store), "--channel", "Nuclei", "--channel", "Membrane"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        present = [
            "otsu_threshold" in pos.zattrs["normalization"][channel]["fov_statistics"]
            for _, pos in plate.positions()
            for channel in ("Nuclei", "Membrane")
        ]
    assert len(present) == 6
    assert all(present)


def test_undo_restores_the_original_block(tmp_path: Path) -> None:
    """``--undo`` is an exact inverse, so the mutation is fully reversible."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    before = _read_norm(store)
    assert main([str(store), "--channel", "Nuclei", "--channel", "Membrane"]) == 0
    assert _read_norm(store) != before
    assert main([str(store), "--channel", "Nuclei", "--channel", "Membrane", "--undo"]) == 0
    assert _read_norm(store) == before


def test_rerun_is_a_noop_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Re-running is idempotent, and ``--dry-run`` leaves the store untouched."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    untouched = _read_norm(store)
    assert main([str(store), "--channel", "Nuclei", "--dry-run"]) == 0
    assert _read_norm(store) == untouched

    assert main([str(store), "--channel", "Nuclei"]) == 0
    written = _read_norm(store)
    assert main([str(store), "--channel", "Nuclei"]) == 0
    assert _read_norm(store) == written


def test_unknown_channel_fails_loudly(tmp_path: Path) -> None:
    """A typo'd channel exits non-zero rather than silently writing nothing."""
    store = tmp_path / "tiny.zarr"
    _make_store(store)
    assert main([str(store), "--channel", "Nucleii"]) == 2


@pytest.mark.parametrize("constant", [0.0, 7.5])
def test_constant_fov_uses_the_constant_itself(tmp_path: Path, constant: float) -> None:
    """Otsu is undefined on a constant input; the constant is stored instead.

    ``generate_fg_masks`` thresholds with ``>=``, so this marks the FOV entirely
    foreground rather than crashing — the same choice the stock path makes.
    """
    store = tmp_path / "flat.zarr"
    with open_ome_zarr(store, layout="hcs", mode="w-", channel_names=_CHANNELS) as plate:
        plate.zattrs.update({"normalization": _norm_block()})
        pos = plate.create_position("A", "1", "0")
        pos.create_image("0", np.full((1, len(_CHANNELS), 4, 32, 32), constant, dtype=np.float32))
        pos.zattrs.update({"normalization": _norm_block()})

    assert main([str(store), "--channel", "Nuclei"]) == 0
    with open_ome_zarr(store, mode="r") as plate:
        _, pos = next(iter(plate.positions()))
        assert pos.zattrs["normalization"]["Nuclei"]["fov_statistics"]["otsu_threshold"] == constant
