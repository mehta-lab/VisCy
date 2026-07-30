r"""Integration tests for ``build_temporal_subset_zarr.py``.

Each test writes a tiny synthetic HCS OME-Zarr in ``tmp_path`` whose voxel values
encode ``(position, timepoint, channel)``, then runs the real builder (no mocks) and
reopens the result. Because the subset is a pure copy, the assertions are exact
equality — a mis-indexed selection rule cannot hide behind a tolerance.

The store deliberately mirrors the two shapes the A549 pools actually have: a
uniform ``T=10`` plate (nucleus ``H2B_all.zarr``) and a mixed ``T=7``/``T=10`` plate
(ER ``SEC61B_all.zarr``), plus the ``normalization`` /
``hpi_values`` / ``native_frame_indices`` zattrs those stores carry.

Run::

    uv run --no-sync pytest applications/dynacell/tools/build_temporal_subset_zarr_test.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import TransformationMeta, open_ome_zarr

# The tools/ directory is not a Python package; add it to sys.path so the module is
# importable by short name (mirrors extract_focus_slab_store_test.py).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from build_temporal_subset_zarr import (  # noqa: E402
    build_temporal_subset_zarr,
    early_timepoints,
    main,
    spread_timepoints,
    verify_temporal_subset,
)

_CHANNELS = ["Phase3D", "Brightfield", "Nuclei"]
_Z, _Y, _X = 4, 8, 8


def _voxel_value(pos_ordinal: int, t: int, channel_idx: int) -> float:
    """Return the constant fill value identifying one (position, timepoint, channel)."""
    return float(1000 * pos_ordinal + 10 * t + channel_idx)


def _make_store(path: Path, frames_per_position: list[int]) -> None:
    """Write a synthetic HCS store whose voxels encode (position, timepoint, channel).

    ``frames_per_position`` gives each position's ``T``, so a mixed-length plate (the
    ER case) is expressible. Every position also carries the temporal + normalization
    zattrs the real A549 pools have, on the same 2 h / 5 hpi grid.
    """
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=_CHANNELS, version="0.5") as plate:
        plate.zattrs.update({"plate_id": "synthetic_temporal"})
        for ordinal, n_frames in enumerate(frames_per_position):
            data = np.empty((n_frames, len(_CHANNELS), _Z, _Y, _X), dtype=np.float32)
            for t in range(n_frames):
                for c in range(len(_CHANNELS)):
                    data[t, c] = _voxel_value(ordinal, t, c)
            pos = plate.create_position("0", "0", f"fov{ordinal:04d}")
            pos.create_image(
                "0",
                data,
                chunks=(1, 1, _Z, _Y, _X),
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 0.174, 0.1494, 0.1494])],
            )
            pos.zattrs.update(
                {
                    "condition": ["mock", "DENV", "ZIKV"][ordinal % 3],
                    "hpi_start": 5.0,
                    "effective_delta_t_h": 2.0,
                    "hpi_values": [5.0 + 2.0 * t for t in range(n_frames)],
                    "native_frame_indices": list(range(n_frames)),
                    "normalization": {
                        ch: {
                            "fov_statistics": {"mean": float(ordinal), "std": 1.0},
                            "timepoint_statistics": {
                                str(t): {"mean": _voxel_value(ordinal, t, c), "std": 1.0} for t in range(n_frames)
                            },
                        }
                        for c, ch in enumerate(_CHANNELS)
                    },
                }
            )


def test_early_timepoints_takes_the_front_of_the_movie() -> None:
    """Early mode is the first n frames, independent of position ordinal."""
    assert early_timepoints(10, 2) == [0, 1]
    assert early_timepoints(7, 2) == [0, 1]
    assert early_timepoints(10, 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="cannot keep"):
        early_timepoints(1, 2)


def test_spread_timepoints_matches_the_documented_rule() -> None:
    """Spread picks ordinal % T and steps by T // n_keep, wrapping."""
    # T=10, n_keep=2 -> step 5 (a 10 h separation on the real 2 h grid).
    assert spread_timepoints(0, 10, 2) == [0, 5]
    assert spread_timepoints(3, 10, 2) == [3, 8]
    assert spread_timepoints(7, 10, 2) == [2, 7]
    # T=7, n_keep=2 -> step 3.
    assert spread_timepoints(0, 7, 2) == [0, 3]
    assert spread_timepoints(5, 7, 2) == [1, 5]
    with pytest.raises(ValueError, match="cannot keep"):
        spread_timepoints(0, 1, 2)


def test_spread_timepoints_are_distinct_and_cover_the_course() -> None:
    """Over a 30-position uniform plate every timepoint is used equally often."""
    picks = [spread_timepoints(i, 10, 2) for i in range(30)]
    assert all(len(set(p)) == 2 for p in picks)
    counts = np.bincount([t for p in picks for t in p], minlength=10)
    # 30 positions x 2 frames / 10 timepoints -> exactly 6 uses each.
    assert counts.tolist() == [6] * 10


def test_early_subset_copies_the_right_frames_and_channels(tmp_path: Path) -> None:
    """Early mode drops unrequested channels and keeps frames 0..n-1 byte-exactly."""
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst_t01.zarr"
    _make_store(source, [10] * 3)

    summary = build_temporal_subset_zarr(source, dest, channels=["Phase3D", "Nuclei"], mode="early", n_timepoints=2)
    assert summary.n_positions == 3
    assert summary.n_pairs == 6
    assert set(summary.kept.values().__iter__().__next__()) == {0, 1}

    # The real verifier is the strongest assertion available: every frame equal.
    verify_temporal_subset(source, dest, summary)

    with open_ome_zarr(dest, mode="r") as dst:
        assert dst.channel_names == ["Phase3D", "Nuclei"]
        pos = dst["0/0/fov0001"]
        assert pos.data.shape == (2, 2, _Z, _Y, _X)
        # Channel 1 of the destination is source channel 2 (Nuclei), not Brightfield.
        assert pos.data[0, 1, 0, 0, 0] == _voxel_value(1, 0, 2)
        assert pos.data[1, 0, 0, 0, 0] == _voxel_value(1, 1, 0)


def test_spread_subset_reindexes_timepoint_statistics(tmp_path: Path) -> None:
    """Spread re-keys timepoint_statistics to the new indices, not the source ones.

    This is the trap the tool exists to avoid: copied verbatim, destination frame 0
    of a spread store would carry the parent's frame-0 stats while holding the
    parent's frame-``ordinal`` pixels.
    """
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst_tspread.zarr"
    _make_store(source, [10] * 4)

    summary = build_temporal_subset_zarr(source, dest, channels=["Phase3D", "Nuclei"], mode="spread", n_timepoints=2)
    assert summary.kept["0/0/fov0002"] == [2, 7]
    verify_temporal_subset(source, dest, summary)

    with open_ome_zarr(dest, mode="r") as dst:
        attrs = dict(dst["0/0/fov0002"].zattrs)
        tp_stats = attrs["normalization"]["Nuclei"]["timepoint_statistics"]
        assert sorted(tp_stats) == ["0", "1"]
        # New index 0 carries source t=2's stats, new index 1 carries source t=7's.
        assert tp_stats["0"]["mean"] == _voxel_value(2, 2, 2)
        assert tp_stats["1"]["mean"] == _voxel_value(2, 7, 2)
        # fov_statistics are deliberately the parent's, unchanged.
        assert attrs["normalization"]["Nuclei"]["fov_statistics"]["mean"] == 2.0
        # Per-timepoint temporal metadata is filtered to the kept frames.
        assert attrs["hpi_values"] == [9.0, 19.0]
        assert attrs["native_frame_indices"] == [2, 7]
        assert attrs["temporal_subset"]["source_timepoints"] == [2, 7]
        assert attrs["condition"] == "ZIKV"


def test_mixed_length_plate_keeps_every_position(tmp_path: Path) -> None:
    """A T=7/T=10 plate (the ER shape) yields an equal frame count per position."""
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst_tspread.zarr"
    _make_store(source, [7, 7, 10, 10])

    summary = build_temporal_subset_zarr(source, dest, channels=["Phase3D", "Nuclei"], mode="spread", n_timepoints=2)
    assert summary.n_positions == 4
    assert summary.n_pairs == 8
    # Step follows each position's own length: 3 for T=7, 5 for T=10.
    assert summary.kept["0/0/fov0001"] == [1, 4]
    assert summary.kept["0/0/fov0002"] == [2, 7]
    verify_temporal_subset(source, dest, summary)


def test_provenance_sidecar_records_the_selection(tmp_path: Path) -> None:
    """The colocated .provenance.json names the source and every kept index."""
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst_t01.zarr"
    _make_store(source, [10] * 2)
    summary = build_temporal_subset_zarr(source, dest, channels=["Phase3D", "Nuclei"], mode="early", n_timepoints=2)

    sidecar = json.loads((tmp_path / "dst_t01.provenance.json").read_text())
    assert sidecar["kind"] == "temporal_subset"
    assert sidecar["mode"] == "early"
    assert sidecar["source_store"] == str(source)
    assert sidecar["n_pairs"] == summary.n_pairs
    assert sidecar["positions"]["0/0/fov0000"] == [0, 1]


def test_refuses_to_overwrite_an_existing_store(tmp_path: Path) -> None:
    """A training store is never silently replaced."""
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst.zarr"
    _make_store(source, [10])
    build_temporal_subset_zarr(source, dest, channels=["Phase3D"], mode="early", n_timepoints=2)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_temporal_subset_zarr(source, dest, channels=["Phase3D"], mode="early", n_timepoints=2)


def test_missing_channel_raises(tmp_path: Path) -> None:
    """Asking for a channel the source lacks fails loudly rather than dropping it."""
    source = tmp_path / "src.zarr"
    _make_store(source, [10])
    with pytest.raises(ValueError, match="Structure"):
        build_temporal_subset_zarr(
            source, tmp_path / "dst.zarr", channels=["Phase3D", "Structure"], mode="early", n_timepoints=2
        )


def test_cli_builds_and_verifies(tmp_path: Path) -> None:
    """The CLI path runs the same build + verify the campaign will invoke."""
    source = tmp_path / "src.zarr"
    dest = tmp_path / "dst_t01.zarr"
    _make_store(source, [10] * 2)
    rc = main(
        [
            "--source",
            str(source),
            "--dest",
            str(dest),
            "--channels",
            "Phase3D",
            "Nuclei",
            "--mode",
            "early",
            "--n-timepoints",
            "2",
        ]
    )
    assert rc == 0
    with open_ome_zarr(dest, mode="r") as dst:
        assert [name for name, _ in dst.positions()] == ["0/0/fov0000", "0/0/fov0001"]
        assert dst["0/0/fov0000"].data.shape[0] == 2
