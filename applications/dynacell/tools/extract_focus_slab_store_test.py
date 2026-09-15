r"""Integration tests for ``extract_focus_slab_store.py``.

Each test builds a tiny synthetic 3D OME-Zarr in ``tmp_path`` whose ``Phase3D``
channel has a *known* in-focus plane (a broadband texture that is progressively
Gaussian-blurred away from a chosen z), so the real
:func:`dynacell.evaluation.focus.estimate_focus_plane` returns that plane. The
extractor is then run for real (no mocks) and the written store is reopened and
asserted against.

Run (worktree, cpdino-eval env, PYTHONPATH shadowing)::

    PYTHONPATH="$WT/packages/viscy-models/src:$WT/packages/viscy-utils/src:\
$WT/packages/viscy-transforms/src:$WT/packages/viscy-data/src:$WT/applications/dynacell/src" \\
        "$V/bin/python" -m pytest applications/dynacell/tools/extract_focus_slab_store_test.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from extract_focus_slab_store import (  # noqa: E402
    clamp_shift_slab,
    extract_focus_slab_store,
    main,
)
from iohub.ngff import TransformationMeta, open_ome_zarr
from scipy.ndimage import gaussian_filter

_CHANNELS = ["Phase3D", "Structure"]
_PIXEL_SIZE = 0.1494
_Z_SCALE = 0.174


def _focus_stack(z_total: int, y: int, x: int, z_focus: int, seed: int) -> np.ndarray:
    """Return a ``(Z, Y, X)`` volume sharpest (max midband power) at ``z_focus``.

    A single broadband texture is progressively Gaussian-blurred with sigma growing
    with ``|z - z_focus|``, so the sharpest plane (most transverse-band power) is
    ``z_focus`` — exactly what the waveorder focus estimator picks.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((y, x)).astype(np.float32)
    vol = np.empty((z_total, y, x), dtype=np.float32)
    for z in range(z_total):
        sigma = 0.9 * abs(z - z_focus)
        vol[z] = base if sigma == 0 else gaussian_filter(base, sigma=sigma)
    return vol


def _make_synthetic_store(
    path: Path,
    *,
    z_focus_per_pos: dict[str, int],
    t: int = 2,
    z_total: int = 12,
    y: int = 64,
    x: int = 64,
    with_normalization: bool = True,
) -> None:
    """Write a tiny 3D HCS OME-Zarr with a known per-position Phase3D focus plane.

    ``z_focus_per_pos`` maps an HCS ``row/col/fov`` position name to its injected
    focus z (same across timepoints). The ``Structure`` channel carries a distinct,
    z-varying pattern so channel copying is verifiable.
    """
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=_CHANNELS) as plate:
        if with_normalization:
            plate.zattrs.update({"normalization": {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}}})
        for seed, (name, z_focus) in enumerate(z_focus_per_pos.items()):
            row, col, fov = name.split("/")
            phase = np.stack([_focus_stack(z_total, y, x, z_focus, seed=seed + 100 * s) for s in range(t)])
            # Structure: a per-z ramp so each plane is uniquely identifiable after the copy.
            structure = np.broadcast_to(
                np.arange(z_total, dtype=np.float32)[None, :, None, None], (t, z_total, y, x)
            ).copy()
            data = np.stack([phase, structure], axis=1)  # (T, C=2, Z, Y, X)
            pos = plate.create_position(row, col, fov)
            pos.create_image(
                "0",
                data,
                chunks=(1, 1, z_total, y, x),
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, _Z_SCALE, _PIXEL_SIZE, _PIXEL_SIZE])],
            )
            if with_normalization:
                pos.zattrs.update(
                    {
                        "normalization": {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}},
                        "condition": "mock",
                    }
                )


def test_clamp_shift_slab_centered_and_edges() -> None:
    """clamp_shift_slab yields a full 2*hw+1 window, shifted inward at the caps."""
    # Centered: window straddles z_focus.
    assert clamp_shift_slab(6, 12, 2) == slice(4, 9)
    # Low edge: shift the whole window to start at 0 (not a clipped, shorter slab).
    assert clamp_shift_slab(0, 12, 2) == slice(0, 5)
    assert clamp_shift_slab(1, 12, 2) == slice(0, 5)
    # High edge: shift so the window ends at z_total.
    assert clamp_shift_slab(11, 12, 2) == slice(7, 12)
    # halfwidth=0 -> single plane at the focus.
    assert clamp_shift_slab(5, 12, 0) == slice(5, 6)
    # Every case spans exactly 2*hw+1 planes.
    for zf in range(12):
        sl = clamp_shift_slab(zf, 12, 2)
        assert sl.stop - sl.start == 5


def test_clamp_shift_slab_too_thin_raises() -> None:
    """A stack shorter than the slab width cannot yield a full slab."""
    with pytest.raises(ValueError):
        clamp_shift_slab(1, 3, 2)  # needs 5 planes, only 3
    with pytest.raises(ValueError):
        clamp_shift_slab(1, 10, -1)  # negative halfwidth


def test_output_shape_channels_and_centering(tmp_path: Path) -> None:
    """Output is (T, C, 2*hw+1, Y, X), channels preserved, slab centered on the focus plane."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6}, t=2, z_total=12, y=64, x=64)

    summary = extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2)

    assert summary.n_positions == 1
    assert summary.n_planes == 5
    assert summary.pixel_size == pytest.approx(_PIXEL_SIZE)
    assert summary.focus_planes["0/0/fov0000"] == [6, 6]
    assert summary.slab_starts["0/0/fov0000"] == [4, 4]

    with open_ome_zarr(src, mode="r") as src_plate, open_ome_zarr(out, mode="r") as out_plate:
        assert out_plate.channel_names == _CHANNELS
        _, src_pos = next(src_plate.positions())
        _, out_pos = next(out_plate.positions())
        assert out_pos.data.shape == (2, 2, 5, 64, 64)
        assert not np.isnan(np.asarray(out_pos.data)).any()
        # Center plane (index 2 of 5, hw=2) is the source focus plane z=6, all channels.
        np.testing.assert_array_equal(np.asarray(out_pos.data[:, :, 2]), np.asarray(src_pos.data[:, :, 6]))
        # Structure ramp confirms the copied z-range is [4, 9).
        np.testing.assert_array_equal(np.asarray(out_pos.data[0, 1, :, 0, 0]), np.arange(4, 9, dtype=np.float32))


def test_clamp_shift_edge_position_still_full_depth(tmp_path: Path) -> None:
    """A focus plane at the stack edge still yields a full 2*hw+1 slab (window shifted)."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    # Low edge (focus at z=0) and high edge (focus at z=11) in one store.
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 0, "0/0/fov0001": 11}, t=2, z_total=12)

    summary = extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2)

    assert summary.focus_planes["0/0/fov0000"] == [0, 0]
    assert summary.slab_starts["0/0/fov0000"] == [0, 0]  # shifted inward, not -2
    assert summary.focus_planes["0/0/fov0001"] == [11, 11]
    assert summary.slab_starts["0/0/fov0001"] == [7, 7]  # shifted inward, ends at z_total

    with open_ome_zarr(out, mode="r") as out_plate:
        for _, pos in out_plate.positions():
            assert pos.data.shape[2] == 5  # uniform full depth despite edge focus
            assert not np.isnan(np.asarray(pos.data)).any()


def test_limit_positions(tmp_path: Path) -> None:
    """--limit-positions caps the number of extracted positions."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6, "0/0/fov0001": 5, "0/0/fov0002": 4}, t=1, z_total=12)

    summary = extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2, limit_positions=1)

    assert summary.n_positions == 1
    with open_ome_zarr(out, mode="r") as out_plate:
        assert len([n for n, _ in out_plate.positions()]) == 1


def test_zattrs_and_omero_preserved(tmp_path: Path) -> None:
    """Custom zattrs (normalization) and omero survive; provenance zattr is written."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6}, t=1, z_total=12, with_normalization=True)

    extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2)

    with open_ome_zarr(out, mode="r") as out_plate:
        assert dict(out_plate.zattrs).get("normalization") == {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}}
        prov = dict(out_plate.zattrs)["focus_slab_extraction"]
        assert prov["phase_channel"] == "Phase3D"
        assert prov["halfwidth"] == 2
        assert prov["n_planes"] == 5
        _, out_pos = next(out_plate.positions())
        pos_attrs = dict(out_pos.zattrs)
        assert pos_attrs["condition"] == "mock"
        assert pos_attrs["normalization"] == {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}}
        assert [c.label for c in out_pos.metadata.omero.channels] == _CHANNELS


def test_overwrite_guard(tmp_path: Path) -> None:
    """A second run without overwrite raises; with overwrite it succeeds."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6}, t=1, z_total=12)

    extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2)
    with pytest.raises(FileExistsError):
        extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2)
    # Overwrite succeeds and rewrites the store.
    summary = extract_focus_slab_store(src, out, phase_channel="Phase3D", halfwidth=2, overwrite=True)
    assert summary.n_positions == 1


def test_missing_phase_channel_raises(tmp_path: Path) -> None:
    """A phase channel absent from the store is a hard error (not a silent skip)."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6}, t=1, z_total=12)
    with pytest.raises(ValueError, match="phase channel"):
        extract_focus_slab_store(src, out, phase_channel="NotAChannel", halfwidth=2)


def test_cli_main_smoke(tmp_path: Path) -> None:
    """The argparse CLI runs end to end and writes a loadable focus store."""
    src = tmp_path / "src.zarr"
    out = tmp_path / "src_focus.zarr"
    _make_synthetic_store(src, z_focus_per_pos={"0/0/fov0000": 6, "0/0/fov0001": 5}, t=1, z_total=12)

    rc = main(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--phase-channel",
            "Phase3D",
            "--halfwidth",
            "2",
            "--limit-positions",
            "1",
        ]
    )
    assert rc == 0
    with open_ome_zarr(out, mode="r") as out_plate:
        names = [n for n, _ in out_plate.positions()]
        assert names == ["0/0/fov0000"]
        _, pos = next(out_plate.positions())
        assert pos.data.shape[2] == 5
