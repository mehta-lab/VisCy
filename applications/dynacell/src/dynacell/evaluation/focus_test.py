"""Unit tests for the nucleus-area focus anchor and slab MIP (evaluation/focus.py)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from dynacell.evaluation.focus import (
    focus_slab_from_plane,
    nucleus_area_plane,
    resolve_focus_instance_planes,
    resolve_nucleus_area_planes,
    slab_mip,
)


def _blob_stack(z_widest: int, z_total: int = 40, size: int = 64) -> np.ndarray:
    """(Z, Y, X) with a centered disk whose radius peaks at ``z_widest`` (nuclear equator)."""
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - size / 2) ** 2 + (xx - size / 2) ** 2
    vol = np.zeros((z_total, size, size), dtype=np.float32)
    for z in range(z_total):
        radius = max(0, 12 - abs(z - z_widest))
        vol[z] = (r2 < radius**2).astype(np.float32)
    return vol


def test_nucleus_area_plane_picks_widest_plane() -> None:
    """The plane of maximum nuclear foreground area is the widest cross-section."""
    assert nucleus_area_plane(_blob_stack(20)) == 20
    assert nucleus_area_plane(_blob_stack(15)) == 15


def test_nucleus_area_plane_clips_to_guard_band() -> None:
    """An edge-focused stack is clipped into the guard band, never to the out-of-focus cap."""
    z_total = 40
    lo = round(0.15 * z_total)  # 6
    hi = round(0.85 * z_total)  # 34
    assert nucleus_area_plane(_blob_stack(1, z_total)) == lo  # bottom cap -> guard floor
    assert nucleus_area_plane(_blob_stack(39, z_total)) == hi - 1  # top cap -> guard ceiling


def test_nucleus_area_plane_constant_volume_falls_back_to_center() -> None:
    """A constant/empty volume has no threshold and no nuclei -> stack center."""
    assert nucleus_area_plane(np.zeros((40, 32, 32), dtype=np.float32)) == 20
    assert nucleus_area_plane(np.full((30, 16, 16), 7.0, dtype=np.float32)) == 15


def test_nucleus_area_plane_single_plane() -> None:
    """A single-plane volume trivially returns index 0."""
    assert nucleus_area_plane(np.ones((1, 16, 16), dtype=np.float32)) == 0


def test_resolve_nucleus_area_planes_per_timepoint() -> None:
    """Per-timepoint planes track each timepoint's widest nuclear plane."""
    tzyx = np.stack([_blob_stack(18), _blob_stack(24)])  # (T=2, Z, Y, X)
    assert resolve_nucleus_area_planes(tzyx, t_count=2) == [18, 24]


def test_slab_mip_halfwidth_zero_is_single_plane() -> None:
    """halfwidth=0 reproduces the single-plane index byte-for-byte."""
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(3, 40, 8, 8)).astype(np.float32)  # (T, Z, Y, X)
    z_idx = [10, 20, 30]
    single = np.stack([vol[t, z_idx[t]] for t in range(3)])
    np.testing.assert_array_equal(slab_mip(vol, z_idx, 0), single)


def test_slab_mip_halfwidth_one_max_projects_slab() -> None:
    """halfwidth=1 MIPs the 3-plane slab centered on each plane, clipped at the edges."""
    vol = np.zeros((1, 10, 4, 4), dtype=np.float32)
    vol[0, 4, 0, 0] = 5.0  # only in the neighbor plane, not the center plane z=5
    out = slab_mip(vol, [5], 1)  # slab z in [4, 6]
    assert out.shape == (1, 4, 4)
    assert out[0, 0, 0] == 5.0  # neighbor plane's peak is captured by the MIP
    # edge plane: slab clipped to [0, 1]
    assert slab_mip(vol, [0], 1).shape == (1, 4, 4)


def test_focus_slab_from_plane_edges() -> None:
    """Slab is centered and clipped to [0, z_total); halfwidth 0 is a single plane."""
    assert focus_slab_from_plane(20, 40, 0) == slice(20, 21)
    assert focus_slab_from_plane(20, 40, 2) == slice(18, 23)
    assert focus_slab_from_plane(0, 40, 2) == slice(0, 3)
    assert focus_slab_from_plane(39, 40, 2) == slice(37, 40)


def test_resolve_focus_instance_planes_nucleus_area() -> None:
    """The nucleus_area anchor resolves planes from the supplied nucleus volume."""
    config = OmegaConf.create({"segmentation": {"focus_anchor": "nucleus_area"}})
    nuc = np.stack([_blob_stack(22)])  # (T=1, Z, Y, X)
    assert resolve_focus_instance_planes(config, t_count=1, pos_gt=None, pos_name="p", nucleus_vol=nuc) == [22]


def test_resolve_focus_instance_planes_nucleus_area_requires_volume() -> None:
    """nucleus_area with no nucleus volume is a hard error, not a silent fallback."""
    config = OmegaConf.create({"segmentation": {"focus_anchor": "nucleus_area"}})
    with pytest.raises(ValueError, match="nucleus_area"):
        resolve_focus_instance_planes(config, t_count=1, pos_gt=None, pos_name="p", nucleus_vol=None)


def test_resolve_focus_instance_planes_unknown_anchor() -> None:
    """An unrecognized anchor raises rather than silently defaulting."""
    config = OmegaConf.create({"segmentation": {"focus_anchor": "bogus"}})
    with pytest.raises(ValueError, match="focus_anchor"):
        resolve_focus_instance_planes(config, t_count=1, pos_gt=None, pos_name="p", nucleus_vol=np.zeros((1, 4, 4, 4)))
