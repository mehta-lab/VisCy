"""The torch focus-plane port in SegConsistencyLoss picks the plane and slab the eval segments."""

import numpy as np
import torch

from dynacell.evaluation.focus import nucleus_area_plane, slab_mip
from viscy_utils.losses.seg_consistency import _slab_mip, nucleus_area_planes


def _blob_volume(rng: np.random.Generator, z: int) -> np.ndarray:
    """Disks peaked at a random plane, on noise, at a random scale and offset."""
    yy, xx = np.mgrid[:48, :48]
    zz = np.arange(z)[:, None, None]
    vol = np.zeros((z, 48, 48), np.float32)
    for _ in range(int(rng.integers(1, 6))):
        cy, cx, r = rng.uniform(6, 42), rng.uniform(6, 42), rng.uniform(3, 9)
        prof = np.exp(-(((zz - rng.uniform(0, z)) / rng.uniform(0.7, 6.0)) ** 2))
        vol += prof * ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2)
    vol += rng.normal(0, rng.uniform(0.01, 0.2), vol.shape)
    return (vol * rng.uniform(0.1, 500) + rng.uniform(-10, 10)).astype(np.float32)


def test_focus_plane_and_slab_match_the_eval():
    rng = np.random.default_rng(0)
    vols = [_blob_volume(rng, int(z)) for z in rng.integers(1, 41, size=48)]
    vols += [np.full((7, 48, 48), 3.0, np.float32), _blob_volume(rng, 1)]  # constant -> center; Z = 1
    planes_seen = set()
    for vol in vols:
        expected = nucleus_area_plane(vol)
        got = int(nucleus_area_planes(torch.from_numpy(vol)[None])[0])
        assert got == expected, (vol.shape, got, expected)
        planes_seen.add((vol.shape[0], got))
        for halfwidth in (0, 1, 2):
            ours = _slab_mip(torch.from_numpy(vol)[None], torch.tensor([got]), halfwidth)[0].numpy()
            np.testing.assert_array_equal(ours, slab_mip(vol[None], [expected], halfwidth)[0])
    # The volumes exercise more than the center fallback.
    assert sum(p != z // 2 for z, p in planes_seen) >= 10
