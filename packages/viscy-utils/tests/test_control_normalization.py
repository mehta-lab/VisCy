import anndata as ad
import numpy as np
import pandas as pd
import pytest

from viscy_utils.evaluation.control_normalization import (
    ControlReference,
    control_reference_stats,
    control_reference_stats_multi,
)


def _make_adata(seed: int = 0) -> ad.AnnData:
    """Two plates, each with control + perturbed cells drifting over HPI.

    Plate A control centroid drifts linearly with HPI; plate B has a constant
    offset. Perturbed cells share the plate/time drift plus an extra shift.
    """
    rng = np.random.default_rng(seed)
    d = 8
    rows = []
    x_blocks = []
    for plate, drift in [("A", 1.0), ("B", 0.0)]:
        offset = 5.0 if plate == "B" else 0.0
        for hpi in np.arange(0.0, 12.0, 0.5):
            center = drift * hpi + offset
            for pert, shift in [("uninfected", 0.0), ("ZIKV", 3.0)]:
                n = 40
                block = rng.normal(loc=center + shift, scale=1.0, size=(n, d))
                x_blocks.append(block)
                rows.extend(
                    {"experiment": plate, "perturbation": pert, "hours_post_perturbation": hpi} for _ in range(n)
                )
    obs = pd.DataFrame(rows)
    return ad.AnnData(X=np.vstack(x_blocks).astype(np.float32), obs=obs)


def test_reference_recovers_control_center():
    adata = _make_adata()
    refs = control_reference_stats(adata, bin_hours=2.0, min_cells=10)
    assert set(refs) == {"A", "B"}
    ref_a = refs["A"]
    # Plate A drifts as 1.0 * hpi; bin 0 covers hpi in [0, 2), centered ~1.0.
    bin0 = ref_a.bin_indices.tolist().index(0)
    assert ref_a.median[bin0].mean() == pytest.approx(1.0, abs=0.5)
    # Scaled MAD of a unit-scale normal ~1.0; positive and finite everywhere.
    assert np.all(ref_a.mad > 0)
    assert float(np.median(ref_a.mad)) == pytest.approx(1.0, abs=0.2)
    assert ref_a.median.shape == (len(ref_a.bin_indices), adata.n_vars)


def test_apply_centers_controls():
    adata = _make_adata()
    refs = control_reference_stats(adata, bin_hours=2.0, min_cells=10)
    ctrl = adata[(adata.obs["experiment"] == "A") & (adata.obs["perturbation"] == "uninfected")]
    z = refs["A"].apply(np.asarray(ctrl.X), ctrl.obs["hours_post_perturbation"].to_numpy())
    # Control cells expressed against their own reference sit near 0.
    assert abs(float(np.median(z))) < 0.3
    # Perturbed cells (shifted +3) land clearly above control after normalization.
    pert = adata[(adata.obs["experiment"] == "A") & (adata.obs["perturbation"] == "ZIKV")]
    zp = refs["A"].apply(np.asarray(pert.X), pert.obs["hours_post_perturbation"].to_numpy())
    assert float(np.median(zp)) > float(np.median(z))


def test_nearest_bin_fallback():
    ref = ControlReference(
        bin_hours=2.0,
        bin_indices=np.array([0, 3]),
        median=np.array([[0.0], [10.0]]),
        mad=np.array([[1.0], [1.0]]),
    )
    # hpi=5 -> bin 2, unoccupied; nearest occupied is bin 3 (dist 1) over bin 0 (dist 2).
    z = ref.apply(np.array([[10.0]]), np.array([5.0]))
    assert z[0, 0] == pytest.approx(0.0)
    # hpi=1 -> bin 0, occupied.
    z0 = ref.apply(np.array([[0.0]]), np.array([1.0]))
    assert z0[0, 0] == pytest.approx(0.0)


def test_zero_mad_is_floored():
    rng = np.random.default_rng(1)
    n = 60
    obs = pd.DataFrame(
        {
            "experiment": ["P"] * n,
            "perturbation": ["uninfected"] * n,
            "hours_post_perturbation": [1.0] * n,
        }
    )
    x = rng.normal(size=(n, 3)).astype(np.float32)
    x[:, 0] = 7.0  # constant dimension -> MAD 0 -> must be floored to 1.0
    refs = control_reference_stats(ad.AnnData(X=x, obs=obs), bin_hours=2.0, min_cells=10)
    assert refs["P"].mad[0, 0] == 1.0
    z = refs["P"].apply(np.array([[7.0, 0.0, 0.0]]), np.array([1.0]))
    assert np.isfinite(z).all()


def test_multi_bin_widths():
    adata = _make_adata()
    multi = control_reference_stats_multi(adata, bin_hours=(1.0, 2.0), min_cells=10)
    assert set(multi) == {1.0, 2.0}
    # Finer bins yield at least as many occupied bins as coarser ones.
    assert len(multi[1.0]["A"].bin_indices) >= len(multi[2.0]["A"].bin_indices)


def test_missing_control_raises():
    adata = _make_adata()
    adata.obs["perturbation"] = "ZIKV"  # no controls anywhere
    with pytest.raises(ValueError, match="No control cells"):
        control_reference_stats(adata, bin_hours=2.0)


def test_plate_without_control_raises():
    adata = _make_adata()
    # Strip controls from plate B only: it now has perturbed cells but no reference.
    mask = (adata.obs["experiment"] == "B") & (adata.obs["perturbation"] == "uninfected")
    adata = adata[~mask.to_numpy()].copy()
    with pytest.raises(ValueError, match="Plates with no control"):
        control_reference_stats(adata, bin_hours=2.0, min_cells=10)


def test_apply_floors_zero_mad():
    # A hand-built reference with a zero-MAD dimension must not divide by zero.
    ref = ControlReference(
        bin_hours=2.0,
        bin_indices=np.array([0]),
        median=np.array([[0.0, 0.0]]),
        mad=np.array([[0.0, 2.0]]),
    )
    z = ref.apply(np.array([[5.0, 4.0]]), np.array([1.0]))
    assert np.isfinite(z).all()
    assert z[0, 0] == pytest.approx(5.0)  # floored MAD 1.0
    assert z[0, 1] == pytest.approx(2.0)  # 4 / 2
