"""Foreground-limited pixel metrics (``FG_*``) against the whole-image ``SI_*`` family.

Real cubic throughout: the invariant under test is that a unit weight map reproduces
the whole-image columns cubic computes, so a stub would test nothing.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("cubic.metrics")

from cubic.metrics import ssim as cubic_ssim  # noqa: E402

from dynacell.evaluation.metrics import (  # noqa: E402
    FOREGROUND_COLUMNS,
    compute_pixel_metrics,
    foreground_pixel_metrics,
    foreground_weight,
)

_GOLDEN = Path(__file__).parent / "data" / "pixel_metrics_golden.npz"
_PAIRS = (("FG_PCC", "PCC"), ("FG_SI_SSIM", "SI_SSIM"), ("FG_SI_NRMSE", "SI_NRMSE"), ("FG_SI_PSNR", "SI_PSNR"))


def _blobs(shape=(16, 64, 64), seed=0) -> tuple[np.ndarray, np.ndarray]:
    """A dim noisy background with bright ellipsoidal blobs; returns (image, blob mask)."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    mask = np.zeros(shape, dtype=bool)
    for cz, cy, cx in ((8, 16, 16), (8, 44, 20), (7, 30, 46)):
        mask |= ((zz - cz) / 5.0) ** 2 + ((yy - cy) / 9.0) ** 2 + ((xx - cx) / 9.0) ** 2 <= 1.0
    texture = rng.normal(0.0, 0.3, shape)
    image = np.where(mask, 3.0 + texture, 0.2 + 0.05 * rng.normal(size=shape))
    return image.astype(np.float32), mask


@pytest.mark.parametrize(("dtype", "rel"), [(np.float32, 1e-6), (np.float64, 1e-12)])
def test_unit_weight_reproduces_whole_image_metrics(dtype, rel):
    """``weight == 1`` scores PCC/SI_SSIM/SI_NRMSE/SI_PSNR as the whole-image columns, on real data.

    float64 pins the formulas (window, crop, sample-covariance factor, SI fit) to
    machine precision; float32, the production dtype, differs only by rounding.
    """
    if not _GOLDEN.exists():
        pytest.skip("golden fixture not generated")
    g = np.load(_GOLDEN)
    pred, target = g["pred"].astype(dtype), g["target"].astype(dtype)
    whole = compute_pixel_metrics(pred, target, spacing=list(g["_spacing"]), use_gpu=False)
    fg = foreground_pixel_metrics(pred, target, np.ones_like(target))
    for fg_key, key in _PAIRS:
        assert fg[fg_key] == pytest.approx(float(whole[key]), rel=rel), (fg_key, fg[fg_key], whole[key])
    assert fg["FG_frac"] == 1.0


def _reference_foreground_metrics(pred, target, weight) -> dict[str, float]:
    """Brute-force ``FG_*`` from their definitions, sharing no code with ``metrics.py``.

    Weighted moments over the whole volume give the Pearson correlation and the
    weighted least-squares affine fit. SSIM is evaluated voxel by voxel over
    skimage's border crop: an explicit 11-voxel Gaussian window (sigma 1.5,
    truncate 3.5) times the weight gives local moments normalized by the local
    weight mass, and the SSIM map is averaged with the weight. The crop keeps every
    window inside the volume, so no boundary mode is involved.
    """
    p, t, w = (np.asarray(a, dtype=np.float64) for a in (pred, target, weight))

    def mean(x):
        return float((w * x).sum() / w.sum())

    t_centred, p_centred = t - mean(t), p - mean(p)
    t_std = math.sqrt(mean(t_centred**2))
    pcc = mean(t_centred * p_centred) / math.sqrt(mean(t_centred**2) * mean(p_centred**2))
    t_norm = t_centred / t_std
    p_fit = mean(t_norm * p_centred) / mean(p_centred**2) * p_centred
    data_range = float(np.ptp(t[w > 0.5])) / t_std
    mse = mean((t_norm - p_fit) ** 2)

    radius = 5
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / 1.5) ** 2)
    kernel = taps[:, None, None] * taps[None, :, None] * taps[None, None, :]
    cov_norm = kernel.size / (kernel.size - 1.0)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = den = 0.0
    for corner in np.ndindex(*(n - 2 * radius for n in t.shape)):
        window = tuple(slice(c, c + 2 * radius + 1) for c in corner)
        centre = tuple(c + radius for c in corner)
        if w[centre] == 0:
            continue
        kw = kernel * w[window]
        a, b = t_norm[window], p_fit[window]
        mu_a, mu_b = (kw * a).sum() / kw.sum(), (kw * b).sum() / kw.sum()
        var_a = cov_norm * ((kw * a * a).sum() / kw.sum() - mu_a**2)
        var_b = cov_norm * ((kw * b * b).sum() / kw.sum() - mu_b**2)
        cov_ab = cov_norm * ((kw * a * b).sum() / kw.sum() - mu_a * mu_b)
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
        num += w[centre] * ssim
        den += w[centre]
    return {
        "FG_PCC": pcc,
        "FG_SI_SSIM": num / den,
        "FG_SI_NRMSE": math.sqrt(mse) / data_range,
        "FG_SI_PSNR": 10 * math.log10(data_range**2 / mse),
    }


def test_soft_weight_matches_brute_force_reference():
    """A non-binary weight scores every FG_* column as the brute-force weighted definitions.

    The unit-weight test cannot see how the weight enters: a weight of 1 everywhere
    makes the local weight-mass normalization, the weighted SSIM-map average and the
    weighted Pearson indistinguishable from their unweighted forms. Here the weight
    spans [0, 1] (with exact zeros and ones) and the prediction is noisier where the
    weight is low, so each of those choices moves the values.
    """
    rng = np.random.default_rng(7)
    shape = (13, 16, 18)
    target = rng.normal(0.0, 1.0, shape) + 2.0
    weight = np.clip(rng.uniform(-0.6, 1.4, shape), 0.0, 1.0)
    pred = 0.5 * target + rng.normal(0.0, 1.0, shape) * (1.5 - weight) + 3.0
    got = foreground_pixel_metrics(pred, target, weight)
    expected = _reference_foreground_metrics(pred, target, weight)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9), (key, got[key], value)


def test_foreground_off_leaves_pixel_metrics_unchanged():
    """No ``foreground`` adds no column; with it, the base columns keep their exact values."""
    image, _ = _blobs()
    pred = image + np.random.default_rng(1).normal(0.0, 0.5, image.shape).astype(np.float32)
    spacing = [0.29, 0.108, 0.108]
    off = compute_pixel_metrics(pred, image, spacing=spacing, use_gpu=False)
    on = compute_pixel_metrics(
        pred,
        image,
        spacing=spacing,
        use_gpu=False,
        foreground={"source": "smooth_otsu", "smooth_sigma_um": 0.5, "feather_sigma_um": 0.3},
    )
    assert not any(k.startswith("FG_") for k in off)
    assert list(on) == [*off, *FOREGROUND_COLUMNS]
    assert all(on[k] == off[k] for k in off)


def test_identical_foreground_with_noisy_background_scores_perfect():
    """A prediction exact inside the GT foreground scores ~1 there while whole-image metrics drop."""
    image, _ = _blobs()
    weight = foreground_weight(image, [1.0, 1.0, 1.0], source="otsu", feather_sigma_um=0.0)
    noise = np.random.default_rng(2).normal(0.0, 2.0, image.shape).astype(np.float32)
    pred = np.where(weight > 0, image, image + noise)
    whole = compute_pixel_metrics(pred, image, spacing=[1.0, 1.0, 1.0], use_gpu=False)
    fg = foreground_pixel_metrics(pred, image, weight)
    assert fg["FG_PCC"] == pytest.approx(1.0, abs=1e-6)
    assert fg["FG_SI_SSIM"] == pytest.approx(1.0, abs=1e-5)
    assert fg["FG_SI_PSNR"] > 100.0  # float32 rounding leaves a ~1e-8 residual, not exact zero
    assert whole["PCC"] < 0.9 and whole["SI_SSIM"] < 0.5


def test_foreground_error_drops_foreground_metrics_more():
    """Error confined to the foreground costs the FG columns more than the whole-image ones."""
    image, mask = _blobs()
    noise = np.random.default_rng(3).normal(0.0, 0.6, image.shape).astype(np.float32)
    pred = np.where(mask, image + noise, image)
    whole = compute_pixel_metrics(pred, image, spacing=[1.0, 1.0, 1.0], use_gpu=False)
    weight = foreground_weight(image, [1.0, 1.0, 1.0], source="otsu", feather_sigma_um=0.0)
    fg = foreground_pixel_metrics(pred, image, weight)
    assert fg["FG_PCC"] < whole["PCC"]
    assert fg["FG_SI_SSIM"] < whole["SI_SSIM"]
    assert fg["FG_SI_PSNR"] < whole["SI_PSNR"]


def test_thin_foreground_keeps_a_defined_ssim():
    """One-voxel tubes erode to nothing under cubic's masked SSIM; ``FG_SI_SSIM`` stays defined."""
    rng = np.random.default_rng(4)
    image = rng.normal(0.0, 0.05, (16, 64, 64)).astype(np.float32)
    tubes = np.zeros(image.shape, dtype=bool)
    tubes[:, 10::12, :] = True
    image[tubes] += 2.0
    pred = image + rng.normal(0.0, 0.2, image.shape).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of an empty slice
        eroded = cubic_ssim(image, pred, mask=tubes, gaussian_weights=True, data_range=float(np.ptp(image)))
    assert math.isnan(eroded)
    fg = foreground_pixel_metrics(pred, image, tubes.astype(np.float32))
    assert np.isfinite(fg["FG_SI_SSIM"]) and 0.0 < fg["FG_SI_SSIM"] < 1.0


def test_empty_foreground_is_nan_with_zero_fraction():
    """A constant target has no foreground: every FG_* is NaN and FG_frac is 0."""
    target = np.full((16, 32, 32), 5.0, dtype=np.float32)
    pred = np.random.default_rng(5).normal(size=target.shape).astype(np.float32)
    weight = foreground_weight(target, [1.0, 1.0, 1.0], smooth_sigma_um=1.0, feather_sigma_um=1.0)
    assert not weight.any()
    fg = foreground_pixel_metrics(pred, target, weight)
    assert fg["FG_frac"] == 0.0
    assert all(math.isnan(fg[k]) for k in FOREGROUND_COLUMNS[:-1])


def test_constant_prediction_scores_zero_pcc_instead_of_vanishing():
    """A prediction collapsed to a constant scores FG_PCC 0 and finite SI columns (gain 0)."""
    image, _ = _blobs()
    weight = foreground_weight(image, [1.0, 1.0, 1.0], source="otsu", feather_sigma_um=1.0)
    fg = foreground_pixel_metrics(np.full_like(image, 0.7), image, weight)
    assert fg["FG_PCC"] == 0.0
    assert all(np.isfinite(fg[k]) for k in ("FG_SI_SSIM", "FG_SI_NRMSE", "FG_SI_PSNR"))
    # The best affine fit of a constant is the target's weighted mean, i.e. 0 in
    # standardized units, so the residual RMS is the standardized target's: 1.
    hard = weight > 0.5
    data_range = float(np.ptp(image[hard])) / float(np.sqrt(np.cov(image.ravel(), aweights=weight.ravel(), ddof=0)))
    assert fg["FG_SI_NRMSE"] == pytest.approx(1.0 / data_range, rel=1e-5)


def test_foreground_weight_recipes():
    """Hard mask without feather, soft weights in [0, 1] with it; ``otsu`` ignores the smoothing sigma."""
    image, mask = _blobs()
    hard = foreground_weight(image, [0.5, 0.25, 0.25], source="smooth_otsu", smooth_sigma_um=0.0)
    assert set(np.unique(hard)) == {0.0, 1.0}
    assert (hard.astype(bool) == mask).mean() > 0.99
    soft = foreground_weight(image, [0.5, 0.25, 0.25], smooth_sigma_um=0.5, feather_sigma_um=0.5)
    assert soft.dtype == np.float32 and soft.min() >= 0.0 and soft.max() <= 1.0
    assert ((soft > 0) & (soft < 1)).any()
    raw = foreground_weight(image, [0.5, 0.25, 0.25], source="otsu", smooth_sigma_um=3.0)
    np.testing.assert_array_equal(raw, foreground_weight(image, [0.5, 0.25, 0.25], source="otsu"))
    with pytest.raises(ValueError, match="source"):
        foreground_weight(image, [1.0, 1.0, 1.0], source="cell")
    with pytest.raises(ValueError, match=">= 0"):
        foreground_weight(image, [1.0, 1.0, 1.0], feather_sigma_um=-1.0)


@pytest.mark.parametrize("source", ["otsu", "smooth_otsu"])
def test_zero_padded_planes_do_not_become_the_background_class(source):
    """All-zero leading planes (A549 SEC61B/TOMM20 GT padding) stay out of the Otsu split.

    Without the exclusion Otsu separates the padding from everything else and
    calls ~95% of the volume foreground.
    """
    image, mask = _blobs()
    image += 100.0  # camera offset: real GT sits far above the zero padding
    padded = image.copy()
    padded[:2] = 0.0
    weight = foreground_weight(padded, [1.0, 1.0, 1.0], source=source, smooth_sigma_um=0.5)
    assert not weight[:2].any()
    assert weight.mean() < 2 * mask.mean()
    reference = foreground_weight(image, [1.0, 1.0, 1.0], source=source, smooth_sigma_um=0.5)
    assert (weight[4:] == reference[4:]).mean() > 0.99


def test_foreground_weight_sigma_is_physical():
    """Sigmas are in um: halving the spacing doubles the voxel sigma, i.e. widens the feather."""
    image, _ = _blobs()
    coarse = foreground_weight(image, [1.0, 1.0, 1.0], source="otsu", feather_sigma_um=1.0)
    fine = foreground_weight(image, [0.5, 0.5, 0.5], source="otsu", feather_sigma_um=1.0)
    assert ((fine > 0) & (fine < 1)).sum() > ((coarse > 0) & (coarse < 1)).sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device parity needs a CUDA GPU")
def test_gpu_matches_cpu():
    """CuPy inputs give the CPU values (float32 reduction order only)."""
    from cubic.cuda import ascupy

    image, _ = _blobs()
    pred = image + np.random.default_rng(6).normal(0.0, 0.5, image.shape).astype(np.float32)
    kwargs = {"source": "smooth_otsu", "smooth_sigma_um": 0.6, "feather_sigma_um": 0.4}
    spacing = [0.5, 0.25, 0.25]
    cpu = foreground_pixel_metrics(pred, image, foreground_weight(image, spacing, **kwargs))
    gpu_image = ascupy(image)
    gpu = foreground_pixel_metrics(ascupy(pred), gpu_image, foreground_weight(gpu_image, spacing, **kwargs))
    for key in FOREGROUND_COLUMNS:
        assert gpu[key] == pytest.approx(cpu[key], rel=1e-5), key
