"""Integration tests for ``lite_checks.check_b2``.

A tiny synthetic 3-system A549-style bucket (12 FOVs, one timepoint each) is written
as real eval dirs and scored by the real ``check_b2``. The per-FOV ``SI_SSIM`` values
are built so the paired resolvability of every pair is known in advance:

* ``A`` carries a large FOV-to-FOV spread (``10 * ordinal``), shared by all systems.
* ``B - A`` is set per FOV by the test (the pair under test).
* ``C - A`` alternates ``+-5`` with zero mean, so ``(A, C)`` and ``(B, C)`` are never
  resolvable.

Run::

    uv run --no-sync pytest applications/dynacell/tools/lite_checks_test.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lite_checks import check_b2

_BUCKET = "a549__mock"
_FOVS = [f"0/0/fov{i:04d}" for i in range(12)]
_MODELS = {"A": "fcmae_vscyto3d_scratch", "B": "fcmae_vscyto3d_pretrained", "C": "unetvit3d"}


def _write_bucket(root: Path, b_minus_a: np.ndarray) -> None:
    """Write the three systems' eval dirs under ``root``."""
    base = 10.0 * np.arange(len(_FOVS))
    values = {"A": base, "B": base + b_minus_a, "C": base + 5.0 * np.array([1, -1] * (len(_FOVS) // 2))}
    for key, model in _MODELS.items():
        path = root / "nucleus" / model / "ipsc" / _BUCKET
        path.mkdir(parents=True)
        pd.DataFrame({"FOV": _FOVS, "Timepoint": 0, "SI_SSIM": values[key]}).to_csv(
            path / "pixel_metrics.csv", index=False
        )
        pd.DataFrame({"FOV": _FOVS, "Timepoint": 0}).to_csv(path / "mask_metrics.csv", index=False)
        pd.DataFrame({"Dataset_DINOv3_KID": [0.0]}).to_csv(path / "feature_metrics.csv", index=False)


def _si_ssim_row(result: pd.DataFrame) -> pd.Series:
    """Return the single ``SI_SSIM`` result row."""
    rows = result[result.metric == "SI_SSIM"]
    assert len(rows) == 1
    return rows.iloc[0]


def test_paired_se_resolves_a_constant_offset_and_full_size_retains_it(tmp_path: Path) -> None:
    """A 0.01 offset under a 110-unit FOV spread is resolvable only with a paired bootstrap.

    ``B - A`` is 0.01 on every FOV, so the paired SE is 0 while each system's own SE is
    ~9. The full-size design (all 12 FOVs, their only timepoint) must keep exactly that
    pair resolvable, with the right sign.
    """
    _write_bucket(tmp_path, np.full(len(_FOVS), 0.01))
    result = check_b2("nucleus", _BUCKET, ["fov12_t0"], tmp_path, data_root=tmp_path)
    row = _si_ssim_row(result)
    assert row.n_sys == 3
    assert row.n_full_resolvable == 1
    assert row.retention == 1.0
    assert row.lite_only_resolvable == 0
    assert row.confidently_wrong == 0
    assert row.n_rows == len(_FOVS)
    assert (tmp_path / f"checkB2_nucleus_{_BUCKET}.csv").exists()


def test_lite_subset_that_reverses_the_gap_loses_the_pair(tmp_path: Path) -> None:
    """Known positive: a lite whose FOVs reverse the B - A gap cannot retain the pair.

    ``B - A`` is +1 on FOVs 0-8 and -0.5 on FOVs 9-11: resolvable on the full set (mean
    0.625, paired SE ~0.19). The lite ``{9, 10, 11}`` resolves it with the opposite sign
    (zero lite SE), so it is lost and counted confidently wrong; the lite ``{0, 1, 2}``
    is the positive control that keeps it.
    """
    _write_bucket(tmp_path, np.array([1.0] * 9 + [-0.5] * 3))
    kept = check_b2("nucleus", _BUCKET, ["fov03_t0"], tmp_path, data_root=tmp_path, fixed_fovs=_FOVS[:3])
    assert _si_ssim_row(kept).n_full_resolvable == 1
    assert _si_ssim_row(kept).retention == 1.0
    lost = check_b2("nucleus", _BUCKET, ["fov03_t0"], tmp_path, data_root=tmp_path, fixed_fovs=_FOVS[9:])
    row = _si_ssim_row(lost)
    assert row.n_full_resolvable == 1
    assert row.retention == 0.0
    assert row.confidently_wrong == 1
    assert row.n_rows == 3
