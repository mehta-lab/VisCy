"""Integration tests for ``scan_nan_microssim_caches.py``.

Each test lays out real eval save dirs in ``tmp_path`` (sidecar written by
``write_metrics_provenance``, then restamped) and runs the tool's ``main``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scan_nan_microssim_caches import main  # noqa: E402

from dynacell.evaluation.provenance import PROVENANCE_FILENAME, write_metrics_provenance


def _save_dir(root, rel, microssim, cubic="0.9.0a1"):
    save_dir = root / rel
    save_dir.mkdir(parents=True)
    write_metrics_provenance(save_dir, cp_reference_sha256=None, cp_space_sha256=None)
    sidecar = save_dir / PROVENANCE_FILENAME
    payload = json.loads(sidecar.read_text())
    payload["versions"]["cubic"] = cubic
    sidecar.write_text(json.dumps(payload))
    pd.DataFrame({"FOV": ["a", "b"], "MicroMS3IM": microssim}).to_csv(save_dir / "pixel_metrics.csv", index=False)
    return save_dir


def test_all_nan_cache_fails(tmp_path, capsys):
    """An all-NaN cache is reported and fails the scan; a partial NaN is not."""
    hit = _save_dir(tmp_path, "er/fnet3d_paper/a549/a549__denv", [np.nan, np.nan])
    _save_dir(tmp_path, "er/fnet3d_paper/a549/a549__zikv", [0.8, np.nan])
    assert main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"ALL-NAN\t{hit}" in out
    assert "a549__zikv" not in out
    assert "2 sidecars, 1 all-NaN MicroMS3IM stamped cubic 0.9.0a1" in out


def test_mito_a549_mock_is_no_longer_exempt(tmp_path, capsys):
    """Calibration now drops constant GT slices, so an all-NaN mito A549 mock cache is stale too."""
    canonical = _save_dir(tmp_path, "mito/fnet2d/joint/a549__mock", [np.nan, np.nan])
    legacy = _save_dir(tmp_path, "a549/evaluations_with_embeddings/eval_phase_mitochondria_mock", [np.nan, np.nan])
    assert main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"ALL-NAN\t{canonical}" in out
    assert f"ALL-NAN\t{legacy}" in out


def test_other_cubic_stamps_and_zarr_contents_are_skipped(tmp_path, capsys):
    """Only the requested cubic stamp is scanned, and zarr stores are not descended into."""
    _save_dir(tmp_path, "er/fnet3d_paper/a549/a549__denv", [np.nan, np.nan], cubic="0.9.0a2")
    _save_dir(tmp_path, "er/fnet3d_paper/a549/prediction.zarr/inner", [np.nan, np.nan])
    assert main(["--root", str(tmp_path)]) == 0
    assert "1 sidecars, 0 all-NaN" in capsys.readouterr().out
