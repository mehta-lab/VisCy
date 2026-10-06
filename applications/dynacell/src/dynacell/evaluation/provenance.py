"""Numeric-provenance contract for eval outputs.

Metric *values* depend on which ``cubic`` is installed, and nothing in a
saved CSV/NPY recorded that. Between 2026-06-30 and 2026-07-29 two eval
venvs coexisted — ``cpdino-eval`` (cubic 0.8.0a2) and
``cpdino-eval-cubic090a1`` (0.9.0a1) — and because the launcher defaulted to
the older one, jobs run after the switch silently kept using it. Measured
across the pin on one real A549 nucleus pair, four columns move:

===========================  ==============
Column                       relative delta
===========================  ==============
``Z_FSC_Resolution``         +30.9%
``XY_FSC_Resolution``        +16.7%
``Spectral_PCC``             <=0.57%
``FRC_Resolution``           +0.52%
===========================  ==============

Everything else is bit-identical: ``PCC``, ``SSIM``, ``NRMSE``, ``PSNR``,
every ``SI_*``, ``PerCell_*``, every ``AP_*`` / ``mAP`` / ``instance_dice``,
``MicroMS3IM`` (5e-8) and the CP feature columns (2e-15).

This module makes that boundary detectable and non-repeatable:

* :func:`check_cubic_pin` rejects unsupported runtime versions. Target-specific
  workflow capabilities are checked separately at startup, so old environments
  can still run targets that do not need the new workflows.
* :func:`write_metrics_provenance` stamps the versions beside the metrics,
  and :func:`metrics_provenance_matches` lets the final-metrics cache gate
  refuse a cache built by a different ``cubic``.

A version bump measured to move no value beyond a stated tolerance lists the old
version under the new one in :data:`CUBIC_VERSIONS_EQUIVALENT_TO`, so its caches
stay reusable instead of forcing every leaf to recompute. 0.9.0a1 -> 0.9.0a2 and
0.9.0a2 -> 0.9.0a3 are such bumps: the first rewrites only the MicroSSIM RI-factor
fit, the second only single-image FSC/FRC paths that production does not run.

The same sidecar stamps the content hash of the CP reference the CP feature
metrics were scored in (:mod:`dynacell.evaluation.cp_reference`). CP KID/FID/
cosine move whenever the reference is rebuilt, so a cache stamped with another
reference -- or with none, i.e. written before the shared CP space existed -- is
refused the same way.

It also stamps the ``prediction_sources_sha256_12`` of the prediction that was scored
(:func:`dynacell.evaluation.cache.prediction_sources_sha256_12`): a re-predict writes
into the same ``io.pred_path``, so the path alone cannot tell the final-metrics cache
gate that its rows describe an older prediction.

Feature FID records its solver identity independently of cubic. A requested FID
accepts only the current or a measured-equivalent solver. Missing identities
denote the original torch-fidelity helper; unknown explicit identities are refused.

When the foreground-limited pixel columns are on, the resolved
``pixel_metrics.foreground`` recipe is stamped too (``pixel_foreground``), with the
``FOREGROUND_METRICS_VERSION`` of the code and the pixel spacing: the ``FG_*`` values
depend on all three, so a cache scored with another recipe, code or spacing -- or
without a recipe -- is refused. A run with the columns off needs nothing from the
stamp and reuses a cache scored with them. With the columns off the key is absent and the payload is unchanged.
"""

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dynacell.evaluation.cache import prediction_sources_sha256_12, source_predates

#: The ``cubic`` version this repo is built against. Must equal the pin in
#: ``applications/dynacell/pyproject.toml``; ``provenance_test.py`` asserts
#: they cannot drift apart.
REQUIRED_CUBIC_VERSION = "0.9.0a4"

# These installed versions may run code paths whose API requirements they meet.
# ER/mitochondria additionally require the new workflows at eval startup.
# Cache equivalence is a separate, measured contract below.
CUBIC_RUNTIME_VERSIONS = frozenset({"0.9.0a3", "0.9.0a2", "0.9.0a1"})

#: For each declared ``cubic`` version, the earlier versions whose metric values it
#: reproduces within a stated tolerance, so their caches stay reusable. Keyed by the
#: declared version so a bump starts with no equivalents until one is measured
#: against it.
#:
#: 0.9.0a1 -> 0.9.0a2 changes only the MicroSSIM RI-factor fit: it reduces the
#: objective in bounded chunks, accumulating in float64 where 0.9.0a1 took a
#: float32 ``.mean()``, and raises ``ValueError`` on an empty pool. Tolerance: MicroMS3IM relative delta <= 1e-6, far
#: below the tables' 2-decimal rounding. Measured 2026-09-28 on the A549-trained
#: ``fnet3d_paper`` ER ``a549__denv`` leaf at the production calibration settings
#: (``max_pairs=12``, seed 42, a 576x640x960 float32 pool): the fitted alpha is
#: 17.883591651916504 under both versions, and MicroMS3IM over all 108 (FOV, t)
#: differs by 0 abs / 0 rel. The fit objective itself moves ~7e-8 relative between
#: the versions; the root finder lands on the same alpha. The other
#: ``pixel_metrics`` / ``mask_metrics`` columns do not use the RI-factor code and
#: were bit-identical on 1 FOV x 7 timepoints.
#:
#: 0.9.0a2 -> 0.9.0a3 changes single-image FSC/FRC only: it judges the split
#: sampling floor per direction, splits binomial counts before isotropic
#: resampling, and warns when a single-image checkerboard result is finer than
#: 2.83 x the pixel size. Production never enables the binomial path, and the
#: two-image FSC/FRC that ``compute_pixel_metrics`` runs is untouched. Tolerance:
#: exact. Measured 2026-09-30 on the A549-trained ``fnet3d_paper`` ER and nucleus
#: ``a549__denv`` leaves (FOV 0/0/fov0000, 7 + 10 timepoints, production eval.yaml
#: and spectral_pcc/base.yaml kwargs, A40): every pixel column, XY/Z FSC,
#: FRC_Resolution and Spectral_PCC differ by 0 abs / 0 rel. The binomial +
#: resample_isotropic control moves 1.8-6.1% under the same harness, so the
#: comparison can see a change. 0.9.0a1 stays listed: it reproduces 0.9.0a2
#: (entry above), which 0.9.0a3 reproduces exactly.
#: 0.9.0a3 -> 0.9.0a4 ports ER/mitochondria workflows and accelerates metric
#: reductions. MEASURED 2026-10-02 on A40 / AMD EPYC 7313P (AVX2, no AVX-512):
#: five full 48x640x960 A549 volumes (ER mock/DENV predictions, ER mock GT,
#: mitochondria DENV prediction and mock GT) have CPU/GPU mask XOR 0 voxels;
#: eight private cold VisCy cache cases write/read the same masks. A one-voxel
#: perturbation is detected, and the AICS baseline mutates its input while the
#: cubic path does not. On captured A549 mock-ER metric inputs, nine pixel
#: columns and ten MicroMS3IM timepoints are exact; Spectral_PCC changes by
#: 7.6400518e-9 absolute / 4.3105157e-7 relative (tolerance: 1e-6 relative).
#: Calibration parameters are identical; a 0.01 score perturbation is detected.
#: On thirteen captured cells, four per-cell PCC/SSIM values and seven GLCM
#: columns are exact; all twenty-two CP columns differ by <=3.4e-15 absolute
#: (tolerance: 1e-12 absolute). A flipped-image feature control changes by 2.596.
#: Other cell lines, AVX-512 hosts and genuinely 2D inputs were not measured.
#: The older entries remain by the measured equivalences above.
CUBIC_VERSIONS_EQUIVALENT_TO = {
    "0.9.0a3": frozenset({"0.9.0a2", "0.9.0a1"}),
    "0.9.0a4": frozenset({"0.9.0a3", "0.9.0a2", "0.9.0a1"}),
}

#: Sidecar written next to ``pixel_metrics.csv`` by :func:`write_metrics_provenance`.
PROVENANCE_FILENAME = "metrics_provenance.json"

#: Packages recorded in the sidecar. Only ``cubic`` gates cache reuse — it is the
#: one whose version was measured to move published columns. ``numpy`` and
#: ``scikit-image`` are recorded because they underpin the same metric code and a
#: future discrepancy would otherwise be just as unattributable, but they are not
#: gated on absent evidence that they move a value.
_RECORDED_PACKAGES = ("cubic", "numpy", "scikit-image")

#: FID solver identity, independent of the cubic pin. Both feature-metric callers
#: use float64 sample-space SVD only when max(n_pred, n_target) < d, and otherwise
#: keep the original torch-fidelity eigvals helper.
FID_IMPLEMENTATION_VERSION = "sample-space-svd-v1"

#: Legacy sidecars carry no FID identity because the only solver was this helper.
_LEGACY_FID_IMPLEMENTATION = "torch-fidelity-eigvals-v1"

#: Numerical compatibility policy: |new - old| <= 1e-5 + 2e-6 * |old|. The
#: absolute allowance is needed near zero, where the legacy non-symmetric
#: covariance eigvals lose accuracy; this is not bitwise equivalence. Measured
#: 2026-10-02 on captured A549 mock ER inputs: 120 (FOV, t) cohorts for each of
#: CP, DINOv3, DynaCLR, CELL-DINO and MorphEm, plus their five pooled datasets.
#: CP is selected, standardized and clipped with the bound production reference.
#: Every cohort passed this contract; all five dataset values use the dense path
#: and are exactly unchanged. An unknown explicit solver identity is refused.
FID_IMPLEMENTATIONS_EQUIVALENT_TO = {
    FID_IMPLEMENTATION_VERSION: frozenset({_LEGACY_FID_IMPLEMENTATION}),
}


def installed_versions() -> dict[str, str]:
    """Return the installed versions of the packages recorded in the sidecar.

    Returns
    -------
    dict[str, str]
        Mapping of distribution name to installed version string.
    """
    return {name: version(name) for name in _RECORDED_PACKAGES}


def _accepted_cubic_versions() -> frozenset[str]:
    """Return the declared ``cubic`` and the versions measured to reproduce its values."""
    return CUBIC_VERSIONS_EQUIVALENT_TO.get(REQUIRED_CUBIC_VERSION, frozenset()) | {REQUIRED_CUBIC_VERSION}


def check_cubic_pin() -> None:
    """Require the declared runtime or a supported older environment.

    Cache equivalence does not grant runtime API capabilities. Eval entry points
    additionally call ``require_cubic_workflows`` for their target.

    Raises
    ------
    RuntimeError
        If the installed ``cubic`` version is not :data:`REQUIRED_CUBIC_VERSION`
        or listed in :data:`CUBIC_RUNTIME_VERSIONS`. Fails closed
        on purpose: a mismatched stack writes plausible values under the wrong
        numeric contract, which is exactly the failure this module exists to prevent.
    """
    installed = version("cubic")
    if installed not in CUBIC_RUNTIME_VERSIONS | {REQUIRED_CUBIC_VERSION}:
        raise RuntimeError(
            f"cubic {installed!r} is installed but this repo declares "
            f"{REQUIRED_CUBIC_VERSION!r}. Metric values are not comparable across "
            "cubic versions (FSC/FRC/Spectral_PCC move), so refusing to write "
            "metrics. Sync the project environment from uv.lock "
            "(`uv sync --package dynacell --extra eval --extra eval_gpu`)."
        )


def write_metrics_provenance(
    save_dir: Path,
    *,
    cp_reference_sha256: str | None,
    cp_space_sha256: str | None,
    prediction_digest: str,
    pixel_foreground: dict[str, Any] | None = None,
    compute_fid: bool = False,
) -> None:
    """Write the numeric-provenance sidecar into ``save_dir``.

    Parameters
    ----------
    save_dir : pathlib.Path
        Directory that receives the metric CSV/NPY files.
    cp_reference_sha256 : str or None
        Content hash of the CP reference the feature metrics were scored in
        (recorded for audit); ``None`` when the run computed no feature metrics.
    cp_space_sha256 : str or None
        ``DatasetCPSpace.binding_sha256`` of the space the run scored in: the
        reference bound to one dataset and the GT cells it was fit on. This is the
        value cache reuse compares. ``None`` exactly when ``cp_reference_sha256`` is.
    prediction_digest : str
        :func:`~dynacell.evaluation.cache.prediction_sources_sha256_12` of the
        prediction the metrics were scored on, taken before scoring and stored as
        ``prediction_sources_sha256_12``; the final-metrics cache gate compares it with
        the store's current one.
    pixel_foreground : dict or None
        Resolved ``pixel_metrics.foreground`` recipe the ``FG_*`` columns were scored
        with, plus the FG code version and pixel spacing, stored under
        ``pixel_foreground``; ``None`` (columns off) writes no key.
    compute_fid : bool
        Whether feature FID was requested. When True, record the implementation
        under ``fid_implementation``; False writes no key.

    Raises
    ------
    ValueError
        If only one of the two hashes is given.
    """
    if (cp_reference_sha256 is None) != (cp_space_sha256 is None):
        raise ValueError("cp_reference_sha256 and cp_space_sha256 must both be given or both be None")
    payload = {
        "versions": installed_versions(),
        "cp_reference_sha256": cp_reference_sha256,
        "cp_space_sha256": cp_space_sha256,
        "prediction_sources_sha256_12": prediction_digest,
    }
    if pixel_foreground is not None:
        payload["pixel_foreground"] = pixel_foreground
    if compute_fid:
        payload["fid_implementation"] = FID_IMPLEMENTATION_VERSION
    (save_dir / PROVENANCE_FILENAME).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def metrics_provenance_matches(
    save_dir: Path,
    *,
    cp_space_sha256: str | None,
    prediction_sources: dict[str, dict[str, Any]],
    pixel_foreground: dict[str, Any] | None = None,
    compute_fid: bool = False,
) -> bool:
    """Return True when ``save_dir``'s metrics were built by the running ``cubic`` and CP reference, from this prediction.

    A missing sidecar returns False. Every cache written before this stamp
    existed predates the fix and is exactly the ambiguous case that has to be
    recomputed, so "unknown" must not read as "compatible" here — unlike the
    per-extractor ``preprocess_version`` bootstrap, which treats an untagged
    entry as unconstrained.

    Parameters
    ----------
    save_dir : pathlib.Path
        Directory holding the metric CSV/NPY files.
    cp_space_sha256 : str or None
        ``DatasetCPSpace.binding_sha256`` of the space the current run would score
        in, or ``None`` when it computes no feature metrics -- then no CP value is
        reused and the recorded hash (or its absence) is irrelevant. When given, a
        sidecar without a ``cp_space_sha256`` (written before the binding existed)
        never matches.
    prediction_sources : dict
        :func:`~dynacell.evaluation.cache.prediction_sources` of the store the current
        run would score. A re-predict writes into the same path, so the sidecar's
        ``prediction_sources_sha256_12`` must equal their digest. A sidecar written
        before that field existed is dated instead: reusable only when every position
        predates it (:func:`~dynacell.evaluation.cache.source_predates` against its
        mtime); a blank position (no stored chunk) cannot be dated and forces one
        recompute. That misses one legacy case: metrics saved after a re-predict but
        scored from pred caches it had left stale. The grouped leaves'
        ``force_recompute.final_metrics: true`` covers it, and stale legacy caches were
        audited and recomputed.
    pixel_foreground : dict or None
        Resolved ``pixel_metrics.foreground`` recipe (with FG code version and pixel
        spacing) the current run would score with, ``None`` when the ``FG_*`` columns
        are off. When given, it must equal the recorded one (absent = ``None``), so
        turning the columns on, or changing a sigma, the code version or the spacing,
        recomputes. ``None`` accepts any recorded recipe: the cache's extra ``FG_*``
        columns cost a run without them nothing.
    compute_fid : bool
        When True, require the current or a measured-equivalent FID solver.
        A missing identity denotes the legacy torch-fidelity implementation;
        unknown explicit identities are refused. False reuses no FID value and
        ignores its solver identity.

    Returns
    -------
    bool
        True when the recorded ``cubic`` version equals the installed one (or both
        are the declared pin or listed under it in :data:`CUBIC_VERSIONS_EQUIVALENT_TO`),
        if ``cp_space_sha256`` is given the recorded binding equals it, if
        ``pixel_foreground`` is given the recorded recipe equals it, and the
        prediction check above passes. When ``compute_fid`` is True the recorded
        FID solver must also be current or measured-equivalent.
    """
    path = save_dir / PROVENANCE_FILENAME
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    recorded = payload.get("versions", {}).get("cubic")
    installed = version("cubic")
    accepted = _accepted_cubic_versions()
    if recorded != installed and not (recorded in accepted and installed in accepted):
        return False
    if cp_space_sha256 is not None and payload.get("cp_space_sha256") != cp_space_sha256:
        return False
    if pixel_foreground is not None and payload.get("pixel_foreground") != pixel_foreground:
        return False
    if compute_fid:
        fid_implementation = payload.get("fid_implementation", _LEGACY_FID_IMPLEMENTATION)
        accepted_fid = FID_IMPLEMENTATIONS_EQUIVALENT_TO.get(FID_IMPLEMENTATION_VERSION, frozenset()) | {
            FID_IMPLEMENTATION_VERSION
        }
        if not isinstance(fid_implementation, str) or fid_implementation not in accepted_fid:
            return False
    digest = payload.get("prediction_sources_sha256_12")
    if digest is not None:
        return digest == prediction_sources_sha256_12(prediction_sources)
    saved_ns = path.stat().st_mtime_ns
    return all(source_predates(source, saved_ns) for source in prediction_sources.values())
