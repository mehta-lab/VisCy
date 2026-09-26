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

* :func:`check_cubic_pin` fails a run whose environment disagrees with the
  version the repo declares, instead of silently producing values from a
  different numeric stack.
* :func:`write_metrics_provenance` stamps the versions beside the metrics,
  and :func:`metrics_provenance_matches` lets the final-metrics cache gate
  refuse a cache built by a different ``cubic``.

The same sidecar stamps the content hash of the CP reference the CP feature
metrics were scored in (:mod:`dynacell.evaluation.cp_reference`). CP KID/FID/
cosine move whenever the reference is rebuilt, so a cache stamped with another
reference -- or with none, i.e. written before the shared CP space existed -- is
refused the same way.
"""

import json
from importlib.metadata import version
from pathlib import Path

#: The ``cubic`` version this repo is built against. Must equal the pin in
#: ``applications/dynacell/pyproject.toml``; ``provenance_test.py`` asserts
#: they cannot drift apart.
REQUIRED_CUBIC_VERSION = "0.9.0a1"

#: Sidecar written next to ``pixel_metrics.csv`` by :func:`write_metrics_provenance`.
PROVENANCE_FILENAME = "metrics_provenance.json"

#: Packages recorded in the sidecar. Only ``cubic`` gates cache reuse — it is the
#: one whose version was measured to move published columns. ``numpy`` and
#: ``scikit-image`` are recorded because they underpin the same metric code and a
#: future discrepancy would otherwise be just as unattributable, but they are not
#: gated on absent evidence that they move a value.
_RECORDED_PACKAGES = ("cubic", "numpy", "scikit-image")


def installed_versions() -> dict[str, str]:
    """Return the installed versions of the packages recorded in the sidecar.

    Returns
    -------
    dict[str, str]
        Mapping of distribution name to installed version string.
    """
    return {name: version(name) for name in _RECORDED_PACKAGES}


def check_cubic_pin() -> None:
    """Raise when the installed ``cubic`` is not the declared one.

    Raises
    ------
    RuntimeError
        If the installed ``cubic`` version differs from
        :data:`REQUIRED_CUBIC_VERSION`. Fails closed on purpose: a mismatched
        stack writes plausible values under the wrong numeric contract, which
        is exactly the failure this module exists to prevent.
    """
    installed = version("cubic")
    if installed != REQUIRED_CUBIC_VERSION:
        raise RuntimeError(
            f"cubic {installed!r} is installed but this repo declares "
            f"{REQUIRED_CUBIC_VERSION!r}. Metric values are not comparable across "
            "cubic versions (FSC/FRC/Spectral_PCC move), so refusing to write "
            "metrics. Point UV_PROJECT_ENVIRONMENT / DYNACELL_EVAL_VENV at an "
            "environment holding the declared version."
        )


def write_metrics_provenance(save_dir: Path, *, cp_reference_sha256: str | None) -> None:
    """Write the numeric-provenance sidecar into ``save_dir``.

    Parameters
    ----------
    save_dir : pathlib.Path
        Directory that receives the metric CSV/NPY files.
    cp_reference_sha256 : str or None
        Content hash of the CP reference the feature metrics were scored in;
        ``None`` when the run computed no feature metrics.
    """
    payload = {"versions": installed_versions(), "cp_reference_sha256": cp_reference_sha256}
    (save_dir / PROVENANCE_FILENAME).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def metrics_provenance_matches(save_dir: Path, *, cp_reference_sha256: str | None) -> bool:
    """Return True when ``save_dir``'s metrics were built by the running ``cubic`` and CP reference.

    A missing sidecar returns False. Every cache written before this stamp
    existed predates the fix and is exactly the ambiguous case that has to be
    recomputed, so "unknown" must not read as "compatible" here — unlike the
    per-extractor ``preprocess_version`` bootstrap, which treats an untagged
    entry as unconstrained.

    Parameters
    ----------
    save_dir : pathlib.Path
        Directory holding the metric CSV/NPY files.
    cp_reference_sha256 : str or None
        Content hash of the CP reference the current run would score in, or
        ``None`` when it computes no feature metrics -- then no CP value is
        reused and the recorded hash (or its absence) is irrelevant. When given,
        a sidecar written before the CP reference existed carries no such key
        and never matches.

    Returns
    -------
    bool
        True when the recorded ``cubic`` version equals the installed one and,
        if ``cp_reference_sha256`` is given, the recorded hash equals it.
    """
    path = save_dir / PROVENANCE_FILENAME
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    recorded = payload.get("versions", {}).get("cubic")
    if recorded != version("cubic"):
        return False
    if cp_reference_sha256 is None:
        return True
    return "cp_reference_sha256" in payload and payload["cp_reference_sha256"] == cp_reference_sha256
