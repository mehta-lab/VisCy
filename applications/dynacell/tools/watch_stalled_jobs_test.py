"""Tests for the stalled-job detector, driven by real measured job traces."""

import sys
from pathlib import Path

import pytest

# The tools/ directory is not a Python package; add it to sys.path so the
# watchdog module is importable by short name.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from watch_stalled_jobs import (  # noqa: E402
    JobState,
    Sample,
    parse_slurm_duration,
)

HOUR = 3600.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1-14:47:33", 139653.0),
        ("21:45:07", 78307.0),
        ("00:00.274", 0.274),
        ("10-15:50:06", 921006.0),
        ("UNLIMITED", None),
        ("", None),
    ],
)
def test_parse_slurm_duration_handles_every_field_shape(text: str, expected: float | None) -> None:
    """Real ``Elapsed``/``AveCPU`` values, including the ``MM:SS.frac`` short form."""
    parsed = parse_slurm_duration(text)
    if expected is None:
        assert parsed is None
    else:
        assert parsed == pytest.approx(expected)


def test_parse_slurm_duration_rejects_a_truncated_field() -> None:
    """``AveCPU`` truncates to ``10-15:41:+`` unless polled with ``-P``.

    Silently accepting it would parse ``41`` as seconds and understate CPU time
    by hours, which reads as a stall on a perfectly healthy job.
    """
    with pytest.raises(ValueError, match="truncated"):
        parse_slurm_duration("10-15:41:+")


def test_hung_pix2pix_predict_is_flagged() -> None:
    """Trace of job 35083019_0, which held a GPU for 17.5 h after finishing.

    Wall/CPU pairs are the measured values: it burned 21.75 h of CPU over its
    21.18 h predict loop, then stopped consuming CPU entirely while wall time
    ran to 38.79 h.
    """
    state = JobState(name="P2P_EMA_REST_g44", node="gpu-b-4")
    state.add(Sample(wall_s=21.18 * HOUR, cpu_s=21.75 * HOUR))
    state.add(Sample(wall_s=22.00 * HOUR, cpu_s=21.75 * HOUR))

    verdict = state.stall_report()

    assert verdict is not None
    fraction, prior_efficiency, _ = verdict
    assert fraction == pytest.approx(0.0)
    assert prior_efficiency > 1.0


def test_healthy_multithreaded_fit_is_not_flagged() -> None:
    """Trace of fit 35083472: 32.6 h wall against 255.8 h of CPU across 16 CPUs.

    Absolute efficiency far exceeds 1.0 here, which is why the detector keys on
    the derivative rather than a cumulative ratio.
    """
    state = JobState(name="FNet3DT01_A549_NUCL", node="gpu-b-3")
    state.add(Sample(wall_s=32.60 * HOUR, cpu_s=255.80 * HOUR))
    state.add(Sample(wall_s=32.95 * HOUR, cpu_s=258.55 * HOUR))

    assert state.stall_report() is None


def test_startup_staging_is_not_flagged() -> None:
    """A job staging a 14 GB store off NFS is legitimately at ~0% CPU.

    It has never demonstrated CPU progress, so the prior-efficiency gate keeps
    it out of the report even though its derivative is flat.
    """
    state = JobState(name="FNet3DTSpread_A549_SEC61B", node="gpu-b-5")
    state.add(Sample(wall_s=0.60 * HOUR, cpu_s=0.01 * HOUR))
    state.add(Sample(wall_s=0.95 * HOUR, cpu_s=0.01 * HOUR))

    assert state.stall_report() is None


def test_young_job_is_not_flagged_even_when_flat() -> None:
    """Short evals finish in minutes; judging them would flag normal startup."""
    state = JobState(name="EV_nucleus", node="gpu-f-4")
    state.add(Sample(wall_s=0.05 * HOUR, cpu_s=0.05 * HOUR))
    state.add(Sample(wall_s=0.40 * HOUR, cpu_s=0.05 * HOUR))

    assert state.stall_report() is None


def test_narrow_window_defers_the_verdict() -> None:
    """Two samples minutes apart are not enough; the lookback is 15 min."""
    state = JobState(name="P2P_EMA_REST_g44", node="gpu-b-4")
    state.add(Sample(wall_s=21.18 * HOUR, cpu_s=21.75 * HOUR))
    state.add(Sample(wall_s=21.20 * HOUR, cpu_s=21.75 * HOUR))

    assert state.stall_report() is None
