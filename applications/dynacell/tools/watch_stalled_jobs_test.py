"""Tests for the stalled-job detector, driven by real measured job traces."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import watch_stalled_jobs
from watch_stalled_jobs import (  # noqa: E402
    STATE_VERSION,
    JobState,
    Sample,
    _load_states,
    parse_slurm_duration,
)

HOUR = 3600.0
SCRIPT = Path(watch_stalled_jobs.__file__)


def _write_state_file(path: Path, **payload) -> None:
    """Write a state file as another writer would; keys override the defaults."""
    path.write_text(json.dumps({"version": STATE_VERSION, "user": "alex.kalinin", "states": {}} | payload))


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


def test_step_boundary_cpu_reset_is_not_flagged() -> None:
    """A batched predict crossing an srun step boundary must not read as stalled.

    ``sstat`` reports only the RUNNING step, so ``AveCPU`` restarts near zero
    each time ``submit_benchmark_batch`` advances to the next of its N
    sequential steps. The unguarded subtraction made that a negative rate,
    which is below any stall threshold — so the watchdog would flag a job that
    is burning CPU as hard as ever, and ``--once`` would exit 1.
    """
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=3.00 * HOUR, cpu_s=2.90 * HOUR))
    # Step 2 starts: same allocation, fresh AveCPU counter.
    state.add(Sample(wall_s=3.60 * HOUR, cpu_s=0.05 * HOUR))

    assert state.stall_report() is None

    # The stale pre-reset baseline is dropped, so the next window measures the
    # new step honestly rather than against a counter that no longer exists.
    state.add(Sample(wall_s=4.20 * HOUR, cpu_s=0.65 * HOUR))
    assert state.stall_report() is None


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


def test_new_step_can_demonstrate_progress_after_a_long_allocation() -> None:
    """Step-local CPU must be compared with step-local elapsed time."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=9.9 * HOUR, cpu_s=9.0 * HOUR))
    state.add(Sample(wall_s=10 * HOUR, cpu_s=0))
    state.add(Sample(wall_s=10 * HOUR + 1800, cpu_s=1700))
    assert state.stall_report() is None

    state.add(Sample(wall_s=10 * HOUR + 2700, cpu_s=1700))
    verdict = state.stall_report()
    assert verdict is not None
    assert verdict[0] == pytest.approx(0.0)
    assert verdict[1] == pytest.approx(1700 / 1800)

    # The demonstrated progress survives both a long stall and sample pruning.
    for elapsed in range(3600, 18001, 900):
        state.add(Sample(wall_s=10 * HOUR + elapsed, cpu_s=1700))
        assert state.stall_report() is not None


def test_new_step_staging_does_not_inherit_previous_progress() -> None:
    """A CPU reset clears the previous step's evidence of active compute."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=10 * HOUR, cpu_s=9 * HOUR))
    state.add(Sample(wall_s=10 * HOUR + 100, cpu_s=5))
    state.add(Sample(wall_s=11 * HOUR, cpu_s=5))
    assert state.stall_report() is None
    assert all(sample.cpu_s == 5 for sample in state.samples)


def test_new_step_work_before_first_observation_can_establish_progress() -> None:
    """The first sample after a reset may contain all of the new step's work."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=10 * HOUR, cpu_s=9 * HOUR))
    state.add(Sample(wall_s=10 * HOUR + 1800, cpu_s=1700))
    assert state.stall_report() is None

    state.add(Sample(wall_s=10 * HOUR + 2700, cpu_s=1700))
    assert state.stall_report() is None

    state.add(Sample(wall_s=10 * HOUR + 3600, cpu_s=1700))
    verdict = state.stall_report()
    assert verdict is not None
    assert verdict[1] == pytest.approx(1700 / 1800)


@pytest.mark.parametrize("next_wall_s", [22 * HOUR, 24 * HOUR])
def test_once_reuses_history_between_invocations(tmp_path, monkeypatch, capsys, next_wall_s) -> None:
    """Separate CLI invocations can identify the measured hung predict."""
    monkeypatch.setattr(
        "sys.argv",
        ["watch_stalled_jobs", "--once", "--state-file", str(tmp_path / "watch.json")],
    )
    clock = iter([21.18 * HOUR, next_wall_s])
    monkeypatch.setattr(
        watch_stalled_jobs,
        "running_jobs",
        lambda user: [("35083019_0", "P2P_EMA_REST_g44", next(clock), "gpu-b-4")],
    )
    monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda jobid: 21.75 * HOUR)

    assert watch_stalled_jobs.main() == 0
    assert "collecting history" in capsys.readouterr().out
    assert watch_stalled_jobs.main() == 1
    assert "STALLED 35083019_0" in capsys.readouterr().out


def test_once_persists_new_step_progress(tmp_path, monkeypatch, capsys) -> None:
    """Persist the current step's origin and prior CPU progress with its samples."""
    monkeypatch.setattr(
        "sys.argv",
        ["watch_stalled_jobs", "--once", "--state-file", str(tmp_path / "watch.json")],
    )
    samples = iter([(9.9 * HOUR, 9 * HOUR), (10 * HOUR, 0), (10 * HOUR + 1800, 1700), (10 * HOUR + 2700, 1700)])
    for expected_status in (0, 0, 0, 1):
        wall_s, cpu_s = next(samples)
        monkeypatch.setattr(
            watch_stalled_jobs,
            "running_jobs",
            lambda user, wall_s=wall_s: [("123", "ER_PREDICT_batch", wall_s, "gpu-f-3")],
        )
        monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda jobid, cpu_s=cpu_s: cpu_s)
        assert watch_stalled_jobs.main() == expected_status
    assert "STALLED 123" in capsys.readouterr().out


def test_once_sparse_cpu_reset_waits_for_observed_step_age(tmp_path, monkeypatch, capsys) -> None:
    """A sparse CPU reset proves prior work but cannot prove the step's age."""
    monkeypatch.setattr(
        "sys.argv",
        ["watch_stalled_jobs", "--once", "--state-file", str(tmp_path / "watch.json")],
    )
    # A multithreaded step can consume 3000 CPU seconds shortly before the
    # first post-reset observation, even when observations are an hour apart.
    samples = [(10 * HOUR, 9 * HOUR), (11 * HOUR, 3000), (11 * HOUR + 900, 3000), (11 * HOUR + 1800, 3000)]
    for (wall_s, cpu_s), expected_status in zip(samples, (0, 0, 0, 1), strict=True):
        monkeypatch.setattr(
            watch_stalled_jobs,
            "running_jobs",
            lambda user, wall_s=wall_s: [("123", "ER_PREDICT_batch", wall_s, "gpu-f-3")],
        )
        monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda jobid, cpu_s=cpu_s: cpu_s)
        assert watch_stalled_jobs.main() == expected_status
        output = capsys.readouterr().out
        assert ("STALLED 123" in output) == bool(expected_status)
    assert "was 0.83 before" in output


def test_requeued_job_uses_its_reported_wall_age() -> None:
    """A wall-clock reset supplies the new job's elapsed time directly."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=10 * HOUR, cpu_s=9 * HOUR))
    state.add(Sample(wall_s=1800, cpu_s=1700))
    state.add(Sample(wall_s=2700, cpu_s=1700))
    verdict = state.stall_report()
    assert verdict is not None
    assert verdict[1] == pytest.approx(1700 / 1800)


def test_once_restarts_history_from_an_unversioned_state_file(tmp_path, monkeypatch, capsys) -> None:
    """A state file from an earlier schema is discarded, not rehydrated.

    Rehydrating it unguarded tracebacked on the first field change, and under
    the exit-code contract that traceback exited 1 -- indistinguishable from a
    stall -- on every run until the cache was deleted by hand.
    """
    state_file = tmp_path / "watch.json"
    state_file.write_text(json.dumps({"user": "alex.kalinin", "states": {"1": {"field_from_v1": 1.0}}}))
    monkeypatch.setattr("sys.argv", ["watch_stalled_jobs", "--once", "--state-file", str(state_file)])
    monkeypatch.setattr(
        watch_stalled_jobs,
        "running_jobs",
        lambda user: [("35083019_0", "P2P_EMA_REST_g44", 21.18 * HOUR, "gpu-b-4")],
    )
    monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda jobid: 21.75 * HOUR)

    assert watch_stalled_jobs.main() == 0

    assert f"schema None is not {STATE_VERSION}; starting a new history" in capsys.readouterr().out
    rewritten = json.loads(state_file.read_text())
    assert rewritten["version"] == STATE_VERSION
    assert set(rewritten["states"]) == {"35083019_0"}


def test_load_states_rejects_another_users_file(tmp_path) -> None:
    """Same schema, different user is a real error rather than a fresh start."""
    state_file = tmp_path / "watch.json"
    _write_state_file(state_file, user="someone.else")

    with pytest.raises(ValueError, match="another user"):
        _load_states(state_file, "alex.kalinin")


def test_cli_exits_2_on_a_tool_error(tmp_path) -> None:
    """A tool failure must not exit 1, which the contract reserves for a stall.

    Pointing ``--user`` at another user's state file raises before any
    ``squeue`` call, so the script fails without touching the scheduler.
    """
    state_file = tmp_path / "watch.json"
    _write_state_file(state_file)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--once", "--user", "someone.else", "--state-file", str(state_file)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "ValueError: state file" in proc.stderr
