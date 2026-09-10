"""Tests for the stalled-job detector, driven by real measured job traces."""

import json
import socket
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
    _save_states,
    parse_slurm_duration,
    running_steps,
    step_cpu_seconds,
)

HOUR = 3600.0
SCRIPT = Path(watch_stalled_jobs.__file__)


def _write_state_file(path: Path, **payload) -> None:
    """Write a state file as another writer would; keys override the defaults."""
    defaults = {"version": STATE_VERSION, "user": "alex.kalinin", "host": socket.gethostname(), "states": {}}
    path.write_text(json.dumps(defaults | payload))


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


def test_running_steps_joins_step_clocks_to_filtered_job_names(monkeypatch) -> None:
    """Steps carry their own elapsed time; names and the interactive filter come from the job list.

    ``squeue -s`` prints ``%j`` as the STEP name (``uv``), so filtering the step
    listing by name would never match ``nomachine``. Array tasks print as
    ``36130452_579.1``; the text before the last ``.`` is the job listing's id.
    ``.batch``/``.extern`` never carry the compute, and a step whose job is not
    in the filtered list is interactive or ended between the two queries.
    """
    outputs = {
        ("squeue", "-u", "alex.kalinin", "-h", "-t", "RUNNING", "-o", "%i|%j"): (
            "36130452_579|ER_PREDICT_batch\n36115356|nomachine\n36120000|FNet3DT01_A549_NUCL\n"
        ),
        ("squeue", "-s", "-u", "alex.kalinin", "-h", "-t", "RUNNING", "-o", "%i|%M|%N"): (
            "36130452_579.batch|3:00:00|gpu-f-3\n"
            "36130452_579.extern|3:00:00|gpu-f-3\n"
            "36130452_579.1|12:34|gpu-f-3\n"
            "36115356.0|1-02:00:00|gpu-e-2\n"
            "36120000.0|2:00:00|gpu-b-3\n"
            "36199999.0|0:30|gpu-b-9\n"
        ),
    }

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=outputs[tuple(argv)], stderr="")

    monkeypatch.setattr(watch_stalled_jobs.subprocess, "run", fake_run)

    assert running_steps("alex.kalinin") == [
        ("36130452_579.1", "ER_PREDICT_batch", 754.0, "gpu-f-3"),
        ("36120000.0", "FNet3DT01_A549_NUCL", 2 * HOUR, "gpu-b-3"),
    ]


def test_step_cpu_seconds_reads_the_single_step_row(monkeypatch) -> None:
    """``sstat -j <step> -P`` returns that step's row; a failed call (step ended) yields None."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[2] == "36115356.1":
            return subprocess.CompletedProcess(argv, 0, stdout="JobID|AveCPU\n36115356.1|22:22:04\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="sstat: error: no steps running\n")

    monkeypatch.setattr(watch_stalled_jobs.subprocess, "run", fake_run)

    assert step_cpu_seconds("36115356.1") == pytest.approx(22 * HOUR + 22 * 60 + 4)
    assert step_cpu_seconds("36115356.2") is None
    assert calls[0] == ["sstat", "-j", "36115356.1", "-P", "--format=JobID,AveCPU"]


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


def test_new_step_starts_its_own_history() -> None:
    """A batched predict advancing to its next srun step must not read as stalled.

    ``sstat`` reports only the RUNNING step, so ``AveCPU`` restarts near zero
    each time ``submit_benchmark_batch`` advances to the next of its N
    sequential steps. Each step is its own key with the scheduler's elapsed
    time for that step, so the new step is judged on its own clocks rather
    than against a counter that no longer exists.
    """
    first = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    first.add(Sample(wall_s=3.00 * HOUR, cpu_s=2.90 * HOUR))
    assert first.stall_report() is None

    # Step 2, first seen 3 min into its own life with a fresh AveCPU counter.
    second = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    second.add(Sample(wall_s=180, cpu_s=180))
    assert second.stall_report() is None
    second.add(Sample(wall_s=0.65 * HOUR, cpu_s=0.65 * HOUR))
    assert second.stall_report() is None


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
    """A later step of a long allocation is judged on its own elapsed time."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=60, cpu_s=0))
    state.add(Sample(wall_s=1860, cpu_s=1700))
    assert state.stall_report() is None

    state.add(Sample(wall_s=2760, cpu_s=1700))
    verdict = state.stall_report()
    assert verdict is not None
    assert verdict[0] == pytest.approx(0.0)
    assert verdict[1] == pytest.approx(1700 / 1800)

    # The demonstrated progress survives both a long stall and sample pruning.
    for elapsed in range(3600, 18001, 900):
        state.add(Sample(wall_s=60 + elapsed, cpu_s=1700))
        assert state.stall_report() is not None


def test_step_already_hung_when_first_seen_is_flagged() -> None:
    """A watcher started mid-allocation must judge the step by its own elapsed time.

    A step 3 h old with 2 h of CPU is 0.67 efficient. Dividing the step's CPU
    by the JOB's 30 h elapsed gave 0.067, below ``MIN_PRIOR_EFFICIENCY``, so a
    ``submit_benchmark_batch`` chain whose current step was already hung was
    never flagged no matter how long it sat.
    """
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=3 * HOUR, cpu_s=2 * HOUR))
    assert state.stall_report() is None

    state.add(Sample(wall_s=3 * HOUR + 900, cpu_s=2 * HOUR))
    verdict = state.stall_report()
    assert verdict is not None
    assert verdict[0] == pytest.approx(0.0)
    assert verdict[1] == pytest.approx(2 / 3)


def test_startup_import_burst_does_not_qualify_a_staging_step() -> None:
    """A 2-min import burst followed by ~0% NFS staging must never read as a stall.

    The burst put 60 s of CPU into a 120 s old step (0.5 efficient). Seeding
    ``prior_efficiency`` from that window and freezing it with ``max()`` made
    the legitimately idle staging phase that followed alert as soon as the
    step turned 30 min old. Only a window at least ``MIN_AGE_S`` long may
    qualify a step.
    """
    state = JobState(name="FNet3DTSpread_A549_SEC61B", node="gpu-b-5")
    for wall_s, cpu_s in ((120, 60), (720, 65), (1320, 65), (1920, 65)):
        state.add(Sample(wall_s=wall_s, cpu_s=cpu_s))
        assert state.stall_report() is None
    for wall_s in range(2820, int(4 * HOUR) + 1, 900):
        state.add(Sample(wall_s=wall_s, cpu_s=65))
        assert state.stall_report() is None


def test_early_burst_on_a_fresh_step_does_not_qualify_it() -> None:
    """A fresh step first seen at 10 min with a burst then staging must never alert.

    200 s of CPU in a 600 s old step is 0.33 -- over ``MIN_PRIOR_EFFICIENCY``
    -- yet the window is far too short to prove the step does real work.
    """
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    for wall_s, cpu_s in ((600, 200), (1500, 280), (2400, 280), (3300, 280)):
        state.add(Sample(wall_s=wall_s, cpu_s=cpu_s))
        assert state.stall_report() is None


def test_qualified_step_stays_flagged_through_a_long_stall() -> None:
    """Positive control: 3500 s of CPU over a 3600 s old step qualifies it for good.

    The step is old enough on first sight for its cumulative ratio to count,
    and ``max()`` keeps that evidence while the idle stretch runs for hours.
    """
    state = JobState(name="P2P_EMA_REST_g44", node="gpu-b-4")
    state.add(Sample(wall_s=3600, cpu_s=3500))
    assert state.stall_report() is None

    for wall_s in range(4500, int(3600 + 8 * HOUR) + 1, 900):
        state.add(Sample(wall_s=wall_s, cpu_s=3500))
        verdict = state.stall_report()
        assert verdict is not None
        assert verdict[1] == pytest.approx(3500 / 3600)


def test_requeued_step_restarts_its_history() -> None:
    """A requeue reuses the step id; the wall-clock decrease restarts the history."""
    state = JobState(name="ER_PREDICT_batch", node="gpu-f-3")
    state.add(Sample(wall_s=10 * HOUR, cpu_s=9 * HOUR))
    state.add(Sample(wall_s=1800, cpu_s=1700))
    assert state.samples == [Sample(wall_s=1800, cpu_s=1700)]
    assert state.stall_report() is None

    state.add(Sample(wall_s=2700, cpu_s=1700))
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
        "running_steps",
        lambda user: [("35083019_0.0", "P2P_EMA_REST_g44", next(clock), "gpu-b-4")],
    )
    monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda step_id: 21.75 * HOUR)

    assert watch_stalled_jobs.main() == 0
    assert "collecting history, tracking 1: 35083019_0.0(1)" in capsys.readouterr().out
    assert watch_stalled_jobs.main() == 1
    assert "STALLED 35083019_0.0 P2P_EMA_REST_g44 on gpu-b-4" in capsys.readouterr().out


def test_once_persists_new_step_progress(tmp_path, monkeypatch, capsys) -> None:
    """A chain's next step is tracked under its own id, with its own clocks, across invocations."""
    monkeypatch.setattr(
        "sys.argv",
        ["watch_stalled_jobs", "--once", "--state-file", str(tmp_path / "watch.json")],
    )
    polls = [("123.0", 9.9 * HOUR, 9 * HOUR), ("123.1", 60, 0), ("123.1", 1860, 1700), ("123.1", 2760, 1700)]
    for (step_id, wall_s, cpu_s), expected_status in zip(polls, (0, 0, 0, 1), strict=True):
        monkeypatch.setattr(
            watch_stalled_jobs,
            "running_steps",
            lambda user, step_id=step_id, wall_s=wall_s: [(step_id, "ER_PREDICT_batch", wall_s, "gpu-f-3")],
        )
        monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda step_id, cpu_s=cpu_s: cpu_s)
        assert watch_stalled_jobs.main() == expected_status
    assert "STALLED 123.1 ER_PREDICT_batch" in capsys.readouterr().out
    # The finished step is dropped from the history once it leaves the queue.
    assert set(json.loads((tmp_path / "watch.json").read_text())["states"]) == {"123.1"}


def test_once_flags_a_step_already_hung_when_first_seen(tmp_path, monkeypatch, capsys) -> None:
    """Sparse one-shot polls judge a mid-allocation step by the scheduler's step age."""
    monkeypatch.setattr(
        "sys.argv",
        ["watch_stalled_jobs", "--once", "--state-file", str(tmp_path / "watch.json")],
    )
    for wall_s, expected_status in ((3 * HOUR, 0), (3 * HOUR + 900, 1)):
        monkeypatch.setattr(
            watch_stalled_jobs,
            "running_steps",
            lambda user, wall_s=wall_s: [("123.1", "ER_PREDICT_batch", wall_s, "gpu-f-3")],
        )
        monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda step_id: 2 * HOUR)
        assert watch_stalled_jobs.main() == expected_status
        output = capsys.readouterr().out
        assert ("STALLED 123.1" in output) == bool(expected_status)
    assert "was 0.67 before" in output


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
        "running_steps",
        lambda user: [("35083019_0.0", "P2P_EMA_REST_g44", 21.18 * HOUR, "gpu-b-4")],
    )
    monkeypatch.setattr(watch_stalled_jobs, "step_cpu_seconds", lambda step_id: 21.75 * HOUR)

    assert watch_stalled_jobs.main() == 0

    assert f"schema None is not {STATE_VERSION}; starting a new history" in capsys.readouterr().out
    rewritten = json.loads(state_file.read_text())
    assert rewritten["version"] == STATE_VERSION
    assert set(rewritten["states"]) == {"35083019_0.0"}


def test_load_states_rejects_another_users_file(tmp_path) -> None:
    """Same schema, different user is a real error rather than a fresh start."""
    state_file = tmp_path / "watch.json"
    _write_state_file(state_file, user="someone.else")

    with pytest.raises(ValueError, match="another user"):
        _load_states(state_file, "alex.kalinin")


def test_load_states_rejects_a_file_written_on_another_host(tmp_path) -> None:
    """``~/.cache`` is NFS with ``local_lock=all``: flock excludes only same-host processes.

    A daemon on one node and a ``--once`` check on another would interleave
    their read-modify-write of the history unguarded, so the file names the
    host that owns it.
    """
    state_file = tmp_path / "watch.json"
    _write_state_file(state_file, host="gpu-x-9")

    with pytest.raises(ValueError, match="written on gpu-x-9; flock is node-local on NFS"):
        _load_states(state_file, "alex.kalinin")


def test_state_file_round_trips_on_the_same_host(tmp_path) -> None:
    """Saving then loading on the writing host returns the same histories."""
    state = JobState(name="P2P_EMA_REST_g44", node="gpu-b-4")
    state.add(Sample(wall_s=21.18 * HOUR, cpu_s=21.75 * HOUR))
    state.add(Sample(wall_s=22.00 * HOUR, cpu_s=21.75 * HOUR))
    state_file = tmp_path / "watch.json"

    _save_states(state_file, "alex.kalinin", {"35083019_0.0": state})

    assert _load_states(state_file, "alex.kalinin") == {"35083019_0.0": state}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["watch.json"]


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
