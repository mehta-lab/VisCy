"""Detect SLURM jobs that hold an allocation without consuming CPU.

A Lightning job can finish its work and then never exit. The failure that
motivated this tool: a ``pix2pix3d`` predict wrote its last chunk, reached
``4920/4920``, and then sat on a GPU + 32 CPUs + 256 GB for 17.5 h until it
was noticed by hand. Nothing in ``squeue`` distinguishes that from healthy
compute -- the job is ``RUNNING`` either way, and its wall clock keeps
climbing.

The discriminator is CPU time. Measured across 509 allocations over 26 days,
every healthy job spent CPU seconds at >= 0.93x wall seconds; the hung one sat
at 0.561 and had spent *zero* additional CPU since the moment its work
finished. There is no overlap between the two populations.

This polls ``sstat`` per running step and flags a step when its CPU time stops
advancing while its elapsed time keeps going. Histories are per step, keyed by
the ``<jobid>.<step>`` id from ``squeue -s`` and judged against that step's own
elapsed time: a ``submit_benchmark_batch`` chain advancing to its next ``srun``
step, or a requeue, starts a fresh history instead of comparing counters across
the boundary. It only reports -- it never cancels. Killing a job
with ``afterok`` dependents strands them in ``DependencyNeverSatisfied``, so
the remediation order matters and is left to a human; the report prints it.

Usage
-----
    uv run python applications/dynacell/tools/watch_stalled_jobs.py --once
    uv run python applications/dynacell/tools/watch_stalled_jobs.py --interval 600

Exit codes: 0 = no stall detected, 1 = something is stalled, 2 = the tool
itself failed (traceback on stderr). A tool error must never read as a stall
verdict, so it gets its own code.
"""

import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Interactive sessions legitimately idle for days. Never flag them -- and never
# cancel them (see the "cancel all jobs means batch only" house rule).
INTERACTIVE_NAMES = re.compile(r"^(nomachine|gpu-hold|interactive|bash|sh|srun)$", re.IGNORECASE)

# A step must have run this long, and must have *already* proven it can burn
# CPU at this rate over a window at least this long, before a flat stretch
# counts as a stall. Without the prior-efficiency condition a job staging a
# 14 GB store off NFS at startup -- legitimately ~0% CPU -- would trip. Without
# the window length a few seconds of import-time CPU on a young step would
# qualify it, and that same staging phase would then read as a stall.
MIN_AGE_S = 1800.0
MIN_PRIOR_EFFICIENCY = 0.30

# Flat means "spent < 10% of a core over the lookback". Healthy jobs sit at
# >= 93%, so this is an order of magnitude of margin.
STALL_CPU_FRACTION = 0.10
LOOKBACK_S = 900.0

# Bump whenever the persisted payload changes shape. The state file is a
# derived cache: an old schema is discarded and the next poll rebuilds a
# baseline, instead of every run tracebacking until someone deletes the file.
STATE_VERSION = 2


def parse_slurm_duration(text: str) -> float | None:
    """Parse a Slurm duration into seconds.

    Handles the ``[DD-]HH:MM:SS[.frac]`` and ``MM:SS.frac`` forms Slurm emits.

    Parameters
    ----------
    text : str
        Duration field from ``squeue``/``sstat``, e.g. ``"1-14:47:33"``.

    Returns
    -------
    float or None
        Seconds, or None when the field carries no duration (``"UNLIMITED"``,
        ``"Unknown"``, ``"INVALID"``, empty).
    """
    text = text.strip()
    if not text or text in {"UNLIMITED", "Unknown", "INVALID", "N/A"}:
        return None
    if text.endswith("+"):
        # Truncated by a narrow --format width; always poll with -P instead.
        raise ValueError(f"truncated Slurm duration {text!r}; widen the format field")
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head)
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


@dataclass(frozen=True)
class Sample:
    """One observation of a step's elapsed and CPU clocks."""

    wall_s: float
    cpu_s: float


@dataclass
class JobState:
    """Rolling samples for one running step.

    ``sstat`` reports the running step's CPU and ``squeue -s`` that step's
    elapsed time, so both clocks belong to the step and restart together when
    a chain advances to its next ``srun`` step (a new key) or the job is
    requeued (the same key, with both counters decreasing).
    """

    name: str
    node: str
    samples: list[Sample] = field(default_factory=list)
    progress_start: Sample | None = None
    prior_efficiency: float = 0.0

    def add(self, sample: Sample) -> None:
        """Track the step's CPU progress and keep a rolling stall baseline."""
        if self.samples and (sample.wall_s < self.samples[-1].wall_s or sample.cpu_s < self.samples[-1].cpu_s):
            # A step id is reused only when the job is requeued. The new step's
            # counters cannot be compared with the old ones, so start over.
            self.samples.clear()
            self.progress_start = None
            self.prior_efficiency = 0.0
        if self.progress_start is None:
            self.progress_start = sample
        else:
            elapsed = sample.wall_s - self.progress_start.wall_s
            if elapsed >= MIN_AGE_S:
                efficiency = (sample.cpu_s - self.progress_start.cpu_s) / elapsed
                # A long idle period must not erase previously observed work.
                self.prior_efficiency = max(self.prior_efficiency, efficiency)
        if sample.wall_s >= MIN_AGE_S and all(s.wall_s < MIN_AGE_S for s in self.samples):
            # First sample taken once the step is old enough for its cumulative
            # ratio to span a real window: CPU over the step's own elapsed time
            # proves activity even when the watcher first saw the step too young
            # to judge, or mid-allocation when it was already hung. Wall time is
            # monotonic within a step and pruning keeps the newest samples, so
            # "no retained mature sample" means this is the first one.
            self.prior_efficiency = max(self.prior_efficiency, sample.cpu_s / sample.wall_s)
        self.samples.append(sample)
        # Keep the closest baseline older than the lookback even when one-shot
        # invocations are far apart; nothing reads samples older than that.
        cutoff = sample.wall_s - LOOKBACK_S
        older = [s for s in self.samples if s.wall_s < cutoff]
        self.samples = older[-1:] + [s for s in self.samples if s.wall_s >= cutoff]

    def stall_report(self) -> tuple[float, float, Sample] | None:
        """Return ``(delta_fraction, prior_efficiency, baseline)`` when stalled.

        Compares the newest sample against the closest one at least
        ``LOOKBACK_S`` older. Returns None when the step is too young, the
        window is not yet wide enough, the step never demonstrated CPU
        progress, or CPU time is still advancing.
        """
        if not self.samples:
            return None
        newest = self.samples[-1]
        if newest.wall_s < MIN_AGE_S:
            return None
        baselines = [s for s in self.samples if newest.wall_s - s.wall_s >= LOOKBACK_S]
        if not baselines:
            return None
        baseline = baselines[-1]
        if self.prior_efficiency < MIN_PRIOR_EFFICIENCY:
            return None
        wall_delta = newest.wall_s - baseline.wall_s
        cpu_delta = newest.cpu_s - baseline.cpu_s
        fraction = cpu_delta / wall_delta
        if fraction >= STALL_CPU_FRACTION:
            return None
        return fraction, self.prior_efficiency, baseline


def running_steps(user: str) -> list[tuple[str, str, float, str]]:
    """Return ``(step_id, job_name, step_wall_s, node)`` for the user's running steps.

    Names and the interactive-session filter come from the job listing: in the
    step listing ``%j`` is the step's own name (e.g. ``uv``). Only ``.extern``
    is skipped. ``.batch`` is tracked like any other step because
    ``submit_benchmark_batch --parallel`` and ``run_eval_direct.slurm`` run
    their compute directly in the batch script with no ``srun`` step; for
    ``srun``-based jobs the batch step never burns CPU, so it never qualifies.
    A step whose job is missing from the filtered job list is either
    interactive or ended between the queries.
    """
    jobs = subprocess.run(
        ["squeue", "-u", user, "-h", "-t", "RUNNING", "-o", "%i|%j"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = {}
    for line in jobs.splitlines():
        jobid, name = line.split("|")
        if not INTERACTIVE_NAMES.match(name.strip()):
            names[jobid.strip()] = name.strip()
    steps = subprocess.run(
        ["squeue", "-s", "-u", user, "-h", "-t", "RUNNING", "-o", "%i|%M|%N"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    running = []
    for line in steps.splitlines():
        step_id, elapsed, node = line.split("|")
        step_id = step_id.strip()
        jobid, _, step = step_id.rpartition(".")
        if step == "extern" or jobid not in names:
            continue
        wall_s = parse_slurm_duration(elapsed)
        if wall_s is not None:
            running.append((step_id, names[jobid], wall_s, node.strip()))
    return running


def step_cpu_seconds(step_id: str) -> float | None:
    """Return the step's ``AveCPU`` in seconds, or None when ``sstat`` has no row for it.

    Once the step has ended -- which can happen between the ``squeue`` listing
    and this call -- ``sstat`` either exits non-zero or exits 0 with only the
    header line and its error on stderr. ``-P`` is required because the
    default ``AveCPU`` width truncates long durations to ``"10-15:41:+"``.
    """
    proc = subprocess.run(
        ["sstat", "-j", step_id, "-P", "--format=JobID,AveCPU"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    rows = proc.stdout.splitlines()[1:]
    if not rows:
        return None
    # A second row for a single step id would be a scheduler contract change.
    (row,) = rows
    _, ave = row.rsplit("|", 1)
    return parse_slurm_duration(ave)


def _format_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}h"


def poll_once(user: str, states: dict[str, JobState]) -> list[str]:
    """Sample every running step and return one report line per stalled step."""
    alerts = []
    live = set()
    for step_id, name, wall_s, node in running_steps(user):
        live.add(step_id)
        cpu_s = step_cpu_seconds(step_id)
        if cpu_s is None:
            continue
        state = states.setdefault(step_id, JobState(name=name, node=node))
        state.add(Sample(wall_s=wall_s, cpu_s=cpu_s))
        verdict = state.stall_report()
        if verdict is None:
            continue
        fraction, prior, baseline = verdict
        idle_s = wall_s - baseline.wall_s
        alerts.append(
            f"STALLED {step_id} {name} on {node}: "
            f"wall {_format_hours(wall_s)}, cpu {_format_hours(cpu_s)}, "
            f"burned {fraction:.3f} core-s/s over the last {_format_hours(idle_s)} "
            f"(was {prior:.2f} before). Verify the output store is complete, then "
            f"clear dependents' Dependency= BEFORE scancel."
        )
    for step_id in set(states) - live:
        del states[step_id]
    return alerts


def _load_states(path: Path, user: str) -> dict[str, JobState]:
    """Rehydrate the persisted step histories.

    Empty when nothing was saved yet or when the file was written by another
    schema version. A file written for another user, or on another host, is an
    error: ``flock`` on this NFS mount only excludes processes on the same
    host, so the daemon and its one-shot checks must share a node.
    """
    if not path.exists():
        return {}
    with path.open() as saved:
        payload = json.load(saved)
    found = payload.get("version")
    if found != STATE_VERSION:
        print(f"state file {path}: schema {found!r} is not {STATE_VERSION}; starting a new history", flush=True)
        return {}
    if payload["user"] != user:
        raise ValueError(f"state file {path} belongs to another user")
    host = payload["host"]
    if host != socket.gethostname():
        raise ValueError(
            f"state file {path} was written on {host}; flock is node-local on NFS, so run the daemon "
            f"and its --once checks on {host} or pass a different --state-file"
        )
    states = {}
    for step_id, data in payload["states"].items():
        data["samples"] = [Sample(**sample) for sample in data["samples"]]
        if data["progress_start"] is not None:
            data["progress_start"] = Sample(**data["progress_start"])
        states[step_id] = JobState(**data)
    return states


def _save_states(path: Path, user: str, states: dict[str, JobState]) -> None:
    """Replace the persisted histories atomically."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "version": STATE_VERSION,
        "user": user,
        "host": socket.gethostname(),
        "states": {step: asdict(s) for step, s in states.items()},
    }
    with temporary.open("w") as saved:
        json.dump(payload, saved)
    temporary.replace(path)


def main() -> int:
    """Poll until interrupted, printing an alert line per stalled step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="alex.kalinin")
    parser.add_argument("--interval", type=float, default=600.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument(
        "--state-file",
        type=Path,
        help="sample history JSON (default: $XDG_CACHE_HOME/viscy/watch_stalled_jobs-USER.json)",
    )
    args = parser.parse_args()

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    state_path = args.state_file or cache_root / "viscy" / f"watch_stalled_jobs-{args.user}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Serialise the daemon and one-shot checks sharing this history. On
        # this NFS mount flock only excludes processes on the same host, so
        # the state file records its host and _load_states enforces it.
        with state_path.with_name(state_path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            states = _load_states(state_path, args.user)
            alerts = poll_once(args.user, states)
            _save_states(state_path, args.user, states)
        if alerts:
            for alert in alerts:
                print(f"[{stamp}] {alert}", flush=True)
        else:
            tracked = ", ".join(f"{j}({len(s.samples)})" for j, s in sorted(states.items()))
            collecting = any(s.samples[-1].wall_s - s.samples[0].wall_s < LOOKBACK_S for s in states.values())
            status = "collecting history" if collecting else "no stall detected"
            print(f"[{stamp}] {status}, tracking {len(states)}: {tracked}", flush=True)
        if args.once:
            return 1 if alerts else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Exit 1 is the stall verdict. A tool error must not collide with it,
        # so this is the one deliberate broad except: report and exit 2.
        traceback.print_exc()
        sys.exit(2)
