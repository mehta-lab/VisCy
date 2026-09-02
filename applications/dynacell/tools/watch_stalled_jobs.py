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

This polls ``sstat`` and flags a job when its CPU time stops advancing while
its wall clock keeps going. It only reports -- it never cancels. Killing a job
with ``afterok`` dependents strands them in ``DependencyNeverSatisfied``, so
the remediation order matters and is left to a human; the report prints it.

Usage
-----
    uv run python applications/dynacell/tools/watch_stalled_jobs.py --once
    uv run python applications/dynacell/tools/watch_stalled_jobs.py --interval 600
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

# Interactive sessions legitimately idle for days. Never flag them -- and never
# cancel them (see the "cancel all jobs means batch only" house rule).
INTERACTIVE_NAMES = re.compile(r"^(nomachine|gpu-hold|interactive|bash|sh|srun)$", re.IGNORECASE)

# A job must have run this long, and must have *already* proven it can burn CPU,
# before a flat stretch counts as a stall. Without the second condition a job
# staging a 14 GB store off NFS at startup -- legitimately ~0% CPU -- would trip.
MIN_AGE_S = 1800.0
MIN_PRIOR_EFFICIENCY = 0.30

# Flat means "spent < 10% of a core over the lookback". Healthy jobs sit at
# >= 93%, so this is an order of magnitude of margin.
STALL_CPU_FRACTION = 0.10
LOOKBACK_S = 900.0


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
    """One observation of a job's wall and CPU clocks."""

    wall_s: float
    cpu_s: float


@dataclass
class JobState:
    """Rolling samples for one job."""

    name: str
    node: str
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        """Append a sample, dropping ones older than the lookback window."""
        self.samples.append(sample)
        cutoff = sample.wall_s - LOOKBACK_S * 4
        self.samples = [s for s in self.samples if s.wall_s >= cutoff]

    def stall_report(self) -> tuple[float, float, Sample] | None:
        """Return ``(delta_fraction, prior_efficiency, baseline)`` when stalled.

        Compares the newest sample against the oldest one at least
        ``LOOKBACK_S`` older. Returns None when the job is too young, the
        window is not yet wide enough, the job never demonstrated CPU
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
        prior_efficiency = baseline.cpu_s / baseline.wall_s if baseline.wall_s else 0.0
        if prior_efficiency < MIN_PRIOR_EFFICIENCY:
            return None
        wall_delta = newest.wall_s - baseline.wall_s
        cpu_delta = newest.cpu_s - baseline.cpu_s
        if cpu_delta < 0:
            # sstat reports only the RUNNING step, so AveCPU resets to ~0 at
            # every step boundary -- and submit_benchmark_batch renders N
            # sequential srun steps per allocation. A negative delta means the
            # counter restarted, not that the job stopped burning CPU; treating
            # it as a stall would flag a healthy multi-step predict (and exit 1
            # under --once). Drop the stale baseline and wait for two samples
            # inside the current step.
            self.samples = [newest]
            return None
        fraction = cpu_delta / wall_delta
        if fraction >= STALL_CPU_FRACTION:
            return None
        return fraction, prior_efficiency, baseline


def running_jobs(user: str) -> list[tuple[str, str, float, str]]:
    """Return ``(jobid, name, wall_s, node)`` for the user's running jobs."""
    out = subprocess.run(
        ["squeue", "-u", user, "-h", "-t", "RUNNING", "-o", "%i|%j|%M|%N"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    jobs = []
    for line in out.splitlines():
        jobid, name, elapsed, node = line.split("|")
        if INTERACTIVE_NAMES.match(name.strip()):
            continue
        wall_s = parse_slurm_duration(elapsed)
        if wall_s is not None:
            jobs.append((jobid.strip(), name.strip(), wall_s, node.strip()))
    return jobs


def step_cpu_seconds(jobid: str) -> float | None:
    """Return the largest per-step ``AveCPU`` for a running job, in seconds.

    The compute step is the one that matters; ``.batch`` and ``.extern`` sit at
    zero. ``-P`` is required because the default ``AveCPU`` width truncates
    long durations to ``"10-15:41:+"``.
    """
    proc = subprocess.run(
        ["sstat", "-j", jobid, "-a", "-P", "--format=JobID,AveCPU"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    best = None
    for line in proc.stdout.splitlines()[1:]:
        if "|" not in line:
            continue
        _, ave = line.rsplit("|", 1)
        seconds = parse_slurm_duration(ave)
        if seconds is not None and (best is None or seconds > best):
            best = seconds
    return best


def _format_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}h"


def poll_once(user: str, states: dict[str, JobState]) -> list[str]:
    """Sample every running job and return one report line per stalled job."""
    alerts = []
    live = set()
    for jobid, name, wall_s, node in running_jobs(user):
        live.add(jobid)
        cpu_s = step_cpu_seconds(jobid)
        if cpu_s is None:
            continue
        state = states.setdefault(jobid, JobState(name=name, node=node))
        state.add(Sample(wall_s=wall_s, cpu_s=cpu_s))
        verdict = state.stall_report()
        if verdict is None:
            continue
        fraction, prior, baseline = verdict
        idle_s = wall_s - baseline.wall_s
        alerts.append(
            f"STALLED {jobid} {name} on {node}: "
            f"wall {_format_hours(wall_s)}, cpu {_format_hours(cpu_s)}, "
            f"burned {fraction:.3f} core-s/s over the last {_format_hours(idle_s)} "
            f"(was {prior:.2f} before). Verify the output store is complete, then "
            f"clear dependents' Dependency= BEFORE scancel."
        )
    for jobid in set(states) - live:
        del states[jobid]
    return alerts


def main() -> int:
    """Poll until interrupted, printing an alert line per stalled job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="alex.kalinin")
    parser.add_argument("--interval", type=float, default=600.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    args = parser.parse_args()

    states: dict[str, JobState] = {}
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        alerts = poll_once(args.user, states)
        if alerts:
            for alert in alerts:
                print(f"[{stamp}] {alert}", flush=True)
        else:
            tracked = ", ".join(f"{j}({len(s.samples)})" for j, s in sorted(states.items()))
            print(f"[{stamp}] ok, tracking {len(states)}: {tracked}", flush=True)
        if args.once:
            return 1 if alerts else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
