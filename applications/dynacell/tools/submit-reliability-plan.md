# Dynacell Job Submission — Reliability Plan

Plan to close 3 gaps in `applications/dynacell/tools/submit_benchmark_job.py` +
`sbatch_template.sbatch` exposed by the 2026-04-21 NCCL hangs on `gpu-d-1`
(4 sibling FCMAE jobs wasted ~1 h each on a 30-min NCCL watchdog timeout
before aborting).

**Scope (v2.4):** Gap 1 + Gap 3 only. Gap 2 (auto-requeue on NCCL watchdog
hang) is **deferred** — plan preserved below for future pickup, but not
implemented this round. Files touched this round: `submit_benchmark_job.py`,
`sbatch_template.sbatch`, new `nccl_smoke_test.py`, and the test file. No
behavior change for jobs that run cleanly.

**v2.4 — updated 2026-04-21 after deferring Gap 2.** Changes listed in
[Appendix: Review findings addressed](#appendix-review-findings-addressed).

---

## Gap 1 — `--exclude` as optional SBATCH directive

**Decision:** mirror how `constraint` is handled. `exclude` is optional,
renders as `--exclude=<value>` when set, skipped when null/absent.

**Changes to `applications/dynacell/tools/submit_benchmark_job.py`:**

```python
_SBATCH_DIRECTIVE_ORDER = (
    ("job_name", "--job-name"),
    ("time", "--time"),
    ("nodes", "--nodes"),
    ("ntasks_per_node", "--ntasks-per-node"),
    ("partition", "--partition"),
    ("cpus_per_task", "--cpus-per-task"),
    ("gpus", "--gpus"),
    ("mem", "--mem"),
    ("constraint", "--constraint"),
    ("exclude", "--exclude"),     # NEW
)

_OPTIONAL_SBATCH_DIRECTIVES = frozenset({"constraint", "exclude"})
```

Rendering already handles null via `_OPTIONAL_SBATCH_DIRECTIVES`; the
`--constraint` quoting branch stays as-is (don't quote `--exclude`, SLURM
accepts bare comma-separated hostlists).

**Usage** — either set in a profile, in the leaf, or via `--override`:

```sh
uv run python .../submit_benchmark_job.py <leaf.yml> \
    --override launcher.sbatch.exclude=gpu-d-1
```

**Tests** (extend `applications/dynacell/tests/test_submit_benchmark_job.py`):

- `exclude="gpu-d-1"` → rendered output contains `#SBATCH --exclude=gpu-d-1`
  and this line appears after `#SBATCH --constraint=...` and before
  `#SBATCH --output=...` (content-based, not positional `[-1]`).
- `exclude=None` or absent → no `--exclude` substring in rendered output.
- Use content-based assertions (`"--exclude=" in rendered`, line-search) so
  future optional directives don't break the test.

**Verify:**
`uv run pytest applications/dynacell/tests/test_submit_benchmark_job.py -k exclude`

---

## Gap 2 — auto-requeue on NCCL watchdog hang [DEFERRED]

> **DEFERRED in v2.4.** Not implementing this round. The full plan below is
> preserved as-is (four rounds of review) so it can be picked up later without
> re-derivation. Implementing Gap 1 + Gap 3 alone still closes the motivating
> failure mode: Gap 3's preflight kills bad-node jobs in ~60 s instead of
> burning a 30-min NCCL watchdog, and Gap 1 lets the operator add
> `--exclude=gpu-d-1` on the manual resubmit. The `#SBATCH --requeue`
> directive and all shell shared-vars (ERR_LOG / BAD_LIST / COUNTER_FILE /
> PREFLIGHT_MARKER / MAX_REQUEUE) are **not** added to the template in v2.4
> — they only exist to serve the requeue path below.

**Decisions:**

1. **Single-node only.** Guard on `$SLURM_JOB_NUM_NODES -eq 1`. On
   multi-node jobs, exit with the original rc — no auto-requeue. Our hardware
   profiles are all `nodes: 1` today; when/if a multi-node profile lands, the
   bad-node identification strategy must be revisited (parse the `[Rank N]`
   hostname from the watchdog stderr line).
2. **Trust `$SLURM_JOB_NODELIST`**, not `squeue`. It's set at allocation time,
   stable across state transitions, and matches a single hostname on
   single-node jobs.
3. **Counter in a file under `run_root/slurm/`**, keyed on `$SLURM_JOB_ID`.
   `SLURM_RESTART_COUNT` is unreliable across `scontrol requeue` — some Slurm
   configs don't increment it on manual requeues.
4. **Disable the EXIT trap before `scontrol requeue`.** Requeued jobs keep
   the same `$SLURM_JOB_ID`, so the existing trap's
   `rm -rf /dev/shm/$SLURM_JOB_ID` would wipe the mmap preload cache that the
   requeued run expects to reuse (`scratch_dir: /dev/shm`, `mmap_preload:
   true` in the ipsc_confocal train set).
5. **Use `scontrol requeuehold` → `update ExcNodeList` → `release`**, not
   `update` on a RUNNING job. Some Slurm versions reject `ExcNodeList` updates
   while the job is RUNNING. Requeuehold parks the job in held-pending,
   guaranteeing the update applies before scheduling.
6. **Record preflight failures in a separate marker file**, not the Slurm
   stderr. Writing to `%j.err` races with slurmstepd, and pollutes the user's
   error log.

**Changes to `applications/dynacell/tools/sbatch_template.sbatch`:**

Add `#SBATCH --requeue` as a **literal line directly in the template**,
immediately after the `@@sbatch_directives` substitution line. `--requeue` is
valueless and always-on for dynacell jobs (no per-leaf configurability), so
it belongs in the template next to the other always-on scaffolding (`umask
0002`, `ml uv`, cleanup trap) — not routed through
`_render_sbatch_directives()`, which only emits keyed directives from
`_SBATCH_DIRECTIVE_ORDER`.

Concretely, the top of the template becomes:

```bash
#!/bin/bash

@@sbatch_directives
#SBATCH --requeue
```

**No change to `submit_benchmark_job.py` for the `--requeue` directive.**
`_SBATCH_DIRECTIVE_ORDER` stays keyed-only. If we ever need another valueless
SBATCH directive, we can either add another literal line or extend the
render function — but not for this change.

Declare shared vars **near the top of the script body** (before the preflight
block from Gap 3) so Gap 3 and Gap 2 can reference them:

```bash
ERR_LOG=@@run_root/slurm/${SLURM_JOB_ID}.err
BAD_LIST=@@run_root/slurm/${SLURM_JOB_ID}.bad_nodes
COUNTER_FILE=@@run_root/slurm/${SLURM_JOB_ID}.requeue_count
PREFLIGHT_MARKER=@@run_root/slurm/${SLURM_JOB_ID}.preflight_failed
MAX_REQUEUE=3

mkdir -p @@run_root/slurm
```

Replace the tail `srun ...` with (order: preflight → main srun → requeue
decision):

```bash
# --- (Gap 3 preflight block goes here, may set PREFLIGHT_MARKER + SRUN_RC=134) ---

if [ ! -f "$PREFLIGHT_MARKER" ]; then
    srun uv run python -m dynacell @@mode --config @@resolved_config
    SRUN_RC=$?
fi

# Requeue decision. Scope: single-node, watchdog signature matched, under cap.
REQUEUE_COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)
SHOULD_REQUEUE=0
if [ "$SRUN_RC" -ne 0 ] && [ "$REQUEUE_COUNT" -lt "$MAX_REQUEUE" ] \
   && [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    if [ -f "$PREFLIGHT_MARKER" ] \
       || grep -q "Watchdog caught collective operation timeout" "$ERR_LOG" 2>/dev/null; then
        SHOULD_REQUEUE=1
    fi
fi

if [ "$SHOULD_REQUEUE" -eq 1 ]; then
    BAD_NODE=$SLURM_JOB_NODELIST
    echo "$BAD_NODE" >> "$BAD_LIST"
    NEW_EXCL=$(sort -u "$BAD_LIST" | paste -sd,)
    echo $((REQUEUE_COUNT + 1)) > "$COUNTER_FILE"
    echo "[auto-requeue] NCCL hang on $BAD_NODE; requeue $((REQUEUE_COUNT+1))/$MAX_REQUEUE, excl: $NEW_EXCL" >&2
    # Clear preflight marker so the requeued job re-runs preflight.
    rm -f "$PREFLIGHT_MARKER"
    # Disable cleanup trap so /dev/shm/$SLURM_JOB_ID (mmap cache) survives
    # the requeued run (same SLURM_JOB_ID reused on `scontrol requeue`).
    trap - EXIT

    # Explicit state machine for the Slurm mutation sequence. Requirements:
    #   1. If requeuehold succeeds, ALWAYS attempt release before exiting,
    #      else the job stays stuck in held-pending forever.
    #   2. Only exit 0 when all three steps succeed (requeue actually took
    #      effect with the exclusion applied).
    #   3. On any failure, fall through to cleanup + exit $SRUN_RC so the
    #      failure stays visible in Slurm accounting.
    REQUEUE_OK=0
    if scontrol requeuehold "$SLURM_JOB_ID"; then
        if scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$NEW_EXCL" \
           && scontrol release "$SLURM_JOB_ID"; then
            REQUEUE_OK=1
        else
            # requeuehold succeeded but update or release failed — do NOT leave
            # the job held. Best-effort release; warn if even that fails so the
            # operator can manually `scontrol release $JID`.
            scontrol release "$SLURM_JOB_ID" \
                || echo "[auto-requeue] WARNING: job remains held; run: scontrol release $SLURM_JOB_ID" >&2
        fi
    fi

    if [ "$REQUEUE_OK" -eq 1 ]; then
        exit 0
    fi
    echo "[auto-requeue] scontrol sequence failed; exiting with rc=$SRUN_RC so failure stays visible" >&2
    # Trap was disabled above; run cleanup manually before exiting so
    # /tmp/$SLURM_JOB_ID and /dev/shm/$SLURM_JOB_ID don't leak on the
    # non-requeued failure path.
    cleanup
fi

exit $SRUN_RC
```

**Assumption to verify before shipping Gap 2:** on our Slurm install,
`scontrol requeuehold` + `update ExcNodeList` + `release` reliably applies
the exclusion to the next run. Probe once with a dummy job before enabling
this path. If the probe fails, fall back to appending to `bad_nodes.log` only
and require operator resubmit (Gap 1 unblocks this fallback).

**Verify:** dry-render the template (`--print-script` on a known leaf), diff
against the previous version; ensure the main srun line, cleanup trap, and
`@@resolved_config` substitution are intact.

---

## Gap 3 — pre-flight NCCL smoke test

**Decisions:**

1. **Absolute path to the smoke test**, injected via a new `@@repo_root`
   template substitution. The rendered sbatch does not `cd` and sbatch is
   submitted from arbitrary CWDs — relative paths break.
2. **Map Slurm env vars to torch.distributed env vars in the Python script**,
   not the shell. Slurm sets `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`;
   `torch.distributed.init_process_group(backend="nccl", init_method="env://")`
   needs `RANK`, `WORLD_SIZE`, `LOCAL_RANK`. Doing the mapping in Python keeps
   the template slim and testable.
3. **Per-job deterministic `MASTER_PORT`** derived from `$SLURM_JOB_ID`, same
   pattern Lightning's `SLURMEnvironment` uses — avoids collisions when two
   jobs share a host.
4. **Unset `MASTER_ADDR`/`MASTER_PORT` before the main srun** so Lightning's
   SLURMEnvironment can set them itself without inheriting preflight values.
5. **Preflight failure exits the script immediately** (v2.4). With Gap 2
   deferred, there is no marker file and no auto-requeue — the job fails with
   the smoke-test's non-zero rc, and the operator resubmits manually (using
   `--override launcher.sbatch.exclude=...` from Gap 1 if needed). If/when
   Gap 2 is picked up, swap `exit $SMOKE_RC` for `touch "$PREFLIGHT_MARKER";
   SRUN_RC=$SMOKE_RC` so the requeue decision block downstream can handle it.

**Submitter changes** — `applications/dynacell/tools/submit_benchmark_job.py`
around the current `SbatchTemplate(...).substitute(...)` call:

```python
repo_root = Path(__file__).resolve().parents[3]  # VisCy/

rendered = SbatchTemplate(template_text).substitute(
    sbatch_directives=_render_sbatch_directives(job_name, str(run_root), sbatch),
    run_root=str(run_root),
    env_block=_render_env_block(env),
    mode=mode,
    resolved_config=str(resolved_path),
    repo_root=str(repo_root),     # NEW
)
```

**New file `applications/dynacell/tools/nccl_smoke_test.py`:**

```python
"""60-second NCCL all-reduce smoke test for dynacell training preflight.

Maps Slurm's per-task env vars (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID) to
the env:// init variables torch.distributed expects (RANK, WORLD_SIZE,
LOCAL_RANK), then runs a single all_reduce + barrier with a 60-second init
timeout. Exits non-zero (via unhandled RuntimeError, which is the intended
signal per project "prefer raising errors" policy) on hang or any NCCL error.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> int:
    """Initialize NCCL, all_reduce a ones tensor, barrier, and exit."""
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
    os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"[nccl-smoke] OK world_size={dist.get_world_size()} sum={t.item()}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Template preflight block** — inserted **before** the existing
`srun uv run python -m dynacell ...` line (no shared-vars prerequisite in
v2.4 since Gap 2 is deferred):

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

echo "[preflight] NCCL smoke test..."
srun --kill-on-bad-exit=1 uv run python @@repo_root/applications/dynacell/tools/nccl_smoke_test.py
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "[preflight] smoke test FAILED on $SLURM_JOB_NODELIST (rc=$SMOKE_RC); exiting before main srun" >&2
    exit $SMOKE_RC
fi

# Lightning's SLURMEnvironment picks its own port from SLURM_JOB_ID; don't
# leak the preflight values into the training srun.
unset MASTER_ADDR MASTER_PORT
```

`exit $SMOKE_RC` fires the cleanup trap (unchanged from the current
template), so `/tmp/$SLURM_JOB_ID` and `/dev/shm/$SLURM_JOB_ID` are cleaned
up on preflight failure just like any other exit path.

**Verify:** allocate an interactive session first so `SLURM_JOB_ID` is set
(lets us derive `MASTER_PORT` the same way the template does — no
hardcoded port):

```sh
salloc --nodes=1 --ntasks=4 --gpus=4 --time=00:10:00
# inside the allocation:
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))
srun uv run python /hpc/mydata/alex.kalinin/VisCy/applications/dynacell/tools/nccl_smoke_test.py
```

Should print `[nccl-smoke] OK world_size=4 sum=4.0` in <10 s. Without
`MASTER_ADDR`/`MASTER_PORT`, `torch.distributed.init_process_group` with
`init_method="env://"` (the default) will error out — the template exports
them, so this manual command must too. The derive-from-`SLURM_JOB_ID` port
formula matches the template exactly.

---

## Tests — expanded scope

No existing test needs rewriting in v2.4. With Gap 2 deferred, the main
dynacell `srun ...` line remains the last line of the rendered template
(Gap 3's preflight srun is inserted *before* it), so
`test_rendered_sbatch_has_srun_at_expected_resolved_path` keeps passing
unchanged. `test_render_sbatch_directives_matches_dihan_order` is also
unaffected (exclude only appears when set; default leaves leave it absent).

New tests to add in
`applications/dynacell/tests/test_submit_benchmark_job.py`:

- `test_rendered_sbatch_has_preflight_srun_absolute_path` — preflight
  `srun ...nccl_smoke_test.py` line references an absolute path rooted at
  the repo (no bare `applications/...`).
- `test_exclude_directive_rendered_when_set` — sbatch dict with
  `exclude="gpu-d-1"` produces `#SBATCH --exclude=gpu-d-1`.
- `test_exclude_directive_skipped_when_absent` — absent or `None` → no
  `--exclude` substring.
- `test_repo_root_substituted_in_preflight_path` — rendered preflight srun
  contains the absolute repo-root path (assert starts with `/` and ends with
  `/applications/dynacell/tools/nccl_smoke_test.py`).
- `test_preflight_failure_exits_before_main_srun` — rendered template
  contains `exit $SMOKE_RC` on the preflight-failure branch, ahead of the
  main dynacell srun.

---

## Order of execution & risk

1. **Gap 1 first, stand-alone commit.** Lowest risk. Unblocks today's
   resubmits via `--override launcher.sbatch.exclude=...`. Tests: 2 new cases.
2. **Gap 3 second** (smoke test file + template preflight wrapper + submitter
   `repo_root` substitution). Moderate risk: template change, new Python
   module. Verify: dry-render + diff rendered sbatch; run the smoke test on
   a 4-GPU interactive allocation before enabling preflight for benchmark
   submissions.
3. **Gap 2 deferred.** Not shipped this round. See the Gap 2 section for the
   full design; pick up later after a Slurm `requeuehold/update/release`
   probe confirms the sequence works on our cluster.

---

## Out of scope

- **Gap 2 (auto-requeue on NCCL watchdog hang) — deferred in v2.4.** Plan
  preserved above for pickup after a Slurm `requeuehold/update/release`
  probe. Gap 1 + Gap 3 together close the motivating GPU-hour loss (bad
  nodes fail fast in preflight; operator resubmits with `--exclude`), so
  Gap 2's incremental value is automation-only, not correctness.
- Pattern divergence with
  `applications/dynaclr/configs/training/slurm/train.sh` (the 4th gap from
  the original review). Fix here stays dynacell-local; revisit
  cross-application unification separately if the same NCCL-hang failure
  mode appears in dynaclr runs.
- Multi-node auto-requeue. Explicitly guarded off by
  `SLURM_JOB_NUM_NODES -eq 1` in the deferred Gap 2 design. Revisit when a
  multi-node hardware profile lands.

---

## Appendix: Review findings addressed

Plan stress-tested by 3 internal subagents (correctness, regression, merge
compatibility) + external review. Changes from v1:

### Blocking (addressed)

- **Smoke-test script path was repo-relative.** v1 ran
  `uv run python applications/dynacell/tools/nccl_smoke_test.py` under a
  sbatch script that never `cd`'s. v2: inject `@@repo_root` substitution in
  submitter; template uses absolute path.
- **Smoke test missed `RANK`/`WORLD_SIZE`.** v1's `init_process_group` relied
  on `env://` default but only exported `MASTER_ADDR`/`MASTER_PORT`. Slurm
  sets `SLURM_PROCID`/`SLURM_NTASKS`, not `RANK`/`WORLD_SIZE`. v2: map in
  Python via `os.environ.setdefault`.
- **Existing test asserts the tail is a single srun line.** v1 only planned
  `exclude` test additions. v2: explicit rewrite of
  `test_rendered_sbatch_has_srun_at_expected_resolved_path` to content-based
  assertion, plus 5 new tests for `--requeue`, preflight, requeue block,
  exclude on/off, and repo-root substitution.

### Important (addressed)

- **`squeue -o %N` for multi-node jobs.** v2 guards on
  `SLURM_JOB_NUM_NODES -eq 1` and uses `$SLURM_JOB_NODELIST` directly.
- **`SLURM_RESTART_COUNT` unreliable on manual requeue.** v2 uses a
  file-backed counter keyed on `SLURM_JOB_ID`.
- **`scontrol update ExcNodeList` may be rejected on RUNNING jobs.** v2 uses
  `requeuehold` → `update` → `release` sequence.
- **Cleanup trap wipes `/dev/shm/$SLURM_JOB_ID` on manual requeue** (same
  jobid reused → mmap cache lost). v2: `trap - EXIT` before `scontrol requeue`.
- **Preflight failure writing synthetic marker into `$ERR_LOG`** races with
  slurmstepd + pollutes user's error log. v2: separate `PREFLIGHT_MARKER`
  file, Gap 2 checks both marker and stderr grep.
- **`MASTER_PORT=29500` hardcoded.** v2: derive from `SLURM_JOB_ID` (same
  pattern as Lightning's `SLURMEnvironment`).
- **`MASTER_PORT` leaking to main srun** would override Lightning's
  auto-picked port. v2: `unset MASTER_ADDR MASTER_PORT` after preflight.
- **Variable ordering (Gap 2 vars referenced by Gap 3 preflight).** v2:
  declare `ERR_LOG`/`PREFLIGHT_MARKER`/`COUNTER_FILE`/`BAD_LIST` in a shared
  top-of-script block before both preflight and main srun.

### Additional review pass (external, after v2 draft) → v2.1

- **Requeue path was best-effort with unconditional `exit 0`.** Any failure
  in the `scontrol requeuehold/update/release` trio would silently mask a
  real failed training run as a green job, or requeue without the exclusion
  actually applied. v2.1: chained `if scontrol ... && scontrol ... && scontrol
  ... ; then exit 0 ; fi` — on any scontrol failure, fall through to
  `exit $SRUN_RC` so the job stays visibly failed in Slurm accounting.
- **Standalone smoke-test verify command was incomplete.** It relied on
  `env://` init but didn't export `MASTER_ADDR`/`MASTER_PORT`, and used a
  repo-relative script path. v2.1: verify command now exports both env vars
  and invokes the script by absolute path, matching the rendered template.

### Scope change → v2.4

- **Gap 2 deferred.** User decision to ship Gap 1 + Gap 3 only this round.
  Gap 2 section preserved intact (four rounds of review still valid for
  future pickup), but marked `[DEFERRED]`.
- **Gap 3 preflight simplified** from "touch marker and let Gap 2 handle it"
  to fail-fast `exit $SMOKE_RC`. Cleanup trap handles tmp/shm on the exit
  path — no leak.
- **Shared-vars block dropped.** `ERR_LOG` / `BAD_LIST` / `COUNTER_FILE` /
  `PREFLIGHT_MARKER` / `MAX_REQUEUE` only existed to serve the deferred
  requeue decision block.
- **`#SBATCH --requeue` dropped.** It was paired with Gap 2's manual requeue;
  without that, it only covers Slurm-side NODE_FAIL / preempt, which is out
  of the motivating scope (NCCL hangs don't trigger `--requeue`).
- **Two planned tests dropped.** `test_rendered_sbatch_has_requeue_directive`
  and `test_rendered_sbatch_has_requeue_block` — both exercised Gap 2
  artifacts that no longer render.
- **Existing test rewrite dropped.**
  `test_rendered_sbatch_has_srun_at_expected_resolved_path` uses
  `rendered.splitlines()[-1]`, which still works: Gap 3's preflight srun is
  inserted *before* the main dynacell srun, so the dynacell srun remains the
  last line of the rendered template.

### Fourth external review pass → v2.3

- **`#SBATCH --requeue` emission path was underspecified.** v2.2 said the
  directive "flows through `@@sbatch_directives` substitution — handled at
  render time", but `_render_sbatch_directives()` only emits keyed directives
  from `_SBATCH_DIRECTIVE_ORDER` (key + value pairs) plus `--output`/`--error`
  — there is no code path for a bare valueless directive. Implementing the
  plan as written would have produced a rendered sbatch with no `--requeue`
  line, silently disabling the Slurm-side auto-requeue Gap 2 depends on.
  v2.3: `#SBATCH --requeue` is a literal line in `sbatch_template.sbatch`
  directly after `@@sbatch_directives` (same pattern as `umask 0002`, `ml
  uv`, cleanup trap — always-on scaffolding belongs in the template). No
  submitter change; `_SBATCH_DIRECTIVE_ORDER` stays keyed-only. Test
  `test_rendered_sbatch_has_requeue_directive` is unaffected (content-based).

### Third external review pass → v2.2

- **Held-pending leak on partial scontrol failure.** v2.1's `&&` chain
  (`requeuehold && update && release`) short-circuits: if `requeuehold`
  succeeds but `update` fails, `release` never runs and the job stays stuck
  in held-pending. v2.2: explicit state machine — if `requeuehold` returns
  0, always attempt `release` before exiting, even on subsequent failures;
  warn the operator if `release` itself fails (manual recovery needed).
  `REQUEUE_OK=1` gates the `exit 0` so a successful-hold-but-failed-update
  path can't report green.
- **Cleanup leaked on non-requeued failure path.** `trap - EXIT` was
  disabled unconditionally before scontrol, so on any scontrol failure the
  job exited `$SRUN_RC` without running cleanup, leaking
  `/tmp/$SLURM_JOB_ID` and `/dev/shm/$SLURM_JOB_ID`. v2.2: call `cleanup`
  explicitly before `exit $SRUN_RC` on the scontrol-failure branch. The
  normal-failure branch (SHOULD_REQUEUE=0) still has the EXIT trap intact,
  so cleanup runs there unchanged.
- **Manual verify command hardcoded `MASTER_PORT=29500`.** v2.2: switched to
  `salloc` + `export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))`,
  matching the template's per-job-id derivation. No fixed-port footguns.

### Nitpicks (not adopted)

- Gratuitous `set -e`/`set +e` toggles — v2 avoids them entirely by letting
  the original un-errexit'd shell semantics hold. Strict error checking is
  scoped only to the scontrol trio (v2.1), where it matters for correctness.
- Grep regex brittleness for watchdog signature — current `"Watchdog caught
  collective operation timeout"` is the stable c10d message; accept
  brittleness to PyTorch upgrades (noisy upgrade will be caught by
  preflight, not the post-hoc grep).
- `BAD_LIST` collision across leaves — each leaf's `run_root` is unique;
  scope by `$SLURM_JOB_ID` in the filename belt-and-suspenders.

### Positive findings (no action)

- No merge conflicts with `main` or other open PRs; all three files this
  plan touches are only edited on `dynacell-models` and only by prior
  commits, not in-flight work.
- Hardware profile YAMLs don't need parallel edits — `exclude` flows through
  `--override launcher.sbatch.exclude=...` without requiring a profile
  change.
