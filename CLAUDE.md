# VisCy — Claude Code Reference

## Project

VisCy is a **uv workspace monorepo** for virtual staining and computational microscopy. Sub-packages live under `packages/`.

## Packages vs Applications

- **Shared code belongs in `packages/`**, not in applications.
- **Applications must not import from each other.** If two applications need the same logic, move it to an existing package or create a new one.
- Applications are consumers of packages — the dependency graph always flows `applications/ → packages/`, never sideways.

---

## Development

### Environment Setup

Use `uv` package manager. Run commands with `uv run <command>`. Edit `pyproject.toml` to modify dependencies and sync to update `uv.lock`.

On HPC, symlink the uv cache out of your home directory first:
```sh
mkdir -p /hpc/mydata/firstname.lastname/.cache/uv && ln -s /hpc/mydata/firstname.lastname/.cache/uv ~/.cache/uv
```

For full setup instructions (installing uv, creating a venv, syncing dependencies), see [CONTRIBUTING.md](./CONTRIBUTING.md).

### SLURM scripts for Lightning DDP jobs

When hand-writing `.slurm` scripts that launch Lightning via `srun`, always use `--ntasks-per-node=N` (not `--ntasks=N`). Lightning's `SLURMEnvironment` validates `SLURM_NTASKS_PER_NODE` at trainer init and raises `RuntimeError: You set --ntasks=N in your SLURM bash script, but this variable is not supported. HINT: Use --ntasks-per-node=N instead.` — the job then dies seconds into the allocation.

Invariant: `#SBATCH --ntasks-per-node=N` must equal `trainer.devices` in the YAML config and `#SBATCH --gpus=N` (single-node) or `#SBATCH --gpus-per-node=N` (multi-node).

The dynacell launcher (`applications/dynacell/tools/submit_benchmark_job.py`) already emits `--ntasks-per-node` correctly; this note is for hand-written scripts (e.g., `applications/cytoland/examples/configs/*/run_*.slurm`).

### Job monitoring and inspection

**Process state ≠ training completeness.** Wandb's `state: finished` only means `wandb.finish()` was called — Lightning calls it on clean SIGTERM teardown via `SLURMEnvironment`, so a `scancel`'d run shows `finished` identically to one that hit `max_epochs`. Always cross-check.

**Liveness check (is the job alive *right now*?):** `wandb.Api().run(...).heartbeatAt` is authoritative. Do not infer liveness from `last.ckpt` mtime, internal step counter, or a single `nvidia-smi` snapshot.

**Completeness check (did the job finish its goal?):** combine three sources, all required:
1. `sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,TimeLimit` — `CANCELLED+` with `ExitCode 0:0` is the signature of a user `scancel`; `TIMEOUT` is wall-time hit; `COMPLETED` with ExitCode 0:0 is the only unambiguous success.
2. The resolved fit YAML at `/hpc/projects/comp.micro/virtual_staining/models/dynacell/.../resolved/fit_*_<timestamp>.yml` — read `trainer.max_epochs` / `trainer.max_steps`. The wandb `r.config` dict only stores model init args, **not trainer args**.
3. Wandb run summary — compare final `epoch` to `trainer.max_epochs`. Final epoch of e.g. `135/200` is killed mid-training, not done; `200/200` is the only credible success indicator.

The `output.log` / `wandb-output.log` for the run will contain `Received SIGTERM: 15` if Lightning's signal handler caught a scancel — a useful confirmation when sacct is ambiguous.

**Before cancelling jobs:** the job name in `squeue` is not a complete description. `FCMAE_VSCyto3D_Pretrained_A549_Membrane` could be a fit run OR a predict run — they share the trained-model directory naming. Verify the actual purpose via:
- The `Comment` field (`squeue -j <id> -o "%k"`) if the launcher set one
- The resolved YAML path (fit vs predict subdirectory)
- The wandb run config for that job ID

When the user says "cancel all jobs," scope it to **batch jobs only**, never the interactive nomachine session. Read job names carefully — a job that has been alive for >24 h on a multi-GPU allocation is almost certainly training, not a predict run that should be ~hours.

**Subagent prompts for job status:** ask for completeness vs config, not just liveness. A prompt like "check the liveness of wandb run X" returns `state: finished` for a SIGTERM'd run and reads as success. Phrase it as "is run X complete relative to its configured `max_epochs`, and what was the exit reason (clean finish, scancel, OOM, timeout, exception)?"

### Hung-but-allocated jobs

**A job can finish its work and never exit.** Job `35083019_0` (a `pix2pix3d` predict) wrote its last chunk at 2026-07-30 19:20, then held an A6000 + 32 CPUs + 256 GB for **17.5 h** doing nothing. `squeue` shows `RUNNING` with a climbing wall clock — indistinguishable from healthy compute — and the rich progress bar only reaches `.out` at exit, so the log looks normal too. Measured rate: 1 in 509 allocations over 26 days.

**The discriminator is CPU time, and only its derivative.** Across those 509 allocations every healthy job spent CPU at >= 0.93x wall; the hung one sat at 0.561 with a *zero* incremental rate. Absolute ratio alone is useless (a 16-CPU fit runs at ~7.8x wall), so compare two samples: `sstat -j <id> -a -P --format=JobID,AveCPU` — `-P` is mandatory, the default width truncates to `10-15:41:+` and misparses.

**Using the stall watchdog.** `applications/dynacell/tools/watch_stalled_jobs.py` automates exactly that comparison. Start it whenever a campaign has long jobs in flight and leave it running:

```sh
# one-shot check: exit 0 = all healthy, exit 1 = something is stalled
uv run --no-sync python applications/dynacell/tools/watch_stalled_jobs.py --once

# continuous, 10-min poll (launch in the background; it runs until killed)
uv run --no-sync python applications/dynacell/tools/watch_stalled_jobs.py --interval 600
```

Quiet polls print `ok, tracking N: <jobid>(<samples>) ...`; a stalled job prints one `STALLED <jobid> <name> on <node>: wall Xh, cpu Yh, burned Z core-s/s over the last Wh` line per poll. Reading it:

- It **only reports — it never cancels.** Cancelling a job with `afterok` dependents strands them, so follow the kill order above by hand.
- **Interactive sessions are excluded by name** (`nomachine`, `gpu-hold`, `interactive`, bare `bash`/`sh`/`srun`). A renamed interactive session would get flagged — it still would not be cancelled, but don't act on the alert without checking.
- It needs **two samples >= 15 min apart** and a job **>= 30 min old**, so expect no verdict on a fresh job for the first couple of polls.
- A job that has never burned CPU is never flagged: startup NFS staging is legitimately ~0% CPU, so the check requires the job to have previously demonstrated CPU progress.
- `--user` defaults to `alex.kalinin`; `sstat` only works on your own running jobs, so it cannot watch someone else's.

**Predict wall limits are per family, sized from measurement** (`launcher_profiles/`):

| Profile | `time` | Basis |
|---|---|---|
| `hardware_predict_any_gpu.yml` | 2 days | longest single-pass predict measured 21.9 h |
| `hardware_predict_celldiff.yml` | 7 days | CELL-Diff runs 5.8-95.6 h; 8 predicts TIMEOUTed at the old 4-day cap on 2026-07-19 |
| `hardware_h200_single.yml` | 4 days | **shared with 40 fit leaves** — do not re-tune from predict data |

A new slow family gets its **own profile**; raising a shared cap to cover it makes the cap meaningless. `test_predict_leaf_wall_limit_matches_its_family` pins all 421 predict leaves to this table, and `generate_hek_predict_configs.py` picks the profile from `_HARDWARE_PROFILE` so regeneration cannot revert it. Fits time out at 4 days too (4 FCMAE joint fits on 2026-07-10/12) — a separate, still-open issue.

**Two traps when killing one:**
- **Lightning swallows SIGTERM.** `signal_connector.py` installs a handler that logs `Received SIGTERM` / `Bypassing SIGTERM` and sets a flag — it does not exit. `scancel` alone cannot stop a Lightning process outside its training loop; it dies on the KILL escalation (`ExitCode 0:9`), which took ~6 min here.
- **Clear dependents' `Dependency=` BEFORE `scancel`**, or `afterok` chains land in `DependencyNeverSatisfied` permanently. `scontrol update JobId=<dep> Dependency=` to drop it, or `Dependency=afterok:<other>` to re-point the chain.

**Forensics — capture this BEFORE cancelling, or the cause is unknowable:**
1. `grep State /proc/<pid>/status` and `cat /proc/<pid>/task/*/wchan` (both readable without root, `ptrace_scope=0` here). `State: D` names an uninterruptible storage/driver call; `S` means a Python-level block.
2. `py-spy dump --pid <pid>` (`uv tool install py-spy`) gives the full Python stack. If it errors `Failed to find python version from target process`, that is *itself* the answer: the process already reached interpreter finalization, so `trainer.predict()` returned and the writer's `plate.close()` completed.
3. **Do not rely on SIGABRT + `PYTHONFAULTHANDLER=1`** — verified to produce no dump once the interpreter is finalizing.

**Known unbounded wait in the predict teardown path:** zarr 3.2.1 `core/sync.py:89` registers `cleanup_resources` via `atexit`, which calls `_executor.shutdown(wait=True)` with no timeout — one stuck `zarr_pool` thread blocks interpreter exit forever at 0% CPU. (The same function caps its io-thread join at `timeout=0.2` "to avoid hanging"; the executor shutdown was left unbounded.) That is the proximate frame, but not the whole story here: a pure Python join still lets the SIGTERM handler run and log, and job `35083019_0` never logged it, so the terminal block was a C-level uninterruptible call underneath.

### Joint vs single-set training batch semantics

`HCSDataModule` and `BatchedConcatDataModule` produce the same number of GPU samples per training step — but the YAML `batch_size` value that gets there is **different by a factor of `num_samples`**. Easy to misread either by skimming.

| DataModule | `train_dataloader` divides by `num_samples`? | Samples per step |
|---|---|---|
| `HCSDataModule` (single-set) | yes (`hcs.py` `train_dataloader`) | `batch_size` |
| `ConcatDataModule` (parent class) | yes (`combined.py` `train_dataloader`) | `batch_size` |
| `BatchedConcatDataModule` (joint) | **no** (`combined.py` overrides; uses `batch_size` as-is) | `batch_size * num_samples` |

To match the same effective per-step samples between a single-set and a joint config, **set `joint.batch_size = single_set.batch_size / num_samples`**.

Examples (verified against the `applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/model/data_overlays/` overlays + their joint leaves):

- FCMAE (`fcmae_vscyto3d_*`): single-set `batch_size: 32, num_samples: 4` → joint `batch_size: 8, num_samples: 4` → both yield **32 samples/step**.
- FNet3D (`fnet3d_paper`): single-set `batch_size: 48, num_samples: 8` → joint `batch_size: 6, num_samples: 8` → both yield **48 samples/step**.

`HCSDataModule._train_transform` enforces `batch_size % num_samples == 0` for single-set use because `train_dataloader` would otherwise round down silently. The check is suppressed for `BatchedConcatDataModule` children via the `_is_batched_concat_child` flag set in the wrapper's `setup()` — joint configs are free to pick any `(batch_size, num_samples)` pair as long as the product is the desired sample count. **Do not** "fix" a joint config by raising `batch_size` to satisfy the divisibility rule; it would multiply effective samples by `num_samples`.

When in doubt, read both `train_dataloader` overrides directly — they are short. Don't infer from comments alone.

### Testing

Prefer `{file}_test.py` in the same directory as `{file}.py`, unless there are import issues, in which case use `tests/`.

---

## Project Conventions

- Ruff config is centralized in the root `pyproject.toml` only. Sub-packages must NOT have their own `[tool.ruff.*]` sections. Ruff does not inherit config — any `[tool.ruff.*]` in a sub-package silently overrides the entire root config (including `lint.select`, `per-file-ignores`, etc.).
- Run `uvx prek run --files {files_you_edited}` (unless the change was simple) and fix typing and linting errors. Use `# type: ignore` as needed. The precommit will give you type errors which is useful — especially to know if you have incorrect code — but for many minor changes it's better to do this after testing. Use a subagent to apply complex fixes.

---

## Engineering Standards

### Git Workflow

- **NEVER** use `git commit --amend` or `git push --force` / `--force-with-lease` unless the user explicitly requests it. Always create NEW commits.
- ALWAYS use atomic commits: one logical change per commit. Never bundle unrelated changes.
- Never use `git add -A` or `git add .`. Always stage specific files by name.
- Always pull before pushing. If push is rejected, pull and retry — never force-push.

### Code Style

- Use a subagent to run tests and complex bash commands, especially those expected to return complex output.
- Run independent tasks (multi-file edits across separate concerns, cross-cutting verifications, distinct review angles) in parallel via concurrent subagents in a single message. Subagent startup overhead is negligible relative to sequential blocking. Only sequence subagents when a later task needs an earlier task's output.

#### Avoid Backwards Compatibility

In most cases it is incorrect to maintain backwards compatibility with a previous pipeline. This is a research codebase — changes are expected and encouraged. Keeping backwards compatibility risks MORE bugs, since someone can unknowingly run old code.

If you believe it is important to maintain backwards compatibility, explicitly ask the user if you should do so during the planning stage. If the user says no, then do not maintain backwards compatibility.

Delete and remove old code that is not used.

#### Use Context Managers for Resources

Always use context managers (`with` statements) when opening external resources like zarr stores, files, or database connections. Never assign them to a variable without a context manager — this leaks file handles and locks.

```python
# correct
with open_ome_zarr(path, mode="r") as plate:
    ...

# wrong — resource never closed
plate = open_ome_zarr(path, mode="r")
```

#### Prefer Raising Errors

Prefer raising errors instead of silently catching them. Errors are good and warn us of issues. For example, prefer `value = my_dictionary['key']` over `value = my_dictionary.get('key')` since the former will raise a `KeyError` to signal that the underlying data is not behaving as expected.

Only catch errors when there is a good reason to do so: for example, catching HTTP errors in order to retry a request.

If you find yourself writing an if statement, fallback, or except statement designed to avoid errors, ask yourself if it would be better to raise the error as a signal to the user.

#### Use Real Integration Tests

Tests should directly *import* the actual code we are trying to test. For example, if you are trying to test `my_function` on some sample data, your test should directly import `my_function` and run it on the sample data. Avoid testing "key behavior" or components in isolation when an integration test would catch more bugs.

Ask yourself if your test is actually covering the true function.

#### Imports

- Import at the top of the file. No inline imports without strong reason.
- Use absolute imports (`from packages.my_directory.my_file`) instead of relative.
- Do not modify `sys.path` for imports.

### Coding Philosophy

#### 1. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 2. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.
