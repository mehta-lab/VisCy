# Reef/Kelp (CoreWeave) — Job Submission Guide

Reef and Kelp are CZI's CoreWeave-hosted SLURM clusters. They are **beta** and
**preemptible by default** — this is the core thing that makes job scripts
here different from Bruno (our home-institution cluster, which is not
preemptible). Everything below is written for adapting VisCy's existing
Bruno SLURM patterns (`applications/dynaclr/configs/training/slurm/train.sh`,
`applications/dynacell/tools/sbatch_template*.sbatch`,
`applications/cytoland/examples/configs/*/run_*.slurm`) to run on Reef, not
for writing from scratch.

Source: internal "AI Research Cluster Reef/Kelp User Guide" (work in
progress, expect breaking changes).

## Bruno vs Reef, at a glance

| Aspect | Bruno (home institution) | Reef/Kelp (CoreWeave) |
|---|---|---|
| Preemption | Not preemptible | `--qos mid` / `--qos low` **are** preemptible; `--qos dev` and team QOS are not |
| GPU limits | Constraint-based (`--constraint='h200\|h100'`) | `--qos dev` caps you at 8 GPUs; `mid`/`low` uncapped but cost/don't-cost fairshare |
| Partitions | `gpu`, `cpu` | GPU: named reserved pools e.g. `h100_reserved`, `h200_reserved`; CPU: `cpu`, `cpu-turin-gp-l` |
| Filesystem | `/hpc/mydata/<user>`, `/hpc/projects/<project>` | Single filesystem on `/bio`; home is `/mnt/main0/home/<user>` |
| RunAI/CoreWeave PVC data | N/A | Mounted directly at `/mnt/runai-<pvc-name>` (same physical storage as RunAI) |
| Cross-cluster data | N/A | Bruno data does **not** auto-sync to Reef — must be copied manually |
| AWS credentials | Manual (`AWS_PROFILE`, etc.) | Auto-provisioned via OIDC on login + compute nodes — **remove** manual AWS env vars from `.bashrc` |
| Job launch | Hand-written `sbatch` scripts | Either hand-written `sbatch`/`srun`, or `slurm_run` (snapshots repo state at submit time) |
| Env manager | `uv` (see root `CLAUDE.md`) | `uv` or `pixi`, same idea |
| Access | Direct SSH | Tailscale-gated; personal login node `login-reef-<email-shortname>` |
| Lightning strategy env | `lightning.pytorch.plugins.environments.SLURMEnvironment` | Same — no change needed |

## Access

- Cluster access + Tailscale setup is one-time (Okta "Coreweave Slurm
  Cluster" request, Tailscale on the `biohub.org` tailnet). Not a per-job
  concern.
- Use your **personal** login node (`login-reef-<shortname>`), not the shared
  one — shared login nodes are being deprecated due to reliability issues.
- Interactive GPU node for debugging a script before submitting a batch job:
  ```sh
  srun -c 8 --mem 96G -N 1 --gres gpu:h100:1 --qos dev -t 1-0 -J interactive --pty zsh -i
  ```
- Interactive CPU node (data prep, quick checks):
  ```sh
  srun --partition=cpu -c 8 --mem 64G -N 1 --qos dev -t 1-0 -J interactive --pty zsh -i
  ```

## QOS — read this before choosing one for a training job

QOS is Reef's stand-in for Bruno's constraint-based scheduling, but it also
controls **preemption**, which Bruno jobs never had to handle:

- `--qos dev` — max 8 GPUs, non-preemptible. Use for interactive/debug and
  short smoke tests, not long training runs (fairshare will eventually
  deprioritize you, but you won't be killed mid-run).
- `--qos teamA`-style (team allocation) — non-preemptible, ask if VisCy/CZ
  Biohub has one before defaulting to `mid`.
- `--qos mid` — no GPU limit, costs fairshare, **preemptible** by `dev`/team
  QOS jobs.
- `--qos low` — no GPU limit, free (no fairshare cost), **preemptible** by
  everything above.

**Implication for VisCy training scripts:** any job submitted at `mid` or
`low` can be killed and requeued at any time. Long DynaCLR/cytoland training
runs on Reef should:
1. Checkpoint frequently (Lightning's `ModelCheckpoint` already does this in
   our configs — just make sure the interval is short enough that a
   preemption doesn't lose much progress).
2. Resume from checkpoint automatically. `train.sh` already supports this via
   `CKPT_PATH` and `WANDB_RUN_ID` — reuse that pattern rather than inventing
   a new one.
3. Add `#SBATCH --requeue` so SLURM automatically resubmits the job on
   preemption instead of leaving it dead in the queue. Bruno scripts don't
   have this flag because it was never needed there.
4. Not assume "job disappeared from `squeue`" means it failed — check
   `sacct -j <id> --format=JobID,State,ExitCode` for `PREEMPTED` vs a real
   failure, same idea as the completeness-check guidance in the root
   `CLAUDE.md`.

## Partitions

- GPU: named reserved pools, e.g. `h100_reserved`, `h200_reserved` (confirm
  exact names available with `sinfo` on the day — Reef is still adding
  pools). This replaces Bruno's `--partition=gpu --constraint='h200|h100'`
  pattern; on Reef pick the partition itself instead of constraining within
  the `gpu` partition.
- CPU-only: `cpu` (has a default QOS) or `cpu-turin-gp-l` (no default QOS —
  **you must pass `--qos` explicitly** or the job is rejected). Use for
  dataset prep / ETL / lightweight inference, same role as CPU-only Bruno
  jobs. `cpu-turin-gp-l` runs with `OverSubscribe=NO`, i.e. CPUs are
  exclusively allocated (no oversubscription like some Bruno CPU nodes).

## Filesystem & data paths

- Single filesystem on `/bio`; user home is `/mnt/main0/home/<user>` (this is
  where `slurm_run` also drops job scripts and logs by default:
  `/mnt/main0/home/<user>/slurm/<date>/slurm-<jobid>.out`).
- CoreWeave/RunAI PVC-backed datasets are mounted at
  `/mnt/runai-<pvc-name>`, e.g. a `dynamic-imaging-models-120t` PVC is at
  `/mnt/runai-dynamic-imaging-models-120t`. This is the same physical storage
  RunAI jobs used — no copy needed if the data already lives there.
- **Bruno data is not auto-synced to Reef.** Any dataset referenced by
  `WORKSPACE_DIR`/`MODEL_ROOT`-style paths in our Bruno scripts
  (`/hpc/mydata/...`, `/hpc/projects/...`) must be manually copied to `/bio`
  or the appropriate `/mnt/runai-*` PVC before a Reef job can read it. Don't
  assume a path that works on Bruno resolves on Reef.

## Environment setup

- `uv` (already our standard, see root `CLAUDE.md`) or `pixi` both work.
  Symlink the cache out of `$HOME` the same way we do on Bruno-style HPC:
  ```sh
  mkdir -p /bio/<project-storage>/<user>/.cache/uv && ln -s /bio/<project-storage>/<user>/.cache/uv ~/.cache/uv
  ```
  (adjust the target to wherever project storage lands on `/bio` or the
  relevant `/mnt/runai-*` PVC — installing envs on a login node is slow, do
  it from a compute node if it's dragging.)
- AWS credentials are auto-provisioned via OIDC on both login and compute
  nodes. **Remove** any `AWS_PROFILE`/`aws-oidc` setup from `.bashrc` —
  leftover Bruno-style AWS env vars can shadow the auto-provisioned ones and
  break S3 access.
- **Nextflow**: unlike Bruno (`module load nextflow/24.10.5`), Reef has no
  `nextflow` module — install it yourself via `micromamba` (already present
  on Reef login nodes; no root needed):
  ```sh
  micromamba create -y -n nextflow -c bioconda -c conda-forge "nextflow=24.10.5"
  ```
  Pin the version to `24.10.5` to match Bruno's module and this repo's
  `applications/dynaclr/nextflow/` DAGs — the latest bioconda build (26.x)
  turns on Nextflow's "strict parser" by default, which rejects the
  `-entry <name>` flag these DAGs rely on
  (`ERROR ~ The '-entry' option is not supported with the strict parser`).
  Run it via `micromamba run -n nextflow nextflow run ...` or
  `micromamba activate nextflow` first.

## Two ways to launch: `slurm_run` vs hand-written `sbatch`

**`slurm_run`** (from `github.com/evolutionaryscale/slurm_run`, install via
`git clone` + `make install` — not yet a pip package) snapshots the repo
state at submit time, so it's a good fit for one-off training/eval launches
where you want the exact commit reproducible. For a VisCy job:
```sh
slurm_run submit \
  --venv=uv \
  --cpus=<cpus_per_task> \
  --gpus=<num_gpus>            # 1-gpu-per-task; sets the number of tasks \
  --partition=<h100_reserved|h200_reserved|...> \
  --qos=<dev|mid|low> \
  -- uv run dynaclr fit --config <path/to/config.yml>
```
`slurm_run jobs` lists your recent submissions.

**Hand-written `sbatch`/`srun` scripts** are still the right choice when you
need our existing patterns — `train.sh`'s config-copying/checkpoint-resume
logic, or `dynacell`'s NCCL preflight smoke test — that `slurm_run` doesn't
know about. When adapting one of our Bruno `.slurm` files for Reef, change:
- `#SBATCH --partition=...` to a Reef GPU/CPU partition name (see above).
- `#SBATCH --constraint=...` (Bruno-only) → drop it; pick the partition
  instead.
- Add `#SBATCH --qos=<dev|mid|low>` (Reef requires an explicit QOS; Bruno
  scripts often didn't set one).
- Add `#SBATCH --requeue` if running at `mid`/`low`, so preemption resubmits
  instead of dying silently.
- Update any `/hpc/mydata/...` or `/hpc/projects/...` path to its `/bio` or
  `/mnt/runai-*` equivalent (see Filesystem section).
- Keep `--ntasks-per-node=N` (not `--ntasks=N`) for any Lightning DDP job —
  this is a Lightning `SLURMEnvironment` requirement, not Bruno- or
  Reef-specific, and the existing invariant in the root `CLAUDE.md` (must
  match `trainer.devices` and `--gpus`/`--gpus-per-node`) still applies
  unchanged on Reef.

## Monitoring

- `slurm_run jobs`, or standard `squeue`/`sacct`, work as usual.
- Web dashboards: "All Jobs Metrics" (click a Job ID for per-job metrics) and
  "Cluster Utilization" — both beta, report bad numbers to Sashidhar Guntury.
- Reminder from root `CLAUDE.md` still applies here, and matters *more* on
  Reef: `wandb` `state: finished` doesn't distinguish a clean finish from a
  preempted/`scancel`'d run. On Reef, also check `sacct` for `PREEMPTED`
  specifically before assuming a dead job failed outright.

## Current rollout limits (beta caveats)

As of this writing Reef only has confirmed support for: single-GPU jobs,
single-node multi-GPU (DDP), small multi-node (2 nodes × 8 GPUs),
checkpoint+resume, and W&B logging — which covers current VisCy training
jobs. Large-scale sweeps, queue-based worker patterns, and general ETL
pipelines on Reef are still on the roadmap (not yet documented) — don't
assume they work without checking `sinfo`/the guide for updates first.
