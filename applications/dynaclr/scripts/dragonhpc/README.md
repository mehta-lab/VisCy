# DragonHPC × DynaCLR Handoff

## Goal

Hand DragonHPC a single, reproducible DynaCLR training run that they can
use as a substrate for swapping in their distributed dataloader. We
want an apples-to-apples comparison between our current MONAI
`ThreadDataLoader` + tensorstore pipeline and a Dragon-backed
replacement.

The recommended starter is **DynaCLR-2D-MIP-BagOfChannels** (the 2D MIP
bag-of-channels run). It's the same training we just profiled
end-to-end (see
`.ed_planning/dynaclr/profiling/2026-04-24-defaults-and-restart.md`),
so the baseline numbers and bottleneck analysis are fresh and
grounded.

| | DynaCLR-2D-MIP | DynaCLR-3D |
|---|---|---|
| Train anchors | ~2.7 M | ~1.0 M |
| Per-batch wire data | ~5 GB | ~10 GB |
| Patch shape | `(B, 1, 1, 160, 160)` (after MIP) | `(B, 1, 32, 160, 160)` |
| Throughput baseline | 132–144 samples/s | not yet profiled |
| GPU util | ~0.5% (I/O bound) | I/O bound |
| Use as starter | **yes** | only if 2D wins are confirmed |

If DragonHPC wants 3D as a stretch goal, the same handoff applies —
only the config path changes (see "3D variant" below).

---

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This handoff doc. |
| `train_2d_mip_minimal.py` | Notebook-style Python entry point — builds `MultiExperimentDataModule` + `ContrastiveModule` + `pl.Trainer` directly, no Lightning CLI. The narrowest place to swap in a Dragon DataLoader. |

## Quick start

```sh
# 5-step smoke test on the tiny parquet (single GPU, no logger).
uv run python applications/dynaclr/scripts/dragonhpc/train_2d_mip_minimal.py --mode fastdev

# 20-batch short run on the production parquet (single GPU, no logger).
uv run python applications/dynaclr/scripts/dragonhpc/train_2d_mip_minimal.py --mode short
```

For the full DDP production run, use the SLURM launcher instead:

```sh
sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.sh
```

---

## Repository layout

```
viscy/                                  # uv workspace monorepo
├── packages/
│   ├── viscy-data/                     # FlexibleBatchSampler, ThreadDataLoader plumbing
│   ├── viscy-models/                   # ConvNeXt-Tiny + NTXent loss
│   └── viscy-transforms/               # GPU-batched augmentations
├── applications/dynaclr/
│   ├── src/dynaclr/data/
│   │   ├── datamodule.py               # MultiExperimentDataModule (Lightning)
│   │   ├── dataset.py                  # MultiExperimentTripletDataset (__getitems__)
│   │   └── experiment.py               # ExperimentRegistry (per-FOV zarr open + zattrs)
│   ├── configs/training/
│   │   ├── recipes/trainer/fit.yml     # logger, callbacks, log cadence
│   │   ├── recipes/topology/           # ddp_2gpu / ddp_4gpu / single_gpu / ...
│   │   ├── recipes/model/contrastive_encoder_convnext_tiny.yml
│   │   ├── DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml   # ← starter run
│   │   ├── DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.sh    # ← SLURM launcher
│   │   ├── debug/DynaCLR-2D-MIP-BagOfChannels-fastdev.yml  # 5-step smoke
│   │   └── slurm/train.sh              # shared srun wrapper
│   └── scripts/
│       ├── dragonhpc/                  # ← this folder
│       └── profiling/
│           ├── benchmark_boc2d_real.py # production-config dataloader benchmark
│           └── profile_num_workers.py  # num_workers sweep
└── pyproject.toml                      # uv workspace root
```

The dataloader hot path is exactly:

```
MultiExperimentDataModule.train_dataloader()
    └─ ThreadDataLoader(
          MultiExperimentTripletDataset,         # __getitems__ returns batched dict
          batch_sampler=FlexibleBatchSampler,    # group/stratify by parquet columns
          collate_fn=lambda x: x,                # dataset already batched, skip collate
          num_workers=4, buffer_size=1, prefetch_factor=2,
       )
```

The dataset reads patches directly with
`tensorstore.stack(...).read().result()` — one batched I/O call per
batch. Augmentation happens on GPU in `on_after_batch_transfer`,
**not** in the worker. That's deliberate — keeps CPU workers I/O-bound.
Don't move augmentation back to the workers when swapping the
dataloader.

---

## Environment

VisCy is a `uv` workspace. Install once:

```sh
cd /hpc/mydata/eduardo.hirata/repos/viscy   # or your clone
uv venv -p 3.13
uv sync --all-packages --all-extras
```

Sanity-check the install:

```sh
uv run viscy --help
uv run pytest packages/viscy-data/ -x
```

If `uv` is not installed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On HPC, redirect the uv cache out of `$HOME` first (one-time setup):

```sh
mkdir -p /hpc/mydata/${USER}/.cache/uv
ln -s /hpc/mydata/${USER}/.cache/uv ~/.cache/uv
```

---

## Data — already prepared

The starter run uses a single parquet "cell index" that fully describes
the training set (one row per `(cell, timepoint, channel)`):

| | Path |
|---|---|
| Parquet (training input) | `/hpc/projects/organelle_phenotyping/models/collections/DynaCLR-2D-MIP-BagOfChannels-v3.parquet` |
| Zarr stores referenced from parquet | various `/hpc/projects/organelle_phenotyping/datasets/*.zarr` on VAST |
| Tracking zarr (per dataset) | `tracking.zarr` next to each dataset |

DragonHPC does **not** need to rebuild the parquet — it is a
checked-in artifact derived deterministically from
`applications/dynaclr/configs/collections/DynaCLR-2D-MIP-BagOfChannels-v3.yml`
via:

```sh
uv run dynaclr build-cell-index   <collection.yml> <out.parquet> --num-workers 8
uv run dynaclr preprocess-cell-index <out.parquet> --focus-channel Phase3D
```

(Documented in `applications/dynaclr/docs/DAGs/training.md`.)

The parquet is self-contained: every per-cell value the model needs at
training time (focus slice index, normalization stats, channel name,
perturbation label) lives in a parquet column. The only zarr metadata
read at training startup is each plate's `focus_slice` zattrs entry,
done once per FOV in `ExperimentRegistry`.

---

## How a real training run is launched

End-user-facing entry point — exactly what we run for production:

```sh
# Baseline production launch (4× H100 / H200, 1 node)
sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.sh
```

That `.sh` does three things:

1. Sets `PROJECT`, `RUN_NAME`, and `CONFIGS` env vars.
2. Sources `applications/dynaclr/configs/training/slurm/train.sh`.
3. The shared `train.sh` runs:

```sh
srun uv run --project "$WORKSPACE_DIR" viscy fit \
    --config <config.yml> \
    --trainer.default_root_dir=$RUN_DIR \
    --trainer.logger.init_args.project=$PROJECT \
    --trainer.logger.init_args.name=$RUN_NAME \
    --trainer.logger.init_args.save_dir=$RUN_DIR
```

`viscy fit` is a Lightning CLI entry point; it constructs the trainer,
the `ContrastiveModule` model, and the `MultiExperimentDataModule` from
the YAML.

### What the YAML config sets

`DynaCLR-2D-MIP-BagOfChannels.yml` is the full source of truth (read
it — it's ~150 lines and well-commented). Key knobs DragonHPC will
care about:

```yaml
trainer:
  strategy: ddp
  devices: 4
  precision: bf16-mixed
  max_epochs: 150
  limit_train_batches: 800     # bounded epoch length
  limit_val_batches: 200

data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    cell_index_path: /hpc/.../DynaCLR-2D-MIP-BagOfChannels-v3.parquet
    z_window: 1
    z_extraction_window: 16    # per-batch wire = 16 Z slices
    yx_patch_size: [256, 256]
    final_yx_patch_size: [160, 160]
    channels_per_sample: 1     # bag-of-channels mode
    batch_size: 256
    num_workers: 4
    prefetch_factor: 2
    buffer_size: 1
    file_io_concurrency: 32
```

---

## Smoke test (5-step fastdev) via the YAML

For a quick check with no Python entry point at all:

```sh
uv run viscy fit \
    --config applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml \
    --config applications/dynaclr/configs/training/debug/DynaCLR-2D-MIP-BagOfChannels-fastdev.yml
```

This:

- Uses a 20k-row sliced parquet (`tmp/boc_tiny.parquet`) — the full
  parquet takes ~3 min just to load at setup time.
- Sets `fast_dev_run: 5` (5 train + 5 val batches), `devices: 1`,
  `num_workers: 0`.
- Disables the W&B logger.

Should finish in a couple of minutes. Catches API regressions, NCCL
config issues, and OOM at the smallest scale.

A DDP version exists for the multi-GPU path:

```sh
sbatch applications/dynaclr/configs/training/debug/DynaCLR-2D-MIP-BagOfChannels-fastdev-ddp.sh
```

---

## Where to plug in Dragon

The narrowest replacement surface is the `train_dataloader` /
`val_dataloader` methods on `MultiExperimentDataModule`
(`applications/dynaclr/src/dynaclr/data/datamodule.py:632`). They
return MONAI `ThreadDataLoader`s that wrap:

- `MultiExperimentTripletDataset` (`dynaclr.data.dataset`,
  `__getitems__` returns a batched dict).
- `FlexibleBatchSampler` (`viscy_data.sampler`, yields lists of indices
  grouped/stratified per epoch).

**Contract for any replacement**: each iterator yield must be a `dict`
of CPU tensors with at least the keys produced by the dataset's
`__getitems__`. The Lightning module's `on_after_batch_transfer` then
moves it to GPU and runs normalization + augmentation. The simplest
swap path:

```python
from dynaclr.data.datamodule import MultiExperimentDataModule

class DragonDataModule(MultiExperimentDataModule):
    def train_dataloader(self):
        # Build a Dragon-backed iterable.
        # Required: yields a dict of CPU tensors with the same keys as
        # MultiExperimentTripletDataset.__getitems__ produces.
        ...

    def val_dataloader(self):
        ...
```

Then instantiate `DragonDataModule` instead of
`MultiExperimentDataModule` in
`train_2d_mip_minimal.py:build_datamodule`.

The dataset already does the heavy lifting (one batched
`tensorstore.stack().read()` call per batch). What Dragon would change
is **how that read is dispatched and prefetched across nodes** —
specifically:

- Replace MONAI's `ThreadDataLoader` (single-process, GIL-bound,
  effective ceiling ~2.2 threads).
- Keep the dataset's `__getitems__` API (Dragon controls _which_ batch
  index list to fetch and _how to overlap_; the dataset still does the
  read).
- Optionally: replace `tensorstore.stack(...).read()` itself with a
  Dragon zarr service (the `czbio-napari-plugin/zarr_service/` dir
  suggests they have one).

---

## Baseline performance (current pipeline)

From `.ed_planning/dynaclr/profiling/2026-04-24-defaults-and-restart.md`:

| metric | value |
|---|---|
| Throughput, `num_workers=2` | 127 samples/s |
| Throughput, `num_workers=4` (current default) | 132.5 samples/s |
| Throughput, `nw=4` + `file_io_concurrency=128` + `ts.Batch` | **143.9 samples/s** |
| GPU utilization | 0.5% (mean over 7 min) |
| NFS throughput from VAST | ~1.3 GB/s (ceiling 2–4 GB/s) |
| Per-batch wire data | ~5 GB |
| Per-batch useful data | ~2.5 GB (1.94× chunk amplification) |
| Cold start (`setup("fit")`) | 219 s |
| Effective parallel-thread ceiling | ~2.2 (Amdahl, GIL-bound) |
| 1-epoch wall time, 4-GPU DDP | ~85 min |

Levers exhausted in single-axis tuning of `ThreadDataLoader` ×
tensorstore (see profiling doc for full table). Remaining high-value
levers — and what we expect Dragon to attack:

1. **Process-based DataLoader** (escape the GIL) — 1.5–2× expected.
2. **Cross-node parallel reads** with a real DDP-aware data service
   (Dragon-style) — up to 4× combined.
3. **Rechunk zarrs to YX=160 patch-aligned** — ~2× from chunk-amp
   removal, but infra-side.

---

## Success criteria for DragonHPC's swap

A clean win is:

- ≥ 1.5× throughput improvement on `samples/s` (median over 60 batches
  after 10-batch warmup), measured at `num_workers` and topology
  matched to ours.
- No regression in `loss/val` trajectory across the first ~5 epochs
  (W&B project `DynaCLR-2D-MIP-BagOfChannels`, baseline run names
  start with `2d-mip-ntxent-t0p2-...`).
- Identical `__getitems__` semantics — i.e., the dataset still sees
  batch index lists from `FlexibleBatchSampler` so stratification and
  positive-pair sampling don't drift.

The benchmark harness lives at
`applications/dynaclr/scripts/profiling/benchmark_boc2d_real.py` and
is the canonical way to measure throughput against the production
config.

---

## 3D variant (stretch)

If 2D shows a clean win and DragonHPC wants the bigger workload, swap
the config:

| | 2D MIP starter | 3D variant |
|---|---|---|
| Config | `DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml` | `DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.yml` |
| Parquet | `DynaCLR-2D-MIP-BagOfChannels-v3.parquet` | `DynaCLR-3D-BagOfChannels-v4.parquet` |
| Per-batch wire | ~5 GB | ~10 GB |
| Patch | `(256, 1, 1, 160, 160)` | `(256, 1, 32, 160, 160)` |
| SLURM launcher | `DynaCLR-2D-MIP-BagOfChannels.sh` | `DynaCLR-3D-BagOfChannels-v2.sh` |

Everything else (data module, sampler, trainer recipe) is identical.

---

## Pointers

- Profiling outcomes: `.ed_planning/dynaclr/profiling/2026-04-24-defaults-and-restart.md`
- Training DAG: `applications/dynaclr/docs/DAGs/training.md`
- Production config (the source of truth):
  `applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml`
- Data module: `applications/dynaclr/src/dynaclr/data/datamodule.py`
- Dataset (the `__getitems__` + `tensorstore.stack` hot path):
  `applications/dynaclr/src/dynaclr/data/dataset.py`
- Sampler: `packages/viscy-data/src/viscy_data/sampler.py`
- Benchmark harness:
  `applications/dynaclr/scripts/profiling/benchmark_boc2d_real.py`
- Notebook-style entry point (this folder):
  `applications/dynaclr/scripts/dragonhpc/train_2d_mip_minimal.py`
