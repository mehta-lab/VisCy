# DynaCLR Training Configs

Training configuration stack for LightningCLI `--config`. Later configs
deep-merge into earlier ones (dicts merge, lists replace). Each leaf
YAML declares a `base:` list of recipes to compose on top of.

## Directory layout

```
configs/training/
  DynaCLR-2D/          # 2D (and MIP) time-lapse contrastive runs
    DynaCLR-2D-BagOfChannels-v3.{yml,sh}
    DynaCLR-2D-MIP-BagOfChannels.{yml,sh}
    DynaCLR-2D-MIP-BagOfChannels-single-marker.{yml,sh}
    DynaCLR-2D-MIP-BagOfChannels-single-marker-A40.{yml,sh}
  DynaCLR-3D/          # 3D time-lapse contrastive runs
    DynaCLR-3D-BagOfChannels-v2.{yml,sh}
    DynaCLR-3D-BagOfChannels-v2-single-marker.{yml,sh}
  DINOv3/              # DINOv3 frozen-encoder + MLP probes
    DINOv3-temporal-MLP-2D-BagOfChannels.{yml,sh}
  Phase-contrastive/
    Phase-contrastive-timeaware.{yml,sh}

  recipes/             # Reusable building blocks (referenced via base:)
    trainer.yml        Trainer + logger + common callbacks
    model/             Encoder and head architectures
    data/              Sampling / positive-pair strategies
    augmentations/     Augmentation pipelines (ops_2d_mild, etc.)

  debug/               # Fast-dev-run / tiny configs for reproducing hangs / OOMs
  demo/                # Self-contained single-file demos for smoke tests
  slurm/
    train.sh           Shared launcher sourced by every sbatch script
  preprocess.yml       Preprocessing config (not a training run)
```

Each top-level model family lives in its own folder. The `yml` and `sh`
for a given run share a name and a directory so `CONFIGS=` references
stay local.

## Composition via `base:`

Each leaf YAML starts with a `base:` list pointing at recipe fragments
(paths are relative to the YAML's directory; since all leaf YAMLs live
one level below `recipes/`, they use `../recipes/...`):

```yaml
# DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml
base:
  - ../recipes/trainer.yml
  - ../recipes/model/contrastive_encoder_convnext_tiny.yml
```

`viscy_utils.compose.load_composed_config` walks the `base:` chain,
deep-merges dicts, and replaces lists.

## SLURM scripts

Each experiment is a thin `.sh` that sets `PROJECT`, `RUN_NAME`,
`CONFIGS`, optional `EXTRA_ARGS`, and sources `slurm/train.sh`:

```bash
sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.sh

RUN_NAME=phase2-hcl sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.sh

for TEMP in 0.1 0.2 0.5; do
  RUN_NAME="sweep-temp${TEMP}" \
  EXTRA_ARGS="--model.init_args.loss_function.init_args.temperature ${TEMP}" \
  sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.sh
done
```

`train.sh` handles:
- `export PYTHONNOUSERSITE=1` (prevents `~/.local/` shadowing conda)
- Creates `${MODEL_ROOT}/${PROJECT}/${RUN_NAME}/` output dir
- Rotates `config.yaml` from any previous run
- Copies the calling sbatch script into the run dir for reproducibility
- Sets WandB logger project / name / save_dir via CLI overrides
- Optional `CKPT_PATH` resume and `WANDB_RUN_ID` to continue a run

## Resuming a run

```bash
CKPT_PATH=/hpc/projects/.../checkpoints/last.ckpt \
WANDB_RUN_ID=<wandb_run_id> \
  sbatch --export=ALL,CKPT_PATH,WANDB_RUN_ID \
  applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.sh
```

`WANDB_RUN_ID` appends `--trainer.logger.init_args.id=<id>
--trainer.logger.init_args.resume=must` so metrics land on the same
W&B timeline.

## Adding a new experiment

1. Find the closest existing run in the matching model family
   folder. Copy the `.yml` and `.sh` alongside it with a new name.
2. Edit `base:` in the YAML to pick the right recipes.
3. Override training-specific values in the YAML (or via `EXTRA_ARGS`
   in the sbatch script for one-off sweeps).
4. `sbatch applications/dynaclr/configs/training/<FAMILY>/<NAME>.sh`.

## Debug / demo configs

- `debug/` — fastdev, tiny, and DDP-reproducer configs used to isolate
  SLURM hangs, memory spikes, and DDP sync issues. Launched with
  `uv run viscy fit --config <base>.yml --config debug/<debug>.yml`.
- `demo/` — self-contained single-file configs for quick local smoke
  tests (no base chain).
