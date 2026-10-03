# DynaCell

**An evaluation framework for dynamic 3D virtual staining of live cells.**

- 📦 **Dataset:** [DynaCell on the AWS Open Data Registry](https://registry.opendata.aws/dynacell/)
- 🤗 **Demo:** [Interactive virtual-staining Space on Hugging Face](https://huggingface.co/spaces/biohub/dynacell)
- 📓 **Notebook:** [Get to know the DynaCell dataset](examples/notebooks/get-to-know-dynacell-dataset.ipynb)

Virtual staining predicts fluorescence-like images of cell structures from
label-free microscopy (brightfield, phase contrast), enabling live-cell
profiling without the phototoxicity and photobleaching of direct fluorescent
labeling. Many models exist for this task, but their performance on volumetric
time-lapse data remains poorly characterized. DynaCell addresses this gap with
three linked components:

- **Data** — a new paired label-free and fluorescence 3D time-lapse dataset of
  live A549 cells (Mantis light-sheet microscope, 20-hour imaging, 4 organelles,
  Zika/Dengue perturbations), plus a reprocessed, curated subset of the Allen
  Institute WTC-11 hiPSC dataset for cross-cell-type and cross-microscope
  evaluation.
- **Baseline models** — four regression baselines, a deterministic 3D Pix2Pix
  GAN, and CELL-Diff under conditional-mean and sampled inference, trained under
  a shared protocol across nucleus, plasma membrane, endoplasmic reticulum (ER),
  and mitochondria.
- **Metrics** — a three-tier panel measuring pixel-level fidelity, organelle
  segmentation utility, and single-cell phenotypic similarity.

Across iPSC and A549 training/test domains, an evaluation-only HEK293T ablation,
4 organelles, and 3 perturbation states, DynaCell exposes trade-offs that
single-metric evaluation obscures. Deterministic point predictors preserve
paired spatial structure, while sampled CELL-Diff better captures
population-level phenotype distributions.

This directory is the DynaCell application of the [VisCy](../../README.md)
monorepo. See the paper for the full benchmark description.

## Directory map

- **[configs/](configs/README.md)** — Hydra/LightningCLI configs: reusable recipes, generic examples, and the
  runnable benchmark leaves (`configs/benchmarks/virtual_staining/`).
- **[src/](src/README.md)** — the installable `dynacell` package: engines, data/manifest layer, evaluation
  pipeline, reporting, preprocessing, and the bundled dataset registry.
- **[tools/](tools/README.md)** — operational scripts: leaf submission, config generation, data/artifact builders.
- **[tests/](tests/README.md)** — the pytest suite (integration-first) for the package and tools.
- **[examples/](examples/README.md)** — the dataset notebook and the Hugging Face virtual-staining demo.
- **[CLAUDE.md](./CLAUDE.md)** — conventions reference: code↔paper names, prediction/eval-dir naming, batch
  semantics, eval runtime.

## Installation

DynaCell is part of the VisCy `uv` workspace. From the repository root:

```bash
uv venv -p 3.13
uv sync --all-packages --all-extras
```

See the repository [CONTRIBUTING.md](../../CONTRIBUTING.md) for full setup,
including HPC notes.

## Quickstart

Generic fit/predict configs for each model family live under `configs/examples/`.
Set `data_path` in the config or pass it on the command line:

```bash
cd applications/dynacell/configs/examples

# Regression baselines
uv run dynacell fit -c fnet3d/fit.yml   --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c unext2/fit.yml   --data.init_args.data_path=/path/to/data.zarr
uv run dynacell fit -c unetvit3d/fit.yml --data.init_args.data_path=/path/to/data.zarr

# Generative baseline (flow-matching CELL-Diff)
uv run dynacell fit -c celldiff/fit.yml --data.init_args.data_path=/path/to/data.zarr
```

Predict writes virtual stains to OME-Zarr via `HCSPredictionWriter`:

```bash
uv run dynacell predict -c unetvit3d/predict.yml --data.init_args.data_path=/path/to/data.zarr
```

Evaluate a prediction against its fluorescence target with the three-tier panel:

```bash
uv run dynacell evaluate io.pred_path=/path/to/prediction.zarr \
    io.gt_path=/path/to/target.zarr target_name=sec61b
```

## Architectures

Each baseline is trained separately per organelle. Code names differ from the
paper display names — see [CLAUDE.md](./CLAUDE.md) for the full translation.

### Regression baselines (`DynacellUNet`)

One spatially registered fluorescence estimate per label-free input; natural
references for localization, segmentation, and tracking.

- **FNet3D** — the classic 3D U-Net (Ounkomol et al. 2018); random-flip
  augmentation only.
- **UNeXt2** — a modern U-Net on the ConvNeXt V2 backbone (Cytoland).
- **VSCyto3D** — UNeXt2 with a pretrained FCMAE encoder (Cytoland).
- **UNetViT3D** — a hybrid CNN-Transformer adapted from the CELL-Diff backbone.

### Generative baselines

- **Pix2Pix3D** (`DynacellGAN`) — deterministic adversarial training with the
  UNetViT3D generator and a multiscale 3D PatchGAN discriminator.

- **CELL-Diff** (`DynacellFlowMatching`) — a 2D flow-matching virtual-staining model extended to 3D
  inputs and outputs. Samples plausible fluorescence volumes conditioned on
  phase; uses iterative flow-matching inpainting for large-volume inference, and
  a single-pass mean-prediction mode for point-estimate comparison. The
  flow-matching loss is computed internally — no external loss function needed.

## Evaluation metrics

Three tiers, each interrogating predicted stain quality differently:

- **Pixel fidelity** — PSNR, SSIM, PCC, NRMSE, plus microscopy-aware Fourier
  shell correlation (FSC), MicroSSIM, and a frequency-aware Spectral PCC robust
  to photobleaching-driven noise in fluorescence targets.
- **Segmentation utility** — Cellpose v4 cpdino instances for nucleus and
  membrane, and Allen Institute semantic segmenters for ER and mitochondria;
  masks are compared with overlap and instance-detection metrics.
- **Phenotype similarity** — a 22-feature handcrafted GLCM+ space and learned
  DINOv3, DynaCLR, CELL-DINO, and MorphEm embeddings; median cosine similarity
  between matched cells plus FID and KID between pooled predicted and
  experimental distributions.

## Config structure

- `configs/recipes/` — reusable fragments (model, trainer, data, topology).
- `configs/examples/` — generic fit/predict pair per model family (stubs with
  `#TODO` data-path placeholders).
- `configs/benchmarks/virtual_staining/` — runnable benchmark leaves composed
  from shared axes: one file per (organelle, train_set, model) for fit and one
  per (organelle, train_set, model, predict_set) for predict. See
  [the benchmarks README](configs/benchmarks/virtual_staining/README.md) for the
  layout, composition order, and reserved-key contract.

Benchmark leaves carry two reserved top-level keys (`launcher:` and `benchmark:`)
that are stripped before the config reaches LightningCLI, so
`uv run dynacell fit -c <benchmark-leaf.yml>` works without the submit tool.
`benchmark.dataset_ref` lookups resolve against a manifest registry bundled with
the package (`src/dynacell/_manifests/`), so `predict` works out of the box.

### Benchmark submit

`tools/submit_benchmark_job.py` drives one benchmark leaf end-to-end (compose →
strip launcher metadata → render sbatch → submit):

```bash
LEAF=applications/dynacell/configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/train.yml

# Preview the rendered sbatch — stdout only, no disk writes, safe on any leaf:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-script

# Preview the resolved LightningCLI config (launcher + benchmark keys stripped):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config

# Stage resolved YAML + sbatch under launcher.run_root without submitting:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --dry-run

# Submit:
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF

# Dotlist overrides deep-merge after compose (repeatable, no ${...} interpolation):
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF \
    --override trainer.max_epochs=50 \
    --override data.init_args.batch_size=2
```

`--print-*` is pure preview (no writes, no submission); `--dry-run` writes
artifacts but skips `sbatch`; `--dry-run` with any `--print-*` lets preview win;
a bare invocation writes **and** submits.

## Subcommands

- `fit`, `validate`, `predict` — supported for all architectures. For UNetViT3D
  and CELL-Diff, `yx_patch_size` and `z_window_size` in the data config must
  match the model's `input_spatial_size`.
- `evaluate`, `evaluate-grouped` — run the three-tier metric panel on one or many
  (model, organelle, condition) variants.
- `precompute-gt` — precompute and cache ground-truth segmentation and features.
- `report` — aggregate evaluation results into tables.
- `test` — not supported (raises `MisconfigurationException`; no `test_step`).

## Data and code availability

- **Code** — BSD-3-Clause, [github.com/mehta-lab/VisCy](https://github.com/mehta-lab/VisCy/tree/dynacell-models).
- **Data** — OME-Zarr stores hosted on AWS Open Data. The DynaCell A549 dataset
  is released under CC-BY-4.0; the reprocessed iPSC subset under the Allen
  Institute Terms of Use.
- **Demo** — a small reviewer sample is at
  [dynacell_a549_demo.zip](https://dynacell.s3.us-west-2.amazonaws.com/v1/demo/dynacell_a549_demo.zip).

## Navigation

- Up: [VisCy monorepo](../../README.md)
- Subdirectories: [configs/](configs/README.md) · [src/](src/README.md) · [tools/](tools/README.md) ·
  [tests/](tests/README.md) · [examples/](examples/README.md)
