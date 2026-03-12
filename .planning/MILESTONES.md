# Milestones: VisCy Modularization

## v1.0 — Transforms & Monorepo Skeleton

**Shipped:** 2026-01-29
**Phases:** 1-5 (4 executed, 1 deferred)

**What shipped:**
- uv workspace scaffolding with `packages/` directory
- `viscy-transforms` package extracted with src layout
- Clean import paths: `from viscy_transforms import X`
- hatchling + uv-dynamic-versioning build system
- All 16 transform modules migrated with full test suite
- CI/CD: 9-job test matrix (3 OS x 3 Python) + lint workflow
- prek hooks with ruff formatting

**Deferred:**
- Phase 4: Documentation (Zensical + GitHub Pages)

**Last phase:** 5

---

## v1.1 — Extract viscy-data

**Shipped:** 2026-02-14
**Phases:** 6-9 (4 phases, 9 plans)

**What shipped:**
- `viscy-data` package extracted with src layout (15 modules, 4015 LOC)
- All 13 data modules migrated with clean import paths (45 public exports)
- No cross-package dependency on viscy-transforms
- Optional dependency groups: `[triplet]`, `[livecell]`, `[mmap]`, `[all]`
- Shared utilities extracted from hcs.py into _utils.py
- Lazy import pattern for optional deps with clear error messages
- All existing data tests passing (71 tests)
- Tiered CI: viscy-data 3x3 base + 1x1 extras

**Last phase:** 9

---

## v1.2 — Extract viscy-models

**Shipped:** 2026-02-13
**Phases:** 10-14 (5 phases, 9 plans)

**What shipped:**
- `viscy-models` package with 8 architectures organized by function (unet/, vae/, contrastive/)
- Shared components extracted to `_components/` (stems, heads, blocks)
- Full test coverage: migrated existing + new forward-pass tests for all models
- Pure nn.Module — no Lightning/Hydra coupling
- State dict key compatibility preserved for checkpoint loading
- CI includes viscy-models in test matrix

**Pending todo from v1.2:**
- Fix deconv decoder channel mismatch in UNeXt2UpStage (pre-existing bug, xfailed test)

**Last phase:** 14

---

## v2.0 — DynaCLR Application

**Shipped:** 2026-02-17
**Phases:** 15-17 (3 phases, manual execution)

**What shipped:**
- `viscy-utils` package — shared ML infrastructure (trainer, callbacks, evaluation, cli_utils)
  - EmbeddingWriter callback for prediction writing
  - Linear classifier evaluation pipeline (train, apply, config)
  - Embedding visualization app (Plotly/Dash)
  - cli_utils with format_markdown_table() and load_config()
  - pyyaml added as dependency
- `applications/dynaclr` — DynaCLR self-contained application
  - ContrastiveModule engine (LightningModule for time-aware contrastive learning)
  - MultiModalContrastiveModule (cross-modal distillation)
  - ClassificationModule (downstream supervised classification)
  - `dynaclr` CLI with LazyCommand pattern:
    - `train-linear-classifier` — train logistic regression on cell embeddings
    - `apply-linear-classifier` — apply trained classifier to new embeddings
  - `evaluation/linear_classifiers/` — dataset discovery, config generation, SLURM scripts
  - `examples/configs/` — fit.yml, predict.yml, ONNX export, SLURM templates
  - `examples/DynaCLR-DENV-VS-Ph/` — infection analysis demo (ImageNet vs DynaCLR)
  - `examples/embedding-web-visualization/` — interactive Plotly/Dash visualizer
  - `examples/DynaCLR-classical-sampling/` — pseudo-track generation from 2D segmentation
  - `examples/vcp_tutorials/` — quickstart notebook and script
  - Optional [eval] extras: anndata, natsort, wandb, scikit-learn, phate, umap-learn
- All YAML configs updated with new class_path imports (dynaclr.engine, viscy_models, viscy_data, viscy_transforms)
- All Python scripts updated with new import paths

**Last phase:** 17

---
