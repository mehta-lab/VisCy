# Codebase Structure

**Analysis Date:** 2026-02-17

## Directory Layout

```
VisCy/
├── .github/               # GitHub workflows and issue templates
├── .planning/             # GSD planning documents (this file's parent)
│   └── codebase/          # Generated codebase analysis documents
├── packages/              # Workspace members (uv workspace)
│   ├── viscy-transforms/  # Image transforms (21 modules, 41 exports)
│   │   ├── src/viscy_transforms/
│   │   ├── tests/
│   │   ├── docs/examples/
│   │   └── pyproject.toml
│   ├── viscy-data/        # Data loaders and DataModules (13 modules, 51 exports)
│   │   ├── src/viscy_data/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── viscy-models/      # Pure nn.Module architectures (8 models, 3 families)
│   │   ├── src/viscy_models/
│   │   │   ├── _components/  (stems, heads, blocks, conv_block_2d, conv_block_3d)
│   │   │   ├── unet/         (unext2, fcmae, unet2d, unet25d)
│   │   │   ├── vae/          (beta_vae_25d, beta_vae_monai)
│   │   │   └── contrastive/  (encoder, resnet3d)
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── viscy-utils/       # Shared ML infrastructure (7 exports + subpackages)
│       ├── src/viscy_utils/
│       │   ├── callbacks/     (embedding_writer)
│       │   ├── evaluation/    (linear_classifier, visualization, metrics, etc.)
│       │   ├── cli_utils.py
│       │   ├── cli.py
│       │   ├── trainer.py
│       │   ├── normalize.py
│       │   ├── log_images.py
│       │   ├── precompute.py
│       │   ├── meta_utils.py
│       │   └── mp_utils.py
│       ├── tests/
│       └── pyproject.toml
├── applications/
│   └── dynacrl/           # DynaCLR application
│       ├── src/dynacrl/
│       │   ├── engine.py          (ContrastiveModule, BetaVaeModule LightningModules)
│       │   ├── multi_modal.py     (MultiModalContrastiveModule)
│       │   ├── classification.py  (ClassificationModule)
│       │   ├── vae_logging.py
│       │   └── cli.py             (dynaclr CLI with LazyCommand)
│       ├── configs/               (application-level configs, currently empty)
│       ├── evaluation/
│       │   └── linear_classifiers/  (train, apply, discovery, config gen)
│       ├── examples/
│       │   ├── configs/           (fit.yml, predict.yml, SLURM scripts, ONNX config)
│       │   ├── DynaCLR-DENV-VS-Ph/
│       │   ├── DynaCLR-classical-sampling/
│       │   ├── embedding-web-visualization/
│       │   └── vcp_tutorials/
│       ├── tests/
│       └── pyproject.toml
├── src/                   # Umbrella viscy package (minimal)
│   └── viscy/
│       └── __init__.py    # Version metadata only
├── scripts/               # Utility scripts directory (currently empty, .gitkeep)
├── pyproject.toml         # Workspace root configuration
├── uv.lock                # Locked dependencies (uv)
├── CITATION.cff           # Citation metadata (Zenodo)
├── CONTRIBUTING.md        # Development guidelines
├── LICENSE                # BSD-3-Clause license
└── README.md              # Main project documentation
```

## Directory Purposes

### Packages

**packages/:**
- Purpose: Root directory for uv workspace members
- Contains: Independent packages that can be versioned and published separately
- Key files: Each package has own `pyproject.toml` with version tags

**packages/viscy-transforms/src/viscy_transforms/:**
- Purpose: GPU-accelerated image transforms for microscopy preprocessing
- Contains: 21 transform modules, type definitions, MONAI wrappers
- Key files:
  - `__init__.py`: Public API exports (41 classes/functions)
  - `_typing.py`: Type definitions (Sample, NormMeta, HCSStackIndex, etc.)
  - `_monai_wrappers.py`: Re-exported MONAI transforms with explicit signatures
  - Individual transform files: `_crop.py`, `_flip.py`, `_normalize.py`, etc.
- Pattern: Private implementation files (`_*.py`) re-exported via `__init__.py`

**packages/viscy-data/src/viscy_data/:**
- Purpose: PyTorch Lightning DataModules and Datasets for microscopy data loading
- Contains: 13 data modules covering HCS, triplet, segmentation, classification, GPU augmentation
- Key files:
  - `__init__.py`: Public API exports (51 classes/types/constants)
  - `_typing.py`: Shared type definitions (Sample, NormMeta, ChannelMap, TrackingIndex)
  - `_utils.py`: Internal data utilities
  - `hcs.py`: Core HCSDataModule for OME-Zarr data
  - `triplet.py`: TripletDataModule for contrastive learning
  - `gpu_aug.py`: CachedOmeZarrDataModule and GPUTransformDataModule
  - `combined.py`: ConcatDataModule, BatchedConcatDataModule, CombinedDataModule
  - `cell_classification.py`: ClassificationDataModule for labeled cell data
  - `cell_division_triplet.py`: CellDivisionTripletDataModule
  - `segmentation.py`: SegmentationDataModule
  - `livecell.py`: LiveCellDataModule (requires `[livecell]` extra)
  - `mmap_cache.py`: MmappedDataModule (requires `[mmap]` extra)
  - `ctmc_v1.py`: CTMCv1DataModule
  - `distributed.py`: ShardedDistributedSampler
  - `select.py`: SelectWell transform
- Optional extras: `[triplet]`, `[livecell]`, `[mmap]`, `[all]`

**packages/viscy-models/src/viscy_models/:**
- Purpose: Pure `nn.Module` architectures (no training logic)
- Contains: 8 model classes across 3 families, plus shared components
- Families:
  - `unet/`: UNeXt2, FullyConvolutionalMAE, Unet2d, Unet25d
  - `vae/`: BetaVae25D, BetaVaeMonai
  - `contrastive/`: ContrastiveEncoder, ResNet3dEncoder
- Shared components (`_components/`): stems.py, heads.py, blocks.py, conv_block_2d.py, conv_block_3d.py
- Key files:
  - `__init__.py`: Top-level exports (8 model classes)
  - Each family sub-package has its own `__init__.py`

**packages/viscy-utils/src/viscy_utils/:**
- Purpose: Shared ML infrastructure, training utilities, evaluation tools
- Contains: Training helpers, normalization, logging, evaluation metrics
- Key files:
  - `__init__.py`: Public API (7 exports: detach_sample, render_images, zscore, unzscore, etc.)
  - `trainer.py`: Custom trainer configuration
  - `cli.py`: CLI utilities
  - `cli_utils.py`: CLI helper functions
  - `normalize.py`: zscore, unzscore, hist_clipping functions
  - `log_images.py`: detach_sample, render_images for TensorBoard/WandB
  - `precompute.py`: Normalization statistics precomputation
  - `meta_utils.py`: Metadata handling utilities
  - `mp_utils.py`: Multiprocessing helpers (get_val_stats, mp_wrapper)
- Sub-packages:
  - `callbacks/`: Lightning callbacks (embedding_writer.py)
  - `evaluation/`: Evaluation tools (linear_classifier, visualization, metrics, clustering, dimensionality_reduction, distance, feature, smoothness, annotation, lca, linear_classifier_config)

### Applications

**applications/dynacrl/:**
- Purpose: DynaCLR application -- self-supervised contrastive learning for cellular dynamics
- Contains: Lightning modules, CLI, evaluation pipelines, example configs
- Key files:
  - `src/dynacrl/engine.py`: ContrastiveModule, BetaVaeModule (LightningModule subclasses)
  - `src/dynacrl/multi_modal.py`: MultiModalContrastiveModule (cross-modal distillation)
  - `src/dynacrl/classification.py`: ClassificationModule (downstream task)
  - `src/dynacrl/vae_logging.py`: VAE-specific logging utilities
  - `src/dynacrl/cli.py`: `dynaclr` CLI with LazyCommand pattern for lazy-loading
  - `__init__.py`: Exports BetaVaeModule, ContrastiveModule, ContrastivePrediction
- Sub-directories:
  - `configs/`: Application-level configuration (currently empty)
  - `evaluation/linear_classifiers/`: Train/apply linear classifiers, dataset discovery, config generation
  - `examples/configs/`: fit.yml, predict.yml, SLURM scripts, ONNX export config
  - `examples/DynaCLR-DENV-VS-Ph/`: Dengue infection demo
  - `examples/DynaCLR-classical-sampling/`: Pseudo-track creation for classical sampling
  - `examples/embedding-web-visualization/`: Interactive embedding visualizer
  - `examples/vcp_tutorials/`: VCP quickstart tutorial
  - `tests/`: test_engine.py

### Root-Level

**src/viscy/:**
- Purpose: Umbrella package that ties subpackages together
- Contains: Minimal code (only version metadata)
- Used by: Package imports like `from viscy import __version__`

**.planning/codebase/:**
- Purpose: Generated documentation for codebase analysis
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Generated by: GSD mapper agent, consumed by planner/executor

## Key File Locations

**Entry Points:**
- `pyproject.toml`: Workspace root config, member declaration, Ruff linting rules
- `packages/viscy-transforms/pyproject.toml`: viscy-transforms package config
- `packages/viscy-data/pyproject.toml`: viscy-data package config
- `packages/viscy-models/pyproject.toml`: viscy-models package config
- `packages/viscy-utils/pyproject.toml`: viscy-utils package config
- `applications/dynacrl/pyproject.toml`: dynacrl application config
- `src/viscy/__init__.py`: Umbrella package version only

**Public APIs (package __init__.py files):**
- `packages/viscy-transforms/src/viscy_transforms/__init__.py`: 41 transform exports
- `packages/viscy-data/src/viscy_data/__init__.py`: 51 data exports
- `packages/viscy-models/src/viscy_models/__init__.py`: 8 model exports
- `packages/viscy-utils/src/viscy_utils/__init__.py`: 7 utility exports
- `applications/dynacrl/src/dynacrl/__init__.py`: 3 Lightning module exports

**Configuration:**
- `pyproject.toml`: Build system, dependencies, dev groups, Ruff linting config
- `.pre-commit-config.yaml`: Git pre-commit hooks (linting, formatting)
- `uv.lock`: Locked dependency versions

**Core Transform Logic (viscy-transforms):**
- `packages/viscy-transforms/src/viscy_transforms/_crop.py`: Batched spatial cropping
- `packages/viscy-transforms/src/viscy_transforms/_flip.py`: Batched random flips
- `packages/viscy-transforms/src/viscy_transforms/_normalize.py`: Normalization with precomputed stats
- `packages/viscy-transforms/src/viscy_transforms/_percentile_scale.py`: GPU percentile-based scaling
- `packages/viscy-transforms/src/viscy_transforms/_noise.py`: Batched Gaussian noise on GPU
- `packages/viscy-transforms/src/viscy_transforms/_affine.py`: Affine transforms (Kornia)
- `packages/viscy-transforms/src/viscy_transforms/_zoom.py`: Batched zoom/resize
- `packages/viscy-transforms/src/viscy_transforms/_elastic.py`: 3D elastic deformations
- `packages/viscy-transforms/src/viscy_transforms/_stack_channels.py`: Multi-channel composition
- `packages/viscy-transforms/src/viscy_transforms/_monai_wrappers.py`: Re-exported MONAI transforms

**Core Data Logic (viscy-data):**
- `packages/viscy-data/src/viscy_data/hcs.py`: HCSDataModule (OME-Zarr loading)
- `packages/viscy-data/src/viscy_data/triplet.py`: TripletDataModule (contrastive learning)
- `packages/viscy-data/src/viscy_data/gpu_aug.py`: GPU-accelerated augmentation DataModules
- `packages/viscy-data/src/viscy_data/combined.py`: Combined/Concat DataModules
- `packages/viscy-data/src/viscy_data/cell_classification.py`: ClassificationDataModule
- `packages/viscy-data/src/viscy_data/_typing.py`: Shared type definitions

**Model Architectures (viscy-models):**
- `packages/viscy-models/src/viscy_models/unet/unext2.py`: UNeXt2 architecture
- `packages/viscy-models/src/viscy_models/unet/fcmae.py`: Fully Convolutional MAE
- `packages/viscy-models/src/viscy_models/contrastive/encoder.py`: ContrastiveEncoder
- `packages/viscy-models/src/viscy_models/contrastive/resnet3d.py`: ResNet3dEncoder
- `packages/viscy-models/src/viscy_models/vae/beta_vae_25d.py`: BetaVae25D
- `packages/viscy-models/src/viscy_models/_components/stems.py`: Encoder stems
- `packages/viscy-models/src/viscy_models/_components/heads.py`: Decoder/projection heads
- `packages/viscy-models/src/viscy_models/_components/blocks.py`: Shared building blocks

**Training Infrastructure (viscy-utils):**
- `packages/viscy-utils/src/viscy_utils/trainer.py`: Custom trainer configuration
- `packages/viscy-utils/src/viscy_utils/normalize.py`: zscore/unzscore/hist_clipping
- `packages/viscy-utils/src/viscy_utils/log_images.py`: Image logging for TensorBoard/WandB
- `packages/viscy-utils/src/viscy_utils/precompute.py`: Normalization stats precomputation
- `packages/viscy-utils/src/viscy_utils/callbacks/embedding_writer.py`: Embedding prediction writer
- `packages/viscy-utils/src/viscy_utils/evaluation/linear_classifier.py`: Linear classifier evaluation
- `packages/viscy-utils/src/viscy_utils/evaluation/visualization.py`: Embedding visualization
- `packages/viscy-utils/src/viscy_utils/evaluation/metrics.py`: Evaluation metrics

**DynaCLR Application:**
- `applications/dynacrl/src/dynacrl/engine.py`: ContrastiveModule, BetaVaeModule
- `applications/dynacrl/src/dynacrl/multi_modal.py`: MultiModalContrastiveModule
- `applications/dynacrl/src/dynacrl/cli.py`: `dynaclr` CLI entry point
- `applications/dynacrl/evaluation/linear_classifiers/train_linear_classifier.py`: Train linear classifiers
- `applications/dynacrl/evaluation/linear_classifiers/apply_linear_classifier.py`: Apply classifiers
- `applications/dynacrl/examples/configs/fit.yml`: Training configuration example
- `applications/dynacrl/examples/configs/predict.yml`: Prediction configuration example

**Type Definitions:**
- `packages/viscy-transforms/src/viscy_transforms/_typing.py`: Transform-level TypedDicts
- `packages/viscy-data/src/viscy_data/_typing.py`: Data-level TypedDicts, constants, type aliases

**Testing:**
- `packages/viscy-transforms/tests/`: 8 test files + conftest.py
- `packages/viscy-data/tests/`: 4 test files + conftest.py
- `packages/viscy-models/tests/`: Test directories per family (test_unet/, test_vae/, test_contrastive/, test_components/) + state_dict_compat
- `packages/viscy-utils/tests/`: 2 test files (test_normalize.py, test_mp_utils.py)
- `applications/dynacrl/tests/`: test_engine.py

**Documentation:**
- `README.md`: Project overview, installation, links to Cytoland and DynaCLR
- `CONTRIBUTING.md`: Development setup, guidelines, pre-commit setup
- `packages/viscy-transforms/docs/examples/batched_transforms.ipynb`: Benchmark notebook

## Naming Conventions

**Files (monorepo-wide):**
- Private implementation files: `_<name>.py` (leading underscore)
  - Example: `_crop.py`, `_flip.py`, `_typing.py`, `_utils.py`
- Public modules (viscy-data, viscy-utils, dynacrl): `<name>.py` (no underscore)
  - Example: `hcs.py`, `triplet.py`, `engine.py`, `normalize.py`
- Test files: `test_<name>.py` (one per module or feature group)
  - Example: `test_flip.py`, `test_hcs.py`, `test_engine.py`
- Module exports are re-imported in `__init__.py` at the package level

**Packages/Applications:**
- Package directories: `viscy-<name>` (hyphen-separated)
- Python package names: `viscy_<name>` (underscore-separated, PEP 8)
- Application directories: `<app_name>` (e.g., `dynacrl`)
- Application Python packages: match directory name (e.g., `dynacrl`)

**Functions/Classes:**
- Transform classes: `PascalCase`, optionally with `d` suffix per MONAI convention
  - Tensor variant: `BatchedRandFlip` (operates on Tensor)
  - Dictionary variant: `BatchedRandFlipd` (operates on dict, suffix `d`)
- DataModules: `PascalCase` with `DataModule` suffix (e.g., `HCSDataModule`, `TripletDataModule`)
- Datasets: `PascalCase` with `Dataset` suffix (e.g., `TripletDataset`, `CachedOmeZarrDataset`)
- Model classes: `PascalCase` (e.g., `UNeXt2`, `ContrastiveEncoder`, `BetaVae25D`)
- Lightning modules: `PascalCase` with `Module` suffix (e.g., `ContrastiveModule`, `BetaVaeModule`)
- Internal/private: Leading underscore (e.g., `_match_image()`, `_normalize()`)

**Variables/Parameters:**
- Transform parameters: `snake_case` (e.g., `roi_size`, `random_center`, `spatial_axes`)
- Type hints: PascalCase for classes, `|` for unions (Python 3.10+)
- Dict keys: snake_case (e.g., `"norm_meta"`, `"source"`, `"target"`)

**Types:**
- TypedDict classes: `PascalCase` with descriptive suffixes
  - Example: `NormMeta`, `LevelNormStats`, `ChannelMap`, `HCSStackIndex`, `TrackingIndex`
- Constants: `UPPER_SNAKE_CASE` (e.g., `INDEX_COLUMNS`, `LABEL_INFECTION_STATE`)

## Where to Add New Code

**New Transform:**
1. Create file: `packages/viscy-transforms/src/viscy_transforms/_<transform_name>.py`
2. Implement both tensor and dictionary versions:
   - Tensor version inherits from `Transform` or `RandomizableTransform`
   - Dictionary version inherits from `MapTransform`, calls tensor version internally
3. Pattern to follow:
   - Copy structure from `_flip.py` (paired batch-aware classes)
   - If wrapping MONAI: use pattern from `_monai_wrappers.py`
   - Add docstrings with Parameters/Returns sections (Numpy style per `pyproject.toml`)
4. Export in: `packages/viscy-transforms/src/viscy_transforms/__init__.py`
5. Add tests: `packages/viscy-transforms/tests/test_<transform_name>.py`
6. Run: `pytest packages/viscy-transforms/tests/test_<transform_name>.py`

**New DataModule/Dataset:**
1. Create file: `packages/viscy-data/src/viscy_data/<module_name>.py`
2. Implement `LightningDataModule` subclass and optionally a `Dataset` subclass
3. Pattern to follow:
   - Copy structure from `hcs.py` (core DataModule) or `triplet.py` (contrastive)
   - Use types from `_typing.py` (Sample, NormMeta, ChannelMap, etc.)
   - If optional dependencies needed, add an extra in `pyproject.toml`
4. Export in: `packages/viscy-data/src/viscy_data/__init__.py`
5. Add tests: `packages/viscy-data/tests/test_<module_name>.py`
6. Run: `pytest packages/viscy-data/tests/test_<module_name>.py`

**New Model Architecture:**
1. Choose the appropriate family: `unet/`, `vae/`, or `contrastive/`
   - Or create a new family sub-package if needed
2. Create file: `packages/viscy-models/src/viscy_models/<family>/<model_name>.py`
3. Implement as pure `nn.Module` (no training logic)
4. Use shared components from `_components/` (stems, heads, blocks)
5. Export in the family `__init__.py` and top-level `__init__.py`
6. Add tests: `packages/viscy-models/tests/test_<family>/test_<model_name>.py`
7. Run: `pytest packages/viscy-models/tests/test_<family>/`

**New Utility/Infrastructure:**
1. Add to existing file if closely related (e.g., normalize.py, log_images.py)
2. Otherwise create: `packages/viscy-utils/src/viscy_utils/<module_name>.py`
3. For evaluation tools: add to `packages/viscy-utils/src/viscy_utils/evaluation/`
4. For callbacks: add to `packages/viscy-utils/src/viscy_utils/callbacks/`
5. Export in `__init__.py` if public API
6. Add tests: `packages/viscy-utils/tests/test_<module_name>.py`

**New Application:**
1. Create directory: `applications/<app_name>/`
2. Follow dynacrl structure: `src/<app_name>/`, `tests/`, `examples/`, `pyproject.toml`
3. Lightning modules go in `src/<app_name>/` (engine.py, etc.)
4. Evaluation pipelines go in `evaluation/`
5. Example configs and scripts go in `examples/`
6. Register as workspace member in root `pyproject.toml`

**New Type Definition:**
1. Transform types: add to `packages/viscy-transforms/src/viscy_transforms/_typing.py`
2. Data types: add to `packages/viscy-data/src/viscy_data/_typing.py`
3. Use TypedDict if structure is fixed, dict if flexible
4. Export in `__all__` at top of file and re-export in main `__init__.py`

**New Documentation:**
1. Notebooks go in: `packages/<pkg>/docs/examples/`
2. README updates: `packages/<pkg>/README.md` (package-specific) or main `README.md`
3. Code comments: Follow Numpy docstring style (configured in Ruff)

**New Test:**
1. File: `packages/<pkg>/tests/test_<module>.py` or `applications/<app>/tests/test_<module>.py`
2. Fixtures from `conftest.py` (device, seed)
3. Use pytest parametrize for testing multiple configurations
4. Run per-package: `pytest packages/<pkg>/tests/` or `pytest applications/<app>/tests/`
5. Run full suite: `pytest` from workspace root

## Special Directories

**uv.lock:**
- Purpose: Frozen dependency versions across workspace
- Generated: By `uv lock` command
- Committed: Yes, for reproducible builds
- Edit: Never manually; run `uv lock` to update

**__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Automatically by Python
- Committed: No (in .gitignore)
- Edit: Never; delete with `find . -type d -name __pycache__ -exec rm -r {} +`

**.git/:**
- Purpose: Git version control history
- Generated: `git init` or `git clone`
- Committed: N/A (contains git objects)
- Edit: Use git commands only

**pyc/egg-info/:**
- Purpose: Build artifacts
- Generated: By `pip install -e .` or build tools
- Committed: No (in .gitignore)
- Edit: Never; clean with `rm -rf *.egg-info`

---

*Structure analysis: 2026-02-17*
