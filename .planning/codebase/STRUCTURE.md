# Codebase Structure

**Analysis Date:** 2026-03-27

## Directory Layout

```
/hpc/mydata/eduardo.hirata/repos/viscy/
├── .planning/                      # GSD planning output (artifacts, documentation)
├── packages/                       # Shared packages (reusable across applications)
│   ├── viscy-data/                 # Data loading, DataModules, Datasets, samplers
│   ├── viscy-models/               # Neural network architectures, heads, components
│   ├── viscy-transforms/           # GPU-native batch transforms for augmentation
│   └── viscy-utils/                # ML utilities, callbacks, evaluation, CLI
├── applications/                   # Research applications (consumers of packages)
│   ├── dynaclr/                    # DynaCLR: contrastive learning for dynamics
│   ├── cytoland/                   # Cytoland: data inspection and curation
│   ├── airtable/                   # Airtable integration utilities
│   └── qc/                         # Quality control and analysis tools
├── src/viscy/                      # Umbrella package (empty re-export namespace)
├── pyproject.toml                  # Root workspace config (ruff, pytest, uv)
├── uv.lock                         # Dependency lock file
├── CLAUDE.md                       # Project-specific coding guidelines
├── CONTRIBUTING.md                 # Contribution guidelines
└── README.md                       # Project overview
```

## Directory Purposes

**packages/viscy-data/**
- Purpose: Data loading and Lightning DataModules for microscopy workflows
- Contains: DataModules (HCSDataModule, TripletDataModule, ClassificationDataModule), Datasets, zarr/parquet I/O, samplers, augmentation utilities, type definitions
- Key files: `src/viscy_data/__init__.py` (public API), `src/viscy_data/hcs.py` (HCSDataModule), `src/viscy_data/triplet.py` (TripletDataModule), `src/viscy_data/sampler.py` (FlexibleBatchSampler)

**packages/viscy-models/**
- Purpose: Neural network architectures and loss functions
- Contains: U-Net variants, VAE models, contrastive encoders, foundation model wrappers, building block components
- Key files: `src/viscy_models/__init__.py` (public API), `src/viscy_models/unet/` (UNet implementations), `src/viscy_models/contrastive/` (contrastive encoders/loss), `src/viscy_models/vae/` (VAE models), `src/viscy_models/components/` (reusable blocks)

**packages/viscy-transforms/**
- Purpose: GPU-native batch transforms (augmentation on GPU post-transfer)
- Contains: Batched MONAI wrappers designed for `(B, C, Z, Y, X)` tensors
- Key files: `src/viscy_transforms/__init__.py` (public API), `src/viscy_transforms/_monai_wrappers.py` (wrapper pattern), all `_*.py` files (individual transforms)

**packages/viscy-utils/**
- Purpose: ML infrastructure, callbacks, evaluation, and utilities
- Contains: Callbacks for embedding/prediction writing, evaluation submodules (linear classifiers, clustering, dimensionality reduction), loss functions, normalization, CLI utilities
- Key files: `src/viscy_utils/__init__.py` (minimal re-exports), `src/viscy_utils/callbacks/` (Lightning callbacks), `src/viscy_utils/evaluation/` (evaluation pipelines), `src/viscy_utils/cli.py` (CLI dispatcher), `src/viscy_utils/trainer.py` (trainer helpers)

**applications/dynaclr/**
- Purpose: DynaCLR self-supervised contrastive learning application for cellular dynamics
- Contains: Training entry points, multi-experiment data handling, experiment registry, evaluation subcommands (linear classifiers, smoothness, dimensionality reduction)
- Key files: `src/dynaclr/cli.py` (entry point), `src/dynaclr/engine.py` (ContrastiveModule, BetaVaeModule LightningModules), `src/dynaclr/data/` (dataset, datamodule, experiment registry, cell index building)

**applications/cytoland/**
- Purpose: Data inspection, curation, and batch composition
- Contains: Data discovery, visual inspection tools, dataset composition utilities
- Key files: Check `src/cytoland/` structure

**applications/airtable/**
- Purpose: Airtable integration for experimental metadata management
- Contains: Airtable utilities, registration workflows, metadata schema definitions
- Key files: `src/airtable_utils/` modules for Airtable operations

**applications/qc/**
- Purpose: Quality control and analysis workflows
- Contains: Data quality assessment, visualization, analysis utilities
- Key files: Check `src/qc/` structure

## Key File Locations

**Entry Points:**
- `applications/dynaclr/src/dynaclr/cli.py` — Main DynaCLR CLI dispatcher; lazy-loads training and evaluation commands
- `packages/viscy-utils/src/viscy_utils/cli.py` — Shared utilities CLI (less commonly used)
- Tests discovered via `pytest` in: `packages/*/tests/` and `applications/*/tests/`

**Configuration:**
- `pyproject.toml` — Root workspace config (ruff linting, pytest paths, uv workspace members)
- `applications/dynaclr/pyproject.toml` — DynaCLR-specific dependencies
- YAML files in `applications/dynaclr/configs/training/` — Training experiment configs
- YAML files in `applications/dynaclr/configs/workflow/` — Collection definitions (zarr paths, channels, perturbations)

**Core Logic:**
- `packages/viscy-data/src/viscy_data/hcs.py` — Base HCSDataModule and Dataset classes
- `packages/viscy-data/src/viscy_data/sampler.py` — FlexibleBatchSampler with grouping and stratification
- `applications/dynaclr/src/dynaclr/data/datamodule.py` — MultiExperimentDataModule
- `applications/dynaclr/src/dynaclr/data/dataset.py` — MultiExperimentTripletDataset with batched I/O
- `applications/dynaclr/src/dynaclr/engine.py` — ContrastiveModule and BetaVaeModule (Lightning training)

**Testing:**
- `packages/viscy-data/tests/` — Tests for data loading, samplers, cell index I/O
- `applications/dynaclr/tests/` — Integration tests for training pipeline (datamodule, dataset, engine)
- Test fixtures in `applications/dynaclr/tests/conftest.py`

**Type Definitions:**
- `packages/viscy-data/src/viscy_data/_typing.py` — Central TypedDict/NamedTuple definitions (Sample, TripletSample, CellIndex, NormMeta, ChannelNormStats, etc.)

## Naming Conventions

**Files:**
- Source modules: `{module_name}.py` (e.g., `hcs.py`, `sampler.py`)
- Private modules: `_{module_name}.py` (e.g., `_typing.py`, `_utils.py`)
- Tests: `{module_name}_test.py` in same dir, or `tests/{module_name}/` for complex test suites
- Config YAML: `{experiment_name}.yml` in `applications/*/configs/*/`

**Directories:**
- Packages: hyphenated lowercase (`viscy-data`, `viscy-models`) in `packages/`
- Package source: underscore format (`viscy_data`, `viscy_models`) in `src/`
- Applications: hyphenated lowercase in `applications/` (e.g., `dynaclr`)
- Application modules: underscore format in `src/{app_name}/` (e.g., `src/dynaclr/data/`, `src/dynaclr/evaluation/`)
- Subpackages: functional names (`data/`, `evaluation/`, `callbacks/`, `components/`)

**Classes:**
- PascalCase: `HCSDataModule`, `ContrastiveModule`, `FlexibleBatchSampler`, `MultiExperimentTripletDataset`
- Lightning modules: suffix with `Module` (e.g., `ContrastiveModule`, `BetaVaeModule`)
- Transforms: prefix with `Batched` for GPU-native variants (e.g., `BatchedAffined`, `BatchedGaussianSmoothd`)

**Functions/Variables:**
- snake_case: `validate_cell_index()`, `write_cell_index()`, `batch_group_by`
- Constants: UPPER_SNAKE_CASE: `CELL_INDEX_CORE_COLUMNS`, `LABEL_CELL_DIVISION_STATE`

**Type Definitions:**
- TypedDict/NamedTuple names: PascalCase with brief names (e.g., `Sample`, `TripletSample`, `CellIndex`, `NormMeta`)

## Where to Add New Code

**New Feature (e.g., new augmentation transform):**
- Primary code: `packages/viscy-transforms/src/viscy_transforms/_{transform_name}.py`
- Update: `packages/viscy-transforms/src/viscy_transforms/__init__.py` (export)
- Tests: `packages/viscy-transforms/tests/test_{transform_name}.py`
- Pattern: Follow existing transform pattern (class inheriting from MONAI; handle batch shape `(B,...)`)

**New DataModule/Dataset:**
- Primary code: `packages/viscy-data/src/viscy_data/{module_name}.py`
- Update: `packages/viscy-data/src/viscy_data/__init__.py` (export)
- Tests: `packages/viscy-data/tests/test_{module_name}.py`
- Pattern: Inherit from `torch.utils.data.Dataset`/`lightning.pytorch.LightningDataModule`; export in __init__

**New Model Architecture:**
- Primary code: `packages/viscy-models/src/viscy_models/{arch_name}/{module_name}.py`
- Update: `packages/viscy-models/src/viscy_models/__init__.py` (export)
- Tests: `packages/viscy-models/tests/test_{arch_name}/test_{module_name}.py`
- Pattern: Inherit from `torch.nn.Module` or PyTorch Lightning module

**New Application Workflow:**
- Primary code: `applications/{app_name}/src/{app_name}/{workflow_name}.py`
- CLI entry: Add LazyCommand to `applications/{app_name}/src/{app_name}/cli.py`
- Tests: `applications/{app_name}/tests/test_{workflow_name}.py`
- Pattern: Use Click for CLI; accept YAML config paths; leverage packages for data/models

**Shared Utilities (if used by multiple applications):**
- Move to: `packages/viscy-utils/src/viscy_utils/{module_name}.py`
- Do NOT add to application-specific directories
- Examples: evaluation scripts, callback implementations, normalization utilities

**Configuration/Constants:**
- YAML configs: `applications/{app_name}/configs/{category}/{name}.yml`
- Python constants: Define at module top-level or in `config.py` files
- Example: `applications/dynaclr/configs/training/` for experiment configs

## Special Directories

**checkpoints/**
- Purpose: Saved Lightning model checkpoints (`.ckpt` files)
- Generated: Yes (created by Lightning Trainer during training)
- Committed: No (should be in .gitignore)

**wandb/**
- Purpose: Weights & Biases run artifacts
- Generated: Yes (if using wandb integration)
- Committed: No

**.pytest_cache/**
- Purpose: Pytest internal cache
- Generated: Yes
- Committed: No

**.venv/**
- Purpose: Python virtual environment (created by `uv venv`)
- Generated: Yes
- Committed: No

**.claude/**
- Purpose: Claude-specific configurations and skills (GSD related)
- Generated: Yes (by Claude)
- Committed: No (locally managed)

**.planning/codebase/**
- Purpose: GSD codebase analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by `/gsd:map-codebase`)
- Committed: Yes (tracked in git for reference)

**configs/ (in each application)**
- Purpose: YAML experiment configurations and collection definitions
- Generated: No (manually created)
- Committed: Yes
- Example: `applications/dynaclr/configs/training/`, `applications/dynaclr/configs/workflow/`

## Import Patterns

**Within Packages:**
Use absolute imports from the package root. Examples:
```python
from viscy_data import HCSDataModule, Sample
from viscy_data.sampler import FlexibleBatchSampler
from viscy_transforms import BatchedAffined
from viscy_utils.callbacks import EmbeddingWriter
from viscy_models import Unet25d, ContrastiveEncoder
```

**Applications importing Packages:**
Import from public API (package-level `__init__.py`). Example:
```python
from viscy_data import HCSDataModule, TripletDataModule
from viscy_models import ContrastiveEncoder, NTXentLoss
from viscy_utils.callbacks import EmbeddingWriter
```

**Within Applications:**
Use absolute imports rooted at application. Example (in dynaclr):
```python
from dynaclr.data.datamodule import MultiExperimentDataModule
from dynaclr.data.experiment import ExperimentRegistry
from dynaclr.engine import ContrastiveModule
```

**Do NOT:**
- Import across applications (e.g., dynaclr → cytoland)
- Use relative imports (e.g., `from ..data import Dataset`)
- Modify `sys.path` for imports

## Configuration Management

**Environment:**
- Env vars configured via shell or `.env` (not tracked in git)
- HPC-specific: symlink uv cache as described in CONTRIBUTING.md

**Build:**
- Root `pyproject.toml` — workspace config and tool settings
- Sub-package `pyproject.toml` files — per-package dependencies
- Ruff config: **ONLY in root** (sub-packages inherit; overrides break tooling)

**Runtime (Application):**
- YAML collection files — define experiments, channels, zarr paths
- YAML training configs — model, optimizer, scheduler, datamodule params
- Parsed via `jsonargparse` + `pyyaml`
- Example: `applications/dynaclr/configs/training/DynaCLR-2D-*.yaml`

---

*Structure analysis: 2026-03-27*
