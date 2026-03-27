# Architecture

**Analysis Date:** 2026-03-27

## Pattern Overview

**Overall:** Hierarchical modular pipeline with clear separation between shared infrastructure (`packages/`), domain-specific applications (`applications/`), and Lightning-based training orchestration.

**Key Characteristics:**
- Monorepo structure using `uv` workspace for dependency management
- Unidirectional dependency flow: `applications/ → packages/` (applications never import from each other)
- Lightning-native training with composable DataModules, Transforms, and LightningModules
- Data-centric architecture centered on zarr-based OME formats with parquet cell indices
- GPU-first augmentation pipeline (transforms run on GPU after batch transfer)
- Batched I/O pattern for performance (read multiple patches per single tensorstore call)

## Layers

**Data Layer (`packages/viscy-data/`):**
- Purpose: PyTorch Lightning DataModules, Datasets, and sampler utilities for microscopy data
- Location: `packages/viscy-data/src/viscy_data/`
- Contains: Dataset implementations (HCSDataModule, TripletDataModule, ClassificationDataModule), zarr/parquet loaders, samplers, augmentation classes, type definitions
- Depends on: numpy, torch, lightning, iohub (for zarr I/O), tensorstore, pandas, pyarrow
- Used by: All applications via public API exports in `__init__.py`

**Model Layer (`packages/viscy-models/`):**
- Purpose: Reusable neural network architectures and loss functions
- Location: `packages/viscy-models/src/viscy_models/`
- Contains: UNet variants (Unet2d, Unet25d, UNeXt2), VAE models (BetaVae25D, BetaVaeMonai), Contrastive encoders (ResNet3dEncoder, ContrastiveEncoder), foundation models (DINOv3, OpenPhenom), component building blocks (stems, heads, conv blocks)
- Depends on: torch, monai, torchvision, timm, pytorch-metric-learning
- Used by: Training applications for encoder/decoder instantiation

**Transform Layer (`packages/viscy-transforms/`):**
- Purpose: GPU-native batch transforms for augmentation and preprocessing
- Location: `packages/viscy-transforms/src/viscy_transforms/`
- Contains: Batched*-prefixed transforms (BatchedAffined, BatchedGaussianSmoothd, etc.) designed for `(B, C, Z, Y, X)` tensors; all transforms operate on batches post-GPU-transfer
- Depends on: torch, monai, numpy, scipy
- Used by: DataModules in `on_after_batch_transfer()` hook

**Utilities Layer (`packages/viscy-utils/`):**
- Purpose: ML infrastructure, logging, evaluation, and training utilities
- Location: `packages/viscy-utils/src/viscy_utils/`
- Contains: Callbacks (EmbeddingWriter, PredictionWriter), evaluation modules (linear classifiers, clustering, dimensionality reduction), loss functions, normalization utilities, trainer configuration helpers, CLI utilities
- Depends on: torch, lightning, tensorboard, sklearn, wandb (optional), phate/umap (optional)
- Used by: Training applications for trainer setup, callbacks, and evaluation pipelines

**Application Layer (`applications/*/`):**
- Purpose: Research workflows combining packages for specific tasks
- Locations: `applications/dynaclr/`, `applications/cytoland/`, `applications/airtable/`, `applications/qc/`
- Contains: Training entry points, experiment registries, dataset-specific wrappers, evaluation scripts, CLI commands
- Depends on: All packages + domain-specific libraries (click for CLI, pytorch-metric-learning for contrastive loss)
- Constraint: Applications must never import from each other; shared logic must move to packages

## Data Flow

**Training Pipeline (DynaCLR Example):**

1. **Experiment Registry** (`applications/dynaclr/src/dynaclr/data/experiment.py`)
   - Reads collection YAML or cell_index parquet
   - Resolves zarr paths, channel names, normalization metadata
   - Produces `ExperimentRegistry` with per-experiment configuration

2. **Cell Index** (`packages/viscy-data/src/viscy_data/cell_index.py`)
   - Flat parquet with one row per (cell, timepoint, channel)
   - Columns: cell_id, marker, channel_name, perturbation, hours_post_perturbation, etc.
   - Indexed by `MultiExperimentIndex` (in `applications/dynaclr/src/dynaclr/data/index.py`)

3. **Dataset** (`applications/dynaclr/src/dynaclr/data/dataset.py`: `MultiExperimentTripletDataset`)
   - Reads cell_index parquet via pandas
   - On `__getitems__(indices)`: stacks multiple tensorstore reads into single call
   - Returns pre-batched dict: `{"image": (B, C, Z, Y, X), "index": [...]}`
   - Normalization metadata resolved per-FOV and stacked into `(B,)` tensors

4. **DataLoader & Sampler** (`packages/viscy-data/src/viscy_data/sampler.py`: `FlexibleBatchSampler`)
   - `batch_group_by`: restricts batches to single experiment/marker (optional)
   - `stratify_by`: balances batches across perturbation/condition columns
   - `temporal_enrichment`: concentrates samples around focal timepoints
   - Yields pre-grouped batch indices to dataset

5. **Batch Transfer** (`applications/dynaclr/src/dynaclr/data/datamodule.py`: `on_after_batch_transfer()`)
   - Batch arrives on GPU
   - Applies normalization (via `NormalizeSampled` with pre-resolved `norm_meta`)
   - Applies augmentation (Batched* transforms from viscy-transforms)
   - Returns normalized/augmented batch ready for model

6. **Training Step** (`applications/dynaclr/src/dynaclr/engine.py`: `ContrastiveModule`)
   - Receives normalized batch
   - Encoder produces embeddings
   - Loss function (NTXentLoss, TripletMarginLoss) compares embeddings
   - Backward pass updates weights
   - Logs embeddings, metrics to tensorboard/wandb

**State Management:**
- **Hyperparameters** — YAML collection/config files parsed via jsonargparse/pyyaml
- **Model state** — Lightning checkpoints (`.ckpt`) with optimizer, scheduler, model weights
- **Training state** — Logged to tensorboard/wandb; reproducible via seed management
- **Data state** — Zarr zattrs store per-FOV statistics; parquet stores cell-level metadata
- **Experiment state** — ExperimentRegistry caches zarr metadata in memory; threadlocal tensorstore contexts for distributed reads

## Key Abstractions

**Sample (Type: `viscy_data._typing.Sample`):**
- Purpose: Standardized dict structure for single-sample data
- Structure: `{"image": Tensor(C,Z,Y,X), "index": dict}` optionally with "mask", "label"
- Pattern: Used throughout; all transforms operate on Sample dicts
- Examples: `packages/viscy-data/src/viscy_data/_typing.py` defines TypedDict; `packages/viscy-data/src/viscy_data/hcs.py` produces samples

**TripletSample (Type: `viscy_data._typing.TripletSample`):**
- Purpose: Anchor-positive-negative triplet for contrastive learning
- Structure: `{"anchor": Sample, "positive": Sample, "negative": Sample, "index": CellIndex}`
- Pattern: Produced by TripletDataset; consumed by ContrastiveModule
- Examples: `packages/viscy-data/src/viscy_data/triplet.py`

**CellIndex (Type: `viscy_data._typing.CellIndex`):**
- Purpose: Single cell observation in parquet; one row per (cell, timepoint, channel)
- Fields: cell_id, marker, channel_name, perturbation, hours_post_perturbation, fov_id, experiment_name, etc.
- Pattern: Flattened representation enables efficient DataFrame filtering; no nested structures
- Examples: `packages/viscy-data/src/viscy_data/cell_index.py`

**ExperimentRegistry:**
- Purpose: Configuration LUT mapping experiment names to zarr paths, channels, normalization stats
- Pattern: Built from YAML (collection) or inferred from parquet + zarr zattrs
- Examples: `applications/dynaclr/src/dynaclr/data/experiment.py`

**DictTransform (Type: `viscy_data._typing.DictTransform`):**
- Purpose: Callable that transforms dict[str, Tensor] in-place
- Signature: `Callable[[dict], dict[str, Tensor]]`
- Pattern: All MONAI/custom transforms conform; enable composable pipelines via `Compose`
- Examples: `packages/viscy-transforms/src/viscy_transforms/_normalize.py`

## Entry Points

**DynaCLR Training:**
- Location: `applications/dynaclr/src/dynaclr/cli.py` (Click-based CLI dispatcher)
- Triggers: `dynaclr train` command (lazy-loads training script on invocation)
- Responsibilities: Parse YAML config, instantiate MultiExperimentDataModule, ContrastiveModule, Lightning Trainer, run fit/predict

**DynaCLR Evaluation:**
- Location: `applications/dynaclr/src/dynaclr/cli.py` (LazyCommand pattern)
- Triggers: `dynaclr train-linear-classifier`, `dynaclr evaluate-smoothness`, etc.
- Responsibilities: Load model checkpoint, extract embeddings, run downstream evaluation (linear classification, clustering, dimensionality reduction)

**Shared Utilities CLI:**
- Location: `packages/viscy-utils/src/viscy_utils/cli.py`
- Triggers: `viscy` command (if installed)
- Responsibilities: Configuration loading, data inspection, utility functions

**Testing Entry:**
- Location: Pytest discovers `tests/` and `*_test.py` files (configured in root `pyproject.toml`)
- Paths: `packages/*/tests/`, `applications/*/tests/`
- Responsibilities: Unit and integration tests with real data

## Error Handling

**Strategy:** Prefer raising errors; fail fast with informative messages.

**Patterns:**
- Zarr read failures → iohub/tensorstore exceptions propagate (not caught)
- Missing parquet columns → KeyError or ValueError on DataFrame access (intentional)
- Type mismatches in transforms → AssertionError or TypeError (e.g., expecting `(B,...)` not `(...)`)
- Optional dependencies (e.g., `scikit-learn` for evaluation) → ImportError on import, caught with try/except in __init__.py
- Configuration parsing → jsonargparse/pyyaml exceptions on invalid YAML

**Logging:**
- Module-level logger: `_logger = logging.getLogger(__name__)`
- Stages logged: DataModule setup, batch construction, transform application, training metrics
- Sink: Configured by Lightning (tensorboard by default; wandb optional via callbacks)

## Cross-Cutting Concerns

**Logging:** Lightning's built-in logging system (TensorBoardLogger by default) plus optional wandb integration via `viscy_utils.callbacks`. Custom callbacks in `viscy_utils.callbacks/` handle embedding snapshots and prediction writing.

**Validation:** Minimal validation; rely on type hints and early errors. Cell index validation via `viscy_data.cell_index.validate_cell_index()`. Transform input shapes validated via assertions in transform code.

**Authentication:** None built-in. External systems (Airtable, wandb) managed by application code with env vars.

**Distributed Training:** DDP supported via PyTorch Lightning Trainer; sampler aware of rank/num_replicas (FlexibleBatchSampler). No custom distributed synchronization; Lightning handles all-reduce.

**Performance Optimization:**
- **GPU-first augmentation:** Transforms run post-transfer on GPU (on_after_batch_transfer)
- **Batched I/O:** Dataset.__getitems__ reads multiple patches in one tensorstore call
- **Memory efficiency:** Context managers for zarr/tensorstore (no leaks)
- **Caching:** Optional CPU-side cache (CachedOmeZarrDataModule) for frequently accessed FOVs

---

*Architecture analysis: 2026-03-27*
