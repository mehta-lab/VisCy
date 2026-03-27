# External Integrations

**Analysis Date:** 2026-03-27

## APIs & External Services

**Airtable (Computational Imaging Database):**
- Service: Airtable - Lab metadata, FOV records, marker registry
- What it's used for: Stores experiment metadata, cell line information, channel marker definitions, and dataset registrations
- SDK/Client: `pyairtable` package
- Location: `applications/airtable/src/airtable_utils/database.py`
- Auth: `AIRTABLE_API_KEY` (personal access token), `AIRTABLE_BASE_ID` (workspace base ID)
- Tables Used:
  - "Datasets" - FOV records with experiment metadata
  - "tblmP8l2GmpCeERyD" - Marker Registry (cell_line → channel aliases → protein markers)

**GitHub:**
- Service: GitHub Actions - CI/CD pipeline
- What it's used for: Automated testing across Python 3.11-3.13 on Ubuntu, macOS, Windows
- Location: `.github/workflows/test.yml`, `.github/workflows/lint.yml`
- Runs: Test matrix for packages and applications, linting via Ruff

## Data Storage

**Primary Formats:**
- OME-Zarr (Zarr format) - Multi-dimensional microscopy imaging data
  - Client: `iohub` (wrapper around Zarr for OME-NGFF compliance)
  - Access pattern: Context managers via `open_ome_zarr()` in all data loading code
  - Location: Used throughout `packages/viscy-data/` and applications

**Parquet (Cell Index Storage):**
- Format: Apache Parquet
- What it stores: Flat cell index tables with one row per (cell, timepoint, channel)
- Schema fields: `channel_name`, `marker`, `perturbation`, and other metadata
- Client: PyArrow
- Location: Built by `dynaclr/data/build_cell_index.py`, consumed by datasets

**TensorStore (High-Performance Array Backend):**
- Service: TensorStore library
- What it's used for: Efficient batched data loading via `__getitems__` pattern
- Implementation: `packages/viscy-data/src/viscy_data/triplet.py` uses `ts.stack(...).read().result()` for vectorized I/O
- Supports: Local filesystem, S3, GCS, and other cloud backends
- Read pattern: Conditional import in triplet dataset

**Zarr (Chunk Storage):**
- Format: Zarr - chunked, compressed N-dimensional arrays
- Location: Native format for OME-Zarr imaging data
- Client: `zarr` Python package (underlying iohub)

**File System:**
- Local filesystem only for most deployments
- TensorStore enables optional cloud storage (S3/GCS) at transport layer

## Authentication & Identity

**Auth Provider:**
- Custom (Airtable Token-Based)
  - Implementation: Environment variable-based API key authentication
  - Token variable: `AIRTABLE_API_KEY`
  - Base ID variable: `AIRTABLE_BASE_ID`
  - Location: `applications/airtable/src/airtable_utils/database.py`
  - Validation: Checked at init time in `AirtableDatasets.__init__()`, raises ValueError if missing
  - No OAuth, no session management

## Monitoring & Observability

**Experiment Tracking:**
- Weights & Biases (optional)
  - Used by: DynaCLR training and evaluation (`dynaclr/engine.py`, linear classifier evaluation)
  - Features: Model artifact storage, training metrics, image logging, hyperparameter sweep
  - Client: `wandb` Python SDK
  - Logger: `lightning.pytorch.loggers.WandbLogger`
  - Conditional: Only instantiated if explicitly configured; not required for basic training

**Logs:**
- TensorBoard (local)
  - Used by: All training applications (DynaCLR, Cytoland)
  - Features: Training metrics, image logging, histograms, embeddings
  - Client: `lightning.pytorch.loggers.TensorBoardLogger`
  - Storage: Local filesystem (typically `logs/` directory)
  - Location: Training via Lightning trainer configuration

**Error Handling & Debugging:**
- Python logging module - Standard library logging to console/stderr
- Application-specific: Lightning's built-in logger (`_logger = logging.getLogger("lightning.pytorch")`)

## CI/CD & Deployment

**Hosting:**
- GitHub (source code)
- Shared filesystem or cloud storage for training outputs
- HPC/SLURM for distributed training (no managed hosting)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/`)
  - Test matrix: Python 3.11-3.13 × {Ubuntu, macOS, Windows}
  - Linting: Ruff format and lint checks
  - Testing: pytest with coverage on all packages and applications
  - No automated deployment (manual artifact management via W&B)

**Build System:**
- Hatchling build backend
- `uv` for dependency management and builds
- Version management: `uv-dynamic-versioning` (git-based semver)

**Development Commands:**
```bash
uv sync --all-packages --all-extras  # Install all dependencies
uv run pytest                         # Run all tests
uvx ruff check packages/              # Lint
uvx ruff format packages/             # Format
```

## Environment Configuration

**Required Environment Variables:**
- `AIRTABLE_API_KEY` - Airtable personal access token (for `airtable-utils`)
- `AIRTABLE_BASE_ID` - Airtable workspace base ID (for `airtable-utils`)

**Optional Environment Variables:**
- `PYTHONNOUSERSITE=1` - Set in SLURM scripts to prevent `~/.local/` from shadowing conda env
- WANDB configuration (if using W&B):
  - `WANDB_PROJECT` - Project name for experiment tracking
  - `WANDB_ENTITY` - Workspace/team name
  - `WANDB_API_KEY` - API token (if offline mode not used)

**Secrets Location:**
- Environment variables only (no committed `.env` or `secrets/` directory)
- In development: Set via shell or `.env` (git-ignored)
- On HPC/SLURM: Passed via job submission script or module environment

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- Airtable API writes (via `pyairtable` during registration and metadata updates)
  - Location: `applications/airtable/src/airtable_utils/registration.py` and scripts
- W&B artifact uploads (optional, during training)
  - Location: DynaCLR linear classifier evaluation workflows

---

*Integration audit: 2026-03-27*
