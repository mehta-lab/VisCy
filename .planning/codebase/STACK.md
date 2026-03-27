# Technology Stack

**Analysis Date:** 2026-03-27

## Languages

**Primary:**
- Python 3.11+ - All source code, packages, and applications use Python
- YAML - Configuration files for training, experiments, and workflows

**Secondary:**
- Shell/Bash - SLURM job submission scripts and CLI utilities

## Runtime

**Environment:**
- Python 3.11, 3.12, 3.13, 3.14 (supported versions)

**Package Manager:**
- `uv` (astral-sh/uv) - Fast, modern Python package manager for workspace dependency management
- Lockfile: `uv.lock` (present and committed)

## Frameworks

**Core ML/Training:**
- PyTorch 2.10+ - Deep learning framework, core dependency for all ML packages
- PyTorch Lightning 2.3+ - Distributed training, callbacks, logging, and model management
- TorchMetrics 1+ - Metric computation during training

**Data Loading & Processing:**
- iohub 0.3a2+ - OME-Zarr reader/writer for microscopy image data
- Zarr - Chunked array storage format for multi-dimensional imaging data
- TensorStore - High-performance array storage and I/O (used in `triplet.py` for batched data loading via `__getitems__`)
- Pandas - Tabular data manipulation for cell indices and metadata
- PyArrow - Parquet file support for flat cell index schemas

**Model Architectures:**
- TIMM (timm 1.0.15+) - Pretrained vision transformer and CNN backbones
- HuggingFace Transformers 4.40+ - Vision models and tokenizers
- MONAI 1.5.2+ - Medical imaging augmentations and components
- Kornia 0.8.2+ - Differentiable image transforms
- Waveorder - Wave optics simulation and analysis (installed from GitHub)

**Contrastive Learning:**
- PyTorch Metric Learning - Contrastive loss functions (NT-Xent, triplet loss)

**Configuration & CLI:**
- Click - CLI framework for command-line tools (`dynaclr`, `qc`, `airtable-utils`)
- jsonargparse 4.26+ - Structured argument parsing with type signatures and YAML config support
- Pydantic 2+ - Data validation and schema definition for all domain objects
- PyYAML - YAML configuration parsing

**Visualization & Logging:**
- Matplotlib 3.10+ - Static plotting
- TensorBoard - Training metrics and image logging via PyTorch Lightning
- Weights & Biases (wandb) - Optional experiment tracking, model artifact storage, and dashboard

**Evaluation & Analysis:**
- Scikit-learn - Linear classifiers, dimensionality reduction
- UMAP - Dimensionality reduction visualization
- PHATE - Progressive harmonic analysis for nonlinear dimensionality reduction
- Scikit-image - Image processing utilities
- AnnData - Annotated data structure for single-cell-like analysis (evaluation extras)
- Statsmodels - Statistical analysis and hypothesis testing
- Seaborn - Statistical data visualization
- Cellpose (optional) - Cell segmentation for evaluation metrics

**Image I/O:**
- ImageIO - Image file reading/writing (PNG, TIFF, etc.)
- TiffFile - TIFF-specific I/O support

**Distributed Training:**
- PyTorch distributed training (DDP) - Native distributed data parallel via Lightning
- SLURM support - Integration via `srun` launcher

**Testing:**
- pytest 9.0.2+ - Test runner and framework
- pytest-cov - Code coverage reports

**Development:**
- Ruff - Fast Python linter and formatter
  - Line length: 120 characters
  - Double quotes for strings
  - Numpy-style docstrings (convention: "numpy")
  - Lint rules: D, E, F, I, NPY, PD, W
- Pre-commit - Git hooks for linting and formatting
- uv-dynamic-versioning - Automatic version management from git tags

**Notebooks & Documentation:**
- JupyterLab 4.5.3+ - Interactive notebook environment
- IPykernel 7.1+ - Jupyter kernel for Python

## Key Dependencies

**Critical Core:**
- `torch>=2.10` - Neural network computation
- `lightning>=2.3` - Training orchestration and distributed support
- `iohub>=0.3a2` - OME-Zarr microscopy data access
- `monai>=1.5.2` - Medical imaging transforms and utilities
- `pydantic>=2` - Data validation for all schemas
- `tensorstore` - High-performance array backend for data loading

**Infrastructure:**
- `pandas` - Cell index and metadata manipulation
- `pyarrow` - Parquet format support
- `pyyaml` - YAML configuration
- `tqdm` - Progress bars

**Optional for Evaluation:**
- `wandb` - Experiment tracking and artifact management (used by DynaCLR for pipeline storage)
- `anndata` - Cell-level analysis and embeddings
- `scikit-learn` - Linear classifiers for representation evaluation
- `statsmodels` - Statistical testing

## Configuration

**Environment Variables (Required for some features):**
- `AIRTABLE_API_KEY` - Personal access token for Airtable integration (required by `airtable-utils`)
- `AIRTABLE_BASE_ID` - Airtable workspace base ID (required by `airtable-utils`)
- `PYTHONNOUSERSITE=1` - Set in SLURM scripts to prevent `~/.local/` from shadowing conda env

**Build Configuration:**
- `pyproject.toml` - Root workspace config (centralized Ruff, pytest, uv settings)
- Each package has its own `pyproject.toml` with specific dependencies
- `uv.lock` - Frozen dependency lock file for reproducibility

**Testing Configuration:**
- `pytest.ini_options` in root `pyproject.toml`
- Test paths: `packages/*/tests`, `applications/*/tests`
- Imports use `importlib` mode to avoid sys.path modification

## Platform Requirements

**Development:**
- Linux, macOS, or Windows
- Python 3.11+ (tested on 3.11, 3.12, 3.13)
- Git (for version management)

**Runtime:**
- GPU (NVIDIA/CUDA) strongly recommended for training, optional for inference
- SLURM job scheduler (for distributed training on HPC)
- Zarr-compatible storage (local filesystem or cloud storage via tensorstore)

**Deployment:**
- OME-Zarr format for input data
- TensorStore-compatible backends (local, S3, GCS, etc.)
- TensorBoard or Weights & Biases for monitoring

---

*Stack analysis: 2026-03-27*
