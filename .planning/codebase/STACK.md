# Technology Stack

**Analysis Date:** 2026-02-07

## Languages

**Primary:**
- Python 3.11+ - Core language for all packages
  - Officially supports: 3.11, 3.12, 3.13, 3.14
  - SPEC 0 compliant (Minimum Supported Dependencies)

## Runtime

**Environment:**
- CPython (via astral-sh setup-uv in CI/CD)

**Package Manager:**
- uv 1.0+ (Rust-based package manager for Python)
  - Lockfile: `uv.lock` (present, 481KB, workspace-aware)
  - Build backend: hatchling with uv-dynamic-versioning

## Frameworks

**Core Deep Learning:**
- PyTorch 2.10+ - Core tensor computation and neural network framework
  - Used throughout transforms for GPU-accelerated image operations
  - `torch.nn.functional` for advanced tensor operations
  - Location: `packages/viscy-transforms/src/viscy_transforms/`

**Image Processing & Medical Imaging:**
- MONAI 1.5.2+ - Medical Open Network for AI transforms library
  - Provides base Transform, MapTransform, RandomizableTransform classes
  - Dictionary-based transforms pattern (transforms ending in "d")
  - Key transforms wrapped in `packages/viscy-transforms/src/viscy_transforms/_monai_wrappers.py`
  - Used for: spatial crops, affine transforms, intensity scaling, decollation

- Kornia 0.8.2+ - PyTorch-based computer vision library
  - GPU-accelerated image transforms
  - `kornia.augmentation.RandomAffine3D` for 3D affine transformations
  - `kornia.filters` for Gaussian filters (filter3d, get_gaussian_erf_kernel1d)
  - `kornia.constants.BorderType` for image padding modes

**Testing:**
- pytest 9.0.2+ - Test runner and framework
  - Config: `pyproject.toml` under `[tool.pytest]`
  - Coverage: pytest-cov 7+
  - Test discovery: `packages/*/tests`, `tests/`

**Build/Dev:**
- Ruff 0.14.14+ - Python linter and formatter (astral-sh)
  - Configuration: `pyproject.toml` under `[tool.ruff]`
  - Line length: 120 characters
  - Linting rules: D (docstrings), E (errors), F (PyFlakes), I (imports), NPY, PD, W
  - Format style: double quotes, 4-space indent
  - Docstring convention: NumPy style
  - Pre-commit integration: `astral-sh/ruff-pre-commit@v0.14.14`

- pyproject-fmt - TOML file formatter
  - Pre-commit hook: `tox-dev/pyproject-fmt@v2.11.1`

## Key Dependencies

**Critical:**
- numpy 2.4.1+ - Numerical computing
  - Type hints via `numpy.typing.DTypeLike`
  - NumPy docstring convention used throughout

- torch 2.10+ - PyTorch (listed above under frameworks)

- monai 1.5.2+ - Medical imaging transforms (listed above under frameworks)

- kornia 0.8.2+ - Computer vision (listed above under frameworks)

**Type Hints & Compatibility:**
- typing-extensions - Backported typing features
  - Provides: `Literal`, `Iterable`, `Sequence`, `NotRequired` for Python 3.11 compatibility

**Optional (notebook extras):**
- matplotlib 3.10.8+ - Plotting
- scikit-image 0.26+ - Image processing utilities
- cmap 0.7+ - Colormap utilities
- pooch 1.9+ - Data fetching and caching
- jupyterlab 4.5.3+ - Interactive notebook environment
- ipykernel 7.1+ - Jupyter kernel

## Configuration

**Environment:**
- No `.env` file detected
- All configuration via `pyproject.toml` workspace configuration
- uv workspace defined in `pyproject.toml` with members: `packages/*`

**Build:**
- Build system: hatchling
- Source mapping: `packages/viscy-transforms/src/viscy_transforms/`
- Version source: uv-dynamic-versioning (git-based, PEP 440 style)

**Code Quality:**
- Pre-commit hooks: `.pre-commit-config.yaml`
  - pyproject-fmt, ruff-check, ruff-format, git hooks
- Linting: Ruff with strict rules
- Formatting: Ruff formatter with double quotes, 120 char lines
- Test paths: `packages/*/tests` and `tests/`

## Platform Requirements

**Development:**
- macOS (latest) - Tested in CI
- Ubuntu (latest) - Tested in CI
- Windows (latest) - Tested in CI
- Python 3.11, 3.12, 3.13 minimum (3.14 supported)
- uv 1.0+ package manager

**Production:**
- Any platform supporting Python 3.11+
- Deployment: PyPI package distribution
  - viscy (umbrella package): https://pypi.org/project/viscy
  - viscy-transforms (image transforms): https://pypi.org/project/viscy-transforms

**Hardware:**
- NVIDIA CUDA support optional (PyTorch with CUDA)
- CPU-only mode supported (PyTorch fallback)

---

*Stack analysis: 2026-02-07*
