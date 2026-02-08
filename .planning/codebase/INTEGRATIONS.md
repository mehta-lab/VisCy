# External Integrations

**Analysis Date:** 2026-02-07

## APIs & External Services

**Hugging Face Spaces:**
- Cytoland Demo - Hosted virtual staining demo
  - URL: https://huggingface.co/spaces/chanzuckerberg/Cytoland
  - Purpose: Interactive 2D virtual staining of cell nuclei and membrane from label-free images
  - No SDK integration in code (deployed separately)

**Chan Zuckerberg Initiative - Virtual Cells Platform:**
- Virtual Cells Models API
  - Model card: https://virtualcellmodels.cziscience.com/model/01961244-1970-7851-a4b9-fdbfa2fba9b2
  - Quick-start: https://virtualcellmodels.cziscience.com/quickstart/cytoland-quickstart
  - Purpose: Model serving and inference for trained Cytoland models (VSCyto2D, VSCyto3D, VSNeuromast)
  - Access: Via CLI tutorials and model deployment API
  - No direct SDK integration in core library

**CZ Biohub Public Data:**
- Public data storage
  - URL: https://public.czbiohub.org/comp.micro/viscy/DynaCLR_demo/
  - Purpose: Hosting DynaCLR demo dataset and model checkpoints
  - Access: Direct HTTP download (via tools like curl, wget, or pooch)
  - Client: pooch 1.9+ (for data fetching and caching in notebooks)

## Data Storage

**Databases:**
- Not detected - This is a transforms/inference library, not a data persistence layer

**File Storage:**
- Local filesystem only
  - PyTorch checkpoint files (.pt, .pth)
  - MONAI/Kornia native in-memory operations
  - Data expected to be pre-staged by users
  - Demo checkpoints retrieved via pooch or direct download

**Caching:**
- Pooch 1.9+ (optional, for notebook usage)
  - Provides: Data fetching with local caching
  - Used in: `optional-dependencies.notebook` group
  - Purpose: Cache demo datasets and model checkpoints locally

## Authentication & Identity

**Auth Provider:**
- None required
- Public repositories and open-source model access
- All integrations are unauthenticated HTTP/file-based

## Monitoring & Observability

**Error Tracking:**
- None detected
- Standard Python exception handling and logging

**Logs:**
- Standard Python logging
- No external log aggregation or observability tools detected
- Console output via pytest when running tests

## CI/CD & Deployment

**Hosting:**
- GitHub (source repository)
  - URL: https://github.com/mehta-lab/VisCy
  - Organization: mehta-lab (Chan Zuckerberg Biohub)

**CI Pipeline:**
- GitHub Actions workflows in `.github/workflows/`

**Test Pipeline (test.yml):**
- Trigger: Push to main, pull requests to main
- Matrix testing:
  - OS: ubuntu-latest, macos-latest, windows-latest
  - Python: 3.11, 3.12, 3.13
- Setup: astral-sh/setup-uv@v7 with cache enabled
- Commands:
  ```bash
  uv sync --frozen --all-extras --dev
  uv run --frozen pytest --cov=viscy_transforms --cov-report=term-missing
  ```
- Working directory: `packages/viscy-transforms`

**Lint Pipeline (lint.yml):**
- Trigger: Push to main, pull requests to main
- Setup: Python 3.13 with uv
- Command:
  ```bash
  uvx prek run --all-files
  ```
- Tools: ruff-check, ruff-format, pyproject-fmt, git hooks

**Package Distribution:**
- PyPI (https://pypi.org)
  - viscy (umbrella package)
  - viscy-transforms (transforms library)
- Build system: hatchling
- Version management: uv-dynamic-versioning (git-based)
- License: BSD-3-Clause

## Environment Configuration

**Required env vars:**
- None explicitly required for library usage
- PyTorch optional: CUDA_VISIBLE_DEVICES for GPU selection

**Secrets location:**
- None detected
- No .env files or credential management in library code
- CI/CD uses GitHub Actions secrets for any deployments (not visible in config)

## Webhooks & Callbacks

**Incoming:**
- GitHub webhooks (GitHub Actions triggered by push/PR)
- No external webhooks

**Outgoing:**
- None detected
- No callbacks to external services from library code

---

*Integration audit: 2026-02-07*
