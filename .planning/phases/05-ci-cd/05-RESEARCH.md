# Phase 5: CI/CD - Research

**Researched:** 2026-01-29
**Domain:** GitHub Actions CI/CD for Python monorepo with uv
**Confidence:** HIGH

## Summary

This research investigates the locked decisions from CONTEXT.md for implementing CI/CD workflows in the viscy-transforms monorepo package. The focus areas are: astral-sh/setup-uv for uv installation and caching, re-actors/alls-green for matrix status check aggregation, uvx prek for pre-commit hook execution, and pytest-cov for terminal coverage reporting.

All researched components are well-documented, actively maintained, and work together seamlessly in GitHub Actions. The setup-uv action (v7) provides excellent caching and Python version management. The alls-green pattern solves the critical "skipped checks pass" problem in GitHub branch protection. prek is fully compatible with existing .pre-commit-config.yaml and offers faster execution than traditional pre-commit.

**Primary recommendation:** Use setup-uv v7 with enable-cache for test matrix, alls-green for status check aggregation, uvx prek for linting, and pytest --cov-report=term-missing for coverage output.

## Standard Stack

### Core

| Library/Action | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| astral-sh/setup-uv | v7 | Install uv with caching in GitHub Actions | Official Astral action, built-in cache support, Python version management |
| re-actors/alls-green | release/v1 | Aggregate matrix job statuses for branch protection | Solves "skipped checks pass" problem, widely adopted pattern |
| prek | latest (via uvx) | Run pre-commit hooks faster | Rust-based, compatible with .pre-commit-config.yaml, 50% less disk space |
| pytest-cov | >=7.0.0 | Coverage reporting in pytest | Already in project dependencies, supports terminal output modes |

### Supporting

| Library/Action | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| actions/checkout | v5 | Clone repository | Every workflow job |
| astral-sh/ruff-action | v3 | Alternative to prek for ruff-only linting | If only ruff checks needed (not recommended - prek covers more) |

### Alternatives Considered (Locked - Not Using)

| Standard | Alternative | Why Not Using |
|----------|-------------|---------------|
| setup-uv | actions/setup-python | setup-uv handles Python installation and provides uv integration |
| prek | pre-commit | prek is locked decision - faster, Rust-based |
| prek | j178/prek-action | Direct uvx prek preferred for control and consistency with local workflow |

**Installation:** Built-in to GitHub Actions via `uses:` directives.

## Architecture Patterns

### Recommended Workflow Structure

```
.github/
└── workflows/
    ├── test.yml       # Test matrix + alls-green check job
    └── lint.yml       # Linting via uvx prek
```

### Pattern 1: Test Matrix with alls-green Status Check

**What:** Run tests across Python version matrix, aggregate results via alls-green job
**When to use:** Always for test workflows with matrix strategy
**Example:**
```yaml
# Source: https://github.com/re-actors/alls-green, https://github.com/astral-sh/setup-uv
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: true
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - run: uv sync --frozen --all-extras --dev
        working-directory: packages/viscy-transforms
      - run: uv run --frozen pytest --cov=viscy_transforms --cov-report=term-missing
        working-directory: packages/viscy-transforms

  check:
    name: All tests pass
    if: always()
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: re-actors/alls-green@release/v1
        with:
          jobs: ${{ toJSON(needs) }}
```

### Pattern 2: Lint Workflow with uvx prek

**What:** Run pre-commit hooks via prek in CI
**When to use:** For linting/formatting checks
**Example:**
```yaml
# Source: https://github.com/j178/prek, https://docs.astral.sh/ruff/integrations
name: Lint

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: "3.13"
          enable-cache: true
      - name: Run prek hooks
        run: uvx prek run --all-files
      - name: Check formatting
        run: uvx ruff format --check .
```

### Pattern 3: Concurrency Control for PR Workflows

**What:** Cancel in-progress runs when new push to same PR, but let main branch complete
**When to use:** All workflows triggered by push and pull_request
**Example:**
```yaml
# Source: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}
```

### Anti-Patterns to Avoid

- **No alls-green job:** Without this, skipped matrix jobs appear as "passing" in branch protection
- **cancel-in-progress: true on main:** Can cancel important main branch workflows; use conditional
- **Missing if: always() on check job:** alls-green job will be skipped if upstream fails, defeating purpose
- **Hardcoded Python versions without quotes:** YAML parses 3.10 as 3.1; always use strings: "3.10"

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Matrix status aggregation | Custom shell script to check job results | re-actors/alls-green | Handles edge cases (skipped, allowed-failures), well-tested |
| uv installation in CI | curl install scripts | astral-sh/setup-uv | Caching, version pinning, Python management built-in |
| Pre-commit hook execution | Manual ruff/mypy commands | uvx prek | Matches local workflow, runs all hooks, handles environments |
| Python version management | actions/setup-python + uv | setup-uv with python-version | Single action handles both, better cache integration |

**Key insight:** GitHub Actions has many edge cases (skipped jobs, concurrency, caching). Using established actions handles these automatically.

## Common Pitfalls

### Pitfall 1: Skipped Status Checks Pass Branch Protection

**What goes wrong:** When a matrix job fails, dependent jobs are skipped. GitHub treats "skipped" as passing for required checks.
**Why it happens:** GitHub's default behavior, not a bug
**How to avoid:** Use alls-green pattern with `if: always()` on the check job
**Warning signs:** PRs can be merged even when tests fail

### Pitfall 2: YAML Floating Point Python Versions

**What goes wrong:** Python version "3.10" becomes 3.1 in matrix
**Why it happens:** YAML interprets unquoted numbers as floats, truncates trailing zeros
**How to avoid:** Always quote Python versions: `["3.11", "3.12", "3.13"]`
**Warning signs:** "Python 3.1 not found" errors, tests running on wrong Python

### Pitfall 3: Cache Conflicts in Matrix Jobs

**What goes wrong:** Matrix jobs overwrite each other's cache or get "(409) Conflict" errors
**Why it happens:** Same cache key used across different configurations
**How to avoid:** Include matrix variables in cache-suffix: `cache-suffix: ${{ matrix.os }}-${{ matrix.python-version }}`
**Warning signs:** Intermittent cache restore failures, inconsistent dependencies

### Pitfall 4: Missing working-directory for Monorepo

**What goes wrong:** Commands run in repo root instead of package directory
**Why it happens:** GitHub Actions defaults to repo root
**How to avoid:** Set `working-directory: packages/viscy-transforms` on relevant steps
**Warning signs:** "pyproject.toml not found", wrong dependencies installed

### Pitfall 5: cancel-in-progress on Main Branch

**What goes wrong:** Rapid successive pushes to main cancel important workflows
**Why it happens:** Unconditional `cancel-in-progress: true`
**How to avoid:** Use conditional: `cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}`
**Warning signs:** Main branch workflows showing as "cancelled"

## Code Examples

### setup-uv with All Relevant Options

```yaml
# Source: https://github.com/astral-sh/setup-uv, Context7 /astral-sh/setup-uv
- name: Install uv
  uses: astral-sh/setup-uv@v7
  with:
    # Python version for UV_PYTHON environment variable
    python-version: ${{ matrix.python-version }}

    # Enable GitHub Actions cache for uv
    enable-cache: true

    # Files that invalidate cache when changed
    cache-dependency-glob: |
      **/pyproject.toml
      **/uv.lock

    # Unique cache per OS/Python combo (prevents conflicts)
    cache-suffix: ${{ matrix.os }}-${{ matrix.python-version }}

    # Prune cache to reduce size
    prune-cache: true
```

### alls-green with Allowed Failures

```yaml
# Source: https://github.com/re-actors/alls-green
check:
  name: All tests pass
  if: always()  # CRITICAL: Must run even when upstream fails
  needs: [test, lint]
  runs-on: ubuntu-latest
  steps:
    - uses: re-actors/alls-green@release/v1
      with:
        jobs: ${{ toJSON(needs) }}
        # Optional: jobs that can fail without failing the check
        # allowed-failures: experimental-python
        # Optional: jobs that can be skipped
        # allowed-skips: docs
```

### pytest-cov Terminal Output

```bash
# Source: Context7 /pytest-dev/pytest-cov
# Show coverage with missing line numbers, skip 100% covered files
pytest --cov=viscy_transforms --cov-report=term-missing:skip-covered tests/

# Output example:
# Name                      Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# viscy_transforms/core.py   257     13    94%   45-47, 102-105
# -------------------------------------------------------
# TOTAL                      353     20    94%
```

### uvx prek in CI

```yaml
# Source: https://github.com/j178/prek
- name: Run prek hooks
  run: uvx prek run --all-files

# Alternative: run specific hooks only
- name: Run ruff checks
  run: uvx prek run --all-files ruff-check ruff-format
```

### Ruff Format Check Mode

```bash
# Source: https://docs.astral.sh/ruff/formatter
# Check without modifying files (non-zero exit if unformatted)
ruff format --check .

# With GitHub Actions annotations for inline PR feedback
ruff check --output-format=github .
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| actions/setup-python + pip install uv | astral-sh/setup-uv | 2024 | Single action, better caching |
| pre-commit (Python) | prek (Rust) | 2024-2025 | ~50% faster, less disk usage |
| Manual status check scripts | re-actors/alls-green | 2022+ | Standardized, handles edge cases |
| Individual linter commands in CI | uvx prek (matches local) | 2024+ | Consistency between local and CI |

**Deprecated/outdated:**
- **pip install uv in CI**: Use setup-uv action instead for caching and version management
- **Manual cache configuration for uv**: setup-uv has built-in cache support with enable-cache

## Open Questions

1. **Exact timeout values for jobs**
   - What we know: setup-uv fetches are fast (<10s), test runs depend on test suite size
   - What's unclear: Appropriate timeout for viscy-transforms tests with PyTorch dependencies
   - Recommendation: Start without explicit timeout, add if jobs hang; PyTorch install may take 2-3 minutes

2. **Path filtering for monorepo**
   - What we know: GitHub supports `paths:` filter in workflow triggers
   - What's unclear: CONTEXT.md says "no path filtering for now" - may want later when more packages added
   - Recommendation: Skip for Phase 5 per CONTEXT.md; document pattern for future reference

## Sources

### Primary (HIGH confidence)

- Context7 /astral-sh/setup-uv - Workflow examples, caching configuration, Python version management
- Context7 /pytest-dev/pytest-cov - Coverage report options, terminal output modes
- https://github.com/astral-sh/setup-uv - Official README with all inputs documented
- https://github.com/re-actors/alls-green - Action configuration and usage patterns
- https://github.com/j178/prek - prek documentation, compatibility with pre-commit
- https://docs.astral.sh/uv/guides/integration/github/ - Official uv GitHub Actions guide
- https://docs.astral.sh/ruff/integrations - Ruff CI integration patterns

### Secondary (MEDIUM confidence)

- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency - GitHub concurrency documentation
- https://github.com/j178/prek-action - prek-action alternative (not using, but researched)

### Tertiary (LOW confidence)

- WebSearch results for community patterns - verified against official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components have official documentation and active maintenance
- Architecture: HIGH - Patterns verified against Context7 and official GitHub docs
- Pitfalls: HIGH - Well-documented issues with known solutions

**Research date:** 2026-01-29
**Valid until:** 2026-03-01 (30 days - stable ecosystem, actions follow semver)
