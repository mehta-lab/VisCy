---
phase: 05-ci-cd
plan: 01
subsystem: infrastructure
tags: [github-actions, ci, testing, linting, matrix]
dependency-graph:
  requires:
    - 03-code-migration (code to test/lint)
    - 01-workspace-foundation (pre-commit config)
  provides:
    - Test matrix workflow (Python 3.11/3.12/3.13 x Ubuntu/macOS/Windows)
    - Lint workflow with prek and ruff format check
    - alls-green status check for branch protection
  affects: []
tech-stack:
  added:
    - astral-sh/setup-uv@v7
    - re-actors/alls-green@release/v1
    - actions/checkout@v5
  patterns:
    - GitHub Actions matrix strategy with fail-fast
    - alls-green aggregation pattern for branch protection
    - Conditional cancel-in-progress for PR workflows
key-files:
  created:
    - .github/workflows/test.yml
    - .github/workflows/lint.yml
  modified: []
decisions:
  - name: "Matrix with fail-fast"
    choice: "fail-fast: true"
    rationale: "Quick feedback on failures, save CI minutes"
  - name: "alls-green for status check"
    choice: "re-actors/alls-green@release/v1"
    rationale: "Single status check for branch protection rules"
  - name: "Conditional cancel-in-progress"
    choice: "${{ startsWith(github.ref, 'refs/pull/') }}"
    rationale: "Cancel stale PR runs but never cancel main branch runs"
metrics:
  duration: 94s
  completed: 2026-01-29
---

# Phase 05 Plan 01: CI/CD Workflows Summary

**One-liner:** GitHub Actions CI with 9-job test matrix (3 OS x 3 Python) and lint workflow using uvx prek and ruff format check

## Commits

| Hash | Message |
|------|---------|
| cbd3d95 | feat(05-01): add test workflow with matrix and alls-green |
| 145d94d | feat(05-01): add lint workflow with prek and ruff format check |

## What Was Built

### Test Workflow (.github/workflows/test.yml)

- **Triggers:** Push to main, PRs targeting main
- **Matrix:** 9 jobs (3 OS x 3 Python versions)
  - OS: ubuntu-latest, macos-latest, windows-latest
  - Python: 3.11, 3.12, 3.13
- **Key features:**
  - Uses `astral-sh/setup-uv@v7` with caching (cache-suffix per matrix combination)
  - Runs pytest with coverage in `packages/viscy-transforms/`
  - `alls-green` check job with `if: always()` for branch protection

### Lint Workflow (.github/workflows/lint.yml)

- **Triggers:** Push to main, PRs targeting main
- **Steps:**
  - `uvx prek run --all-files` - runs all pre-commit hooks (ruff-check, ruff-format, ty)
  - `uvx ruff format --check .` - explicit formatting verification
- **Uses:** Python 3.13 (highest supported)

### Concurrency Control

Both workflows use conditional cancel-in-progress:
```yaml
cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}
```
This cancels stale PR runs but never cancels main branch runs.

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- [x] Both YAML files pass validation
- [x] test.yml has 9-job matrix (3 OS x 3 Python)
- [x] test.yml has alls-green check job with `if: always()`
- [x] lint.yml runs `uvx prek run --all-files`
- [x] lint.yml runs `ruff format --check .`
- [x] Both workflows trigger on push/PR to main
- [x] Both workflows have concurrency control

## Next Steps

1. Push to GitHub to verify workflows run successfully
2. Configure branch protection rules to require:
   - "All tests pass" status check (alls-green)
   - "Lint" status check
3. Consider adding workflow for package publishing (future phase)
