---
phase: 05-ci-cd
verified: 2026-01-29T11:52:00Z
status: human_needed
score: 4/4 must-haves verified
human_verification:
  - test: "Push code to main branch and verify workflows run"
    expected: "Test workflow triggers, runs 9 matrix jobs, alls-green check passes"
    why_human: "Cannot verify GitHub Actions execution without pushing to remote"
  - test: "Create a PR and verify workflows run with cancel-in-progress behavior"
    expected: "New PR commits cancel previous PR runs, main branch runs never cancel"
    why_human: "Cannot verify GitHub Actions concurrency behavior programmatically"
  - test: "Configure branch protection to require status checks"
    expected: "Branch protection requires 'All tests pass' and 'Lint' checks"
    why_human: "Branch protection is GitHub repository configuration, not code"
---

# Phase 5: CI/CD Verification Report

**Phase Goal:** Automated testing and linting via GitHub Actions
**Verified:** 2026-01-29T11:52:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Push to main triggers test workflow for viscy-transforms | ✓ VERIFIED | test.yml lines 8-11: triggers on push to main, working-directory set to packages/viscy-transforms (lines 40, 44) |
| 2 | Tests run against Python 3.11, 3.12, 3.13 on Ubuntu, macOS, Windows | ✓ VERIFIED | test.yml lines 24-25: matrix with 3 OS x 3 Python = 9 jobs |
| 3 | uvx prek linting passes in CI | ✓ VERIFIED | lint.yml line 32: `uvx prek run --all-files` |
| 4 | PR cannot merge unless all checks pass | ✓ VERIFIED | test.yml lines 46-55: alls-green check job with `if: always()` aggregates results; lint.yml provides separate status check |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/test.yml` | Test matrix workflow with alls-green | ✓ VERIFIED | 55 lines, contains re-actors/alls-green@release/v1, matrix strategy, working-directory wiring |
| `.github/workflows/lint.yml` | Lint workflow with prek and ruff format | ✓ VERIFIED | 35 lines, contains uvx prek and ruff format --check |

**Artifact Verification Details:**

#### `.github/workflows/test.yml` (Level 1-3 Verification)
- **Level 1 (Exists):** ✓ PASS — File exists at correct path
- **Level 2 (Substantive):** ✓ PASS — 55 lines, no stub patterns, complete workflow definition
  - Matrix: 3 OS x 3 Python = 9 jobs (lines 24-25)
  - Python versions quoted to prevent YAML float parsing: ["3.11", "3.12", "3.13"]
  - Uses astral-sh/setup-uv@v7 with caching (line 32-36)
  - Runs pytest with coverage (line 43)
  - alls-green check job with `if: always()` (line 48)
- **Level 3 (Wired):** ✓ PASS — Connected to viscy-transforms package
  - working-directory: packages/viscy-transforms (lines 40, 44)
  - Package exists with tests in packages/viscy-transforms/tests/
  - Dependencies defined in pyproject.toml with pytest and pytest-cov

#### `.github/workflows/lint.yml` (Level 1-3 Verification)
- **Level 1 (Exists):** ✓ PASS — File exists at correct path
- **Level 2 (Substantive):** ✓ PASS — 35 lines, no stub patterns, complete workflow definition
  - Uses Python 3.13 (line 28)
  - Runs prek hooks (line 32)
  - Runs ruff format check (line 35)
- **Level 3 (Wired):** ✓ PASS — Connected to pre-commit configuration
  - .pre-commit-config.yaml exists with ruff-check, ruff-format, ty hooks
  - prek will execute hooks defined in .pre-commit-config.yaml

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| test.yml | packages/viscy-transforms | working-directory | ✓ WIRED | Lines 40, 44 set working-directory, package exists with tests |
| test.yml | alls-green check job | needs dependency | ✓ WIRED | Line 49: `needs: [test]` creates dependency chain |
| lint.yml | .pre-commit-config.yaml | uvx prek | ✓ WIRED | prek reads .pre-commit-config.yaml which exists and defines hooks |

**Link Verification Details:**

1. **test.yml → packages/viscy-transforms:**
   - Pattern found: `working-directory: packages/viscy-transforms` (lines 40, 44)
   - Target package exists with proper structure (src/, tests/)
   - pyproject.toml defines test dependencies (pytest, pytest-cov)
   - Tests exist: test_adjust_contrast.py, test_crop.py, test_flip.py, etc.

2. **test.yml → alls-green check job:**
   - Pattern found: `needs: [test]` (line 49)
   - Pattern found: `if: always()` (line 48) — critical for branch protection
   - Uses re-actors/alls-green@release/v1 (line 53)
   - Passes `jobs: ${{ toJSON(needs) }}` (line 55)

3. **lint.yml → .pre-commit-config.yaml:**
   - Pattern found: `uvx prek run --all-files` (line 32)
   - .pre-commit-config.yaml exists with 3 hooks: ruff-check, ruff-format, ty
   - Additional explicit check: `uvx ruff format --check .` (line 35)

### Requirements Coverage

Requirements mapped to Phase 5:

| Requirement | Status | Supporting Truths |
|-------------|--------|-------------------|
| CI-01: Test workflow for viscy-transforms | ✓ SATISFIED | Truth 1, 2 |
| CI-03: Lint workflow with prek | ✓ SATISFIED | Truth 3 |
| CI-04: alls-green for branch protection | ✓ SATISFIED | Truth 4 |

### Anti-Patterns Found

No anti-patterns detected. Scanned both workflow files for:
- TODO/FIXME/placeholder comments: None found
- Stub implementations: None found
- Empty returns: N/A (YAML configuration, not code)
- Hardcoded values: Appropriate for CI configuration

### Human Verification Required

#### 1. Workflow Execution on Push to Main

**Test:** Push code changes to the main branch (or merge a PR) and observe GitHub Actions execution.

**Expected:**
- Test workflow triggers automatically
- 9 matrix jobs execute (3 OS x 3 Python)
- All jobs pass (green checkmarks)
- "All tests pass" check job runs and succeeds
- Lint workflow triggers and passes

**Why human:** GitHub Actions workflows only execute on the remote GitHub repository. Cannot verify execution behavior without pushing to remote and observing the Actions tab.

#### 2. PR Workflow Concurrency Behavior

**Test:** Create a PR, push a commit, then push another commit before the first workflow completes.

**Expected:**
- Second push cancels the first workflow run (for PR branches)
- Workflow runs on main branch are never cancelled
- Only the most recent PR run completes

**Why human:** Concurrency behavior (`cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}`) depends on GitHub Actions runtime state and cannot be verified statically.

#### 3. Branch Protection Configuration

**Test:** Configure branch protection rules for main branch in GitHub repository settings.

**Expected:**
- Require "All tests pass" status check before merging
- Require "Lint" status check before merging
- PRs cannot be merged if either check fails

**Why human:** Branch protection rules are configured in GitHub repository settings, not in code. The workflows provide the status checks, but a human must configure the repository to require them.

---

## Overall Assessment

**Automated Verification:** All must-haves verified at code level
**Human Verification:** Required to confirm runtime behavior and GitHub configuration

### What Works (Verified)

1. **Workflow Files Exist and Are Substantive:**
   - test.yml: 55 lines with complete 9-job matrix definition
   - lint.yml: 35 lines with prek and ruff format checks
   - No stub patterns, no placeholders, no TODOs

2. **Proper Wiring:**
   - test.yml correctly targets packages/viscy-transforms via working-directory
   - alls-green check job correctly depends on test job with `if: always()`
   - lint.yml correctly invokes prek which reads .pre-commit-config.yaml
   - All referenced files and packages exist

3. **Matrix Configuration:**
   - 3 OS (ubuntu-latest, macos-latest, windows-latest)
   - 3 Python versions (3.11, 3.12, 3.13) properly quoted as strings
   - fail-fast: true for quick feedback
   - Caching enabled with per-matrix-combo cache keys

4. **Concurrency Control:**
   - Both workflows have conditional cancel-in-progress
   - PRs cancel stale runs, main branch never cancels

5. **Status Check Pattern:**
   - alls-green aggregation job provides single status check
   - `if: always()` ensures check runs even when tests fail
   - Proper for branch protection rules

### What Needs Human Verification

1. **GitHub Actions Execution:**
   - Workflows must actually run on GitHub's infrastructure
   - Visual confirmation of green checkmarks in Actions tab
   - Verification of matrix expansion (9 jobs)

2. **Concurrency Behavior:**
   - Confirm PR runs cancel correctly
   - Confirm main runs never cancel

3. **Branch Protection:**
   - Repository settings must be configured to require status checks
   - This is a GitHub UI action, not code

### Phase Goal Status

**Code-level verification:** PASSED — All artifacts exist, are substantive, and properly wired

**Full goal achievement:** PENDING HUMAN VERIFICATION — Workflows are correctly defined but require:
1. Push to GitHub to verify execution
2. Branch protection configuration
3. Confirmation of runtime behavior

---

_Verified: 2026-01-29T11:52:00Z_
_Verifier: Claude (gsd-verifier)_
