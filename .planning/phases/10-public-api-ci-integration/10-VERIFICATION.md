---
phase: 10-public-api-ci-integration
verified: 2026-02-13T19:15:00Z
status: passed
score: 5/5
re_verification: false
---

# Phase 10: Public API & CI Integration Verification Report

**Phase Goal:** Users can `from viscy_models import ModelName` for all 8 models, with CI verifying the full package
**Verified:** 2026-02-13T19:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                                      | Status     | Evidence                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------ |
| 1   | `from viscy_models import UNeXt2, FullyConvolutionalMAE, ContrastiveEncoder, ResNet3dEncoder, BetaVae25D, BetaVaeMonai, Unet2d, Unet25d` succeeds without error | ✓ VERIFIED | Tested: imports work, `__all__` lists all 8 classes |
| 2   | `uv run --package viscy-models pytest` passes all tests including new compatibility tests                                 | ✓ VERIFIED | 93 tests (92 passed, 1 xfailed pre-existing), including 24 new state dict compat tests                            |
| 3   | CI workflow runs viscy-models tests alongside viscy-transforms tests                                                       | ✓ VERIFIED | `.github/workflows/test.yml` matrix includes both packages (18 jobs: 3 OS x 3 Python x 2 packages)                |
| 4   | State dict keys for all 8 models match their original monolithic counterparts exactly                                     | ✓ VERIFIED | 24 regression tests (3 per model: count + prefixes + sentinels) all pass                                          |
| 5   | Root pyproject.toml already lists viscy-models as workspace dependency (no change needed)                                  | ✓ VERIFIED | Line 28: `dependencies = [ "viscy-models", "viscy-transforms" ]`, Line 52: `viscy-models = { workspace = true }` |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact                                                     | Expected                                                                 | Status     | Details                                                                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------ |
| `packages/viscy-models/src/viscy_models/__init__.py`         | Top-level re-exports of all 8 model classes                             | ✓ VERIFIED | Lines 7-9: imports from subpackages, Lines 11-20: `__all__` with all 8 classes in alphabetical order  |
| `packages/viscy-models/tests/test_state_dict_compat.py`      | State dict key compatibility regression tests for all 8 models          | ✓ VERIFIED | 324 lines, 8 test classes with 3 tests each (count, prefixes, sentinels), all 24 tests pass           |
| `.github/workflows/test.yml`                                 | CI test matrix including viscy-models                                    | ✓ VERIFIED | Line 22: matrix includes `package: [viscy-transforms, viscy-models]`, working-directory uses variable  |

### Key Link Verification

| From                                                 | To                                                       | Via                   | Status     | Details                                                                                  |
| ---------------------------------------------------- | -------------------------------------------------------- | --------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| `packages/viscy-models/src/viscy_models/__init__.py` | `packages/viscy-models/src/viscy_models/unet/__init__.py` | re-export imports     | ✓ WIRED    | Line 8: `from viscy_models.unet import FullyConvolutionalMAE, UNeXt2, Unet2d, Unet25d` |
| `packages/viscy-models/src/viscy_models/__init__.py` | `packages/viscy-models/src/viscy_models/contrastive/__init__.py` | re-export imports | ✓ WIRED    | Line 7: `from viscy_models.contrastive import ContrastiveEncoder, ResNet3dEncoder`      |
| `packages/viscy-models/src/viscy_models/__init__.py` | `packages/viscy-models/src/viscy_models/vae/__init__.py` | re-export imports     | ✓ WIRED    | Line 9: `from viscy_models.vae import BetaVae25D, BetaVaeMonai`                         |
| `.github/workflows/test.yml`                         | `packages/viscy-models`                                  | CI test job working-directory | ✓ WIRED | Line 37: `working-directory: packages/${{ matrix.package }}` with `package: [viscy-transforms, viscy-models]` |

### Requirements Coverage

| Requirement | Description                                                          | Status       | Evidence                                                                              |
| ----------- | -------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------- |
| API-01      | `from viscy_models import UNeXt2` works for all 8 model classes     | ✓ SATISFIED  | Tested successfully, all 8 classes importable from top-level                          |
| API-02      | `uv run --package viscy-models pytest` passes all tests              | ✓ SATISFIED  | 93 tests (92 passed, 1 xfailed pre-existing)                                         |
| API-03      | CI test matrix updated to include viscy-models                       | ✓ SATISFIED  | Matrix expanded from 9 to 18 jobs (3 OS x 3 Python x 2 packages)                     |
| API-04      | Root pyproject.toml updated with viscy-models workspace dependency   | ✓ SATISFIED  | Already present (no change needed): lines 28, 52                                      |
| COMPAT-01   | State dict keys preserved identically for all migrated models        | ✓ SATISFIED  | 24 regression tests verify parameter counts, prefixes, and sentinel keys for all 8 models |

### Anti-Patterns Found

**None** — No TODOs, FIXMEs, placeholders, empty implementations, or stub patterns detected in modified files.

### Human Verification Required

None — All phase goals are programmatically verifiable and have been verified.

---

## Summary

**Phase 10 goal ACHIEVED:** Users can `from viscy_models import ModelName` for all 8 models, with CI verifying the full package.

All 5 observable truths verified:
1. Top-level imports work for all 8 model classes
2. Full test suite passes (93 tests: 92 passed, 1 xfailed pre-existing)
3. CI matrix includes viscy-models alongside viscy-transforms (18 jobs)
4. State dict compatibility regression tests pass for all 8 models
5. Root pyproject.toml already lists viscy-models as workspace dependency

All 5 requirements (API-01, API-02, API-03, API-04, COMPAT-01) satisfied.

All artifacts exist, are substantive (not stubs), and are wired correctly.

**This is the FINAL phase of milestone v1.1** — viscy-models package is feature-complete with public API, full test coverage, and CI. Ready for PR to main branch.

---

_Verified: 2026-02-13T19:15:00Z_
_Verifier: Claude (gsd-verifier)_
