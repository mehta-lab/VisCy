---
phase: 22-batch-sampling
verified: 2026-02-23T04:23:37Z
status: passed
score: 5/5 must-haves verified
re_verification: null
gaps: []
human_verification: []
---

# Phase 22: Batch Sampling Verification Report

**Phase Goal:** Users can compose experiment-aware, condition-balanced, and temporally enriched batch sampling strategies via a single configurable FlexibleBatchSampler
**Verified:** 2026-02-23T04:23:37Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | With experiment_aware=True, every batch contains cells from only a single experiment | VERIFIED | `TestExperimentAware::test_batch_indices_from_single_experiment` passes; code line 270-271 picks experiment via `rng.choice`, line 271 sets pool restricted to that experiment's indices |
| 2 | With condition_balanced=True, each batch has approximately equal condition representation per experiment | VERIFIED | `TestConditionBalanced::test_two_conditions_balanced` and `test_three_conditions_balanced` pass; `_sample_condition_balanced` (lines 402-501) enforces per-condition quotas with remainder correction |
| 3 | With temporal_enrichment=True, batches concentrate cells around a focal HPI with a configurable window while including a global fraction | VERIFIED | `TestTemporalEnrichment::test_enriched_batches_concentrate_near_focal` passes (avg focal fraction >= 0.55 asserted); `_enrich_temporal` (lines 321-396) implements focal/global split with `temporal_window_hours` and `temporal_global_fraction` |
| 4 | FlexibleBatchSampler supports DDP via set_epoch() for deterministic shuffling and rank-aware iteration | VERIFIED | `TestDDPDisjointCoverage` (5 tests) all pass; `set_epoch` at line 231, rank-sliced interleaving at line 250 (`all_batches[self.rank :: self.num_replicas]`) |
| 5 | Leaky > 0.0 allows a configurable fraction of cross-experiment samples in otherwise experiment-restricted batches | VERIFIED | `TestLeakyMixing::test_leaky_injects_cross_experiment` passes; lines 277-293 compute `n_leak = int(batch_size * leaky)` and sample from other experiments |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `packages/viscy-data/src/viscy_data/sampler.py` | 220 | 501 | VERIFIED | FlexibleBatchSampler class with all 5 axes; no stubs or TODOs |
| `packages/viscy-data/tests/test_sampler.py` | 350 | 968 | VERIFIED | 35 tests covering SAMP-01 through SAMP-05 plus validation guards, protocol, determinism |
| `packages/viscy-data/src/viscy_data/__init__.py` | contains FlexibleBatchSampler | present at lines 85, 116 | VERIFIED | Imported and in `__all__` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/test_sampler.py` | `viscy_data/sampler.py` | `from viscy_data.sampler import FlexibleBatchSampler` | VERIFIED | Line 21 of test_sampler.py |
| `viscy_data/sampler.py` | `torch.utils.data.Sampler` | `class FlexibleBatchSampler(Sampler[list[int]])` | VERIFIED | Line 25 of sampler.py |
| `viscy_data/__init__.py` | `viscy_data/sampler.py` | `from viscy_data.sampler import FlexibleBatchSampler` | VERIFIED | Line 85 of __init__.py; "FlexibleBatchSampler" in `__all__` at line 116 |
| `viscy_data/sampler.py` | `valid_anchors DataFrame` | `hours_post_infection` column for temporal enrichment | VERIFIED | `_hpi_values` precomputed at lines 143-145; used in `_enrich_temporal` at lines 350-363 |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SAMP-01: Experiment-aware batching | SATISFIED | `TestExperimentAware` (3 tests) all pass; cascade picks single experiment per batch |
| SAMP-02: Condition balancing | SATISFIED | `TestConditionBalanced` (3 tests) all pass; `_sample_condition_balanced` enforces per-condition ratios |
| SAMP-03: Temporal enrichment | SATISFIED | `TestTemporalEnrichment` (6 tests) all pass; `_enrich_temporal` implements focal/global HPI sampling |
| SAMP-04: DDP support | SATISFIED | `TestDDPDisjointCoverage` (5 tests) + `TestDDPPartitioning` (2 tests) all pass; rank-sliced interleaving |
| SAMP-05: Leaky experiment mixing | SATISFIED | `TestLeakyMixing` (3 tests) all pass; `n_leak = int(batch_size * leaky)` cross-experiment injection |

### Anti-Patterns Found

None. Scanned `sampler.py` for: TODO, FIXME, XXX, HACK, PLACEHOLDER, `return null`, `return {}`, empty handlers. Zero matches.

### Human Verification Required

None. All success criteria are mechanically verifiable:
- Experiment isolation: checked via DataFrame index lookup
- Condition ratios: checked statistically over many batches in tests
- Temporal concentration: checked via mode-HPI proximity in test assertions
- DDP interleaving: verified by comparing rank slices to reference full list
- Package import: verified via `uv run python -c "from viscy_data import FlexibleBatchSampler; print(FlexibleBatchSampler)"` returning `<class 'viscy_data.sampler.FlexibleBatchSampler'>`

### Gaps Summary

No gaps. All 5 observable truths are verified at all three levels (exists, substantive, wired).

## Verification Evidence

### Test Run (35/35 pass)

```
packages/viscy-data/tests/test_sampler.py .............................. [ 85%]
.....                                                                    [100%]
============================== 35 passed in 3.49s ==============================
```

### Full Regression Suite (107/107 pass)

```
============================== 107 passed in 13.70s ============================
```

### Lint

```
uvx ruff check packages/viscy-data/src/viscy_data/sampler.py
All checks passed!
```

### Package Import

```
$ uv run python -c "from viscy_data import FlexibleBatchSampler; print(FlexibleBatchSampler)"
<class 'viscy_data.sampler.FlexibleBatchSampler'>
```

### Commits Verified

All 5 TDD phase commits present in git history:
- `f12e128` test(22-01): add failing tests for FlexibleBatchSampler
- `fe38805` feat(22-01): implement FlexibleBatchSampler with experiment-aware, condition-balanced, leaky mixing
- `4b89f53` refactor(22-01): export FlexibleBatchSampler from viscy_data package
- `7a40b6f` test(22-02): add failing tests for temporal enrichment, DDP coverage, validation
- `7de55ee` feat(22-02): implement temporal enrichment, validation guards, DDP coverage

---

_Verified: 2026-02-23T04:23:37Z_
_Verifier: Claude (gsd-verifier)_
