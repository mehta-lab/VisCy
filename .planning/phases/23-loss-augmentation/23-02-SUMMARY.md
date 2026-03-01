---
phase: 23-loss-augmentation
plan: 02
subsystem: augmentation
tags: [channel-dropout, tau-sampling, exponential-decay, contrastive-learning, gpu-augmentation]

# Dependency graph
requires:
  - phase: 23-01
    provides: "NTXentHCL loss module (DynaCLR loss foundation)"
provides:
  - "ChannelDropout nn.Module for GPU augmentation pipeline in viscy-data"
  - "sample_tau exponential decay temporal offset sampler in dynaclr"
  - "Top-level exports: viscy_data.ChannelDropout, dynaclr.sample_tau"
affects: [23-03, dynaclr-dataset, dynaclr-datamodule, training-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [per-sample-channel-masking, exponential-decay-weighted-sampling]

key-files:
  created:
    - packages/viscy-data/src/viscy_data/channel_dropout.py
    - packages/viscy-data/tests/test_channel_dropout.py
    - applications/dynaclr/src/dynaclr/tau_sampling.py
    - applications/dynaclr/tests/test_tau_sampling.py
  modified:
    - packages/viscy-data/src/viscy_data/__init__.py
    - applications/dynaclr/src/dynaclr/__init__.py

key-decisions:
  - "ChannelDropout clones input tensor (non-destructive) for pipeline safety"
  - "Per-sample independent dropout via torch.rand mask on batch dimension"
  - "Exponential decay tau sampling uses normalized offset for consistent behavior across tau ranges"

patterns-established:
  - "GPU augmentation modules: nn.Module with train/eval mode gating"
  - "Weighted discrete sampling: numpy rng.choice with computed probability vectors"

# Metrics
duration: 3min
completed: 2026-02-23
---

# Phase 23 Plan 02: ChannelDropout and Variable Tau Sampling Summary

**ChannelDropout nn.Module for per-sample channel zeroing on (B,C,Z,Y,X) tensors, plus exponential-decay tau sampling utility for temporal contrastive learning**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-23T19:36:20Z
- **Completed:** 2026-02-23T19:39:33Z
- **Tasks:** 3 (TDD RED/GREEN/REFACTOR)
- **Files modified:** 6

## Accomplishments
- ChannelDropout with per-sample stochastic channel zeroing, train/eval mode gating, and dtype/device preservation
- Exponential decay sample_tau utility biasing temporal positive selection toward small offsets
- 18 comprehensive tests (11 ChannelDropout + 7 tau_sampling) covering edge cases, probabilistic behavior, determinism
- Top-level package exports for both modules

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests** - `0b497e2` (test)
2. **GREEN: Implementation** - `048d0fa` (feat)
3. **REFACTOR: Package exports** - `358ee57` (refactor)

_TDD cycle: test -> feat -> refactor_

## Files Created/Modified
- `packages/viscy-data/src/viscy_data/channel_dropout.py` - ChannelDropout nn.Module for GPU augmentation
- `packages/viscy-data/tests/test_channel_dropout.py` - 11 tests for ChannelDropout behavior
- `applications/dynaclr/src/dynaclr/tau_sampling.py` - Exponential decay tau sampling utility
- `applications/dynaclr/tests/test_tau_sampling.py` - 7 tests for variable tau sampling
- `packages/viscy-data/src/viscy_data/__init__.py` - Added ChannelDropout export
- `applications/dynaclr/src/dynaclr/__init__.py` - Added sample_tau and NTXentHCL exports

## Decisions Made
- ChannelDropout clones input tensor (non-destructive) to avoid corrupting upstream pipeline state
- Per-sample independent dropout via `torch.rand(B)` mask provides proper stochastic regularization
- Exponential decay uses normalized offset `(tau - tau_min) / (tau_max - tau_min)` for consistent behavior regardless of tau range

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `ruff` not available via `uv run ruff` (workspace root); resolved by using `uv tool run ruff` instead
- Linter auto-added `NTXentHCL` import to dynaclr `__init__.py` (from a prior phase's module); included in refactor commit

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ChannelDropout ready for integration into GPU augmentation pipeline (after scatter/gather chain)
- sample_tau ready for use in DynaCLR dataset's temporal positive pair selection
- Both modules have comprehensive test coverage and clean lint

## Self-Check: PASSED

All 5 created files verified on disk. All 3 commit hashes (0b497e2, 048d0fa, 358ee57) found in git log.

---
*Phase: 23-loss-augmentation*
*Completed: 2026-02-23*
