---
phase: 23-loss-augmentation
plan: 01
subsystem: loss
tags: [contrastive-learning, ntxent, hard-negatives, pytorch-metric-learning, hcl]

# Dependency graph
requires:
  - phase: 20-experiment-config
    provides: "ExperimentConfig and ExperimentRegistry for DynaCLR pipeline"
provides:
  - "NTXentHCL nn.Module: NT-Xent loss with hard-negative concentration (beta parameter)"
  - "Drop-in replacement for NTXentLoss via isinstance compatibility"
  - "TDD test suite with 12 test cases for loss behavior"
affects: [23-loss-augmentation, dynaclr-training, contrastive-module]

# Tech tracking
tech-stack:
  added: []
  patterns: ["HCL reweighting via _compute_loss override in pytorch_metric_learning pair-based loss pipeline"]

key-files:
  created:
    - "applications/dynaclr/src/dynaclr/loss.py"
    - "applications/dynaclr/tests/test_loss.py"
  modified:
    - "applications/dynaclr/src/dynaclr/__init__.py (already had export from 23-02)"

key-decisions:
  - "Override _compute_loss (pair-based) rather than forward -- integrates with pytorch_metric_learning's reducer/distance pipeline"
  - "beta=0.0 fast-path delegates to super()._compute_loss for exact numerical identity with NTXentLoss"
  - "HCL weights normalized per-anchor to sum to neg_count, preserving loss magnitude across beta values"
  - "Reweighting uses raw cosine similarities (before temperature scaling) for concentration factor"

patterns-established:
  - "Loss module pattern: subclass NTXentLoss, override _compute_loss, return same dict format"
  - "TDD for loss: verify beta=0 equivalence with atol=1e-6 as numerical identity proof"

# Metrics
duration: 4min
completed: 2026-02-23
---

# Phase 23 Plan 01: NTXentHCL Summary

**NTXentHCL loss with beta-controlled hard-negative concentration, subclassing pytorch_metric_learning NTXentLoss with pair-based _compute_loss override**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T19:36:15Z
- **Completed:** 2026-02-23T19:40:00Z
- **Tasks:** 3 (RED, GREEN, REFACTOR)
- **Files modified:** 2 created, 1 already exported

## Accomplishments
- NTXentHCL with beta=0.0 produces numerically identical output to NTXentLoss (atol=1e-6 verified)
- NTXentHCL with beta>0 concentrates loss on hard negatives via exp(beta*sim) reweighting
- isinstance(NTXentHCL(), NTXentLoss) returns True -- ContrastiveModule's training_step NTXent code path activates without modification
- 12 comprehensive tests covering subclass identity, numerical equivalence, gradient flow, edge cases

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for NTXentHCL** - `0b497e2` (test)
2. **GREEN: Implement NTXentHCL** - `b36e614` (feat)
3. **REFACTOR: Add to package exports** - no commit needed (export already present from 23-02)

_Note: The __init__.py already contained the NTXentHCL export from commit 358ee57 (23-02 plan ran first). Refactor phase verified all checks pass._

## Files Created/Modified
- `applications/dynaclr/src/dynaclr/loss.py` - NTXentHCL class (110 lines) with HCL reweighting in _compute_loss
- `applications/dynaclr/tests/test_loss.py` - 12 test cases (205 lines) covering subclass, beta=0 equivalence, hard negatives, gradients, temperature, edge cases, CUDA

## Decisions Made
- **Override _compute_loss, not forward:** NTXentLoss uses pytorch_metric_learning's GenericPairLoss pipeline (distance -> pairs -> _compute_loss -> reducer). Overriding _compute_loss integrates properly with the distance/reducer chain rather than reimplementing the full forward pass.
- **beta=0.0 fast-path:** Delegates directly to super()._compute_loss() for guaranteed numerical identity with standard NTXentLoss, avoiding any floating-point drift from the custom code path.
- **Weight normalization:** HCL weights are normalized so their sum equals the number of negatives per anchor. This preserves loss magnitude, meaning beta only changes the distribution of gradient signal among negatives, not the overall loss scale.

## Deviations from Plan

None - plan executed exactly as written.

_Note: The plan's REFACTOR step to add NTXentHCL to __init__.py was already satisfied by a prior commit (358ee57 from 23-02). This is not a deviation; it simply means the export was already in place._

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- NTXentHCL is ready to be used as `loss_function=NTXentHCL(temperature=0.07, beta=0.5)` in ContrastiveModule
- Configurable via Lightning CLI YAML with `class_path: dynaclr.loss.NTXentHCL`
- Full dynaclr test suite passes (83 tests, 3 skipped for CUDA/HPC)

## Self-Check: PASSED

- [x] loss.py exists (110 lines >= 60 min_lines)
- [x] test_loss.py exists (205 lines >= 120 min_lines)
- [x] SUMMARY.md exists
- [x] Commit 0b497e2 (RED) exists
- [x] Commit b36e614 (GREEN) exists
- [x] `from dynaclr.loss import NTXentHCL` works
- [x] `class NTXentHCL(NTXentLoss)` verified
- [x] `isinstance(NTXentHCL(), NTXentLoss)` passes

---
*Phase: 23-loss-augmentation*
*Completed: 2026-02-23*
