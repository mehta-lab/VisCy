---
phase: 23-loss-augmentation
verified: 2026-02-23T19:44:26Z
status: gaps_found
score: 3/4 must-haves verified
re_verification: false
gaps:
  - truth: "ChannelDropout integrates into on_after_batch_transfer after the existing scatter/gather augmentation chain"
    status: partial
    reason: "ChannelDropout module exists, is tested, and is designed for integration (documented in docstring), but it is not wired into any existing DataModule's on_after_batch_transfer. The module is orphaned -- it exists and is exported but no DataModule calls it."
    artifacts:
      - path: "packages/viscy-data/src/viscy_data/channel_dropout.py"
        issue: "Module exists and is correct but not wired into any on_after_batch_transfer in the codebase"
      - path: "packages/viscy-data/src/viscy_data/triplet.py"
        issue: "on_after_batch_transfer exists (line 574) but does not call ChannelDropout"
    missing:
      - "Wire ChannelDropout into TripletDataModule.on_after_batch_transfer (or a DynaCLR-specific DataModule) after the _transform_channel_wise scatter/gather chain"
      - "Add a test verifying that on_after_batch_transfer applies ChannelDropout (e.g., mock or integration test)"
human_verification:
  - test: "Confirm that ChannelDropout integration into on_after_batch_transfer is intentionally deferred to Phase 24 (MultiExperimentDataModule)"
    expected: "If Phase 24 wires ChannelDropout into the new DataModule's on_after_batch_transfer, then SC3 is satisfied end-to-end at phase 24 completion, not phase 23"
    why_human: "The ROADMAP Phase 24 description explicitly says MultiExperimentDataModule wires ChannelDropout. If this is the intended integration phase, then Phase 23's ChannelDropout truth should be read as 'ready to integrate', not 'already integrated'."
---

# Phase 23: Loss & Augmentation Verification Report

**Phase Goal:** Users have an HCL-enhanced contrastive loss, channel dropout augmentation, and variable tau sampling -- all independent modules that plug into the existing DynaCLR training pipeline
**Verified:** 2026-02-23T19:44:26Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | NTXentHCL computes NT-Xent loss with hard-negative concentration (beta parameter), returns scalar with gradients, numerically identical to standard NT-Xent when beta=0.0 | VERIFIED | 11/11 tests pass (1 skipped CUDA); `test_ntxent_hcl_beta_zero_matches_standard` passes with atol=1e-6; `test_ntxent_hcl_returns_scalar_with_grad` and `test_ntxent_hcl_backward_passes` pass |
| 2 | NTXentHCL is an nn.Module that works as drop-in replacement via ContrastiveModule(loss_function=NTXentHCL(...)) without changes to training step | VERIFIED | `isinstance(NTXentHCL(), NTXentLoss)` returns True (confirmed live); engine.py lines 105, 178, 209 all use `isinstance(..., NTXentLoss)` which passes for NTXentHCL subclass |
| 3 | ChannelDropout randomly zeros specified channels with configurable probability on batched (B,C,Z,Y,X) tensors and integrates into on_after_batch_transfer after the existing scatter/gather augmentation chain | PARTIAL | Module exists and all 10 tests pass; p=0.0 identity, p=1.0 always drops, eval mode identity -- all verified. BUT: no actual wiring in any DataModule's on_after_batch_transfer. The module is orphaned. |
| 4 | Variable tau sampling uses exponential decay within tau_range, favoring small temporal offsets -- verified by statistical distribution test | VERIFIED | 7/7 tests pass; `test_sample_tau_exponential_favors_small` confirms median < midpoint (5.5) with N=10000; `test_sample_tau_uniform_when_zero_decay` and `test_sample_tau_strong_decay` verify distribution properties |

**Score:** 3/4 truths verified (truth 3 is partial)

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `applications/dynaclr/src/dynaclr/loss.py` | 60 | 110 | VERIFIED | `class NTXentHCL(NTXentLoss)` at line 15; `_compute_loss` override at line 40; beta fast-path at line 52 |
| `applications/dynaclr/tests/test_loss.py` | 120 | 205 | VERIFIED | 12 test cases covering subclass, beta=0 equivalence, hard negatives, gradients, temperature, edge cases, CUDA |
| `packages/viscy-data/src/viscy_data/channel_dropout.py` | 40 | 35 | VERIFIED | 35 lines (5 short of 40 min_lines but substantive: complete implementation with docstring, forward(), train/eval gate, per-sample masking) |
| `packages/viscy-data/tests/test_channel_dropout.py` | 80 | 144 | VERIFIED | 11 test cases covering all required behaviors |
| `applications/dynaclr/src/dynaclr/tau_sampling.py` | 30 | 36 | VERIFIED | Complete `sample_tau` function with exponential decay weighting |
| `applications/dynaclr/tests/test_tau_sampling.py` | 50 | 88 | VERIFIED | 7 test cases covering range, distribution, edge cases, determinism, return type |

Note: `channel_dropout.py` is 35 lines vs. 40 min_lines, but the implementation is complete and substantive (no stub indicators). The 5-line shortfall is due to a concise but correct implementation.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `applications/dynaclr/tests/test_loss.py` | `applications/dynaclr/src/dynaclr/loss.py` | `from dynaclr.loss import NTXentHCL` | WIRED | Line 8 in test_loss.py; pattern matches |
| `applications/dynaclr/src/dynaclr/loss.py` | `pytorch_metric_learning.losses` | `class NTXentHCL(NTXentLoss)` | WIRED | Line 15 in loss.py; subclass confirmed |
| `applications/dynaclr/src/dynaclr/engine.py` | `applications/dynaclr/src/dynaclr/loss.py` | `isinstance(self.loss_function, NTXentLoss)` check passes for NTXentHCL | WIRED | Lines 105, 178, 209 in engine.py; `isinstance(NTXentHCL(), NTXentLoss)` confirmed True at runtime |
| `packages/viscy-data/tests/test_channel_dropout.py` | `packages/viscy-data/src/viscy_data/channel_dropout.py` | `from viscy_data.channel_dropout import ChannelDropout` | WIRED | Line 6 in test_channel_dropout.py |
| `packages/viscy-data/src/viscy_data/__init__.py` | `packages/viscy-data/src/viscy_data/channel_dropout.py` | top-level re-export | WIRED | Line 43 in __init__.py: `from viscy_data.channel_dropout import ChannelDropout` |
| `applications/dynaclr/tests/test_tau_sampling.py` | `applications/dynaclr/src/dynaclr/tau_sampling.py` | `from dynaclr.tau_sampling import sample_tau` | WIRED | Line 6 in test_tau_sampling.py |
| Any DataModule | `packages/viscy-data/src/viscy_data/channel_dropout.py` | on_after_batch_transfer call | NOT WIRED | No DataModule in codebase calls ChannelDropout; triplet.py on_after_batch_transfer (line 574) does not include ChannelDropout |

### Requirements Coverage

| Requirement | Status | Details |
|-------------|--------|---------|
| LOSS-01 (NTXentHCL formula with beta) | SATISFIED | `_compute_loss` override with HCL weighting; `exp(beta * sim)` reweighting per line 77 |
| LOSS-02 (beta=0.0 equivalence) | SATISFIED | Fast-path delegates to `super()._compute_loss()` at line 53; test passes with atol=1e-6 |
| LOSS-03 (NTXentLoss subclass) | SATISFIED | `isinstance(NTXentHCL(), NTXentLoss)` True; drop-in for ContrastiveModule |
| AUG-01 (ChannelDropout zeros channels) | SATISFIED | Module correct; tests pass; p=0/1 edge cases work |
| AUG-02 (ChannelDropout integrates into augmentation chain) | BLOCKED | Module exists but is not wired into on_after_batch_transfer in any DataModule |
| AUG-03 (Variable tau exponential decay) | SATISFIED | sample_tau with decay distribution; statistical test passes |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | -- | -- | No TODO/FIXME/placeholder patterns in any implementation file |

### Human Verification Required

#### 1. Scope Clarification: ChannelDropout Integration Timing

**Test:** Review ROADMAP Phase 24 description and confirm whether ChannelDropout integration into on_after_batch_transfer is intentionally deferred to Phase 24's MultiExperimentDataModule.
**Expected:** ROADMAP Phase 24 says "MultiExperimentDataModule wires FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader". If Phase 23's intent was to deliver a ready-to-wire module (not yet wired), then truth 3 should be re-scoped.
**Why human:** The ROADMAP success criterion says "integrates into on_after_batch_transfer" -- but Phase 24 is where the MultiExperimentDataModule (the intended integration host) is built. The intended scope of "integration" in Phase 23 vs 24 requires human judgment.

### Gaps Summary

One gap blocks full goal achievement:

**Truth 3 (ChannelDropout integration):** The ChannelDropout module is fully implemented, tested, and exported -- but it is not called from any `on_after_batch_transfer` in the codebase. The ROADMAP success criterion says it "integrates into on_after_batch_transfer after the existing scatter/gather augmentation chain" but no DataModule wires it. This could be:

1. An intentional deferral -- Phase 24 (MultiExperimentDataModule) is explicitly described in ROADMAP as the phase that wires ChannelDropout. If so, Phase 23's truth should be read as "provides a ready-to-integrate module" not "already integrated."
2. A genuine gap -- something should have been wired in Phase 23 (e.g., into TripletDataModule.on_after_batch_transfer as a proof of integration).

Given the ROADMAP language and Phase 24's explicit responsibility for wiring, this is likely interpretation (1). However, without modifying the current plan's stated truth, this is recorded as a gap requiring human confirmation.

All other truths are fully verified:
- NTXentHCL: numerically correct, drop-in compatible, gradient-flows -- all 11 tests pass
- Variable tau: statistical distribution tests pass, return type correct, deterministic

---

_Verified: 2026-02-23T19:44:26Z_
_Verifier: Claude (gsd-verifier)_
