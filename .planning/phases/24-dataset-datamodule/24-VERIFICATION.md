---
phase: 24-dataset-datamodule
verified: 2026-02-23T22:05:58Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 24: Dataset & DataModule Verification Report

**Phase Goal:** Users can train DynaCLR across multiple experiments using MultiExperimentTripletDataset and MultiExperimentDataModule, which wire together all sampling, loss, and augmentation components with full Lightning CLI configurability

**Verified:** 2026-02-23T22:05:58Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `__getitems__` returns dict with `anchor`, `positive` Tensor keys of shape (B,C,Z,Y,X), plus `anchor_norm_meta` (consumed by DataModule before engine sees batch) | VERIFIED | `dataset.py` lines 146-171, engine `training_step` reads `batch["anchor"]` and `batch["positive"]`, test `test_getitems_returns_anchor_positive_keys` asserts shape `(2, 2, 1, 32, 32)`, all 7 dataset tests pass |
| 2 | Positive sampling follows lineage through division events -- shared `lineage_id` links parent track and daughter tracks, enabling t+tau sampling across division boundaries | VERIFIED | `_reconstruct_lineage` in `index.py` sets `lineage_id` to root ancestor's `global_track_id` for all descendants; `_find_positive` looks up `(lineage_id, t+tau)` in pre-built lookup -- test `test_positive_through_division` asserts daughters share parent's `lineage_id` and are reachable as positives |
| 3 | MultiExperimentDataModule wires FlexibleBatchSampler + Dataset + ChannelDropout + ThreadDataLoader with `collate_fn=lambda x: x`, and train/val split is by whole experiments | VERIFIED | `datamodule.py` lines 285-320: `FlexibleBatchSampler` as `batch_sampler` for train only, `ThreadDataLoader` for both, `collate_fn=lambda x: x` on both loaders; setup() splits by `exp.name not in self.val_experiments`; tests `test_train_dataloader_uses_flexible_batch_sampler`, `test_val_dataloader_no_batch_sampler`, `test_train_val_split_by_experiment` all pass |
| 4 | All hyperparameters (tau_range, tau_decay_rate, experiment_aware, condition_balanced, temporal_enrichment, hcl_beta, channel_dropout_prob) exposed as `__init__` parameters | VERIFIED | All 7 hyperparameters present in `MultiExperimentDataModule.__init__` signature (lines 105-129) and stored on `self`; test `test_init_exposes_all_hyperparameters` asserts all values, passes |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `applications/dynaclr/src/dynaclr/dataset.py` | MultiExperimentTripletDataset class | VERIFIED | 352 lines; substantive implementation with `__getitems__`, `_sample_positives`, `_find_positive`, `_slice_patches`, `_get_tensorstore`, `_build_lineage_lookup`; imported and used by `datamodule.py` |
| `applications/dynaclr/tests/test_dataset.py` | TDD tests with `test_getitems_returns_anchor_positive` | VERIFIED | 392 lines; 7 tests across 5 classes; `test_getitems_returns_anchor_positive_keys` present; all 7 tests pass |
| `applications/dynaclr/src/dynaclr/datamodule.py` | MultiExperimentDataModule LightningDataModule | VERIFIED | 382 lines; substantive implementation; `setup()`, `train_dataloader()`, `val_dataloader()`, `on_after_batch_transfer()` fully implemented |
| `applications/dynaclr/tests/test_datamodule.py` | TDD tests with `test_train_val_split_by_experiment` | VERIFIED | 445 lines; 6 tests across 6 classes; `test_train_val_split_by_experiment` present; all 6 tests pass |
| `applications/dynaclr/src/dynaclr/__init__.py` | Updated top-level exports with both classes | VERIFIED | Both `MultiExperimentTripletDataset` and `MultiExperimentDataModule` imported and in `__all__`; import verified at CLI |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `dataset.py` | `dynaclr.index.MultiExperimentIndex` | `self.index.valid_anchors` for anchor lookup | WIRED | Line 146: `anchor_rows = self.index.valid_anchors.iloc[indices]`; line 119: iterates `self.index.tracks` in `_build_lineage_lookup` |
| `dataset.py` | `dynaclr.tau_sampling.sample_tau` | temporal offset for positive selection | WIRED | Line 29: `from dynaclr.tau_sampling import sample_tau`; line 237: `sampled_tau = sample_tau(tau_min, tau_max, rng, self.tau_decay_rate)` |
| `dataset.py` | `dynaclr.experiment.ExperimentRegistry` | `channel_maps` for per-experiment channel index remapping | WIRED | Line 315: `channel_map = self.index.registry.channel_maps[exp_name]`; line 316: `channel_indices = [channel_map[i] for i in sorted(channel_map.keys())]` |
| `datamodule.py` | `dataset.py` (MultiExperimentTripletDataset) | creates train and val dataset instances | WIRED | Lines 232, 250: `MultiExperimentTripletDataset(index=..., fit=True, ...)` for both train and val |
| `datamodule.py` | `viscy_data.sampler.FlexibleBatchSampler` | `batch_sampler` for train DataLoader | WIRED | Lines 287-299: `FlexibleBatchSampler(valid_anchors=..., ...)` created and passed as `batch_sampler=sampler` at line 303 |
| `datamodule.py` | `viscy_data.channel_dropout.ChannelDropout` | applied in `on_after_batch_transfer` | WIRED | Line 172: `self.channel_dropout = ChannelDropout(...)` in `__init__`; lines 377-379: `batch[key] = self.channel_dropout(batch[key])` applied to anchor and positive |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| DATA-01: Dataset returns ContrastiveModule-compatible batch dict | SATISFIED | Truth 1 verified; `batch["anchor"]` + `batch["positive"]` as Tensors; engine unchanged |
| DATA-02: Positive sampling follows lineage through division events | SATISFIED | Truth 2 verified; lineage_id propagated to daughters in `_reconstruct_lineage`, tested |
| DATA-03: DataModule wires FlexibleBatchSampler + ChannelDropout + ThreadDataLoader | SATISFIED | Truth 3 verified; all components wired and tested |
| DATA-04: Train/val split by whole experiments, not FOVs | SATISFIED | Truth 3 verified; setup() filters by experiment name; `test_train_val_split_by_experiment` confirms no FOV overlap |
| DATA-05: All hyperparameters exposed as __init__ parameters | SATISFIED | Truth 4 verified; 14 hyperparameters including all 7 named, stored, and passed through to FlexibleBatchSampler / ChannelDropout |

### Anti-Patterns Found

None. No TODOs, FIXMEs, placeholders, empty implementations, or stub returns detected in `dataset.py` or `datamodule.py`.

### Human Verification Required

None. All critical behaviors are covered by automated tests with synthetic zarr fixtures that exercise real I/O paths (not mocked). The tests verified:
- Actual tensor shapes from tensorstore reads
- Real lineage reconstruction through division events
- Real experiment-level train/val split
- Real ChannelDropout behavior in train vs eval mode

### Test Summary

```
applications/dynaclr/tests/test_dataset.py  -- 7 passed in 3.77s
applications/dynaclr/tests/test_datamodule.py  -- 6 passed in 3.38s
Total: 13 passed
```

### Implementation Notes

1. **norm_meta handling:** `__getitems__` returns `anchor_norm_meta` and `positive_norm_meta` in the batch dict. These are consumed by `on_after_batch_transfer` before the engine's `training_step` receives the batch. The engine only reads `batch["anchor"]` and `batch["positive"]`, so the batch format is fully compatible without engine changes.

2. **Division lineage:** `_reconstruct_lineage` in `index.py` sets each track's `lineage_id` to its root ancestor's `global_track_id` via parent graph traversal. Daughters share the parent's `lineage_id`. The dataset's `_build_lineage_lookup` indexes by `(experiment, lineage_id) -> {t: [row_indices]}`, enabling O(1) positive lookup that naturally crosses division boundaries.

3. **collate_fn=lambda x: x:** Both train and val dataloaders use identity collation because `__getitems__` returns an already-batched dict (not a list of individual samples). `FlexibleBatchSampler` provides batched indices.

4. **hcl_beta on DataModule:** Stored for Lightning CLI YAML discoverability but not functionally used by the DataModule. The actual `NTXentHCL` is configured on `ContrastiveModule`. This is intentional per the plan.

---

_Verified: 2026-02-23T22:05:58Z_
_Verifier: Claude (gsd-verifier)_
