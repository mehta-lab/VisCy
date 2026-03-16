# Recipe: Sampling Strategies for DynaCLR

## Overview

`FlexibleBatchSampler` controls **what ends up in each training batch** through
four composable axes. The right combination depends on your scientific question
and dataset structure.

| Axis | Parameter | What it controls |
|------|-----------|------------------|
| Experiment selection | `experiment_aware` | Whether batches are restricted to one experiment |
| Leaky mixing | `leaky` | Fraction of cross-experiment samples injected into experiment-pure batches |
| Stratification | `stratify_by` | Balance batches by column(s) (e.g. condition, organelle) |
| Temporal enrichment | `temporal_enrichment` | Concentrate batches around a focal hours-post-perturbation (HPP) window |

Additionally, the **positive pair** is controlled by `tau_range` and
`tau_decay_rate`, which determine how far in time the positive is from the
anchor.

---

## Recommended configurations

### 1. Temporal contrastive learning (default for infection studies)

**Goal:** Learn representations that capture morphological changes over infection
while distinguishing infected from uninfected cells at the same disease stage.

```yaml
experiment_aware: true
stratify_by: condition
temporal_enrichment: true
temporal_window_hours: 2.0
temporal_global_fraction: 0.3
tau_range: [0.5, 2.0]
tau_decay_rate: 2.0
channel_dropout_prob: 0.5
```

**What each batch looks like:**

- All cells from one experiment (consistent channel semantics)
- ~50% infected, ~50% uninfected (from `stratify_by`)
- ~70% of cells within +/-2h of a randomly chosen focal HPP
- Anchor-positive pairs are the same cell separated by 0.5-2h

**Why this works:** The hardest and most informative negatives are
cross-condition cells at similar HPP. An uninfected cell and an infected cell
at 12h post-perturbation look similar but have different biology. The model must
learn subtle morphological signatures of perturbation response rather than just cell
age or imaging artifacts.

**When to use:** Multi-condition time-lapse experiments where you want
perturbation-aware temporal representations.

---

### 2. Augmentation-only contrastive (SimCLR-style)

**Goal:** Learn augmentation-invariant representations without temporal signal.
Useful as a baseline or when tracking data is unreliable.

```yaml
experiment_aware: true
stratify_by: condition
temporal_enrichment: true
temporal_window_hours: 2.0
temporal_global_fraction: 0.3
tau_range: [0, 0]          # positive = same cell, same frame
channel_dropout_prob: 0.5
```

**What each batch looks like:**

- Same composition as configuration 1
- But the positive is the **same cell at the same timepoint**, with different
  random augmentations (crops, flips, intensity jitter, noise)

**Why this works:** The model learns features invariant to imaging noise and
augmentation while still benefiting from cross-condition negatives at similar
HPP. No temporal continuity is learned.

**When to use:** As a baseline to measure the added value of temporal positives.
Also useful when tracking quality is poor (frequent ID swaps) and temporal
positives would be unreliable.

> **Note:** `tau_range: [0, 0]` is not yet implemented. The current code skips
> `tau=0` in the fallback loop. This will require a code change to support.

---

### 3. Cross-experiment regularization (leaky mixing)

**Goal:** Learn representations that generalize across experiments with different
imaging conditions (staining intensity, illumination, microscope).

```yaml
experiment_aware: true
leaky: 0.3                 # 30% from other experiments
stratify_by: condition
temporal_enrichment: true
temporal_window_hours: 2.0
temporal_global_fraction: 0.3
tau_range: [0.5, 2.0]
channel_dropout_prob: 0.5
```

**What each batch looks like:**

- ~70% cells from one experiment, ~30% from other experiments
- Condition balance and temporal enrichment still apply to the primary pool
- The leaked samples provide cross-experiment negatives

**Why this works:** The leaked cross-experiment samples act as hard negatives
that force the encoder to ignore batch effects (microscope-specific intensity
distributions, background patterns, PSF differences). The model learns features
that transfer across experiments.

**When to use:**

- You have **replicate experiments** with the same perturbation and reporters, and want
  batch-effect-invariant representations
- You have enough experiments (3+) that cross-experiment diversity is meaningful
- **Channel dropout is important here** since different experiments may have
  different fluorescence reporters. The model learns to rely on phase contrast
  which is consistent across experiments

**When NOT to use:**

- You only have 1-2 experiments (not enough diversity to regularize against)
- Experiments have fundamentally different biology (different cell types,
  perturbations) where cross-experiment negatives would be misleading

---

### 4. Multi-column stratification

**Goal:** Balance batches by multiple metadata columns simultaneously.

```yaml
experiment_aware: true
stratify_by: [condition, organelle]   # balance by both
temporal_enrichment: false
tau_range: [0.5, 2.0]
```

**What each batch looks like:**

- All cells from one experiment
- Equal representation of each (condition, organelle) combination
  (e.g., ~25% infected+mito, ~25% infected+ER, ~25% uninfected+mito,
  ~25% uninfected+ER)

**Why this works:** When you have multiple experimental factors, single-column
stratification can leave one factor unbalanced. Multi-column stratification
creates a cross-product of groups and balances all of them.

**When to use:** Experiments with multiple metadata dimensions you want
the model to distinguish (e.g., perturbation x organelle reporter, dose x
timepoint category).

---

### 5. Experiment-mixed (no experiment awareness)

**Goal:** Maximize batch diversity by mixing all experiments freely.

```yaml
experiment_aware: false
stratify_by: condition
temporal_enrichment: false
tau_range: [0.5, 2.0]
```

**What each batch looks like:**

- Cells from any experiment, proportional to experiment size
- Condition-balanced across the global pool

**Why this works:** Every batch contains cross-experiment pairs, providing
maximum diversity. This can help when all experiments share the same channel
semantics and you want to maximize the effective dataset size per batch.

**When to use:**

- All experiments have **identical channel names and semantics**
- You want maximum batch diversity and don't care about experiment identity
- Useful for late-stage fine-tuning after learning experiment-specific
  representations

**When NOT to use:**

- Experiments have **different fluorescence reporters** (GFP vs RFP).
  Mixing them in one batch means the fluorescence channel has different
  biological meaning for different samples, which confuses the encoder

---

### 6. Minimal / fully random (diagnostic baseline)

**Goal:** No structured sampling. Useful only for debugging or as a
lower-bound baseline.

```yaml
experiment_aware: false
stratify_by: null
temporal_enrichment: false
tau_range: [0.5, 2.0]
```

**What each batch looks like:**

- Random cells from any experiment, any condition, any timepoint
- Natural distribution proportional to sample counts

**When to use:** Only as a diagnostic baseline to verify that structured
sampling (configs 1-5) actually improves representation quality. Compare
linear probe accuracy or temporal smoothness metrics.

---

## Decision flowchart

```
Do experiments have different fluorescence reporters?
  YES -> experiment_aware: true
  NO  -> experiment_aware: false is fine

Do you have multiple conditions (infected/uninfected/mock)?
  YES -> stratify_by: condition
  NO  -> stratify_by: null

Is temporal structure important to your question?
  YES -> temporal_enrichment: true
         tau_range: [0.5, 2.0] (temporal positives)
  NO  -> temporal_enrichment: false
         tau_range: [0, 0] (augmentation-only positives)

Do you want cross-experiment generalization?
  YES -> leaky: 0.2-0.3 (with channel_dropout_prob >= 0.5)
  NO  -> leaky: 0.0
```

## Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_aware` | bool | `true` | Restrict each batch to one experiment |
| `stratify_by` | str, list, or null | `"condition"` | Column(s) to balance within batches |
| `leaky` | float | `0.0` | Fraction of batch from other experiments (only with `experiment_aware`) |
| `temporal_enrichment` | bool | `false` | Concentrate batch around focal HPP |
| `temporal_window_hours` | float | `2.0` | Half-width of focal window in hours |
| `temporal_global_fraction` | float | `0.3` | Fraction of batch drawn from all timepoints |
| `tau_range` | [float, float] | `[0.5, 2.0]` | Hours range for temporal positive offset |
| `tau_decay_rate` | float | `2.0` | Exponential decay favoring shorter offsets |
| `channel_dropout_prob` | float | `0.5` | Probability of zeroing fluorescence channel |
