# CELL-DINO RoPE history-shortcut ablation (seed 17)

## Question

Why does five-frame CELL-DINO RoPE improve held-out infection classification while ordered and shuffled histories perform nearly identically?

## Locked hypothesis

The gain is not chronological. It comes either from (a) the current-frame transformer mapping, (b) unordered denoising from additional observations of the same cell, or (c) acquisition/condition context supplied by any matched cells.

## Confirmatory interventions

Use the already-trained five-frame CELL-DINO checkpoints for all five acquisition-held-out folds. Do not retrain. Keep the current/fifth token unchanged in every arm.

1. `ordered`: original causal same-track sequence.
2. `same_track_shuffled`: independently permute the four prior tokens (five deterministic seeds).
3. `current_repeated`: repeat the current token in all five positions.
4. `matched_other_track`: replace the four prior tokens with a different cell's history, matched within acquisition, perturbation, and exact HPI (five deterministic donor seeds).

Each arm gets its own source-control calibration threshold. Primary metrics are held-out within-ZIKV AUROC and raw target-control FPR; calibrated metrics are secondary.

## Locked interpretation thresholds

- Current-frame shortcut: `current_repeated` is within 0.01 mean AUROC and 0.02 mean raw control FPR of `ordered`, with the same direction in at least four of five folds.
- Unordered same-cell denoising: `same_track_shuffled` remains close to `ordered`, `current_repeated` is worse, and `matched_other_track` loses at least 0.01 AUROC or adds at least 0.02 raw FPR in at least four folds.
- Marginal/acquisition-context shortcut: `matched_other_track` is also within the closeness thresholds.
- Temporal-order learning requires ordered performance to exceed the distribution across repeated same-track shuffles. Otherwise attention attribution remains blocked.

## Sanity checks

- Reproduce the saved ordered probabilities/metrics within numerical tolerance.
- Assert exact current-token identity across arms.
- Assert donor track differs from the target track.
- Report donor coverage and any unmatched rows.

This protocol is confirmatory; any additional arms or thresholds are exploratory.
