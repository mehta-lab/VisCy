# control_reference_normalization — when to use it (and when not to)

`control_normalization.py` expresses each cell relative to the **control (untreated)** population
of the **same plate** at the **same hours-post-perturbation (HPI)** bin, using robust statistics
(median + IQR per embedding dimension). Stats-only: `control_reference_stats(adata, bin_hours=...)`
returns a small per-plate table, and `ControlReference.apply(x, hpi)` does `(x - median[bin]) /
iqr[bin]` on the fly — no normalized array is stored.

## What we measured (2026-08, ZIKV/DENV A549, DynaCLR Phase3D + fluorescent markers)

- **Flattens control temporal drift — yes.** Control-centroid std across HPI bins dropped **32–53%**
  on three plates, including a held-out plate the stats were never fit on.
- **Helps downstream classification — no, it slightly hurts.** On a held-out ZIKV plate, all four
  DynaCLR linear classifiers (SEC61B/pAL17 × same-modality/→Phase3D) **lost** AUROC after
  normalization (−0.008 to −0.032). Infection is time-dependent, so flattening the control-relative
  time axis also removes trajectory signal the classifier uses; per-dimension IQR scaling can
  amplify noise on low-variance dimensions.
- **Per-dimension only.** Corrects location + scale per axis, not correlated (multivariate) batch
  directions. True cross-plate batch alignment would need control-covariance whitening (TVN/CORAL) —
  not implemented here. An early plate-prediction "batch" probe was confounded (ZIKV vs DENV differ
  by virus + reporter, so they *should* separate) and is not a valid batch test.

## Guidance

- **Do NOT** put this in front of a classifier that depends on the infection trajectory — it is off
  by default in `LinearClassifiersStepConfig` (`control_normalize=False`) for this reason.
- **Do** use it as a drift diagnostic, or where cross-condition comparability *at matched time* is
  the actual goal — but re-measure your own downstream metric before relying on it.

Full write-up (numbers, figures, dead ends):
`.ed_planning/dynaclr/batch_correction/control_reference_normalization.md`.
