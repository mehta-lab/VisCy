# Straightening on DynaCLR — exploration

**Status (2026-07-27):** promoted to implementation; review findings incorporated.

## Question

Can DynaCLR learn a common biological progression without erasing state organization?

## Working hypothesis

Straightening alone is a per-track geometric prior; it does not directly reward a transition shared across cells. The selected objective therefore combines:

`L = L_NT-Xent + lambda_pred * L_pred + lambda_curv * L_curv`

The shared predictor supplies the common-transition pressure, curvature regularization makes local motion easier to interpret, and NT-Xent resists constant-state collapse.

## Implementation findings

### Identity is stricter than lineage

A lineage may contain a parent and multiple daughters. Sampling any same-lineage row at each timepoint can create a synthetic path that changes physical cells. Temporal stencils must match the anchor's exact `global_track_id`; a division makes the stencil invalid.

### Augmentation is part of the temporal model

Applying random transforms to flattened `B*K` frames independently injects artificial motion. The default must reuse one random realization across the `K` frames of each track while allowing different realizations across tracks. Independent and augmentation-free modes remain useful ablations.

### Masked loss still has distributed semantics

On a rank with no valid sequences, returning a zero connected only to encoder output leaves predictor parameters unused. Passing the empty input through the predictor produces zero gradients for its parameters and keeps DDP iteration state consistent.

### Evaluation bins need explicit boundary semantics

HPI loops use half-open intervals `[lo, hi)`. `np.arange(start, max + width, width)` still omits `max` when it is exactly a boundary. The edge builder must create one additional terminal edge, and widths must be finite and positive before arithmetic.

## Decisions retained

- Operate on encoder features, matching the representation being regularized.
- Use three or more points for curvature; two points provide smoothing only.
- Keep fixed-frame spacing initially for a small, auditable implementation.
- Treat velocity-based phenotype splitting as a possible discovery signal, not automatically as representation damage.
- Evaluate biology and collapse jointly; straightness by itself is not a success criterion.

## Rejected shortcuts

- Same-lineage sampling without exact track identity.
- Independent stochastic transforms as the default temporal input.
- Skipping configured modules on empty masked batches under DDP.
- Using frozen-embedding straightness as a gate for whether training may induce temporal structure.

## Evidence added by review hardening

- Exact-track selection chooses one sibling consistently and invalidates parent-to-daughter stencils.
- `self` positives can emit sequences without a missing lookup.
- Repeated frames remain identical after track-consistent random flips.
- Every predictor parameter receives a zero, non-`None` gradient on all-invalid batches.
- Boundary-aligned maximum HPI values fall inside a bin; invalid widths fail at config construction.

## Open empirical questions

- Does the predictor learn biology rather than acquisition-time drift?
- Which temporal weight schedule preserves contrastive retrieval best?
- Is `K=3` sufficient, or do longer stencils improve robustness enough to justify their sampling cost?
- Do consistent spatial transforms improve temporal metrics without weakening useful augmentation diversity?
