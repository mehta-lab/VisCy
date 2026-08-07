# Temporal regularization of DynaCLR embeddings — plan

**Status (2026-07-27):** implementation, PR-review hardening, and local integration validation complete; remote CI remains.

## Objective

Keep NT-Xent for state separation, then add two losses on encoder features:

- a shared next-state predictor, `L_pred`, to reward transition structure shared across cells;
- a three-frame curvature loss, `L_curv`, to make distance along a track better reflect progression.

The losses create temporal structure; they are not gated on straightness already being present in a frozen contrastive embedding.

## Locked design

- Emit fixed `K`-frame, `tau`-spaced sequences as row-major `B*K` tensors.
- A valid sequence uses one exact `global_track_id` and marker. It stops at divisions and never switches siblings.
- Apply temporal losses directly to encoder features; do not add a separate temporal head.
- Stop-gradient the predictor target at `t+1`.
- Keep temporal weights independently schedulable.
- Default to track-consistent stochastic augmentation. Retain `independent` and `none` as explicit ablations.
- Mask incomplete sequences while keeping every configured module in the distributed backward graph.

## Milestones

- [x] Sequence sampling, rectangular batching, and validity masks.
- [x] Curvature loss and shared predictor.
- [x] Lightning integration, schedules, and configuration.
- [x] Local end-to-end smoke coverage.
- [x] PR review hardening and regressions.
- [x] Run the final combined local suite.
- [ ] Confirm remote CI after push.
- [ ] Compare biology and representation metrics against the NT-Xent baseline.

## PR review resolutions

| Priority | Risk | Resolution |
| --- | --- | --- |
| P1 | A lineage stencil could jump between siblings. | Match every frame by exact `global_track_id` and invalidate at division boundaries (`061d6e31`). |
| P1 | Flattened `B*K` frames received independent random transforms. | Reuse one transform realization per track; expose `consistent`, `independent`, and `none` modes (`061d6e31`). |
| P1 | An all-invalid rank left predictor parameters unused in DDP. | Forward the empty masked tensor through the predictor and backpropagate a graph-connected zero (`651d693c`). |
| P2 | `positive_cell_source=self` skipped the lookup needed by sequence emission. | Build the lineage/timepoint lookup whenever sequences are enabled (`061d6e31`). |
| P2 | Half-open HPI bins omitted a maximum on an exact boundary. | Share an edge builder that always adds a terminal bin (`f9426fab`). |
| P3 | Zero, negative, or non-finite HPI widths were accepted. | Reject invalid widths during Pydantic validation (`f9426fab`). |

## Acceptance checks

- Exact-track and division-boundary sampler regressions pass.
- Self-positive sequence emission passes.
- Track-consistent flip regression passes.
- Predictor parameters receive non-`None`, zero gradients for all-invalid batches.
- HPI terminal-boundary and invalid-width regressions pass.
- Dataset/datamodule, engine, witness-GMM, and final integration suites pass.

Local combined validation passed on 2026-07-27. The two skipped inference-reproducibility tests require external HPC data and CUDA; there were no failures.

## Follow-up experiment

Train matched seeds for NT-Xent-only versus NT-Xent + `L_pred` + `L_curv`. Compare contrastive retrieval, collapse indicators, temporal prediction, per-track progression, and held-out biological separation before promoting the temporal objective to a default recipe.
