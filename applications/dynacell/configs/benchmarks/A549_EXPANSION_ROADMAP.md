# A549 Expansion Roadmap

Multi-stage rollout adding A549/mantis-lightsheet alongside the
existing iPSC/confocal benchmark cells, with a manifest-driven
dataset resolver as the foundation.

## Goal

- **Two training sets per (organelle, model) cell**: `ipsc_confocal`
  and `joint_ipsc_confocal_a549_mantis`.
- **Two held-out evaluation splits per trained model**:
  `ipsc_confocal` and `a549_mantis`. Every trained model evaluates on
  both, regardless of training source, so cross-dataset transfer is
  measurable.

The post-reorg layout (`14f59f1`) supports this — each
`<org>/<model>/<train_set>/` dir is a training experiment with room
for multiple `predict__<predict_set>.yml` and
`eval__<predict_set>.yaml` leaves. The resolver removed the data-path
duplication that would otherwise blow up across ~60 new leaves.

## Status snapshot (2026-04-26)

| Stage | Description | Status |
|---|---|---|
| 1 | Resolver core + 1 migration (`er_sec61b`, `ipsc_confocal`) | **Done** — `38d47b3`, `4bb9f09` |
| 2 | Migrate `mito_tomm20`, `nucleus`, `membrane` to `dataset_ref` | **Done** — `11836c8`, `326b2d0`, `6273439` |
| 3 | Hydra-side hook + 4 eval target YAMLs migrated | **Done** — `8924ab2`, `f5a6e56`, `a984384` |
| 4 | (folded into Stage 3) | n/a |
| 5 | Register a549-mantis manifests | **Partial** — done in dynacell-paper (`aeef64c`, 7 per-plate manifests 2024_10_29 → 2025_08_26); VisCy fixture mirror missing. A549 zarr normalization-stats backfill closed 2026-04-24 (dynacell-paper `f4120e0` + 17-zarr backfill). |
| 6 | Single-dataset a549 predict + eval leaves | **Not started** |
| 7 | Joint training leaves (ipsc + a549) | **In flight — blocked**. First leaf + smoke variants shipped (`er/celldiff`: `9654e2b`, `4d399d5`, `234819a`). 4-GPU DDP smoke still hangs after PR #413 (`0b04b24`) — a second deadlock surface remains; see `.claude/handoffs/handoff-batched-concat-ddp-hang-followup-2026-04-26.md`. |

## Remaining work

### Stage 5 — VisCy bundled manifest registry

The canonical a549-mantis manifests live in `dynacell-paper`. VisCy
ships its own copy of the canonical YAMLs as a bundled registry under
`applications/dynacell/src/dynacell/_manifests/`, registered as a
`dynacell.manifest_roots` entry-point provider in
`applications/dynacell/pyproject.toml`. The resolver auto-discovers
this without any `DYNACELL_MANIFEST_ROOTS` env var configuration —
works on a fresh clone for any Stage 6 a549 leaf. Drift between the
mirror and dynacell-paper canonical is guarded by
`tests/test_manifest_sync.py`, which is skipped unless
`DYNACELL_PAPER_PATH` is set (typical CI / local dev environment).

The a549 zarr normalization-stats gap (every `mantis_v1/<plate>/<split>/<GENE>.zarr`
missing `normalization` zattrs at plate and position level) closed on
2026-04-24: dynacell-paper `f4120e0` adds `generate_normalization_metadata`
as a post-write step in the assembly pipeline, and the 17 pre-hook
zarrs were backfilled in 5.7 min. Joint leaves consuming these stores
no longer fail or asymmetrically normalize at training time. Treat as
done; no VisCy-side action.

### Stage 6 — single-dataset a549 predict + eval leaves

Add `predict__a549_mantis.yml` + `eval__a549_mantis.yaml` to existing
`<organelle>/<model>/ipsc_confocal/` cells so iPSC-trained models can
be evaluated on the a549 test split.

Sub-scope (from the original roadmap, still unresolved):

- **(iv) full-but-predictable-only** — the 8 cells that already have
  `<model>_predict.yml` overlays (celldiff + unetvit3d × 4 organelles).
  Recommended starting point.
- **(iii) full-all-models** — additionally create skeleton
  `fcmae_vscyto3d_predict.yml`, `fnet3d_paper_predict.yml`,
  `unext2_predict.yml` overlays. Defer unless needed.

Each leaf is 5–10 lines:

```yaml
# eval__a549_mantis.yaml
defaults:
  - override /target: er_sec61b
benchmark:
  dataset_ref: {dataset: a549-mantis-2024_11_07, target: sec61b}
io:
  pred_path: /hpc/.../sec61b_celldiff_on_a549.zarr
save:
  save_dir: /hpc/.../eval_sec61b_celldiff_on_a549
```

### Stage 7 — joint training leaf expansion

The joint-loader infrastructure landed in `4bc2e53` (sharded sampler
in `BatchedConcatDataModule`) and `5950576` (split fit overlays). PR
#413 (`0b04b24`) addressed one DDP deadlock surface (the
`use_thread_workers=True` thread-shim under real `init_process_group`)
but the 4-GPU smoke still hangs at the same milestone — a second
deadlock surface remains; see
`.claude/handoffs/handoff-batched-concat-ddp-hang-followup-2026-04-26.md`.
Joint leaf expansion is blocked until this resolves.

The first joint leaf shipped at
`er/celldiff/joint_ipsc_confocal_a549_mantis/train.yml` (`9654e2b`);
smoke variants followed (single-GPU `4d399d5`, 4-GPU DDP `234819a`).
The single-GPU smoke runs end-to-end against `_test48` debug zarrs;
the 4-GPU DDP smoke is the failing reproducer for the open deadlock.

Smoke leaves rely on the `_test48` debug-zarr convention documented
in this app's `CLAUDE.md` and mirrored in `dynacell-paper`'s `CLAUDE.md`:
short-wall validation jobs override `data_path` to the colocated
`<NAME>_test48.zarr` so `mmap_preload` finishes staging in under a
minute instead of 45+ min on the full 500-FOV stores.

Joint leaves bypass the single-dataset `dataset_ref` resolver and
author the data block inline because hparams live on each child.
Shared HCS init_args factor via a YAML merge anchor.

Remaining matrix:

- Other organelles for `celldiff`: `mito`, `nucleus`, `membrane`.
- Other models for `er`: `unetvit3d`, `fcmae_vscyto3d_{scratch,pretrained}`,
  `fnet3d_paper`, `unext2`.
- Cross-product: 4 organelles × 6 models = 24 cells (minus the one
  already shipped).
- Companion leaves per joint cell: `predict__ipsc_confocal.yml`,
  `predict__a549_mantis.yml`, `eval__ipsc_confocal.yaml`,
  `eval__a549_mantis.yaml`.

Decision pending: order of expansion. Reasonable defaults are
"finish the celldiff row first" (organelle sweep on a known-good
model) or "finish the er column first" (model sweep on a known-good
organelle). Pick when the next paper experiment lands.

## Dependency graph

```
Stage 1 ✅ ─> Stage 2 ✅ ─> Stage 3 ✅
                              └─> Stage 6 (predict/eval on a549)
                                      ^
                                      │
Stage 5 (a549 manifest) — partial ────┘
        canonical: done
        VisCy fixture mirror: pending

Stage 7 (joint training leaves) — independent of resolver path
        first leaf + smoke variants: done
        4-GPU DDP smoke: blocked on remaining deadlock (see followup handoff)
        expansion (24 cells + companion leaves): pending P0 deadlock fix
```

Stages 1–3 and 5 (canonical) blocked Stage 6. The remaining gap on
the VisCy side is the fixture mirror. Stage 7 has its own
infrastructure (`BatchedConcatDataModule` + `ShardedDistributedSampler`)
and is orthogonal to the resolver path.

## Non-goals

- FOV-level split resolution (Phase 5D of the dynacell-paper refactor —
  about *FOV membership*, not *dataset facts*).
- New CLI flags on `dynacell fit` / `predict` — the resolver is implicit
  via the composition hook.
- Reporting-side path resolution — reporting consumes eval outputs, not
  source data.
- Changes to `_internal/shared/model/model_overlays/` or
  `launcher_profiles/` — those are model/hardware concerns, orthogonal.
