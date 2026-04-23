# A549 Expansion Roadmap

## Context

The current `virtual_staining/` tree is populated only for `ipsc_confocal`
training and evaluation. Research requirements expand this to:

- **Two training sets per (organelle, model) cell**: `ipsc_confocal` and
  `joint_ipsc_confocal_a549_mantis`.
- **Two held-out evaluation splits per trained model**: `ipsc_confocal`
  and `a549_mantis`. Every trained model evaluates on both, regardless of
  training source, so cross-dataset transfer is measurable.

The post-reorg layout (`14f59f1`) already supports this — each
`<org>/<model>/<train_set>/` dir is a training experiment with room for
multiple `predict__<predict_set>.yml` and `eval__<predict_set>.yaml`
leaves. The gap is the underlying **data-path plumbing**: every leaf
today hardcodes `data_path`, `source_channel`, `target_channel` (for
train/predict) or `gt_path`, `cell_segmentation_path`, `gt_cache_dir`
(for eval). Naively expanding to a549 duplicates all of this across
roughly 60 new leaves, which becomes technical debt the moment any path
moves.

`DATASET_REF_RESOLVER_SPEC.md` proposes exactly the right fix for the
train/predict side: a manifest-driven resolver so leaves reference a
dataset by name (`benchmark.dataset_ref: {dataset, target}`) and the
composition layer splices in paths and channels from a Pydantic
manifest. This roadmap sequences the resolver PR, its follow-ups, and
the a549 expansion so each stage lands on top of a solid foundation.

## How the two workstreams relate

- **Question B from the a549 scoping discussion** (how to disentangle
  organelle identity from dataset-specific paths in `target/` YAMLs) is
  precisely what the resolver solves. The manifest owns the paths; the
  Hydra `target/` group owns only channel names and organelle identity.
- Duplicating a549-variant target YAMLs today (`er_sec61b_a549.yaml` and
  friends) would write exactly the path duplication the resolver is
  designed to eliminate. Don't do it.
- The spec's non-goal #1 is eval-side resolution; the a549 eval
  expansion needs that, so we either wait for the eval-side follow-up
  or write eval leaves with hardcoded paths (debt). This roadmap waits.

## Staged plan

### Stage 1 — Resolver PR (per `DATASET_REF_RESOLVER_SPEC.md`)

Scope exactly as the spec describes:

- `packages/viscy-utils/src/viscy_utils/compose.py`: add optional
  keyword-only `resolver: Callable[[dict], dict] | None` to
  `load_composed_config`.
- `applications/dynacell/src/dynacell/data/manifests.py`: add
  `DatasetRef` Pydantic model.
- `applications/dynacell/src/dynacell/data/resolver.py`: new module with
  `ResolvedDataset`, `discover_manifest_roots`, `resolve_dataset_ref`,
  three error classes.
- `applications/dynacell/src/dynacell/_compose_hook.py`:
  `_dynacell_ref_resolver` that reads `composed["benchmark"]["dataset_ref"]`,
  infers mode, splices `data_path` / `source_channel` / `target_channel`.
- Wire the hook in `dynacell/__main__.py` and `tools/submit_benchmark_job.py`.
- Manifest-root precedence: CLI roots → `DYNACELL_MANIFEST_ROOTS` env
  var → `dynacell.manifest_roots` entry points.
- Collision policy: leaves with both `dataset_ref` and explicit
  `data_path` raise `ValueError`.
- Migrate exactly one target (`er_sec61b` + `ipsc_confocal` train_set)
  to prove end-to-end composition is byte-identical to pre-PR output.

Tests: `test_dataset_ref.py` (unit — resolver + error messaging),
`test_compose.py` (viscy-utils resolver kwarg contract), expansion to
`test_benchmark_config_composition.py` (integration, one migrated
target). Full dynacell suite green.

Exit criteria: one migrated leaf composes identically to today;
remaining 16 train leaves + 8 predict leaves + 8 eval leaves untouched.

### Stage 2 — Migrate remaining train/predict targets

`mito_tomm20`, `membrane`, `nucleus` move onto `dataset_ref` in the
train/predict overlays — bundled in **one VisCy PR** (each target is
a two-line fragment change; splitting fragments the test delta and
buys nothing). Scope: three target fragments, six predict leaves
that drop their explicit `data_path`, a fixture manifest update
adding `nucleus` + `membrane` entries, and a companion
`dynacell-paper` manifest PR that adds the same two entries to the
canonical manifest (required for production runs; VisCy tests pass
against the repo-local fixture). Behavior stays identical; integration
tests parametrize across every model in `TRAIN_LEAVES` /
`PREDICT_LEAVES` that composes a migrated fragment to guard against
drift.

### Stage 3 — Hydra-side hook + migrate all four eval target YAMLs

Extend `dataset_ref` resolver to the Hydra/eval side. Add post-compose
hook (`_ref_hook.py`) at `evaluate_model` / `precompute_gt` entry
points. Migrate all four eval target YAMLs (`er_sec61b`, `mito_tomm20`,
`nucleus`, `membrane`) + `predict_set/ipsc_confocal.yaml` together so
`io.*` and `pixel_metrics.spacing` come from the manifest. Add
`gt_cache_dir` to `StoreLocations`.

Deliverables:

- New `applications/dynacell/src/dynacell/evaluation/_ref_hook.py`.
  The hook fires inside the `evaluate_model()` and `precompute_gt()`
  entry-point function bodies (not during Hydra compose), reads
  `composed["benchmark"]["dataset_ref"]`, and splices the manifest
  fields into the composed config before `pipeline.py` consumes it.
- `ResolvedDataset` extended with `cell_segmentation_path` and
  `gt_cache_dir` fields.
- Four migrated eval target YAMLs (`_internal/shared/eval/target/*.yaml`):
  each keeps only `target_name` and `benchmark.dataset_ref.target`;
  `io.gt_path`, `io.cell_segmentation_path`, `io.gt_channel_name`,
  `io.pred_channel_name`, `io.gt_cache_dir` all come from the manifest.
  `pred_channel_name` is derived in the hook as
  `f"{target_channel}_prediction"` and is not stored in the manifest.
- Migrated `_configs/predict_set/ipsc_confocal.yaml`: contributes only
  `benchmark.dataset_ref.dataset`; `pixel_metrics.spacing` comes from
  the manifest.
- `benchmark: null` placeholder added to `_configs/eval.yaml` so the
  node exists for the hook to populate.
- Hydra-branch error catch wired into `dynacell/__main__.py` so hook
  errors surface as user-facing messages.
- Integration tests extended with Layer 2 entry-point wiring coverage
  (the hook actually runs through `evaluate_model` / `precompute_gt`,
  not just called directly).

The `gt_cache_dir` addition to `StoreLocations` requires a companion
bump to the canonical `dynacell-paper` manifest — see the planned
spec at `~/.claude/plans/dynacell-paper-stage3-gt-cache-dir.md` (or a
companion spec if the path hasn't been published yet). Pydantic's
default `extra="ignore"` on `StoreLocations` makes the ordering
constraint an auditing preference, not a schema-parsing requirement —
older manifests without `gt_cache_dir` still parse.

### Stage 4 — Merged into Stage 3

Stage 4 (migrate eval target YAMLs) has been folded into Stage 3 and
lands in the same PR. See Stage 3 above. Any downstream references to
"after Stage 4" now mean "after the combined Stage 3".

### Stage 5 — Add a549 manifest

Register `aics-a549-mantis` (slug TBD) in the manifest registry:

- `data_path_train`, `data_path_test` per organelle target
- `source_channel`, `target_channel`
- `spacing` (voxel dimensions)
- Eval-side fields: `gt_path`, `cell_segmentation_path`,
  `gt_cache_dir`, `gt_channel_name`

Lives in `dynacell-paper` (or wherever the canonical manifest source
is). Independent of VisCy changes — can proceed in parallel with
Stage 3.

### Stage 6 — A549 leaf expansion

With the resolver in place on both sides, each new leaf is 5-10 lines:

```yaml
# eval__a549_mantis.yaml
# @package _global_
defaults:
  - override /target: er_sec61b
benchmark:
  dataset_ref: {dataset: aics-a549-mantis, target: sec61b}
io:
  pred_path: /hpc/.../sec61b_celldiff_on_a549.zarr
save:
  save_dir: /hpc/.../eval_sec61b_celldiff_on_a549
```

The expansion matrix:

- Fill in every `<org>/<model>/ipsc_confocal/` with
  `predict__a549_mantis.yml` + `eval__a549_mantis.yaml`.
- Create `<org>/<model>/joint_ipsc_confocal_a549_mantis/` with
  `train.yml` + `predict__ipsc_confocal.yml` +
  `predict__a549_mantis.yml` + `eval__ipsc_confocal.yaml` +
  `eval__a549_mantis.yaml`.
- Corresponding `_internal/leaf/` symlinks for every new eval leaf.

Sub-scope to decide when we get here (question A from earlier):

- **(iv) full-but-predictable-only** — add predict/eval leaves only for
  the 8 cells that have `<model>_predict.yml` overlays (celldiff,
  unetvit3d × 4 organelles). Start here.
- **(iii) full-all-models** — additionally create skeleton
  `fcmae_vscyto3d_predict.yml`, `fnet3d_paper_predict.yml`,
  `unext2_predict.yml` overlays. Defer unless needed.

## Dependency graph

```
Stage 1 (resolver core + 1 migration)
  └─> Stage 2 (migrate other train/predict targets)
  └─> Stage 3 (Hydra-side hook + migrate all four eval target YAMLs)
          └─> Stage 6 (a549 leaf expansion)
                  ^
                  │
        Stage 5 (a549 manifest) ─┘
```

Stages 1 → 2 → 3 → 6 are strictly sequential on the VisCy side
(Stage 4 has been merged into Stage 3). Stage 5 is independent and
can proceed in parallel with Stages 1–3, as long as it lands before
Stage 6.

## Why not expand a549 first

Tempting because the data work (Stage 5) is decoupled and a549
experiments may be on the critical path for a paper. But writing a549
leaves before the resolver means:

- ~60 new YAMLs duplicate the paths the resolver will consolidate.
- Eval target YAMLs get a549-variant clones, doubling the `target/`
  group size (the thing Stage 4 is designed to shrink).
- Every a549 leaf gets rewritten once the resolver lands, turning the
  a549 PR into technical debt the moment it merges.

If the research timeline forces Stage 5/6 before Stages 1–4 land, do it
with eyes open: treat the hardcoded-path leaves as transitional and
schedule the rewrite explicitly.

## Non-goals for this roadmap

- FOV-level split resolution (Phase 5D of the dynacell-paper refactor —
  this is about *FOV membership*, not *dataset facts*).
- New CLI flags on `dynacell fit` / `predict` — the resolver is implicit
  via the composition hook.
- Reporting-side path resolution — reporting consumes eval outputs, not
  source data.
- Changes to `_internal/shared/model/model_overlays/` or
  `launcher_profiles/` — those are model/hardware concerns, orthogonal.
