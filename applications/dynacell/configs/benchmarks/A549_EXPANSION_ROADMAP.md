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
train/predict overlays. Each is a small follow-up PR touching one target
group file + one train_set file (if applicable) + any corresponding
leaves that referenced them. Behavior stays identical; test suite
verifies no drift.

### Stage 3 — Extend resolver to Hydra / eval side

The spec defers eval-side resolution. This stage closes it:

- Extend the Pydantic manifest schema (or add a sibling) to cover
  eval-specific fields: `gt_path`, `cell_segmentation_path`,
  `gt_cache_dir`, `gt_channel_name`. These are dataset-specific, so the
  manifest is the right owner.
- Add a Hydra-side resolver hook. Options:
  - A custom OmegaConf resolver that reads the manifest at compose
    time, or
  - A `dynacell.evaluation` pre-compose step that splices the manifest
    fields into the composed dict before `pipeline.py` consumes it.
- Update `pipeline_cache.py` and any callers so manifest access goes
  through the same registry the train-side resolver uses.

This is the architectural precondition for clean a549 eval leaves.
Scope is contained (eval-side only) but requires Hydra + Pydantic + the
resolver lib to agree on a single manifest shape.

### Stage 4 — Migrate eval target YAMLs

Strip `io.gt_path`, `io.cell_segmentation_path`, `io.gt_cache_dir`,
`io.gt_channel_name` out of `_internal/shared/eval/target/*.yaml`.
Those YAMLs keep only channel-name and organelle identity; paths come
from the manifest via `benchmark.dataset_ref`.

Existing eval leaves become thinner (they inherit `dataset_ref` from
the training cell or declare their own), and the current 8 eval leaves
continue to compose identically.

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
  └─> Stage 3 (eval-side resolver)
          └─> Stage 4 (migrate eval target YAMLs)
                  └─> Stage 6 (a549 leaf expansion)
                          ^
                          │
                Stage 5 (a549 manifest) ─┘
```

Stages 1 → 2 → 3 → 4 → 6 are strictly sequential on the VisCy side.
Stage 5 is independent and can proceed in parallel with Stages 1–3, as
long as it lands before Stage 6.

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
