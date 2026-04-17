# LEGACY — Dihan's pre-schema CellDiff / UNetViT3D configs

**Reference-only.** `base:` paths were patched post-move from
`../../../configs/recipes/...` to `../../../../configs/recipes/...` so the
equivalence test in `tests/test_benchmark_config_composition.py` can still
compose them, and the `preload:` kwarg was later renamed in place to
`mmap_preload:` when `HCSDataModule` dropped the ambiguous name. The
patched files are not intended to be launched directly — use the migrated
leaves under `configs/benchmarks/virtual_staining/` via
`submit_benchmark_job.py`.

## Migration map

| Legacy file | New leaf | Equivalence test |
|---|---|---|
| `sec61b/fit_celldiff.yml` | `train/er/ipsc_confocal/celldiff.yml` | `test_train_leaf_matches_legacy[er-sec61b]` |
| `tomm20/fit_celldiff.yml` | `train/mito/ipsc_confocal/celldiff.yml` | `test_train_leaf_matches_legacy[mito-tomm20]` |
| `nucl/fit_celldiff.yml` | `train/nucleus/ipsc_confocal/celldiff.yml` | `test_train_leaf_matches_legacy[nucleus-nucl]` |
| `memb/fit_celldiff.yml` | `train/membrane/ipsc_confocal/celldiff.yml` | `test_train_leaf_matches_legacy[membrane-memb]` |
| `sec61b/predict_celldiff.yml` | `predict/er/ipsc_confocal/celldiff/ipsc_confocal.yml` | `test_predict_leaf_matches_legacy[er-sec61b]` |
| `tomm20/predict_celldiff.yml` | `predict/mito/ipsc_confocal/celldiff/ipsc_confocal.yml` | `test_predict_leaf_matches_legacy[mito-tomm20]` |
| `nucl/predict_celldiff.yml` | `predict/nucleus/ipsc_confocal/celldiff/ipsc_confocal.yml` | `test_predict_leaf_matches_legacy[nucleus-nucl]` |
| `memb/predict_celldiff.yml` | `predict/membrane/ipsc_confocal/celldiff/ipsc_confocal.yml` | `test_predict_leaf_matches_legacy[membrane-memb]` |
| `sec61b/fit_unetvit3d.yml` | `train/er/ipsc_confocal/unetvit3d.yml` | `test_unetvit3d_train_leaf_matches_legacy` |
| *(git-removed)* `sec61b/fit_fnet3d_paper.yml` | `train/er/ipsc_confocal/fnet3d_paper.yml` | `test_fnet3d_paper_leaf_matches_ran_config` |

The `fnet3d_paper` leaf has no source file in LEGACY — the earlier
`fit_fnet3d_paper.yml` was git-removed in commit `42d66d7`. The new leaf
is verified directly against the LightningCLI config.yaml that Lightning
saved when the run trained, at
`/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/config.yaml`.
The equivalent wandb-logged model hyperparameters
(in project `computational_imaging/dynacell`, run group
`FNet3D_iPSC_SEC61B_paper`) match across all 9 runs in the group.

### Notes on `fit_unetvit3d.yml`

The legacy file carries a latent copy-paste bug: `net_config:` nested
under `DynacellUNet`'s `init_args`. `DynacellUNet.__init__` takes
`model_config:`, not `net_config:`, so jsonargparse rejects that
override — the legacy config would fail to load if run today. The
override is also redundant with the recipe's `model_config.input_spatial_size`,
so the new leaf drops it. Runtime-equivalent in every other field.

## Why kept

These are the source-of-truth hyperparameter reference for the migrated
benchmark leaves under `configs/benchmarks/virtual_staining/train/` and
`.../predict/`. The equivalence test
(`tests/test_benchmark_config_composition.py`) asserts that each migrated
leaf composes to the same values these files compose to. Delete this tree
only after:

1. One successful end-to-end `submit_benchmark_job.py` run against a
   migrated leaf (fit or predict), verified on wandb/disk; and
2. 2026-06-30 at the earliest.

Whoever deletes this should note both conditions in the commit message.

## Rerunning these configs

Copy them back out to the original location or fix the `base:` paths
manually. They are preserved exactly as they were when they worked.
