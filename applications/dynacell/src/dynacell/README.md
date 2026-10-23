# dynacell (package)

The installable `dynacell` package: virtual-staining LightningModules, the manifest-driven data layer, the
three-tier evaluation pipeline, reporting, and preprocessing. Consumes the `viscy-*` workspace packages
(`viscy_data`, `viscy_models`, `viscy_transforms`, `viscy_utils`); never imports from sibling applications.

## Top-level modules

- `__init__.py` — lazily exposes the three engines (`DynacellUNet`, `DynacellFlowMatching`, `DynacellGAN`) to
  avoid pulling heavy training deps on import.
- `__main__.py` — CLI entry point. Routes Lightning subcommands (`fit`/`validate`/`predict`) to
  `viscy_utils.cli.main()` and Hydra subcommands (`evaluate`, `evaluate-grouped`, `precompute-gt`, `report`) to
  their entry points; also injects the eval Hydra searchpaths and shared HF cache on a repo checkout.
- `engine.py` — the LightningModules: `DynacellUNet` (regression), `DynacellFlowMatching` (CELL-Diff),
  `DynacellGAN` (pix2pix3d adversarial).
- `celldiff_wrapper.py` — `CELLDiff3DVS`, wrapping `viscy_models.celldiff.CELLDiffNet` with flow-matching
  transport (loss + ODE-based generation).
- `_compose_hook.py` — composition-time resolver spliced into `viscy_utils.compose`; reads
  `benchmark.dataset_ref` and fills concrete `data_path`/channels from the manifest so benchmark leaves run
  standalone.

## Subpackages

- **[data/](data/README.md)** — dataset schemas (collections, manifests, splits) and the manifest-root resolver
  that turns a `DatasetRef` into concrete paths + channel names.
- **[evaluation/](evaluation/README.md)** — the three-tier metric pipeline: segmentation, pixel/mask/feature
  metrics, deep-feature extractors, artifact cache, focus-aware 2D projection, runtime parallelism.
- **[_manifests/](_manifests/README.md)** — bundled dataset registry (A549-Mantis per-organelle/condition +
  AICS iPSC manifests), the default source for the resolver.
- `reporting/` — `dynacell report`: aggregate eval CSV/NPY into paper tables (`tables.py`) and comparison
  figures (`figures.py`); Hydra schema in `_configs/`.
- `preprocess/` — reusable dataset-preprocessing helpers (`load_preprocess_config`, `rewrite_zarr`).

## Conventions

Code↔paper name translation, training-data pooling, and prediction/eval-dir naming live in the application
[CLAUDE.md](../../CLAUDE.md). The evaluation subpackage carries its own [CLAUDE.md](evaluation/CLAUDE.md)
covering the mandatory `cubic` GPU-dispatch pattern.

## Navigation

- Up: [src](../README.md)
- Subdirectories: [data/](data/README.md), [evaluation/](evaluation/README.md), [_manifests/](_manifests/README.md)
