# dynacell._manifests

Bundled dataset registry — the **default** manifest roots the [resolver](../data/README.md) discovers via the
`dynacell.manifest_roots` entry point (declared in `pyproject.toml`), so path resolution works out-of-the-box
on any clone. Override at runtime with `DYNACELL_MANIFEST_ROOTS=/path/to/other/registry`.

**This directory is the source of truth for dataset manifests.** There is no longer a mirrored copy to stay in
sync with: `dynacell-paper` consumed the migrated preprocessing/registry code in its Phase-13 consolidation and
deleted its own `_configs/datasets/` tree, so authoring and content both live here. Edit these YAMLs directly.

## Layout

Each `<dataset>/` holds a `manifest.yaml` (channels, voxel `spacing`, and per-target `stores`:
train/test/cell_segmentation/gt_cache_dir) plus a `splits/` folder of train/val/test position lists.

- `aics-hipsc/` — WTC-11 hiPSC confocal (Allen Institute), all four targets (`nucleus`, `membrane`, `sec61b`,
  `tomm20`); one split file per target.
- `a549-mantis-<marker>-<condition>/` — the A549 Mantis light-sheet registry, one manifest per
  `marker ∈ {caax, h2b, sec61b, tomm20}` × `condition ∈ {mock, denv, zikv}` (12 folders). CAAX = membrane,
  H2B = nucleus.

## Navigation

- Up: [dynacell](../README.md)
