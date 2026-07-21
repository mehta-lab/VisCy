# dynacell._manifests

Bundled dataset registry — the **default** manifest roots the [resolver](../data/README.md) discovers via the
`dynacell.manifest_roots` entry point (declared in `pyproject.toml`), so path resolution works out-of-the-box
on any clone. Override at runtime with `DYNACELL_MANIFEST_ROOTS=/path/to/other/registry`.

VisCy is the source of truth for manifest *content* (this directory); `dynacell-paper` is the source of truth
for manifest *authoring*. New datasets preprocessed there are mirrored back here, and
[`tests/test_manifest_sync.py`](../../../tests/README.md) enforces the parity.

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
