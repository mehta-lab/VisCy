# dynacell.data

Dataset schemas and the manifest-driven resolver that turn a lightweight `DatasetRef` (`{dataset, target}`)
into concrete zarr paths, channel names, and voxel spacing. This is the indirection that lets benchmark leaves
carry `benchmark.dataset_ref` instead of duplicating absolute HPC paths.

## Contents

- `manifests.py` — Pydantic schemas for a dataset manifest: `DatasetManifest`, `TargetConfig`, `StoreLocations`,
  `SplitDefinition`, `VoxelSpacing`, `DatasetRef`, plus `load_manifest` / `load_splits` / `get_target`.
- `resolver.py` — `resolve_dataset_ref` + manifest-root discovery. Root precedence: `cli_roots` arg →
  `DYNACELL_MANIFEST_ROOTS` env → `dynacell.manifest_roots` entry point (defaults to
  [`_manifests/`](../_manifests/README.md)). Looks for `<root>/<dataset>/manifest.yaml`, first hit wins.
  Errors: `ManifestNotFoundError`, `NoManifestRootsError`, `TargetNotFoundError`.
- `collections.py` — `BenchmarkCollection` / `CollectionExperiment` / `ChannelEntry` / `Provenance` schemas +
  `load_collection` for describing multi-experiment collections.
- `specs.py` — `BenchmarkSpec` + `load_benchmark_spec` for reproducible run specs.
- `_yaml.py` — shared YAML loading helpers.

The compose-time splice lives in [`../_compose_hook.py`](../README.md) (train/predict leaves) and
`evaluation/_ref_hook.py` (eval leaves), both of which call this resolver.

## Navigation

- Up: [dynacell](../README.md)
