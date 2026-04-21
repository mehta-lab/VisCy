# VisCy PR Spec: Manifest-Driven Dataset Reference Resolution

## Goal

Add a resolver to VisCy's config composition pipeline that turns `benchmark.dataset_ref: {dataset, target}` into concrete `data_path`, `source_channel`, `target_channel` — read from a Pydantic `DatasetManifest` YAML discovered via manifest roots. Zero changes to Lightning CLI, Hydra, or `submit_benchmark_job.py`'s public surface. Backward compatible.

## Context

Today, core dataset facts are duplicated:

- **dynacell-paper** `configs/datasets/aics-hipsc/manifest.yaml` — canonical source of truth (data paths, channels, spacing, splits).
- **VisCy** `_internal/shared/model/{train_sets,targets}/*.yml` — hardcodes `data_path`, `source_channel`, `target_channel`.

If a zarr store moves, both sides need hand updates. The intent hook `benchmark.dataset_group: aics-hipsc` already sits in `train_sets/ipsc_confocal.yml` as a breadcrumb — nothing reads it.

This PR closes the gap by wiring a resolver at the composition choke point (`viscy_utils.cli._maybe_compose_config` plus `submit_benchmark_job.py`).

## Non-goals

- Eval-side (Hydra) resolution. Eval uses Pydantic manifests already via `pipeline_cache.py`; a follow-up PR can make eval leaves `dataset_ref`-aware.
- Collection-aware data loading (Phase 5D of the dynacell-paper refactor). This PR resolves *dataset facts*, not *FOV membership*.
- Inventing a new CLI flag surface on `dynacell fit` / `predict`. Resolution is implicit via the composition hook.
- Importing anything from `dynacell-paper`. The resolver is generic.

## New types and functions

**`applications/dynacell/src/dynacell/data/manifests.py`** — add one Pydantic model:

```python
class DatasetRef(BaseModel):
    """Reference to a dataset target, resolved against a manifest registry."""

    dataset: str
    target: str
    # Future: override: dict = {} for paper-specific store swaps. Not in v1.
```

**`applications/dynacell/src/dynacell/data/resolver.py`** — new module:

```python
class ManifestNotFoundError(LookupError): ...
class TargetNotFoundError(LookupError): ...
class NoManifestRootsError(RuntimeError): ...

class ResolvedDataset(BaseModel):
    """Flat view of the fields a leaf needs after ref resolution."""

    manifest_path: Path         # for provenance / logging
    data_path_train: Path
    data_path_test: Path
    source_channel: str
    target_channel: str
    spacing: VoxelSpacing

def discover_manifest_roots(
    cli_roots: list[Path] | None = None,
) -> list[Path]:
    """Resolve manifest roots in precedence order (see below).

    Raises NoManifestRootsError if nothing is configured.
    """

def resolve_dataset_ref(
    ref: DatasetRef,
    roots: list[Path] | None = None,
) -> ResolvedDataset:
    """Load the manifest for `ref.dataset` and return the target's fields."""
```

**Manifest root precedence** (highest wins):

1. `cli_roots` parameter (from the compose hook).
2. `DYNACELL_MANIFEST_ROOTS` env var (colon-separated absolute paths).
3. Python entry points under group `dynacell.manifest_roots`; each entry resolves to a package resource directory (e.g. `dynacell_paper._configs.datasets`).

Scan logic: for each root (in order), look for `<root>/<dataset>/manifest.yaml`. First hit wins. No recursion, no globbing.

## Composition hook

**`packages/viscy-utils/src/viscy_utils/compose.py`** — add an optional post-composition hook:

```python
def load_composed_config(
    path: Path,
    *,
    resolver: Callable[[dict], dict] | None = None,
) -> dict:
    """... existing docstring ...

    If `resolver` is provided, it is called on the final merged dict
    before returning. Resolvers are pure: given a dict, return a dict.
    """
```

Keep the existing behavior when `resolver` is `None`. No mutation of `_seen` semantics.

**`packages/viscy-utils/src/viscy_utils/cli.py`** — `_maybe_compose_config`:

```python
# after load_composed_config(path) returns:
composed = load_composed_config(path, resolver=_dynacell_ref_resolver)
# existing: strip reserved keys (launcher, benchmark) before writing tempfile
```

The resolver:

```python
def _dynacell_ref_resolver(composed: dict) -> dict:
    ref_dict = composed.get("benchmark", {}).get("dataset_ref")
    if ref_dict is None:
        return composed                     # no-op for legacy leaves
    ref = DatasetRef.model_validate(ref_dict)
    mode = _infer_mode(composed)            # "fit" | "predict" | "validate"
    resolved = resolve_dataset_ref(ref)
    return _splice_resolved(composed, resolved, mode)
```

### Mode inference (predict vs fit store)

`_infer_mode(composed)` order:

1. If `composed["launcher"]["mode"]` set (benchmark leaves always set this), use it.
2. Else inspect `sys.argv[1]` (`fit` / `predict` / `validate`).

Splicing:

- `mode == "fit"` or `"validate"` → `data.init_args.data_path = resolved.data_path_train`
- `mode == "predict"` → `data.init_args.data_path = resolved.data_path_test`
- Always: `data.init_args.source_channel = resolved.source_channel`
- Always: `data.init_args.target_channel = resolved.target_channel`

`benchmark.spacing` gets filled from `resolved.spacing.as_list()` (handy for eval and any metric-aware callbacks). Since `benchmark:` is stripped before Lightning sees it, this only matters for downstream consumers of the resolved intermediate dict.

### Collision policy

If the composed config has BOTH `benchmark.dataset_ref` AND an explicit `data.init_args.data_path` (or `source_channel` / `target_channel`), raise `ValueError` with both values. Do not silently override either way — the user is giving conflicting signals and should pick.

Exception: predict mode leaves sometimes declare `data.init_args.data_path` pointing at the test store today. Those leaves must migrate to `dataset_ref` in the same PR OR be left alone — they can't do both.

### `submit_benchmark_job.py` pickup

Since it calls `load_composed_config()` on line 159, the resolver is applied automatically — **but only if we pass it through**. Change that call to `load_composed_config(path, resolver=_dynacell_ref_resolver)`. The topology validation (gpus × nodes × ntasks_per_node) runs on the already-resolved dict, which is fine.

## Files touched

```
packages/viscy-utils/src/viscy_utils/compose.py        # +resolver kwarg
packages/viscy-utils/src/viscy_utils/cli.py            # wire resolver
packages/viscy-utils/tests/test_compose.py             # resolver kwarg tests
applications/dynacell/src/dynacell/data/__init__.py    # export DatasetRef, resolve_dataset_ref
applications/dynacell/src/dynacell/data/manifests.py   # +DatasetRef model
applications/dynacell/src/dynacell/data/resolver.py    # new
applications/dynacell/src/dynacell/_compose_hook.py    # _dynacell_ref_resolver wrapper
applications/dynacell/src/dynacell/__main__.py         # pass hook to viscy_utils.cli
applications/dynacell/tools/submit_benchmark_job.py    # pass hook
applications/dynacell/tests/test_dataset_ref.py        # new
applications/dynacell/tests/test_benchmark_config_composition.py  # migrate er_sec61b leaf
applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/model/targets/er_sec61b.yml
  # drop data_path/target_channel, add benchmark.dataset_ref
applications/dynacell/configs/benchmarks/virtual_staining/_internal/shared/model/train_sets/ipsc_confocal.yml
  # drop source_channel
```

The viscy-utils changes stay generic (the `resolver` kwarg is a plain callable). The dynacell-specific resolver lives in `applications/dynacell/src/dynacell/_compose_hook.py` and is injected by dynacell's CLI entry points only. This keeps viscy-utils from depending on dynacell.

## Test matrix

**Unit (`test_dataset_ref.py`):**

| Scenario | Expected |
|---|---|
| `resolve_dataset_ref` happy path (aics-hipsc/sec61b) | returns `ResolvedDataset` with correct paths |
| unknown dataset | `ManifestNotFoundError` listing available datasets across roots |
| unknown target within a known dataset | `TargetNotFoundError` listing available targets |
| `DYNACELL_MANIFEST_ROOTS` unset + no entry points | `NoManifestRootsError` with install hint |
| env var takes precedence over entry points | passes with env-var path |
| invalid manifest YAML (missing `name`) | Pydantic `ValidationError`, file path in message |
| `cli_roots` wins over env var and entry points | verified |

**Integration (`test_benchmark_config_composition.py`):**

| Scenario | Expected |
|---|---|
| `er/ipsc_confocal/celldiff/train.yml` composes with `dataset_ref` | resolved dict has `data_path_train`, correct channels |
| same leaf with `launcher.mode: predict` swapped in | `data_path` resolves to test store |
| leaf with `dataset_ref` + explicit `data_path` | `ValueError` on collision |
| no `dataset_ref` in the composed config | resolver no-op; legacy path still works |
| FCMAE pretrained/scratch parity | still only differs in encoder init |
| topology invariants | still hold |

**CLI routing (`test_cli_routing.py`):**

- `dynacell fit -c <leaf>` → `_dynacell_ref_resolver` is wired into `load_composed_config`.
- Hydra subcommands (`evaluate`, `report`, `precompute-gt`) — unchanged, no hook injected. (Eval-side is out of scope per "Non-goals".)

**viscy-utils tests (`test_compose.py`):**

- `load_composed_config` with `resolver=None` behaves exactly as before.
- `resolver=lambda d: d` roundtrip identity.
- `resolver` receives the post-merge dict, not per-base fragments.

## Backward compatibility

- Leaves with no `benchmark.dataset_ref`: unchanged behavior.
- Leaves with explicit `data_path` and no `dataset_ref`: unchanged behavior.
- The `compose.load_composed_config` signature gains a keyword-only arg with default `None`; existing callers are unaffected.
- `viscy_utils` remains generic — it calls whatever resolver is passed, so other consumers can register their own.

## Failure-mode messaging (explicit)

Because the most common new failure will be "I ran `dynacell fit` and got an unhelpful error":

```
NoManifestRootsError: No dynacell manifest roots configured.

Configure via one of:
  - CLI flag:       --manifest-root /path/to/datasets
  - Env var:        export DYNACELL_MANIFEST_ROOTS=/path/to/datasets
  - Install a provider:  pip install dynacell-paper

Leaf config at /path/to/leaf.yml references:
  benchmark.dataset_ref: {dataset: aics-hipsc, target: sec61b}
```

```
ManifestNotFoundError: dataset 'aics-hipsc' not found in manifest roots.

Searched:
  - /hpc/.../dynacell_paper/_configs/datasets     (no aics-hipsc/manifest.yaml)
  - /tmp/my-roots                                 (no aics-hipsc/manifest.yaml)

Available datasets across configured roots:
  - (none)
```

## Migration plan (in-PR)

1. Land the resolver + tests with all existing leaves untouched. Full test suite green.
2. Migrate ONE target in the same PR: `er_sec61b.yml` + `ipsc_confocal.yml` — prove end-to-end composition produces an identical final dict to today (byte-for-byte if possible; modulo field ordering).
3. Tail PRs (not this one) migrate `tomm20`, `membrane`, `nucleus`, and the mantis_lightsheet train_set when it lands.

## Open questions for review

1. **Should the resolver be part of viscy-utils, or purely in `applications/dynacell`?** This spec keeps the `resolver` kwarg in viscy-utils (generic) but the dynacell-specific resolver function in `applications/dynacell`. Alternative: promote the resolver concept as part of `viscy-utils`. I'd keep it generic (the viscy-utils kwarg takes any callable).
2. **Entry point group name**: `dynacell.manifest_roots` is specific; `viscy.dataset_registries` is generic. Since the schema is dynacell-specific even though it lives in VisCy, I'd go with `dynacell.manifest_roots`.
3. **Should `benchmark.dataset_ref` stay in the final dict (it gets stripped anyway) or get moved to a resolver-only key**? Staying under `benchmark:` is cleanest because it already gets stripped and has provenance value before strip. Keep it there.

## Downstream (dynacell-paper) companion PR

After this lands in VisCy, the dynacell-paper side change is small:

1. Add the entry point in `pyproject.toml`:
   ```toml
   [project.entry-points."dynacell.manifest_roots"]
   default = "dynacell_paper._configs.datasets"
   ```
2. Smoke test: `dynacell fit -c <VisCy leaf with dataset_ref> --trainer.fast_dev_run=true` resolves paths when `dynacell-paper` is installed.
3. No changes to manifest YAML content.
