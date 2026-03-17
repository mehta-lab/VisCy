# Physical Scale Normalization + Microscope Metadata + Cross-Scope Finetuning

**Branch:** `app-dynaclr`
**Date:** 2026-03-17
**Status:** Implemented, pre-commit passing, unit tests passing

---

## Motivation

Two experiments from different microscopes contain the same biology but differ in:
1. **Pixel/voxel size** — different magnifications mean cells appear at different spatial scales
2. **Embedding space** — microscope-specific biases push same-biology cells apart in latent space

The fix is two-pronged:
- **Physical scale normalization** at read time — adjust the pixel window read from disk so that after rescaling the patch is always exactly the target spatial size. No padding, no empty borders.
- **Cross-scope contrastive finetuning** — finetune the projection MLP with cross-microscope positives (same condition + HPI window) mixed with temporal positives for regularization.

---

## Files Changed

| File | Change |
|---|---|
| `packages/viscy-data/src/viscy_data/collection.py` | Added `microscope`, `pixel_size_xy_um`, `pixel_size_z_um` to `ExperimentEntry` |
| `packages/viscy-data/src/viscy_data/_typing.py` | Added `microscope` to `CELL_INDEX_GROUPING_COLUMNS` |
| `packages/viscy-data/src/viscy_data/cell_index.py` | Added `microscope` to `CELL_INDEX_SCHEMA`; write from experiment in `build_timelapse_cell_index` |
| `packages/viscy-data/tests/test_cell_index.py` | Added `microscope: ""` to `_make_valid_df` fixture |
| `applications/dynaclr/src/dynaclr/data/experiment.py` | `reference_pixel_size_*` params; `scale_factors` computed field; fail-fast validation |
| `applications/dynaclr/src/dynaclr/data/index.py` | Pass `microscope` through `_load_experiment_fovs`; fill `microscope=""` in `_align_parquet_columns` for old parquets |
| `applications/dynaclr/src/dynaclr/data/dataset.py` | `_rescale_patch`; scale-adjusted read window in `_slice_patch`; `_find_cross_scope_positive`; `cross_scope_fraction` + `hpi_window` params; `microscope` in `_META_COLUMNS` |
| `applications/dynaclr/src/dynaclr/data/datamodule.py` | Pass `reference_pixel_size_*`, `cross_scope_fraction`, `hpi_window` through to registry and datasets |
| `applications/dynaclr/src/dynaclr/engine.py` | `freeze_backbone: bool = False`; `on_fit_start` freezes backbone params |
| `applications/dynaclr/configs/training/batch_correction_fit.yml` | New example finetuning config |
| `applications/dynaclr/tests/test_dataset.py` | `TestRescalePatch` (3 tests) + `TestCrossScopePositive` (4 tests) |

---

## Design Decisions

### `None` instead of `0.0` for pixel sizes
`pixel_size_xy_um`, `pixel_size_z_um`, `reference_pixel_size_xy_um`, `reference_pixel_size_z_um` all default to `None`.

- `None` = "not provided / no rescaling requested"
- `0.0` was considered but is physically nonsensical and ambiguous
- Fail-fast `ValueError` in `ExperimentRegistry.__post_init__` if reference size is set but any experiment is missing pixel sizes — catches misconfiguration at `setup()` time, before training starts

### Scale factor convention
```
scale = experiment_um / reference_um
```
- `scale > 1` → experiment has larger pixels → read fewer disk pixels to cover same physical area
- `scale = 1` → no-op (short-circuits interpolation entirely)
- Read window: `y_half = round((patch_size // 2) * scale_y)`
- After read: `F.interpolate(..., size=target, mode="nearest-exact")` back to exact target size

### Cross-scope positives
- `cross_scope_fraction: float = 0.0` — fraction of positives per batch that are cross-microscope
- Match criteria: different `microscope`, same `condition`, `|HPI_anchor - HPI_candidate| <= hpi_window`
- Falls back to temporal positive if no cross-scope candidate found
- Validation at dataset init: raises if `cross_scope_fraction > 0` and any experiment has `microscope = ""`

### Freeze backbone
- `freeze_backbone: bool = False` on `ContrastiveModule`
- Implemented in `on_fit_start` — freezes `self.model.backbone.parameters()`
- Only the projection MLP is updated during finetuning

---

## Backwards Compatibility

All new fields default to `None` / `""` / `0.0`:

| Scenario | Behaviour |
|---|---|
| Old collection YAML (no pixel sizes) | `scale_factors = 1.0` → read window unchanged, no rescaling |
| Old parquet (no `microscope` column) | `_align_parquet_columns` fills `microscope = ""` |
| `cross_scope_fraction = 0.0` (default) | Pure temporal positives — no change to existing behaviour |
| `freeze_backbone = False` (default) | No change to optimizer |

---

## Usage

### Collection YAML — add per experiment
```yaml
experiments:
  - name: scope1_exp
    microscope: "scope1"
    pixel_size_xy_um: 0.2028
    pixel_size_z_um: 0.5
    ...
  - name: scope2_exp
    microscope: "scope2"
    pixel_size_xy_um: 0.1625
    pixel_size_z_um: 0.5
    ...
```

### Datamodule config — enable rescaling + cross-scope finetuning
```yaml
data:
  class_path: dynaclr.data.datamodule.MultiExperimentDataModule
  init_args:
    reference_pixel_size_xy_um: 0.2028   # one scope's pixel size as reference
    reference_pixel_size_z_um: 0.5
    cross_scope_fraction: 0.5
    hpi_window: 1.0
```

### Engine config — freeze backbone for finetuning
```yaml
model:
  class_path: dynaclr.engine.ContrastiveModule
  init_args:
    freeze_backbone: true
    lr: 1.0e-5
    ckpt_path: path/to/pretrained.ckpt
```

See full example: `applications/dynaclr/configs/training/batch_correction_fit.yml`

---

## TODO

- [ ] Decide whether to add `pixel_size_xy_um`, `pixel_size_z_um`, `microscope` fields to Airtable so `build-collection` can auto-populate them (currently must be filled manually in the YAML)
- [ ] Run `fast_dev_run` smoke test with `batch_correction_fit.yml` once a two-microscope dataset is available
- [ ] QC: verify `stratify_by=["condition", "microscope"]` produces balanced batches across scopes
