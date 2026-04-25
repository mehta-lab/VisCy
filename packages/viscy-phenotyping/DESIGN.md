# viscy-phenotyping: Design Notes

This document explains the architecture, design choices, and rationale behind
`viscy-phenotyping` as developed. It is intended as a reference for future
contributors and for understanding why things are the way they are.

---

## Purpose

`viscy-phenotyping` computes interpretable image-based features from fluorescence
microscopy patches and 2-D nuclear segmentation masks. It is designed to complement
DynaCLR embeddings — the features are structured so they can be joined with the
DynaCLR AnnData output on `(fov_name, track_id, t, id)` and correlated with the
learned embedding space.

---

## Repository structure

```
packages/viscy-phenotyping/
├── src/viscy_phenotyping/
│   ├── __init__.py
│   ├── cli.py                    # Click CLI (entry points)
│   ├── profiler.py               # Orchestrator: calls all feature modules per cell
│   ├── io.py                     # Border-safe 2-D patch cropping
│   ├── features.py               # Nuclear morphology from full-FOV label images
│   ├── features_shape.py         # Problem 6: nuclear shape / FSDs
│   ├── features_radial.py        # Problems 1 & 3: radial distribution / ring uniformity
│   ├── features_texture.py       # Problem 2: GLCM, LBP, intensity statistics
│   ├── features_density.py       # Problem 4: Otsu spots, granularity spectrum
│   ├── features_structure.py     # Problem 5: Canny edges, skeleton analysis
│   └── features_gradient.py      # Problem 7: Sobel gradient, signal-to-background
├── tests/
│   ├── features_test.py          # Tests for extract_nuclear_morphology
│   └── profiler_test.py          # Tests for all 7 feature modules + orchestrator
├── FEATURES.md                   # Feature reference documentation
├── DESIGN.md                     # This file
└── pyproject.toml
```

---

## The 7 phenotyping problems

The feature set was designed around 7 biological questions:

| # | Problem | Module |
|---|---|---|
| 1 | Radial distribution of signal from nuclear centre | `features_radial.py` |
| 2 | Signal homogeneity / texture | `features_texture.py` |
| 3 | Concentric ring / ER-like pattern uniformity | `features_radial.py` |
| 4 | Signal packing density and spot size | `features_density.py` |
| 5 | Edge count and strand/filament continuity | `features_structure.py` |
| 6 | Nuclear shape and circularity | `features_shape.py` |
| 7 | Gradient sharpness and nuclear-vs-cytoplasmic contrast | `features_gradient.py` |

---

## Key design decision: mask usage

### The problem

An early version of the library applied the nuclear binary mask to restrict all feature
computation to pixels inside the nucleus. This is correct for nuclear features but wrong
for cytoplasmic channels — a channel showing cytoplasmic signal (e.g. ER, mitochondria)
would return near-zero or misleading values if only nuclear pixels were sampled.

### The resolution

The nuclear mask is used in three different ways depending on the feature:

| Usage | Features |
|---|---|
| **Mask IS the object** — shape of the nucleus is what is being measured | `features_shape.py` |
| **Mask provides the centroid only** — radial profile is computed over all patch pixels | `features_radial.py` |
| **Mask separates nucleus from background** — used only for `nucleus_to_cytoplasm_ratio` (mean inside mask vs mean of all pixels outside mask) | `features_gradient.py` |
| **No mask** — features computed on the full fluorescence patch | `features_texture.py`, `features_density.py`, `features_structure.py` |

This means the same library correctly handles both nuclear-localised and
cytoplasmic-localised channels without any reconfiguration.

---

## Output format: CSV not AnnData

The output is a plain CSV file with one row per cell per timepoint. Column layout:

```
fov_name, track_id, t, id, parent_track_id, parent_id, z, y, x, [feature columns...]
```

The index columns match `ULTRACK_INDEX_COLUMNS` from `viscy-data` exactly, so the CSV
can be joined directly with DynaCLR AnnData output using pandas:

```python
import anndata as ad
import pandas as pd

adata = ad.read_zarr("embeddings.zarr")
features = pd.read_csv("features.csv")
merged = adata.obs.merge(features, on=["fov_name", "track_id", "t", "id"])
```

AnnData was considered for the output format but dropped because:
- It adds a heavy dependency to a pure CPU feature computation library
- Plain CSV is easier to inspect, share, and load in any tool
- The merge step is one line of pandas

---

## CLI design

Three commands are exposed via the `viscy-phenotyping` entry point:

```
viscy-phenotyping write-header     # initialise output CSV with correct columns
viscy-phenotyping compute-features # process one or all FOVs
viscy-phenotyping list-fovs        # enumerate FOV names from a zarr store
viscy-phenotyping merge-features   # combine per-FOV CSVs (optional utility)
```

### `--fov-name` for parallelism

`compute-features` accepts an optional `--fov-name` argument. Without it, all FOVs
are processed sequentially in one job. With it, only a single FOV is processed —
intended for SLURM array job parallelism where each task handles one FOV.

### `write-header`

Before any array jobs are submitted, `write-header` runs `compute_cell_features` on a
tiny synthetic patch (64×64, ones image, circular mask) to discover the full set of
output column names. It writes an empty CSV containing only the header row.

This avoids a race condition in the parallel write step: if the shared output CSV did
not pre-exist, two array jobs finishing simultaneously could both try to write the
header, resulting in a duplicate header or corrupt file.

---

## SLURM parallelisation

### The pattern

```
submit_features.sh          ← user runs this once
    │
    ├── write-header         ← creates shared CSV with header
    ├── list-fovs            ← writes fov_list.txt
    └── sbatch --array=0-N   ← one task per FOV
            └── compute_features_worker.sh
                    ├── compute-features --fov-name FOV → /tmp/viscy_XXXX.csv
                    ├── flock -x → tail -n +2 /tmp/... >> shared.csv
                    └── rm /tmp/viscy_XXXX.csv
```

### Concurrent write safety

Each worker:
1. Writes its output to a unique temp file in `/tmp` (local to the compute node, fast)
2. Acquires an exclusive `flock` lock on `{output}.lock`
3. Appends its rows (without header) to the shared CSV
4. Releases the lock and deletes the temp file

This means the shared CSV grows incrementally as jobs complete — you can check
progress with `wc -l features.csv` while the array is running.

### Why `/tmp` for the temp file

Writing to `/tmp` (node-local storage) avoids contention on the shared HPC filesystem
(GPFS/Lustre) during the per-FOV computation. Only the final append touches the shared
filesystem under a lock, minimising I/O bottlenecks.

### Config file

All run parameters live in `features.yml`:

```yaml
dataset_name: 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV
data_path: /hpc/.../registered.zarr
tracks_path: /hpc/.../tracks.zarr
output_csv: /hpc/.../dataset_name_features.csv
source_channels:
  - "raw mCherry EX561 EM600-37"
nuclear_label_channel: "nuclei_prediction_labels_labels"
patch_size: [160, 160]
```

The submit script parses this with a Python heredoc and exports shell variables.
Channel names are joined with `|` before export (to survive shell word-splitting on
spaces) and rebuilt into a bash array in the worker script.

---

## Label ID lookup: `track_id` not `id`

The tracking zarr stores label images where each pixel value equals the `track_id`
of the cell occupying that pixel. The tracking CSV contains both `id` (a large unique
graph node identifier, e.g. `22000013`) and `track_id` (a small integer like `32`).

The mask for a given cell is therefore:

```python
mask = label_patch == int(row["track_id"])   # correct
mask = label_patch == int(row["id"])          # wrong — id never appears in the label image
```

This was discovered by inspecting the label array unique values and comparing them to
the CSV columns.

---

## Label array shape handling

Label arrays in this dataset have shape `(T, 1, 1, Y, X)` — multiple size-1 leading
dimensions from the OME-Zarr storage format. The CLI uses:

```python
label_img = np.squeeze(np.asarray(label_array[t]))
```

`np.squeeze` removes all size-1 dimensions, resulting in `(Y, X)` regardless of how
many leading singleton dimensions the zarr contains. This is more robust than
`raw[0] if raw.ndim == 3 else raw`, which only handles one specific shape.

---

## Correlation analysis

`applications/dynaclr/pc_feature_correlation.py` correlates DynaCLR PCs with computed
features.

### Usage

```bash
uv run python applications/dynaclr/pc_feature_correlation.py \
    --embeddings /path/to/embeddings.zarr \
    --features   /path/to/dataset_features.csv \
    --output     /path/to/correlation_heatmap.svg \
    --n-pcs      8 \
    --top-n-features 5    # optional: top 5 features per PC (union across PCs)
```

Run from the root of the VisCy repository.

### What it does

1. Loads the DynaCLR AnnData zarr and runs PCA on the embeddings (`X` matrix)
2. **Filters to PCs with > 10% variance explained** — lower-variance PCs are dropped
   from the plot. The variance % is shown in each y-axis label (e.g. `PC1 (32.1%)`)
3. Loads the features CSV
4. Joins on `(fov_name, track_id, t, id)`
5. Computes pairwise Spearman rank correlation between each PC and each feature column
   directly — **not** via a full square `.corr()` matrix, so PC↔PC and feature↔feature
   correlations are never computed or displayed
6. Saves a heatmap with PCs on the Y-axis and features on the X-axis

Spearman rank correlation is used (rather than Pearson) because many image features
are non-normally distributed.
