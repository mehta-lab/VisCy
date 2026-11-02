# Cell segmentation and organelle signal quality

This optional pipeline measures fluorescence inside individual cells before
deciding which observations should enter image embeddings, median/MAD
normalization, PCA, and witness-score analyses. Cell masks also reveal how much
of a cell fits inside a square crop centered on its tracked nucleus.

The initial recipe uses existing virtual membrane predictions with Cellpose
CP-DINO (DINOv3 ViT-L). Phase3D focus metadata selects five planes, from one
below focus through three above, for maximum projections of virtual stains and
acquired fluorescence. Invalid focus windows are recorded as missing frames.
Existing nuclear tracks are associated by majority overlap; multiple nuclei may
share one cell. Cytoplasmic measurements exclude nuclear footprints.
Cell IDs are local to each FOV and frame; existing nuclear `track_id` values
provide the temporal links. Several tracks may therefore share one measured
cell. Inspect those cases for true multiple nuclei versus merged masks.

## Pilot

The native-resolution reference recipe is
`applications/dynaclr/configs/phenotyping/cpdino_tomm20.yml`.
Use an environment containing the `dynaclr[phenotyping]` dependencies. The
initial HPC run uses `.venv-phenotyping`, which preserves the existing `.venv`.
Its package versions and checkpoint checksum are recorded in the baseline's
`environment.json` and each attempt's provenance; do not run `uv sync` over the shared working environment to
reproduce this pilot.

Run compute on an allocated GPU node or through Slurm. For example, from the
repository on `gpu-sm02-10`:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-phenotyping/bin/python -m dynaclr.phenotyping.pipeline --config applications/dynaclr/configs/phenotyping/cpdino_tomm20.yml --dry-run
```

The runner supports `--fov`, `--time`, `--annotation-frames`, `--resume`, and
`--summarize`. Keep material changes in new numbered attempts directly under
each dataset's `2-phenotyping/segmentations_membrane` directory. Preserve the
configuration, input identities, checkpoint hash, and source snapshot with the
result. A completed attempt is immutable.

The reviewed 18-frame preview and full April pilot use raw grayscale resized
to 0.58 µm/pixel before inference, then restore labels to the native grid.
Their exact recipes are the dataset-relative `02_grayscale_058_pilot/pilot.yml`
and `03_full_pilot/pilot.yml`. This scale is an empirical pilot choice; the
repository reference recipe retains native resolution. Do not overwrite the
historical `00_cpdino_baseline` output named in that template; choose a fresh
attempt with `--output`.

### Interpreting cell size and crop coverage

**Native pixels** are pixels on the original assembled image grid, before
resampling. April and July use 0.1133 µm/pixel: a 256-native-pixel square spans
29.0 µm. A 256-pixel input at the matrix reference scale of 0.1494 µm/pixel
spans 38.2 µm and reads approximately 338 native pixels per side. Check each
model's resolved scale and crop settings before comparing pixel counts.

The pilot has three distinct measurements:

| Measurement | Meaning |
|---|---|
| Cell bounding-box width and height | The extents of the inferred whole-cell mask, independent of crop center. |
| `nuclear_centered_patch_side_px` | A conservative odd square side centered on the tracked nuclear coordinate that encloses the entire inferred whole-cell bounding box. |
| Cytoplasm area retained by a crop | Cytoplasm pixels inside the specified square divided by all pixels in that cell's cytoplasmic mask. This requires reading the mask; bounding-box containment cannot substitute for it. |

The earlier 512/768-pixel coverage table counts the fraction of nuclear
track/frame anchors whose **entire inferred cell** fits. It does not report
the percentage of cytoplasm retained inside a crop. An eccentric nucleus,
an elongated cell, or a merged segmentation increases the required centered
square. Multiple nuclear anchors may share one inferred cell. These are
geometric diagnostics, not recommended embedding patch sizes or segmentation
accuracy estimates. The 0.58 µm working scale remains under review for merges.

Independent CP-DINO segmentation of `nuclei_prediction` and
`membrane_prediction` can check nuclear footprints and assignments on the
same projection. Subtracting the nuclear foreground produces a cytoplasmic
mask; it does not by itself separate merged cell labels. Preserve candidate
multinucleated cells and review overlapping or floating cells with axial
context before deciding whether to split or reassign them.

### Running FOV arrays

For FOV arrays, initialize once with `--init`, provide one FOV per line, and
use `tools/phenotyping_fov.sbatch`. Each task has a two-hour limit and writes
distinct frame shards. Submit `tools/phenotyping_summary.sbatch` with an
`afterok` dependency on the array to aggregate and compare annotations. These
pilot launchers use an archived runtime under the attempt directory, plus an
archived `provenance/review_cell_signal.py`, so record the source snapshot
before submission.

## Collection and model matrix

`dynaclr phenotyping-batch --config CONFIG -c COLLECTION --dry-run` displays
resolved dataset inputs and Slurm commands. A real submission uses `--submit`
and explicit `--experiment NAME` selections. Full-Z assembly and prepared focus
stores are distinct inputs; derived MIP stores are not interchangeable with
the full-Z source. Fluorescence marker identity must respect well-specific
channel mappings, especially mixed-organelle plates.
The dataset-level batch launcher defaults to 72 hours because it processes
FOVs sequentially; five-plane reads still decompress full-Z source chunks.
Use `--time-limit` to change this for a measured workload. The initial April
pilot uses FOV arrays to distribute those reads across jobs.

For shared matrix execution, set `defaults.phenotyping_config` and explicitly
include `phenotype` in `--stages`. Phenotyping is planned once per dataset,
outside the model/checkpoint loops. Prediction, normalization, and evaluation
depend on the shared jobs; training can run independently. Existing matrix
defaults do not launch phenotyping. Invoke with the phenotyping-capable Python
environment so submitted workers have the same dependencies.

## Signal annotations and eligibility

The April 10 viewer's 600 observations marked `empty` are confident candidate
references; the July 1 viewer's 357 are uncertain supporting evidence. They are
frame-level labels selected using UMAP clusters. Unmarked observations are
unreviewed, not confirmed signal-positive controls. Compare measured raw
cytoplasmic fluorescence with these labels and inspect disagreements; UMAP
separation itself cannot validate the annotations.

Eligibility keys are `dataset`, `fov_name`, `track_id`, `t`, and `channel_name`.
Keys must be nonnull and unique; `track_id` and `t` are integers. `eligible`
is a nullable boolean and `reason` a nullable string. Only explicit `False`
excludes a row. Matched null decisions and missing keys remain eligible, with
separate audit counts. Duplicate keys or invalid decision types raise errors.
Use canonical collection experiment names and raw channel names, such as
`GFP` or `Phase3D`, rather than marker labels or viewer aliases. A fluorescence
exclusion never excludes the corresponding phase row. Before cutoff review,
core eligibility exports remain entirely null. Missing measurements never
count as low signal; track summaries include valid-frame coverage.

### Optional prediction and matrix configuration

After review, `predict-triplet --eligibility /absolute/reviewed.parquet` filters
tracking rows before image patches are read. Empty selected FOVs are omitted.
If an entire reporter is excluded, prediction raises before model/GPU setup
and writes no embedding store for that reporter. The optional flag authorizes
that file; no additional mandatory review manifest or automatic cutoff exists.

Matrix eligibility belongs in `defaults.eligibility` or the model entry;
`predict_flags.eligibility` is rejected. A model-level path overrides the
default; `eligibility: null` disables it for that model. Paths resolve relative
to the submission working directory, so absolute paths are recommended.
The wrapper forwards `DYNACLR_ELIGIBILITY` without adding positional arguments.

Use a **distinct run label** for each eligibility file/content identity and an
explicit checkpoint pointing to the original trained model:

```yaml
defaults:
  collection: /absolute/collection.yml
  datasets_root: /absolute/datasets
  eligibility: /absolute/reviewed-v1.parquet
  eval_config: /absolute/evaluation.yml
models:
  - train_sbatch: /absolute/original-training.sh
    family: DynaCLR
    run: original-run-eligibility-reviewed-v1
    ckpt_name: last
    checkpoint: /absolute/original-run/checkpoints/last.ckpt
  - train_sbatch: /absolute/original-training.sh
    family: DynaCLR
    run: original-run-unfiltered
    ckpt_name: last
    checkpoint: /absolute/original-run/checkpoints/last.ckpt
    eligibility: null
```

Request `--stages predict,normalize,eval` for a full refit, or `--stages predict`
for raw embeddings only. Eligibility does not implicitly schedule normalization.
For checkpoint sweeps, give matching `ckpt_names` and explicit `checkpoints`
lists. Changing `run` without explicit checkpoint paths changes derived
checkpoint lookup. Foundation models use their explicit `training_config`;
Katamari uses its explicit checkpoint. Their eligibility field has the same
defaults/model schema.

Outputs live under
`{dataset}/2-phenotyping/predictions/{family}/{run}/{ckpt_name}/embeddings/{marker}.zarr`.
Both the authoritative parquet file SHA256 and a canonical record SHA256 are
stored in `uns.phenotyping_eligibility` as JSON. Record hashing ignores row
order; file hashing also detects a parquet rewrite with unchanged records.
Raw and normalized cache reuse requires matching provenance. Incompatible
existing output identities are rejected, including with overwrite; use a
fresh run label. Compatible raw stores can skip prediction while missing
normalization is refit only when requested, using the full selected collection.
Normalize/eval-only matrix requests also validate identity before launching
any stage and require existing raw embedding stores.

For existing raw embeddings, the
`dynaclr.evaluation.phenotyping_eligibility` module can audit or export a new
filtered store on compute:

```bash
# Audit: reads only obs, including backed H5AD obs.
.venv/bin/python -m dynaclr.evaluation.phenotyping_eligibility \
  --input /absolute/raw/SEC61B.zarr --eligibility /absolute/reviewed-v1.parquet \
  --dataset CANONICAL_DATASET --channel-name GFP

# Export: reads raw embeddings and writes a new store plus audit sidecar.
.venv/bin/python -m dynaclr.evaluation.phenotyping_eligibility \
  --input /absolute/raw/SEC61B.zarr --eligibility /absolute/reviewed-v1.parquet \
  --dataset CANONICAL_DATASET --channel-name GFP \
  --output /absolute/eligibility-reviewed-v1/SEC61B.zarr
```

Export retains raw `X`, layers, `var`, `raw`, and `X_backbone`/`X_projections`.
`--raw-obsm-key` retains an additional named raw representation. It removes
other `obsm` entries (including normalized PCA/UMAP), all `obsp`, and old `uns`.
Original `uns` is archived in the `.eligibility.json` sidecar; the new store
contains the eligibility audit. Existing output paths are refused and the
source is unchanged. `X` must be raw; normalized-only sources need re-embedding.
Refit control normalization, PCA, UMAP, and witness analyses on the selected
cohort. The pilot does not activate any exclusion threshold or change patch
sizes.

## Research history

The PLAN and exploration live in
`/home/eduardo.hirata/repos/ed-planning/projects/dynaclr/segmentation/cellpose-dino-dynacell/`.
The April 10 pilot must be reviewed before collection-wide execution.
