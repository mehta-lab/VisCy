# Pseudotime DAG

This document describes how the pseudotime pipeline runs. For why it
runs this way — methodology decisions, claims, falsification protocol —
see the source-of-truth discussion document at
`/home/eduardo.hirata/repos/DynaCLR/.planning/dynaclr_dtw_discussion.md`.

## 1. Goals

We measure when SEC61 (ER), G3BP1 (stress granules), and quantitative
phase morphology change relative to per-cell NS3 sensor translocation
in single A549 cells, then compare three alignment tracks to see which
sharpens the population timing readouts.

The pipeline produces three parallel sets of organelle-remodeling
plots, one per track, indexed on the same lineage-reconnected cohort
so the side-by-side comparison is direct.

| Track | Anchor | Alignment | Outputs in |
|---|---|---|---|
| **A-anno** | Human `infection_state` first-positive frame | Per-cell shift, real-time | `3-organelle-remodeling/A-anno/` |
| **A-LC** | Linear classifier `predicted_infection_state` first-positive frame | Per-cell shift, real-time | `3-organelle-remodeling/A-LC/` |
| **B** | NS3 embedding band on transition window | Hybrid DTW: warp transition only, propagated to organelle and phase | `3-organelle-remodeling/B/` |

Path B is the methodological contribution. Paths A-anno and A-LC are
baselines.

### What `t_rel = 0` means

`t_rel = 0` is when the NS3 protease sensor crosses our chosen anchor:
either the human `infection_state` first-positive frame (Path A-anno),
the LC threshold-crossing (Path A-LC), or the NS3 embedding's
half-rise band on the transition window (Path B). All three are
landmarks downstream of viral entry, polyprotein translation, and ER
remodeling — `t_rel = 0` is a fiducial clock, not the start of
infection. See discussion §1.1 and §2.1 for biological framing.

## 2. Pipeline stages

```
0-select_candidates    →  1-build_template   →  2-align_cells   →  3-organelle-remodeling   →  4-compare_tracks
   (lineage-reconnect,    (Path B only —          (Path A-anno,        (per-track per-organelle      (side-by-side
    cohort tagging,        NS3 transition          A-LC, B               readouts: SEC61              comparison,
    manual + auto)         template)               alignments)           cosine-distance,             warp-vs-no-warp,
                                                                         G3BP1 oscillation,           bimodality test,
                                                                         phase distance)              robustness)
```

Stages 0 and 1 share data across tracks. Stages 2, 3, and 4 fork
per-track, then re-converge in Stage 4 for comparison.

## 3. Directory layout

```
applications/dynaclr/
├── configs/pseudotime/
│   ├── datasets.yaml                                # datasets + embedding patterns
│   ├── candidates.yaml                              # candidate sets, cohort tags, lineage rules
│   ├── build_template.yaml                          # Stage 1 (Path B template build)
│   ├── align_cells.yaml                             # Stage 2 (per-track query sets)
│   └── compare_tracks.yaml                          # Stage 4 (cross-track headlines)
├── docs/DAGs/pseudotime.md                          # this file
├── src/dynaclr/pseudotime/                          # library
│   ├── dtw_alignment.py                             # template fit + warp solver
│   ├── io.py                                        # template-zarr layout + provenance
│   ├── alignment.py                                 # lineage reconnection, daughter handling
│   ├── signals.py                                   # extract annotation / LC / embedding signals
│   ├── metrics.py                                   # onset, t_50, peak, oscillation stats
│   └── plotting.py                                  # response curves, heatmaps, comparisons
└── scripts/pseudotime/
    ├── 0-select_candidates/
    │   ├── select_candidates.py                     # auto path (from annotations)
    │   ├── manual_candidates.py                     # manual path (hand-picked tracks)
    │   ├── reconnect_lineages.py                    # mother+daughter stitching, divides flag
    │   ├── tag_cohorts.py                           # productive / bystander / abortive / mock
    │   ├── inspect_candidates.py                    # per-track image montage QC
    │   └── candidates/                              # output: lineage-aware annotation CSVs
    ├── 1-build_template/                            # Path B only
    │   ├── build_template.py                        # NS3 transition template via DBA
    │   ├── evaluate_template.py                     # self-align sanity check
    │   ├── templates/                               # output: template_*.zarr
    │   └── plots/                                   # output: build-set diagnostics
    ├── 2-align_cells/
    │   ├── align_anno.py                            # Path A-anno: real-time shift on infection_state
    │   ├── align_lc.py                              # Path A-LC: real-time shift on LC predictions
    │   ├── align_embedding.py                       # Path B: hybrid DTW + warp propagation
    │   ├── A-anno/alignments/                       # per-track output parquets
    │   ├── A-LC/alignments/
    │   └── B/alignments/
    ├── 3-organelle-remodeling/
    │   ├── readout_sec61.py                         # cosine-distance-from-baseline
    │   ├── readout_g3bp1.py                         # oscillation excursion stats
    │   ├── readout_phase.py                         # phase embedding distance
    │   ├── A-anno/                                  # per-track per-organelle outputs
    │   ├── A-LC/
    │   └── B/
    └── 4-compare_tracks/
        ├── compare_onsets.py                        # side-by-side SEC61, G3BP1, phase headlines
        ├── compare_phase_to_fluor.py                # claim (a') Spearman ρ
        ├── warp_vs_no_warp.py                       # mandatory comparator (Path B only)
        ├── bimodality_check.py                      # dip-test on every back-projected distribution
        └── headline_figure.py                       # the figure-1 of the paper
```

Status of stage scripts as of this revision:

- **Implemented and current:** `0-select_candidates/select_candidates.py`,
  `manual_candidates.py`, `inspect_candidates.py`,
  `1-build_template/build_template.py`, `evaluate_template.py`,
  `2-align_cells/align_cells.py` (current name covers what becomes
  `align_embedding.py` after split), per-stage plotting scripts.
- **Implemented but in worktree, not in DAG structure:**
  `annotation_remodeling.py` and `prediction_remodeling.py` cover
  Path A-anno and Path A-LC respectively; live in
  `.claude/worktrees/cytoland-virtual-staining-examples/applications/dynaclr/scripts/pseudotime/`.
  Need refresh against new directory structure and configs.
- **TODO: not yet implemented:** `reconnect_lineages.py`,
  `tag_cohorts.py`, the per-track split in Stage 2, per-organelle
  readouts as separate scripts, all of Stage 4.

The current scripts still produce useful output but operate on a
single track (Path A-anno-equivalent) without lineage reconnection,
cohort tagging, or cross-track comparison. The phases below describe
the migration.

## 4. Cohorts

Stage 0 emits one annotation CSV per cohort. All four cohorts share the
schema. Each cell carries a `cohort` column.

| Cohort | Definition | Used by |
|---|---|---|
| `productive` | Lineage with NS3 rise within imaging window; manual `[t_before, t_after]` and `t_key_event` from `manual_candidates.py` | Primary cohort; all three tracks |
| `bystander` | Lineage in infected wells with no NS3 rise across imaging duration | Negative control for claims (a), (a'), (b) |
| `abortive` | Lineage with NS3 channel embedding present, no rise | Claim (b) bifurcation comparison (caveat: censored-data category, see discussion §3.2) |
| `mock` | Lineage from uninfected wells | Per-frame null distribution for organelle distance comparisons |

Mock cells do not get a synthetic `t_zero`. Every mock cell × frame
contributes to an FOV-stratified per-frame null distribution.

## 5. Stage 0 — Select candidates and reconnect lineages

Stage 0 emits per-cohort annotation CSVs. Two entry points feed it:
`select_candidates.py` (auto, from existing annotations) and
`manual_candidates.py` (hand-picked tracks).

The single output artifact is `{cohort}_annotations.csv`, one row per
`(dataset_id, fov_name, lineage_id, t)` over the per-cell crop window.

### 5.1 Auto path

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python select_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/candidates.yaml \
    --candidate-set infection_transitioning_nondiv
```

Filters tracks per `config["candidate_sets"][NAME]["filter"]` (anchor
label, anchor_positive, min_pre/post_minutes, crop_window_minutes),
expands each track into per-frame rows, copies real annotation labels
onto each row, then runs lineage reconnection (§5.3) and cohort tagging
(§5.4) before writing.

### 5.2 Manual path

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
python manual_candidates.py
```

Each track spec is `{t_before, t_after, labels: {...}}` in a Python
dict. Frames in `[t_before, t_after]` get the positive label if they
fall inside an interval. The CSV schema is the only contract with
downstream stages.

### 5.3 Lineage reconnection

**TODO: implement.** Stitches mother + daughter chains into single
lineages using `parent_track_id`. Daughter handling regime-dependent:

- Division before `t_zero`: keep both daughters as paired observations;
  siblings are biologically equivalent at infection.
- Division after `t_zero`: keep the daughter with more pre-`t_zero`
  footage. Daughters at this stage carry different viral loads.
- Division during the transition window: tag as separate cohort outside
  DTW alignment; mitotic ER fragmentation distorts templates.

Each lineage record carries `divides ∈ {none, pre, during, post}` and
`lineage_id` (replaces `track_id` as the canonical unit downstream).

### 5.4 Cohort tagging

**TODO: implement.** For each lineage, derive `cohort` from the
NS3 channel signal:

- `productive`: lineage has manual `t_key_event` and survives window
  cropping in Stage 1.
- `bystander`: in infected well, no NS3 rise across imaging window.
- `abortive`: NS3 channel embedding present, no rise.
- `mock`: from uninfected wells.

### 5.5 Inspect

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python inspect_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/candidates.yaml \
    --candidate-set infection_transitioning_nondiv
```

Renders a per-cell-anchored image montage with `t_key_event` marked.
Also writes a sidecar `_qc.csv` with per-track stats (n_frames,
pre_frames, post_frames, fov, divides) for non-visual QC.

## 6. Stage 1 — Build NS3 transition template (Path B only)

Stage 1 produces the canonical NS3 transition template against which
Path B aligns query cells. Paths A-anno and A-LC do not use a
template.

### 6.1 Template build

```bash
cd applications/dynaclr/scripts/pseudotime/1-build_template
uv run python build_template.py \
    --config ../../../configs/pseudotime/build_template.yaml \
    --template infection_nondividing_zikv
```

The builder:

1. Reads `productive` cohort annotations.
2. Crops each lineage to `[t_zero - h_pre, t_zero + h_post]` real-time.
   Default: `h_pre = 240 min`, `h_post = 360 min` (`540 min` for G3BP1
   readout downstream). See discussion §3.6.
3. Pulls NS3 channel embeddings within the transition sub-window
   `[t_zero - k_pre, t_zero + k_post]`. Default: `k_pre = 60 min`,
   `k_post = 120 min`. Use the 10 min/frame cohort if available
   (target frame count ≥ 12 for DBA stability).
4. Computes per-cell pre-baseline = mean NS3 embedding across pre-window
   frames. Cosine distance against this per-cell baseline.
5. Runs DBA on the transition sub-window only.
6. Saves the template zarr.

### 6.2 Template zarr contents

| Path | Type | Description |
|---|---|---|
| `template` | (T, D) array | DBA template in the transition window |
| `time_calibration` | (T,) array | mean `t_relative_minutes` per template position |
| `template_labels/{col}` | (T,) array | per-position label fractions |
| `tau_event_band` | (2,) array | `[τ_lo, τ_hi]`: half-rise band of `||dT/dτ||`. The event identifier per discussion §3.4. |
| `lineage_ids` | list (attrs) | `[dataset_id, fov_name, lineage_id]` per build-set lineage |
| `aggregator` | str (attrs) | `"dba"` |
| `template_duration_minutes` | float (attrs) | `time_calibration[-1] - time_calibration[0]` |
| `build_frame_intervals_minutes` | dict (attrs) | per-dataset frame interval at build time |
| `viscy_git_sha`, `dtaidistance_version`, `scikit_learn_version`, `numpy_version` | str (attrs) | provenance |

**TODO:** add `tau_event_band` to the zarr writer. Currently the
template stores derivative-argmax as a point.

### 6.3 Self-consistency check

```bash
uv run python evaluate_template.py \
    --config ../../../configs/pseudotime/build_template.yaml \
    --template infection_nondividing_zikv
```

Re-aligns the build-set lineages onto the template they built. Sanity
check, not generalization evidence.

## 7. Stage 2 — Align query cells per track

Stage 2 forks per track. Each track produces an alignment parquet with
the same schema (described in §7.4) so Stage 3 readouts and Stage 4
comparisons read uniformly.

### 7.1 Path A-anno: annotation-anchored real-time shift

**TODO: implement** as `align_anno.py`. Replaces `annotation_remodeling.py`'s
alignment step (currently in
`.claude/worktrees/.../annotation_remodeling.py`).

Per-cell `t_zero` = first frame where `infection_state == "infected"`
in the manual or auto annotation CSV. Real-time per-cell shift:
`t_rel = (t - t_zero) * frame_interval_minutes`. No DTW, no template,
no warping.

```bash
# TODO: command shape
uv run python align_anno.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --query-set zikv_07_24
```

### 7.2 Path A-LC: LC-anchored real-time shift

**TODO: implement** as `align_lc.py`. Replaces `prediction_remodeling.py`.

Per-cell `t_zero` = first frame of the longest run of `predicted_infection_state == "infected"` in the NS3 channel embedding zarr. `min_run` parameter (default 3) prevents single-frame flickers from defining `t_zero`.

```bash
# TODO: command shape
uv run python align_lc.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --query-set zikv_07_24 --min-run 3
```

### 7.3 Path B: hybrid DTW + warp propagation

Existing `align_cells.py` covers most of this. Renaming and feature gaps
listed below.

```bash
cd applications/dynaclr/scripts/pseudotime/2-align_cells
uv run python align_embedding.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_zikv \
    --query-set zikv_07_24 \
    --min-match-minutes 360 --max-skew 0.7
```

The aligner:

1. Loads `templates/template_{name}.zarr` and the cohort-tagged query
   annotations.
2. Pulls NS3 channel embeddings, applies build-time L2 normalization
   (no refit at alignment time).
3. For each query lineage, runs subsequence DTW on the transition
   sub-window. The template (length T) must match fully; the query
   floats. Returns a warp path, best-match window `[q_start, q_end]`,
   cost, and `path_skew`.
4. **TODO:** propagate the warp path to organelle and phase channel
   embeddings within the transition sub-window. Pre and post stay in
   real-time.
5. **TODO:** for each cell, back-project the τ_event band to a
   per-cell real-time interval. Report median + IQR per cohort.
6. Applies guards (§7.6) and writes alignment parquet.
7. Writes a sidecar `{template}_on_{qset}.drop_log.json` with
   per-filter drop counts.

### 7.4 Alignment parquet schema (all three tracks)

One row per `(dataset_id, fov_name, lineage_id, t)`. Per-lineage columns
are repeated on every frame so downstream scripts can filter rows
without a separate join.

| Column | Type | Per-frame? | Tracks | Notes |
|---|---|---|---|---|
| `dataset_id`, `fov_name`, `lineage_id`, `t` | ids | yes | all | identifiers (`lineage_id` replaces `track_id`) |
| `cohort` | str | yes | all | `productive` / `bystander` / `abortive` / `mock` |
| `divides` | str | yes | all | `none` / `pre` / `during` / `post` |
| `t_zero` | int | per-lineage | all | per-cell anchor frame |
| `t_rel_minutes` | float | yes | all | real-time minutes from `t_zero` |
| `track_path` | str | per-lineage | all | `A-anno` / `A-LC` / `B` |
| `pseudotime` | float ∈ [0, 1] | yes | B only | warp-path template position |
| `alignment_region` | str | yes | B only | `pre` / `aligned` / `post` |
| `t_rel_minutes_warped` | float | yes | B only | back-projected real-time at template position; equals `t_rel_minutes` outside `aligned` |
| `dtw_cost` | float | per-lineage | B only | raw DTW cost |
| `length_normalized_cost` | float | per-lineage | B only | `dtw_cost / len(warp_path)` |
| `path_skew` | float ∈ [0, 1] | per-lineage | B only | mean deviation from ideal diagonal |
| `match_q_start`, `match_q_end` | int | per-lineage | B only | absolute query frames bounding the matched window |
| `template_id` | str | per-lineage | B only | UUID linking to template zarr |

`t_rel_minutes` is shared across tracks. For Paths A, it's the only
time coordinate. For Path B, it covers pre/post windows; the transition
window also has `t_rel_minutes_warped` from the back-projection.

### 7.5 Diagnostic plots per track

```bash
# Path B only — same scripts as before, renamed
uv run python rank_by_cost.py --query-set zikv_07_24
uv run python plot_top_n_montage.py --query-set zikv_07_24 --top-n 30 --worst-n 10
uv run python plot_pcs_aligned.py --query-set zikv_07_24 --top-n 50
```

### 7.6 Guards and frame-rate invariance

DTW with generous psi can collapse the template onto a single query
frame. Five guards prevent and surface this:

| Guard | CLI flag | Default | Rejects |
|---|---|---|---|
| Non-finite cost | always on | — | tracks too short for the solver |
| Path skew | `--max-skew` | 0.7 | degenerate non-diagonal warps (primary gate per discussion §3.8 #2) |
| Length-normalized cost gate | `--cost-gate` | sweep | stereotypy filter; sweep `{0, 10, 20, 30, 50}%` |
| Minute-based match window | `--min-match-minutes` | 360 | template compressed onto tiny real-time window |
| Pre/post headroom | per query-set YAML | 0 | lineages without real footage on either side |

Path skew is the primary gate; cost is secondary (per discussion §3.8
#2: skew rejects DTW failures without rejecting biological variance).
**TODO:** wire path-skew-as-primary into the existing two-pass filter
(currently cost-only).

`--max-psi-minutes` defaults to half the template duration, read from
template attrs. Per-track psi is `round(max_psi_minutes / dataset_frame_interval_minutes)`.

## 8. Stage 3 — Organelle remodeling readouts

Stage 3 forks per track per organelle. Each readout reads its track's
alignment parquet and produces population curves and per-cell timing
metrics. The plotting scripts in this stage replace the current
`plot_organelle_remodeling.py` with per-organelle scripts.

### 8.1 SEC61 readout

**TODO: implement** as `readout_sec61.py`. Cosine distance of SEC61
embedding from per-cell baseline (= mean SEC61 embedding across
pre-window frames). Per-cell trajectory binned by `t_rel_minutes`.
Population curve = binned median + IQR.

```bash
uv run python readout_sec61.py \
    --track {A-anno|A-LC|B} \
    --query-set zikv_07_24
```

Headline metric: real-time `t_rel` at which productive median exceeds
FOV-paired mock 95th percentile. Reported per-cohort. For Path B,
back-projected real-time IQR is reported alongside.

### 8.2 G3BP1 readout

**TODO: implement** as `readout_g3bp1.py`. Oscillation-aware metrics
on the post-window (real-time, never warped per discussion §3.6 — stress
granule kinetics are sub-frame, warping by NS3 is meaningless):

| Metric | Definition |
|---|---|
| `excursion_count` | Number of distance threshold crossings in the post-window |
| `dwell_time_minutes` | Total time above threshold |
| `largest_excursion_amplitude` | Max distance above baseline |
| `largest_excursion_duration` | Duration of the longest contiguous excursion |

Threshold = mock pulsation 95th percentile, FOV-stratified.

### 8.3 Phase readout

**TODO: implement** as `readout_phase.py`. Cosine distance of phase
embedding from per-cell baseline. Same structure as SEC61.

For claim (a'), phase per-cell `t_50` (or onset-time metric) is
extracted and matched to the same cell's SEC61 fluorescence onset
time. The pair `(t_phase, t_sec61)` per cell feeds Stage 4's Spearman
ρ comparison.

### 8.4 Per-cell timing metrics

**TODO: implement** per-track `compute_timing_metrics.py`. Replaces
the existing single-track version under `3-organelle-remodeling/`.

Per-cell scalars (computed on the aligned region):

| Metric | Definition | Why |
|---|---|---|
| `t_onset_abs` | First `t_rel` where `distance − pre_median` crosses `+0.10` | SNR-robust |
| `t_50` | First `t_rel` crossing `pre_median + 0.5 × Δpeak`, last 2 frames excluded | Half-rise timing |
| `t_peak` | `argmax` of distance over interior aligned region | Time of maximum embedding divergence |
| `delta_peak` | `max(aligned distance) − median(pre distance)` | Amplitude in cosine units |
| `rise_rate_per_hour` | OLS slope of distance vs `t_rel` over aligned region × 60 | Per-cell aggregate speed |

Outputs: `timing/{stem}_per_cell.parquet` + `timing/{stem}_summary.md`.

## 9. Stage 4 — Cross-track comparison and robustness

**TODO: implement.** Stage 4 is the headline-figure stage of the paper.

### 9.1 Side-by-side onset comparison

```bash
cd applications/dynaclr/scripts/pseudotime/4-compare_tracks
uv run python compare_onsets.py \
    --query-set zikv_07_24 \
    --organelles sec61,g3bp1,phase \
    --tracks A-anno,A-LC,B
```

For each `(organelle, track)`, plot the population curve + IQR on a
real-time axis. Three columns (organelles), three rows (tracks). The
methodological-claim verdict is whether Path B's IQR is ≥25% tighter
than the better of A-anno and A-LC at the headline metric (per
discussion §2.2).

Outputs:
- `compare_onsets_{qset}.png`: the headline figure
- `compare_onsets_{qset}_summary.csv`: per `(organelle, track, cohort)`
  headline number with CI, dip-test, dropped-cohort comparison

### 9.2 Phase-to-fluorescence correlation (claim a')

```bash
uv run python compare_phase_to_fluor.py \
    --query-set zikv_07_24 \
    --organelle {sec61|g3bp1} \
    --track B
```

For each productive lineage, extract phase and matched fluorescent
marker `t_50`. Compute Spearman ρ across the cohort. Permutation null
(1000 shuffles) for the p-value. SEC61 carries falsification weight
(per discussion §7 claim a'); G3BP1 expected-null result reports as
positive evidence for fluorescence-and-phase complementarity.

### 9.3 Warp-vs-no-warp comparator (Path B only)

```bash
uv run python warp_vs_no_warp.py \
    --query-set zikv_07_24 \
    --organelles sec61,g3bp1,phase
```

For Path B, regenerate every organelle/phase readout twice: once with
the NS3 warp propagated (current behavior), once without (organelle
embedding kept on its own real-time axis). Side-by-side population
curves and headline numbers. Per discussion §3.8 #10 and §4.5: if the
two agree within 25%, warp propagation is neutral; keep it. If they
diverge, investigate which masks real timing.

### 9.4 Bimodality test

```bash
uv run python bimodality_check.py \
    --input compare_onsets_{qset}_summary.csv
```

Hartigans dip-test (or 1- vs. 2-component GMM BIC) on every
back-projected real-time distribution from Path B. Per discussion
§3.8 #11. Multimodal distributions get reported as mode-stratified
medians or histogram-as-headline rather than a single point.

### 9.5 Robustness panel

| Check | Implements | Reference |
|---|---|---|
| Cost-gate sweep `{0, 10, 20, 30, 50}%` | Path B only | Discussion §3.8 #1 |
| Path-skew gate as primary | Path B only | Discussion §3.8 #2 |
| Window ablation ±50% on `h_pre`, `h_post`, `k_pre`, `k_post` | Path B only | Discussion §3.8 #3 |
| Within-condition shuffle null | All tracks | Discussion §3.8 #4 |
| DBA-init K=10 stability | Path B only | Discussion §3.8 #5 |
| Per-cell baseline-noise bootstrap | Path B only | Discussion §3.8 #6 |
| Funnel transparency table | All tracks | Discussion §3.8 #7 |
| Mock-FOV stability | All tracks | Discussion §3.8 #8 |
| Inter-annotator agreement on `t_zero` | All tracks | Discussion §3.8 #9 |
| Warp-vs-no-warp comparator | Path B only | §9.3 above |
| Bimodality test | Path B only | §9.4 above |
| Cost-gate kept-vs-dropped symmetric reporting | Path B only | Discussion §3.8 #12 |

All robustness outputs land in `4-compare_tracks/robustness/`.

## 10. Configs

Three YAMLs split across `configs/pseudotime/`. Each is loaded with
`datasets.yaml` via the `--datasets` + `--config` CLI pair.

| File | Contains | Used by |
|---|---|---|
| `datasets.yaml` | `data_zarr`, `embeddings` glob patterns, `datasets` list (pred_dir, annotations_path, fov_pattern, frame_interval_minutes) | every stage |
| `candidates.yaml` | `candidate_sets.{name}`, lineage-reconnect rules, cohort-tagging rules | Stage 0 |
| `build_template.yaml` | `templates.{name}` (Path B template build) | Stage 1 |
| `align_cells.yaml` | `query_sets.{name}` per track | Stage 2 |
| `compare_tracks.yaml` | Stage 4 headlines (which `(organelle, track, cohort)` triples to plot, sweep ranges) | Stage 4 |

### 10.1 Example `candidate_sets` entry

```yaml
candidate_sets:
  zikv_productive_07_24:
    datasets: ["2025_07_24_SEC61", "2025_07_24_G3BP1"]
    cohort_filter:
      productive:
        anchor_label: infection_state
        anchor_positive: infected
        anchor_negative: uninfected
        min_pre_minutes: 240
        min_post_minutes: 360
      bystander:
        anchor_label: infection_state
        all_negative: uninfected
        min_imaging_minutes: 600
      abortive:
        # NS3 channel present but no rise; defined operationally per
        # the abortive-detection step
        require_ns3_channel: true
        max_ns3_rise_amplitude: 0.05  # cosine units
      mock:
        well_pattern: "B/*"  # uninfected wells
    lineage_rules:
      reconnect: true
      daughter_handling:
        pre_t_zero: paired
        post_t_zero: longer_pre_window
        during_transition: separate_cohort
    crop_window:
      h_pre_minutes: 240
      h_post_minutes: 360
      h_post_minutes_g3bp1: 540
    max_lineages: 200
```

### 10.2 Example template entry (Path B)

```yaml
templates:
  infection_nondividing_zikv:
    candidate_set: zikv_productive_07_24
    cohort: productive
    channel: sensor
    transition_window:
      k_pre_minutes: 60
      k_post_minutes: 120
    preprocessing:
      zscore: none
      l2_normalize: true
    aggregator: dba
    dba:
      max_iter: 30
      tol: 1.0e-5
      init: medoid
    cost_gate:
      mode: sweep
      values: [0.0, 0.1, 0.2, 0.3, 0.5]
      primary_gate: path_skew
      max_skew: 0.7
    tau_event:
      mode: half_rise_band
      threshold_fraction: 0.5
```

### 10.3 Example query-set entry

```yaml
query_sets:
  zikv_07_24:
    candidate_set: zikv_productive_07_24
    cohorts: [productive, bystander, abortive, mock]
    channel: sensor
    datasets:
      - dataset_id: "2025_07_24_SEC61"
      - dataset_id: "2025_07_24_G3BP1"
    track_paths: [A-anno, A-LC, B]
    min_pre_minutes: 240
    min_post_minutes: 360
```

## 11. Annotations CSV schema

One file per cohort at `0-select_candidates/candidates/{cohort}_{candidate_set}_annotations.csv`.
One row per `(dataset_id, fov_name, lineage_id, t)`.

| Column | Type | Notes |
|---|---|---|
| `dataset_id` | str | matches a key in `config["datasets"]` |
| `fov_name` | str | e.g. `A/2/000000` |
| `lineage_id` | int | replaces `track_id`; reconnects mother + daughter |
| `track_id` | int | original track id, retained for traceability |
| `parent_track_id` | int | from tracking; `-1` if root |
| `t` | int | absolute frame index |
| `cohort` | str | `productive` / `bystander` / `abortive` / `mock` |
| `divides` | str | `none` / `pre` / `during` / `post` |
| `infection_state` | str | `infected` / `uninfected` / blank |
| `organelle_state` | str | `remodeled` / `noremodeled` / blank |
| `cell_division_state` | str | `mitosis` / `interphase` / blank |

Stage 1 derives the per-lineage crop window and `t_key_event` from the
annotations; these are not CSV columns.

## 12. How to run end-to-end

Once all phases land:

```bash
# Stage 0
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python select_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/candidates.yaml \
    --candidate-set zikv_productive_07_24

# Stage 1 (Path B only)
cd ../1-build_template
uv run python build_template.py \
    --config ../../../configs/pseudotime/build_template.yaml \
    --template infection_nondividing_zikv

# Stage 2 (three tracks)
cd ../2-align_cells
uv run python align_anno.py        --query-set zikv_07_24
uv run python align_lc.py          --query-set zikv_07_24 --min-run 3
uv run python align_embedding.py   --query-set zikv_07_24 --template infection_nondividing_zikv

# Stage 3 (per track per organelle)
cd ../3-organelle-remodeling
for track in A-anno A-LC B; do
  uv run python readout_sec61.py  --track $track --query-set zikv_07_24
  uv run python readout_g3bp1.py  --track $track --query-set zikv_07_24
  uv run python readout_phase.py  --track $track --query-set zikv_07_24
done

# Stage 4 (comparison + robustness)
cd ../4-compare_tracks
uv run python compare_onsets.py         --query-set zikv_07_24
uv run python compare_phase_to_fluor.py --query-set zikv_07_24 --organelle sec61
uv run python warp_vs_no_warp.py        --query-set zikv_07_24
uv run python bimodality_check.py       --input compare_onsets_zikv_07_24_summary.csv
```

## 13. Limitations and pointers

This document describes pipeline operations. For the *why* — claims,
falsification protocol, decisions, alternatives considered — see
`/home/eduardo.hirata/repos/DynaCLR/.planning/dynaclr_dtw_discussion.md`.

Key limitations carried by the pipeline:

- No smFISH for per-cell viral RNA → "productive vs abortive" is
  partly a censored-data category, not pure biology.
- No entry inhibitor → phase-tracks-fluorescence (claim a') is
  correlation, not causation, even at high ρ.
- No live cell-cycle marker → the `divides` flag confounds cell-cycle,
  division timing, and survivor bias.
- DENV deferred → claim (e) is not in scope for v1.

See discussion §8 for the full limitations and future-work priority.
