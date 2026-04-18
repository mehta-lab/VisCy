# Pseudotime DAG

Pipeline for DTW-based pseudotime alignment of cell trajectories.
Each stage is a standalone Python script; outputs from one stage feed the next.

The pipeline is organised around three explicit axes:

- **Task** — the event that anchors the time-alignment (`t_key_event`). Derived from the anchor label (e.g. first `infected` frame for `infection_onset`). Planned tasks: cell division, cell death.
- **Channel** — which embedding zarr to align (`phase`, `sensor`, `organelle_sec61`, `organelle_g3bp1`).
- **Annotated candidates** — which cells to build the template from, plus per-frame labels. Expressed as an annotations CSV so users can inspect, curate, or hand-write the list.

### What `t_rel = 0` actually means (infection templates)

The `infection_state` label is derived from the **NS3 protease sensor translocating to the nucleus**: a viral-protease-cleavable reporter that gets transcribed, and once NS3 is expressed and cleaves it, the reporter moves nucleus-ward. So `infected = True` at the single-cell level means "NS3 protease is active in this cell," which is downstream of:

1. Virus entry, endocytosis, RNA release (minutes–hours earlier)
2. Initial translation of the viral polyprotein
3. ER membrane invagination to form replication organelles
4. NS3 accumulation to a level high enough to cleave the sensor
5. Sensor translocation past the detection threshold

**Implication for organelle remodeling.** Organelle changes that happen at `t_rel < 0` are not alignment artifacts or noise — they are the upstream biology of infection. For ZIKV/DENV specifically, ER (SEC61) remodeling *must* precede `t_rel = 0` because the replication organelles are what let NS3 be made in the first place. G3BP1 (stress granules) may show biphasic kinetics: mild rise while the virus suppresses SG formation, then a sharp rise once antiviral response breaks through. See Hofstadter & Cristea 2025 (Annu. Rev. Virol., DOI 10.1146/annurev-virology-092623-094221) for the review. The sensor template gives us a **reproducible, late-stage, cell-intrinsic anchor** — not the start of infection, but a reliable clock we can measure other events against.

## Directory layout

```
applications/dynaclr/
├── configs/pseudotime/
│   ├── datasets.yaml                                # shared infra: datasets + embedding patterns (loaded by every stage via --datasets)
│   ├── build_template.yaml                          # Stage 1 recipe: candidate_sets + templates
│   └── align_cells.yaml                             # Stage 2 recipe: query_sets
├── docs/DAGs/pseudotime.md                          # this file
└── scripts/pseudotime/
    ├── utils.py                                     # shared helpers (load_stage_config, read_focus_slice)
    ├── sweep_pcs.py                                 # PCA sweep — build × align × compare for multiple n_components
    ├── 0-select_candidates/
    │   ├── select_candidates.py                     # Stage 0: auto path (from annotations)
    │   ├── manual_candidates.py                     # Stage 0: manual path (hand-picked tracks)
    │   ├── inspect_candidates.py                    # Stage 0 QC: per-track-anchored image montage + QC CSV
    │   ├── refine_candidates.py                     # Stage 0.5: bootstrap-rank candidates by DTW cost, keep top-N
    │   └── candidates/                              # output: {set}_annotations.csv + _montage.png + _qc.csv + _ranking.csv
    ├── 1-build_template/
    │   ├── build_template.py                        # build DBA template (raw + PCA flavors)
    │   ├── evaluate_template.py                     # self-align (build-set only, sanity check)
    │   ├── plot_pcs.py                              # PCs over pseudotime (post-hoc PCA, self-align)
    │   ├── templates/                               # output: template_*.zarr
    │   ├── alignments/                              # output: self-align parquet
    │   └── plots/                                   # output: self-align montages + PC plots
    ├── 2-align_cells/
    │   ├── align_cells.py                           # subsequence-DTW scan template over query tracks
    │   ├── rank_by_cost.py                          # DTW cost histogram + duration scatter
    │   ├── plot_top_n_montage.py                    # montage of top/worst-N cells anchored at template t=0
    │   ├── plot_pcs_aligned.py                      # PCs vs real time: pre/aligned/post (query cells)
    │   ├── alignments/                              # output: {template}_{flavor}_on_{query_set}.parquet
    │   └── plots/                                   # output: cost diagnostics + montages + PC plots + pca_sweep_*.png/.md
    └── 3-organelle-remodeling/
        ├── plot_organelle_remodeling.py             # Stage 3a: organelle-channel remodeling vs sensor-aligned t_rel
        ├── plot_aligned_montage.py                  # Stage 3b: dual-channel (organelle + sensor) montage, orange border on remodel frames
        ├── compute_timing_metrics.py                # Stage 3c: per-cell timing scalars from embedding cosine distance (t_onset_abs, t50, t_peak, Δpeak, rise_rate_per_hour)
        ├── compute_label_timing.py                  # Stage 3d: per-cell timing scalars from LC predictions (t_first_pos, t_run_start, pos_fraction, flips)
        ├── plots/                                   # output: organelle_remodeling_*.png, aligned_montage_*.png
        ├── timing/                                  # output: compute_timing_metrics per-cell parquet + summary.md + compare_*.png/.md
        └── timing_labels/                           # output: compute_label_timing per-cell parquet + summary.md + compare_*.png/.md
```

## DAG

```
 ┌──────────────── AUTO ────────────────┐   ┌──────── MANUAL (debug/test) ────────┐
 │                                      │   │                                     │
 │ [annotations.csv]  [embedding .zarr] │   │ user-observed phenotypes            │
 │         │                    │       │   │         │                           │
 │         └──────────┬─────────┘       │   │         ▼                           │
 │                    ▼                 │   │ manual_candidates.py                │
 │ select_candidates.py --candidate-set │   │ (hand-picked track specs with       │
 │ (filter tracks, emit per-frame       │   │  [t_on, t_off] label intervals)     │
 │  labels over the crop window)        │   │                                     │
 └──────────────┬───────────────────────┘   └──────────────┬──────────────────────┘
                │                                          │
                └───────────────────┬──────────────────────┘
                                    ▼
                candidates/{candidate_set}_annotations.csv
                (dataset_id, fov_name, track_id, t,
                 infection_state, organelle_state, cell_division_state)
                one row per (cell, frame) over the crop window
                                    │
                                    ▼
        1-build_template/build_template.py --template {name}
        (join CSV with embedding zarr on (dataset_id, fov_name, track_id, t),
         derive per-cell crop window and t_key_event from the annotations,
         apply optional per-experiment z-score + L2-normalize,
         run DTW-DBA (cosine metric) to build TWO template flavors in parallel:
           raw/  — template in 768-D embedding space
           pca/  — template after PCA to N components
         save template zarr)
                   │
                   ▼
        templates/template_{name}.zarr
        ├── raw/template              (T, 768)  DBA template, 768-D
        ├── raw/time_calibration      (T,)      minutes relative to t_key_event
        ├── raw/template_labels/{col} (T,)      per-position label fractions
        ├── pca/template              (T, N)    DBA template, PCA-reduced
        ├── pca/time_calibration      (T,)
        ├── pca/template_labels/{col} (T,)
        ├── pca/components            (N, D)    build-time PCA model
        ├── pca/mean                  (D,)
        ├── pca/explained_variance_ratio (N,)
        ├── zscore_params/{ds_id}/*   (D,)      only if zscore=per_dataset
        ├── t_key_event               (N_cells,)  per-cell anchor frame
        └── attrs: template_cell_ids, l2_normalize, metric, aggregator
                   │
                   ▼
        1-build_template/evaluate_template.py --template {name} --flavor {raw|pca}
        (self-consistency check — re-align the same cells used to build
         the template. Not subsequence DTW; closed-endpoint on both sides.)
                   │
                   ├──► 1-build_template/alignments/template_alignments_{name}_{flavor}.parquet
                   └──► 1-build_template/plots/realtime_montage_{name}_{flavor}_{channel}.png
                        plots/pcs_over_pseudotime_{name}_{flavor}.png
                        (via plot_pcs.py — diagnostic post-hoc PCA on build-set cells)
                   │
                   ▼
──── Stage 2: scan template across query tracks ────────────────────────────────
                   │
        2-align_cells/align_cells.py \
            --template {name} --flavor {raw|pca} --query-set {qset}
        (for every query cell track — NOT in the build set — run
         SUBSEQUENCE DTW: template (length T) must match fully, query
         (length Q ≥ T) endpoints float. Scans the template across the
         query's time axis and picks the window with minimum cost.
         Preprocessing: apply the build-time zscore + PCA + L2 from the
         template zarr — never refit at alignment time.)
                   │
                   ▼
        2-align_cells/alignments/{template}_{flavor}_on_{qset}.parquet
        (one row per query (dataset_id, fov_name, track_id, t):
           pseudotime ∈ [0, 1]            template position from warp path
           alignment_region                "pre" | "aligned" | "post"
           estimated_t_rel_minutes         time_calibration[template_pos]
                                           NaN outside alignment_region == "aligned"
           dtw_cost                        per-track total cost (repeated on rows)
           length_normalized_cost          dtw_cost / len(warp_path)
           match_q_start, match_q_end      absolute query frames bounding the match
           match_duration_minutes          (q_end - q_start) * frame_interval_minutes
        )
                   │
                   ├──► 2-align_cells/rank_by_cost.py --template {name} --flavor {..} --query-set {..}
                   │     (histogram of length_normalized_cost,
                   │      scatter match_duration_minutes vs cost.
                   │      Use to pick a cost cutoff before montage.)
                   │     plots/cost_ranking_{template}_{flavor}_{qset}.png
                   │
                   ├──► 2-align_cells/plot_top_n_montage.py \
                   │        --template {..} --flavor {..} --query-set {..} \
                   │        --top-n 30 --worst-n 10
                   │     (rows = query cells sorted by length_normalized_cost
                   │      ascending; top-N at top, worst-N at bottom for
                   │      contrast. Columns = real time anchored at each
                   │      cell's warped t=0, i.e. the frame where
                   │      estimated_t_rel_minutes crosses 0. Red border at t=0.
                   │      Frames in "pre"/"post" are shown faded.)
                   │     plots/realtime_montage_{template}_{flavor}_{qset}.png
                   │
                   └──► 2-align_cells/plot_pcs_aligned.py \
                            --template {..} --flavor {..} --query-set {..} \
                            --top-n 50
                         (fit diagnostic post-hoc PCA on aligned-region
                          frames of top-N query cells. Plot PCs vs minutes:
                            left  = unaligned: PC vs (t - match_q_start) * frame_interval
                                    — each cell anchored at its own match start;
                                      traces scatter in shape.
                            right = aligned:   PC vs estimated_t_rel_minutes;
                                    traces collapse onto a shared curve.
                          Bottom row: query-truth label fraction (solid red) on
                          BOTH axes + template-build-cells fraction (grey dashed,
                          secondary). A sharper right-panel truth curve = real
                          alignment, not just embedding-shape collapse.)
                         plots/pcs_over_pseudotime_{template}_{flavor}_{qset}.png

──── Stage 3: organelle remodeling vs sensor-aligned t_rel ───────────────────
      (consumes Stage 2's alignment parquet; no new DTW)

        3-organelle-remodeling/plot_organelle_remodeling.py \
            --template {..} --flavor {..} --query-set {..} \
            --organelle-channel {organelle_sec61 | organelle_g3bp1 | phase}
        (REUSE the sensor alignment parquet as a timing skeleton and
         project organelle-channel embeddings onto the sensor-derived
         t_rel_minutes. No new DTW. For each (dataset, fov, track, t)
         in the sensor parquet, look up the organelle embedding from
         its zarr, compute distance-from-pre-baseline (cosine,
         per-cell), and plot vs t_rel.
         Three rows: (A) per-cell organelle distance traces,
         (B) post-hoc PC1/PC2 of organelle embeddings over t_rel,
         (C) ground-truth organelle_state fraction (when available).
         Report remodeling onset offset in title: "SEC61 remodels at
         t_rel = +X min".)
        plots/organelle_remodeling_{template}_{flavor}_{organelle_channel}_{qset}.png
```

## How to run

Run each script from its own directory — scripts resolve output paths relative to their own location.

### Stage 0 — Select candidates

Stage 0 emits a single artifact: `{candidate_set}_annotations.csv`, one row per `(dataset_id, fov_name, track_id, t)` with per-frame label columns. Two independent scripts produce this file — downstream consumers treat the outputs identically.

**Auto — `select_candidates.py`** (from annotations)

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python select_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/build_template.yaml \
    --candidate-set infection_transitioning_nondiv
```

Filters tracks per `config["candidate_sets"][NAME]["filter"]` (anchor label, anchor_positive, min_pre/post_minutes, crop_window_minutes), then expands each selected track into per-frame rows over its crop window, copying real annotation labels onto each row. Writes `candidates/{candidate_set}_annotations.csv`.

**Manual — `manual_candidates.py`** (user-written, for debugging / hand-curated cells)

Each track spec is a `{t_before, t_after, labels: {label_col: [[t_on, t_off], ...]}}` entry in a Python dict. For every frame in `[t_before, t_after]`, the script emits the positive label if that frame falls inside any interval, otherwise the negative label. Columns with no intervals are left blank.

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
python manual_candidates.py
```

This path shares no code with `select_candidates.py`; the CSV schema is the only contract.

**Inspect — `inspect_candidates.py`** (per-track-anchored QC montage + stats CSV)

Reads the candidate annotations CSV and renders a montage where every row is anchored at that cell's `t_key_event` (red border at offset 0), so scanning down rows makes bad candidates obvious. Also writes a sidecar `{candidate_set}_qc.csv` with per-track stats (n_frames, pre_frames, post_frames, fov) for non-visual QC.

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python inspect_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/build_template.yaml \
    --candidate-set infection_transitioning_nondiv
```

### Stage 0.5 — Refine candidates (bootstrap)

`refine_candidates.py` handles the common case of noisy annotations producing a broad candidate set that contains some bad (mislabeled / wrong-cell) tracks. Two-pass filter:

1. **Strict headroom inside the crop**: drops tracks whose `t_key_event` is too close to the window start/end (the "annotation starts at transition" cases where the cell has no genuine uninfected baseline).
2. **Bootstrap self-alignment**: builds an initial DBA template from the surviving tracks, self-aligns each cell against it, ranks by `length_normalized_cost`, and keeps the top-N.

Produces a **refined candidate-set CSV** that the final template build consumes. Cells surviving both filters are simultaneously well-annotated *and* consistent with the population consensus trajectory.

```bash
cd applications/dynaclr/scripts/pseudotime/0-select_candidates
uv run python refine_candidates.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/build_template.yaml \
    --candidate-set infection_transitioning_nondiv_top20
```

The refined set is declared in `build_template.yaml` as a candidate entry with `refine_from: <source-set>`, `min_pre_frames`, `min_post_frames`, and `top_n_by_cost`. See "Example refined-candidate entry" below.

Outputs: `candidates/{refined_set}_annotations.csv`, `{refined_set}_ranking.csv` (full ranking with kept/rejected flags). Run `inspect_candidates.py` on the refined set afterwards to visually QC the surviving cells.

### Stage 1 — Build template

```bash
cd applications/dynaclr/scripts/pseudotime/1-build_template
uv run python build_template.py --config ../../../configs/pseudotime/multi_template.yaml --template infection_nondividing_sensor
```

Outputs `templates/template_{name}.zarr` with **both flavors** (raw and PCA) built from the same input cells. The downstream picks which flavor to use at alignment time.

**What the builder does**

1. Reads `candidates/{candidate_set}_annotations.csv`.
2. Groups by `(dataset_id, fov_name, track_id)`; pulls embedding rows from the channel's zarr.
3. Derives each cell's crop window from `[min(t), max(t)]` and `t_key_event` from the first frame where the anchor label is positive.
4. Applies optional per-dataset z-score.
5. Builds **two templates from the same cells**, in parallel:
   - `raw/` — optional L2-normalize, then DTW-DBA with cosine metric.
   - `pca/` — fits PCA (`n_components`), transforms, optional L2, then DTW-DBA.
6. Saves the combined zarr.

#### 1a — Self-consistency check (`evaluate_template.py`, `plot_pcs.py`)

Both scripts live under `1-build_template/` and operate on the **build set only** — they re-align the cells that built the template onto itself. They are **not** subsequence DTW (template and cell share endpoints). Treat outputs as a sanity check, not as evaluation of generalization.

```bash
cd applications/dynaclr/scripts/pseudotime/1-build_template
uv run python evaluate_template.py --config ../../../configs/pseudotime/multi_template.yaml --template infection_nondividing_sensor --flavor raw
uv run python plot_pcs.py --config ../../../configs/pseudotime/multi_template.yaml --template infection_nondividing_sensor --flavor raw --n-pcs 5
```

Outputs: `alignments/template_alignments_{name}_{flavor}.parquet`, `plots/realtime_montage_{name}_{flavor}_*.png`, `plots/pcs_over_pseudotime_{name}_{flavor}.png`.

Montage optional args: `--pre-minutes 180`, `--post-minutes 420`, `--crop-half 80`, `--n-cells 50` (sorted by DTW cost).
PC plot optional args: `--n-pcs 5`, `--n-bins 20`.

### Stage 2 — Align query cells to the template (subsequence DTW)

This stage takes the template built in Stage 1 and scans it across **new** cell tracks from any dataset (not necessarily the ones used to build the template). Subsequence DTW finds, per query track, the time window where the template best matches — i.e. the time when that cell traverses the same canonical event.

The template's `time_calibration` provides the real-time clock. Once a cell's best-matching window is found, each frame inside the window is mapped to template-relative minutes; frames before/after stay untouched but are labeled `"pre"` / `"post"` for downstream pre-vs-post analysis.

**All alignment, evaluation, and plotting for Stage 2 live under `2-align_cells/` — same convention as Stage 1.**

#### 2a — Align (`align_cells.py`)

```bash
cd applications/dynaclr/scripts/pseudotime/2-align_cells
uv run python align_cells.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor \
    --flavor raw \
    --query-set sensor_all_07_24 \
    --min-match-minutes 360 --max-skew 0.7
```

What it does:

1. Loads `templates/template_{name}.zarr` and reconstructs a `TemplateResult` for the chosen flavor. **Reuses the build-time zscore + PCA + L2 stored in the zarr — never refits at alignment time.**
2. Loads the query set's embedding zarr(s), restricted to the template's channel.
3. For each query track, calls `dtw_align_tracks(..., subsequence=True, frame_interval_minutes=..., max_psi_minutes=...)` so psi is frame-rate invariant — same wall-clock freedom on 10 min/frame and 30 min/frame tracks. The template (length T) must match fully while the query (length Q ≥ T) floats; returns a warp path, best-match window `[q_start, q_end]`, cost, and `path_skew`.
4. Applies guards (see "Guards and frame-rate invariance" below) and writes one row per `(dataset_id, fov_name, track_id, t)`.

#### 2b — Rank cells by DTW cost (`rank_by_cost.py`)

Diagnostic before rendering montages. Length-normalized cost (`dtw_cost / len(path)`) is the correct rank for subsequence DTW because matched windows have variable length.

```bash
uv run python rank_by_cost.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor --flavor raw --query-set sensor_all_07_24
```

Outputs `plots/cost_ranking_{template}_{flavor}_{qset}.png` (histogram + duration-vs-cost scatter).

#### 2c — Top-N realtime montage (`plot_top_n_montage.py`)

```bash
uv run python plot_top_n_montage.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor --flavor raw --query-set sensor_all_07_24 \
    --top-n 30 --worst-n 10
```

Rows = query cells ranked by length-normalized cost; columns = real time anchored at each cell's warped `t=0` (the frame where `estimated_t_rel_minutes` crosses 0). Top-N at top, worst-N at bottom for contrast. Pre/post frames are shown faded. Red border at `t=0`.

Outputs `plots/realtime_montage_{template}_{flavor}_{qset}.png`.

#### 2d — PCs over real time, pre / aligned / post (`plot_pcs_aligned.py`)

Fits a diagnostic post-hoc PCA on the **aligned-region** frames of the top-N query cells, then projects pre / aligned / post frames through the same basis so trajectories extend on both sides of the event window. Plots top PCs vs minutes:

- **Left (unaligned):** PC vs `(t - match_q_start) * frame_interval_minutes` — anchored at each cell's own match start.
- **Right (aligned):** PC vs `estimated_t_rel_minutes`; pre/post frames are extrapolated off either end using `time_calibration[0]` / `time_calibration[-1]` as anchors.
- Points are coloured by `alignment_region` (grey = pre, blue = aligned, red = post); legend is written to a separate `*.legend.png` so the main grid isn't squeezed.

`--exclude-template-cells` drops query cells that match the template build-set (honest generalization reporting). Without it, build-set cells will always score best since they're matching themselves.

The bottom row carries **two** curves so alignment quality can be judged honestly:

- **Solid red — query truth**: fraction of query cells where `obs[truth_column] == truth_positive` at each bin. Present on BOTH axes. Left bins by `(t - match_q_start) * frame_interval`; right bins by `estimated_t_rel_minutes` restricted to `alignment_region == "aligned"`. A sharper right-panel curve than left = DTW is genuinely moving the annotated transition into alignment with template t=0.
- **Dashed grey — template fraction**: label fractions stored in the template zarr (`raw/template_labels/{col}`). This is a property of the build-set cells only, not the query. Included as a secondary reference; do NOT treat it as evidence of query-side alignment.

`--truth-column` / `--truth-positive` pick the label. Use human `infection_state` on 07_22/07_24 when available; `predicted_infection_state` on 08_26/01_28 (or 07_22 where human labels are sparse).

```bash
uv run python plot_pcs_aligned.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor --flavor raw --query-set sensor_all_07_24 \
    --top-n 50 --n-pcs 5 --exclude-template-cells \
    --truth-column infection_state --truth-positive infected
```

Outputs `plots/pcs_over_pseudotime_{template}_{flavor}_{qset}.png` + `.legend.png`.

### Stage 3 — Organelle remodeling vs sensor-aligned t_rel (`3-organelle-remodeling/plot_organelle_remodeling.py`)

Stage 3 is a **consumer** of Stage 2's alignment parquet. It runs no new DTW — it joins the sensor-channel alignment parquet with an organelle-channel embedding zarr and plots organelle dynamics on the sensor-derived time axis. Lives in its own `3-organelle-remodeling/` directory so the scope (read Stage 2 artifacts, write new plots) is obvious.

**Scientific question.** The sensor channel tells us *when* the NS3 protease sensor translocates to the nucleus (via template alignment; see "what t=0 actually means" note at the top of the doc). Do the organelle channels (SEC61 ER, G3BP1 stress granules) show coordinated remodeling around that same t=0, and at what offset — before, after, or simultaneous?

**Design decision: reuse the sensor alignment as a timing skeleton** (option a, not build a separate organelle template). Rationale: the claim is "organelle remodeling *relative to* infection onset," which requires a single shared clock. A sensor-derived t=0 is meaningful; a SEC61-derived t=0 would be tautological.

**Inputs**

- Sensor alignment parquet: `infection_nondividing_sensor_{raw|pca}_on_{qset}.parquet`.
- One organelle embedding zarr resolved via `datasets.yaml.embeddings.{organelle_channel}`. Supported channels: `organelle_sec61`, `organelle_g3bp1`, `phase`.

**Organelle channels live in disjoint FOV groups.** Each fluorophore was only acquired in its dedicated wells — on 07_24, SEC61 is only in A/1 + A/2 and G3BP1 only in C/1 + C/2. A sensor-query row from `2025_07_24_G3BP1` therefore has **no** SEC61 embedding and vice versa; those rows are dropped at join time. This is not a bug — it's the microscopy design. The per-organelle plot effectively restricts to the subset of sensor-aligned cells that were imaged in that organelle's wells.

**Pipeline**

1. Join the sensor parquet with the organelle embedding on `(dataset_id, fov_name, track_id, t)`.
2. Compute **distance-from-baseline** per frame. Baseline = mean organelle embedding across `alignment_region == "pre"` frames per cell. Per-frame scalar = cosine distance from that per-cell baseline.
3. Render three panels stacked:
   - **Panel A**: per-cell organelle-distance traces vs `estimated_t_rel_minutes`, colored by pre/aligned/post. Binned median + IQR overlay.
   - **Panel B**: post-hoc PC1/PC2 of the organelle embeddings (fitted on aligned-region frames, projected onto pre + post) vs `estimated_t_rel_minutes`. Mirror of `plot_pcs_aligned.py` but in organelle-embedding space.
   - **Panel C**: `organelle_state` fraction vs `estimated_t_rel_minutes` (when the query obs has the column). Same truth-binning convention as Stage 2d.
4. Compute the **remodeling onset offset**: the `t_rel_minutes` where Panel A's binned median crosses a threshold (default: 2σ above the pre-baseline distance distribution). Report in the plot title — e.g. `"SEC61 remodels at t_rel = +60 min"`.

**Preprocessing**

Organelle embeddings are used as-is. `--flavor` only selects which sensor alignment parquet to join on (different warp paths yield different t_rel mappings); the organelle distance metric (and Panel B's post-hoc PCA) are computed per-run on the joined organelle embeddings.

**Template cells are not excluded by default.** The sensor template was built on sensor embeddings — organelle embeddings from the same cells aren't "self-alignment" in any meaningful sense. Keep the full top-N by sensor DTW cost.

**How to run (Phase 1 — Panel A)**

```bash
cd applications/dynaclr/scripts/pseudotime/3-organelle-remodeling
uv run python plot_organelle_remodeling.py \
    --datasets ../../../configs/pseudotime/datasets.yaml \
    --config ../../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor --flavor raw \
    --query-set sensor_all_07_24 \
    --organelle-channel organelle_sec61 \
    --top-n 30
```

Outputs `plots/organelle_remodeling_{template}_{flavor}_{organelle_channel}_{qset}.png`.

**Delivery plan**

1. ✅ Panel A only, SEC61 + G3BP1 on `sensor_all_07_24` — sanity-check the join + baseline subtraction.
2. Add Panels B + C (organelle-space PCA + `organelle_state` truth curve; CLI grows `--n-pcs`, `--truth-column`, `--truth-positive`).
3. Sweep across organelle channels × query sets (07_24 + 07_22 + 01_28; 08_26 missing labels).
4. Replicate check: does the remodeling offset hold across datasets? Emit a summary table analogous to the cross-dataset sensor results above.

**Phase 1 results (Apr 2026, infection_nondividing_sensor, raw flavor, 07_24, top-30 by sensor cost, template cells NOT excluded)**

Two distinct organelle kinetics visible in cosine distance from per-cell pre-baseline:

| Organelle | Cells kept | Pre (t≈-400) | Onset of divergence | At sensor t=0 | Post |
|---|---:|---:|---:|---:|---:|
| **SEC61 (ER)** | 15 / 30 (A/2 only) | ~0.025 | **~-250 min** — gradual, monotonic | ~0.09 | ~0.24, still rising |
| **G3BP1 (stress granules)** | 15 / 30 (C/2 only) | ~0.03 | biphasic: gentle rise from ~-300 min, plateau around t=0, **sharp kink at ~+200 min** | ~0.10 | ~0.28, plateaus ~0.28 by t≈+400 |

**Two qualitatively different kinetics.**

- **SEC61 (ER) — steady, one-way remodeling.** The cosine distance from baseline rises monotonically from ~-250 min through the entire post window, with no return toward baseline. This matches the biology of ER-derived replication organelles: once the ER is restructured into invagination-type ROs for flavivirus replication, it stays restructured for as long as the virus is replicating. We don't expect the ER to "snap back" during the observation window — SEC61 remodeling is a persistent, one-way structural change upstream of the NS3 sensor signal.
- **G3BP1 (stress granules) — transient, comes-and-goes.** The distance curve shows small, repeated up-and-down excursions through the pre + early-aligned region (gentle rises, mini-plateaus), then a sharp rise around t≈+200 min, and finally a plateau rather than continued growth. This matches the biology of stress granules: they are phase-separated membraneless condensates that **assemble and disassemble** on minute timescales. Flavivirus NS3 and capsid proteins actively suppress SG formation early (so translation of viral proteins can continue) — hence the low, flickering pre-phase — and then once the antiviral response overwhelms that suppression, SGs form persistently and the signal jumps. The plateau (not continued rise) is expected: SG mass is bounded by the available G3BP1 pool, unlike ER membrane area.

The **SEC61 steady climb vs G3BP1 transient-then-step** contrast is exactly the kind of temporal signature the pipeline was built to surface. Same sensor clock, different organelle grammars.

Per Hofstadter & Cristea 2025 (Annu. Rev. Virol., DOI 10.1146/annurev-virology-092623-094221): "Flaviviruses (including ZIKV) actively suppress stress granule formation to maintain translation of viral proteins" — consistent with the suppressed early G3BP1 signal and the late breakthrough. ER invagination happening before the sensor readout is consistent with "ZIKV/DENV form replication organelles from ER membranes" being an upstream prerequisite for NS3 expression.

### Stage 3c — Per-cell embedding-timing metrics (`compute_timing_metrics.py`)

Reduces each cell's per-frame cosine-distance-from-pre-baseline curve to five scalars, then pools into a per-organelle distribution so distributions (not cells, since FOVs are disjoint) can be compared across organelles.

**Per-cell scalars (computed on the aligned region only, with interior restriction):**

| metric | definition | why |
|---|---|---|
| `t_onset_abs` | first `t_rel` where `distance − pre_median` crosses `+0.10` (cosine units) | SNR-robust: cells with small Δpeak can't fake an early onset by their noise floor crossing a normalized fraction |
| `t50` | first `t_rel` where distance crosses `pre_median + 0.5 × Δpeak`, last 2 aligned frames excluded | half-rise timing, interior-restricted to dodge DTW endpoint pile-up |
| `t_peak` | `argmax` of distance over interior aligned region | time of maximum embedding divergence |
| `delta_peak` | `max(aligned distance) − median(pre distance)` | amplitude of remodeling in cosine units |
| `rise_rate_per_hour` | OLS slope of distance vs `t_rel` over aligned region × 60 | per-cell aggregate speed of change |

**Outputs:** `timing/{stem}_per_cell.parquet` + `timing/{stem}_summary.md` (per-well medians + pooled bootstrap CI). Run `compute_timing_metrics.py compare` on multiple per-cell parquets to emit strip plots + pairwise rank-sum tests (writes `timing/{out_stem}.png/.md`).

### Stage 3d — Per-cell label-timing metrics (`compute_label_timing.py`)

Parallel to Stage 3c but uses **linear classifier predictions** (`predicted_{state}`, the dense LC output per frame) instead of embedding distance. Supervised projection → collapses off-axis embedding noise (cell cycle, focus, photobleaching) that cosine distance would catch.

**Per-cell scalars on the binarized predicted-label trajectory (1 = positive):**

| metric | definition | region |
|---|---|---|
| `t_first_pos` | first `t_rel` with a positive prediction | whole track |
| `t_run_start` | first `t_rel` entering a run of ≥ `min_run` (default 3) consecutive positives | whole track |
| `t_run_end` | last `t_rel` in the run | whole track |
| `pos_duration` | `t_run_end − t_run_start` | whole track |
| `pos_fraction` | fraction of aligned frames predicted positive | **aligned only** |
| `flips` | number of 0↔1 transitions across the track | whole track |

**Aligned-vs-whole-track asymmetry is intentional** — `pos_fraction` is the aligned-period fingerprint (density of the positive state during DTW-mappable frames); the timing scalars run across the whole track so "LC fires before sensor translocation" can be measured as a negative `t_first_pos`.

**Example: SEC61 vs G3BP1 `predicted_organelle_state==remodel` on `sensor_all_07_24` (n=15 each)**

| metric | SEC61 median [CI] | G3BP1 median [CI] | p (MW-U) |
|---|---|---|---|
| `t_first_pos` (min) | **-207 [-354, -158]** | +221 [+198, +341] | **4.7e-4** |
| `t_run_start` (min) | **-72 [-170, +3]** | +221 [+198, +221] | 0.048 |
| `pos_fraction` | **0.81 [0.52, 0.93]** | 0.00 [0.00, 0.03] | **1.6e-4** |
| `flips` | 3 [3, 6] | 1 [0, 4] | 0.028 |

Signal that was suggestive but not significant in Stage 3c (embedding-timing ΔT ≈ 120 min, p ≈ 0.4) becomes sharp in Stage 3d because the LC was trained on the `remodel` label directly. Biologically consistent with Hofstadter & Cristea 2025: SEC61 (ER) remodels early for replication-organelle formation; G3BP1 (stress granules) is actively suppressed by flavivirus NS3/capsid during infection.

**Caveat.** A near-zero G3BP1 `pos_fraction` could be real suppression or LC blind spot (if trained on SEC61-dominated data). Before interpreting as biology, verify the LC's training set covered the G3BP1 channel and morphology.

### Delivery plan for modular multi-dataset + virus comparison (next)

Current Stage 3c/3d take one `--query-set` (one alignment parquet → one population). Next iteration moves the pooling to be **dataset-group-aware** so the same templates can be evaluated across:

1. **A configurable dataset pool** — pass a list of datasets to pool (all will have LC predictions; only some have human annotations). The script should error softly when a requested label column is missing from a dataset rather than silently NaN-ing those rows.
2. **Virus-stratified comparison** — ZIKV vs DENV. Cells from `2025_01_28_ZIKV_DENV` carry a `perturbation` column (`infected`, `mock`) plus a `virus` column; per-organelle distributions should split on `virus` and the compare step should render side-by-side strips.
3. **Artifact caching** — because each stage writes its own parquet, re-running only the comparison step on different pool/virus filters should be cheap (no re-computation of per-cell metrics). Confirm this already holds with the current output layout.

### Guards and frame-rate invariance

Subsequence DTW with generous psi relaxation can collapse the template onto a single query frame (near-zero cost, no biological meaning). Four guards prevent and surface this:

| guard | CLI flag | default | what it rejects |
|---|---|---|---|
| Non-finite cost | (always on) | — | tracks too short for the solver to find any valid path |
| Minimum match window | `--min-match-minutes` or `--min-match-ratio` | ratio 0.5 | template compressed onto a tiny real-time window |
| Path skewness | `--max-skew` | 0.8 | L-shaped / non-diagonal warps that slip past psi |
| Pre/post headroom | query-set YAML `min_pre_minutes` / `min_post_minutes` | 0 | cells without real footage on either side of the event |

**Minute-based guards supersede frame-based ones when both are set.** When query datasets have heterogeneous frame intervals (e.g. 07_22 at 10 min/frame vs 07_24 at 30 min/frame), use `--min-match-minutes` and `--max-psi-minutes` instead of `--min-match-ratio` and the implicit `t_template // 2` psi: minute-based thresholds apply the same wall-clock requirement regardless of frame rate.

`--max-psi-minutes` defaults to **half the template duration**, read from `template_duration_minutes` in the template zarr attrs. Per-track psi is then `round(max_psi_minutes / dataset_frame_interval_minutes)`.

### PCA sweep — finding the sweet spot (`sweep_pcs.py`)

Sweeps `n_components` for one template, rebuilding the template at each value and re-running Stage 2a against a fixed query set. Produces a 2×2 summary plot + a markdown table sidecar:

- Cost distribution vs n_components (boxplot)
- Tracks kept vs n_components
- Spearman rank correlation to the RAW 768-D reference (the sweet-spot indicator)
- PCA explained variance vs n_components

```bash
cd applications/dynaclr/scripts/pseudotime
uv run python sweep_pcs.py \
    --datasets ../../configs/pseudotime/datasets.yaml \
    --build-config ../../configs/pseudotime/build_template.yaml \
    --align-config ../../configs/pseudotime/align_cells.yaml \
    --template infection_nondividing_sensor \
    --query-set sensor_all_07_24 \
    --n-components 5,10,20,30,50 \
    --min-match-ratio 0.7 --max-skew 0.7
```

Outputs `plots/pca_sweep_{template}_{qset}.png` and `.md`.

## Key config fields

Three YAMLs split across `configs/pseudotime/`, each loaded alongside `datasets.yaml` via the `--datasets` + `--config` CLI pair:

| File | Contains | Used by |
|---|---|---|
| `datasets.yaml` | `data_zarr`, `embeddings` glob patterns, `datasets` list (pred_dir, annotations_path, fov_pattern, `frame_interval_minutes`) | every stage (passed via `--datasets`) |
| `build_template.yaml` | `candidate_sets.{name}`, `templates.{name}` | Stage 0 (auto), Stage 1 |
| `align_cells.yaml` | `query_sets.{name}` | Stage 2 |

Field reference:

| Field | Purpose |
|---|---|
| `data_zarr` (top-level) | source image zarr for cell crop montages (Stage 0 inspect, Stage 2c) |
| `embeddings.{channel}` | glob pattern → zarr per channel |
| `datasets[].frame_interval_minutes` | real-time spacing between adjacent `t` values; used for minute→frame conversions |
| `datasets[].fov_pattern` | substring selecting FOVs from that dataset's zarr (e.g. `A/2`) |
| `candidate_sets.{name}` | anchor label + minute-based filters + `crop_window_minutes` + `max_tracks` |
| `templates.{name}` | candidate_set reference, channel, anchor label, preprocessing, DBA params |
| `query_sets.{name}` | channel (must match template), datasets, `min_pre_minutes` / `min_post_minutes`, optional `track_filter` |

### Example candidate-set entry

```yaml
candidate_sets:
  infection_transitioning_nondiv:
    datasets: ["2025_07_24_SEC61", "2025_07_24_G3BP1"]
    filter:
      anchor_label: infection_state
      anchor_positive: infected
      anchor_negative: uninfected
      min_pre_minutes: 120            # need ~4 frames before onset (at 30 min/frame)
      min_post_minutes: 180
      crop_window_minutes: 240        # ± half-window around the onset
    max_tracks: 50                    # cap for speed
```

### Example template entry

```yaml
templates:
  infection_nondividing_sensor:
    candidate_set: infection_transitioning_nondiv   # → candidates/{..}_annotations.csv
    channel: sensor                                  # key in datasets.yaml embeddings:
    anchor_label: infection_state                    # determines t_key_event
    anchor_positive: infected

    preprocessing:
      zscore: none                                   # {none, per_dataset}
      pca:
        n_components: 20                             # pca/ flavor; raw/ always built. Use sweep_pcs.py to pick.
      l2_normalize: true                             # applied last — on both flavors

    aggregator: dba                                  # {dba, median}
    dba:
      max_iter: 30
      tol: 1.0e-5
      init: medoid
    metric: cosine                                   # {cosine, euclidean}
```

`track_filter`, `min_track_minutes`, `crop_window_minutes`, per-template `datasets` are all **gone** — they're baked into the annotations CSV by Stage 0.

### Example query-set entry (Stage 2)

Query sets describe which cells to **scan the template over** — typically cells from other datasets, or cells you deliberately withheld from the build set.

```yaml
query_sets:
  sensor_all_07_24:
    channel: sensor                                  # must match templates.{name}.channel
    datasets:
      - dataset_id: "2025_07_24_SEC61"
      - dataset_id: "2025_07_24_G3BP1"
    # Pre/post headroom (minutes, per-cell). Pass 1 (_load_query_embeddings)
    # requires the track to hold template + pre + post frames; pass 2 (after DTW)
    # requires the matched window to sit with real footage on both sides.
    min_pre_minutes: 120
    min_post_minutes: 180
    min_track_minutes: 120                           # floor; the template+headroom calculation takes the max
    track_filter: {}                                 # optional obs-column equality filters
```

Unlike `candidate_sets`, query sets do **not** require an `anchor_label` — we are *estimating* `t_key_event` for each query cell via DTW, not reading it off annotations.

## Annotations CSV schema

One file per candidate set, at `0-select_candidates/candidates/{candidate_set}_annotations.csv`. One row per `(dataset_id, fov_name, track_id, t)` over the hand-picked or auto-selected crop window.

| column | type | notes |
|---|---|---|
| `dataset_id` | str | matches a key in `config["datasets"]` |
| `fov_name` | str | e.g. `A/2/000000` |
| `track_id` | int | |
| `t` | int | absolute frame index |
| `infection_state` | str | `"infected"` / `"uninfected"` / blank |
| `organelle_state` | str | `"remodeled"` / `"noremodeled"` / blank |
| `cell_division_state` | str | `"mitosis"` / `"interphase"` / blank |

Positive/negative values per label are defined in `manual_candidates.py::LABEL_VALUES`. Additional label columns can be added by extending that dict.

### Derived at read time (not stored in the CSV)

Stage 1 computes the following from the annotations CSV; they are **not** CSV columns:

- **Crop window** per cell: `[t_before, t_after] = [min(t), max(t)]` across that cell's rows.
- **`t_key_event`** per cell: the first `t` where the anchor label (configured per template) takes its positive value.

## Template zarr contents

Every build produces **both flavors** from the same input cells.

| Path | Type | Description |
|---|---|---|
| `raw/template` | (T, D) array | DBA template in raw embedding space (D = 768 after optional z-score + L2). |
| `raw/time_calibration` | (T,) array | mean `t_relative_minutes` at each raw-template position |
| `raw/template_labels/{col}` | (T,) array | per-position label fraction for each label column |
| `pca/template` | (T, N) array | DBA template in PCA-reduced space |
| `pca/time_calibration` | (T,) array | analogous, warping paths differ |
| `pca/template_labels/{col}` | (T,) array | analogous |
| `pca/components` | (N, D) array | build-time PCA components (downstream alignment must apply these) |
| `pca/mean` | (D,) array | build-time PCA mean |
| `pca/explained_variance_ratio` | (N,) array | fraction of variance per component |
| `zscore_params/{ds_id}/mean` | (D,) array | only present when `zscore=per_dataset`. Shared across flavors. |
| `zscore_params/{ds_id}/std` | (D,) array | only present when `zscore=per_dataset` |
| `t_key_event` | (N_cells,) array | per-cell anchor frame |
| attrs `template_cell_ids` | list | `[dataset_id, fov_name, track_id]` per input cell |
| attrs `l2_normalize` | bool | whether L2 was applied before DTW |
| attrs `metric` | str | `"cosine"` — downstream alignment must match |
| attrs `aggregator` | str | `"dba"` or `"median"` |
| attrs `template_duration_minutes` | float | `time_calibration[-1] - time_calibration[0]`; used by Stage 2 to default `max_psi_minutes = template_duration_minutes / 2` |
| attrs `build_frame_intervals_minutes` | dict | `{dataset_id: frame_interval_minutes}` — records the real-time scale of each build dataset |

The `pca/` entries are the **build-time** PCA that maps raw embeddings into the `pca/` flavor's feature space. This is distinct from the Stage 2d diagnostic PCA (`plot_pcs_aligned.py`), which is fit post-hoc on the aligned-region frames of query cells for plotting only and is not stored in the template zarr.

## Stage 2 alignment parquet schema

One row per `(dataset_id, fov_name, track_id, t)`. Per-track columns (`dtw_cost`, `length_normalized_cost`, `path_skew`, `match_q_start`, `match_q_end`, `match_duration_minutes`) are repeated on every frame so downstream scripts can filter rows without a separate join.

| column | type | per-track? | notes |
|---|---|---|---|
| `dataset_id`, `fov_name`, `track_id`, `t` | ids | per-frame | identifiers |
| `pseudotime` | float ∈ [0, 1] | per-frame | warp-path template position, unit-free |
| `alignment_region` | str | per-frame | `"pre"` / `"aligned"` / `"post"` |
| `estimated_t_rel_minutes` | float | per-frame | `time_calibration[template_pos]`; `NaN` outside `aligned` (see `plot_pcs_aligned.py` for the extrapolation it uses for plotting only) |
| `dtw_cost` | float | yes | raw DTW cost at the best-path endpoint |
| `length_normalized_cost` | float | yes | `dtw_cost / len(warp_path)` — the correct ranking signal |
| `path_skew` | float ∈ [0, 1] | yes | mean deviation of warp path from ideal diagonal; ported from the old `find_best_match_dtw_bernd_clifford` |
| `match_q_start`, `match_q_end` | int | yes | absolute query frames bounding the matched window |
| `match_duration_minutes` | float | yes | `(q_end - q_start) * dataset.frame_interval_minutes` |
| `warping_speed` | float | per-frame | discrete derivative of `pseudotime` |
| `propagated_{label}_label` | float | per-frame | template label fraction propagated via warp path; `NaN` outside `aligned` |
| `template_id` | str | per-frame | UUID linking to template zarr |

## Example refined-candidate entry (Stage 0.5)

```yaml
candidate_sets:
  infection_transitioning_nondiv_top20:
    refine_from: infection_transitioning_nondiv   # parent candidate set
    channel: sensor                                # channel used for bootstrap alignment
    min_pre_frames: 4                              # stricter than the parent's min_pre_minutes
    min_post_frames: 6
    top_n_by_cost: 20                              # keep cells with lowest DTW cost against the initial template
```

The final template entry references the *refined* set:

```yaml
templates:
  infection_nondividing_sensor:
    candidate_set: infection_transitioning_nondiv_top20
    channel: sensor
    anchor_label: infection_state
    anchor_positive: infected
    preprocessing:
      pca:
        n_components: 20
      l2_normalize: true
    dba:
      max_iter: 30
      init: medoid
    metric: cosine
```

## Cross-dataset results (reference — refined 20-cell template, Apr 2026)

Template built from 20 hand-picked+bootstrap-refined cells from 07_24 (SEC61 A/2 + G3BP1 C/2), 17 frames × 30 min = 455 min.

| Query set | Frame rate | Virus | Tracks kept | Cost p50 |
|---|---:|---|---:|---:|
| `sensor_all_07_24` (build datasets) | 30 min | ZIKV | 96 | **0.206** |
| `sensor_07_22_zikv` (cross frame rate) | 10 min | ZIKV | 49 | 0.207 |
| `sensor_08_26_zikv` (new replicate) | 30 min | ZIKV | 92 | 0.232 |
| `sensor_01_28_zikv_denv` (cross-virus) | 30 min | ZIKV+DENV | 136 | 0.292 |

Ordering is the expected signal: build ≈ cross-frame-rate < cross-replicate < cross-virus.

### Template selection (Apr 2026): keep both `manual_debug_sensor` and `infection_nondividing_sensor`

Both templates are maintained. They serve different purposes:

| Template | Build set | Use case |
|---|---|---|
| `manual_debug_sensor` | 4 hand-picked cells on 07_24 A/2 | Debug / smoke-test. Sharpest in-distribution PC collapse; useful for verifying new code paths. |
| `infection_nondividing_sensor` | 20 bootstrap-refined cells on 07_24 (A/2 + C/2) | Production. Monotonic query-truth curves on every dataset with per-frame labels. Use this for organelle-remodeling and cross-dataset analyses. |

**Honest query-truth comparison with the updated Stage 2d plot** (raw flavor, query-truth curve binned by `estimated_t_rel_minutes` on `alignment_region == "aligned"`):

| Query set | Truth col | `manual_debug` right-panel | `infection_nondiv` right-panel |
|---|---|---|---|
| `sensor_all_07_24` | `infection_state` | sharp rise to ~0.95, width ~200 min | rise to ~0.75, width ~350 min |
| `sensor_07_22_zikv` | `predicted_infection_state` | modest rise 0.2 → 0.85 | sharp rise 0.1 → 0.95 |
| `sensor_08_26_zikv` | — (no per-frame labels) | — | — |
| `sensor_01_28_zikv_denv` | `predicted_infection_state` | **non-monotonic** (rises, falls, rises; overfits to ZIKV-only trajectory) | roughly monotonic rise 0.15 → 0.7 |

So `manual_debug` wins in-distribution but breaks on cross-virus; `infection_nondiv` gives monotonic alignment everywhere the labels exist. Neither is "the right answer" universally — pick based on the analysis target. For organelle remodeling we use `infection_nondividing_sensor` because the question spans multiple replicates.

08_26 is currently uninformative for truth-curve evaluation because its embedding zarr obs lacks `predicted_infection_state`. Running the infection classifier on that zarr is the gating step to close the cross-replicate picture.

## Next steps & known gaps

### Outstanding

- **Stage 2e — Organelle remodeling (the main goal).** Design locked (option a: reuse the sensor alignment parquet as the timing skeleton; no separate organelle template). Full spec in the "Stage 2e" section above. Implementation delivered in phases: Panel A → add Panels B + C → sweep channels × query sets → cross-dataset offset replication.
- **UMAP/PHATE colored by pseudotime.** Once organelle plots land, this is the natural next exploratory step.
- **Run infection classifier on 08_26 embedding zarr.** The 2025_08_26 sensor zarr obs lacks `predicted_infection_state`, so Stage 2d/2e truth curves can't be evaluated on that dataset. Gating step for closing the cross-replicate picture.
- **Stage 1a PC plots** still use the old closed-endpoint `evaluate_template.py` pipeline. Works, but the left-column "unaligned" curve now uses the true annotation (fixed Apr 2026). When ready for a deeper refactor, switch Stage 1a to use subsequence DTW like Stage 2 for consistency.
- **07_22 build-set integration.** 07_22 annotations use an older tracking version that doesn't match the embedding zarr's track_ids. Re-tracking 07_22 with the current version would let us include it in the template build (not just as a query set).
- **Cleanup of swept template zarrs.** `sweep_pcs.py` leaves `template_*_pc5/10/20/30/50.zarr` (~50 MB each) under `1-build_template/templates/`. Add a `--cleanup` flag or document manual deletion.

### Followups / fragility

- **Sakoe-Chiba band** (`--sakoe-chiba-ratio`) as an optional 4th guard alongside psi, skew, min_match — only wire up if we see more collapse symptoms.
- **Per-dataset `data_zarr` in `datasets.yaml`** is populated for 07_22/07_24 but not for 08_26/01_28 (query-only — no image montages needed). Adding them would enable Stage 2c montages on those datasets.
- **Annotation noise (±2-3 frames around true onset)** is handled by DBA averaging, but a systematic bias across annotators would shift the template's t=0. No known bias today; worth re-checking if a new annotator starts contributing.
- **Stage 2d truth curve** (`plot_pcs_aligned.py --truth-column`) falls back gracefully to a placeholder when the query obs doesn't have the requested label column. 08_26/01_28 have `predicted_infection_state` only; 07_24/07_22 have human `infection_state`. Use `--truth-column infection_state --truth-positive infected` when human labels exist; `predicted_infection_state` otherwise.

### Bugs fixed this cycle (Apr 2026)

- **Psi collapse**: unconstrained psi let DTW collapse the template onto a single query frame (cost ~0, no biology). Capped at `t_template // 2`.
- **Minute-based psi was wrong**: initial `max_psi_minutes` scaling used the *query* frame interval, which over-relaxed on cross-frame-rate datasets. Psi is a template-axis budget; the frame-unit default handles all frame rates correctly.
- **Label propagation** set pre/post frames to 0.0/1.0; now `NaN` (matches `estimated_t_rel_minutes` convention).
- **Stage 1a truth curve** was using `propagated_*_label` (template-warped) instead of the candidate CSV ground-truth. Fixed to read from CSV directly.
- **Stage 2d truth curve** rendered a placeholder "(no ground-truth for query cells)"; now reads from query obs (`--truth-column`).
- **Stage 2d right-panel was misleading**: the bottom-right curve plotted the template's own stored `template_labels/{col}` fraction, which is a property of 4-20 build-set cells and always looks sharp (goes from 0 to 1 within one template step). This read as "alignment is perfect" when in reality no query labels were involved. Fix: right panel now plots query-truth binned by `estimated_t_rel_minutes` (restricted to `alignment_region == "aligned"`) as the primary solid-red curve, and demotes the template fraction to a dashed grey secondary reference. This is how we caught the `manual_debug_sensor` cross-virus failure on 01_28 (right-panel truth curve became non-monotonic — the real signal).
- **DBA medoid init** subsampled randomly; could pick a short track as medoid, truncating the template. Now picks the longest N.
- **Dead code deleted**: `evaluation.py` (broken `onset_concordance` metric) and untracked `classification.py`.
