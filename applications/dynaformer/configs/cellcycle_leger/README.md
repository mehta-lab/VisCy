# Cell-cycle regression on the Leger et al. RPE1 tracks

The cell-cycle regression models of `../cellcycle` (frozen foundation-model embeddings → RoPE transformer T → MLP
head f), trained and scored on the RPE1 FUCCI dataset of Leger et al., MELBA 2026
(`/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data`, see its `README.md`): brightfield 64×64 movies of
whole mitosis-to-mitosis tracks in, the two FUCCI reporters (Geminin, Cdt1; per-track min-max scaled) out. Scored by
`Andrea_mlbi/code/reproduce`, the harness that scores the paper's released checkpoints and the August dynaformer runs
of `../examples/leger`.

## What differs from `../cellcycle`

Only the data. Every leaf composes `../cellcycle/recipes/tasks/regression.yml` and the NIH3T3 trainer, so model,
optimizer, budget (15k steps) and checkpoint selection (validation Pearson r) are identical;
`tests/test_cellcycle_leger.py` checks each leaf against its NIH3T3 namesake.

- **Splits.** The published splits have no validation set: movie `0608` (311 tracks) is held out of `train` for it, as
  in the August runs; `test` (Regular, 358 tracks) and `test_drugged` (Drug, 73) are only scored.
- **Clips.** Up to 192 frames (the longest window Leger et al. trained on), ~32 clips per step (`frames_per_batch`
  6144); evaluation runs whole tracks (80–868 frames).
- **Input.** Brightfield, per-frame z-scored as published, 64 px resized to 224 by the encoder. The RGB models
  (DINOv3, C-RADIO) map z-scores through `rgb_window` (-6, 4), brightfield's 0.1st–99.95th percentile, instead of the
  phase window (-4, 8). MorphEm and CELL-DINO normalize their input themselves.
- **Targets.** Geminin and Cdt1 in stored order, already in [0, 1] (scale 1); no PIP-FUCCI phase scores during
  training, the harness scores the phases.

## Pipeline

The commands that produced the results below (VAST environment, from this folder):

```sh
DATA=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data
python -m dynaformer.leger_embed index --data-dir $DATA --out $DATA/embeddings   # index.csv, labels.npy (once)
./embed.sh morphem 450000 --constraint="h200|h100|a100"         # plan, chunk jobs (3), finalize job
./embed.sh vit-untrained 450000 --constraint="h200|h100|a100"   # 3 chunks
./embed.sh cell-dino 150000 --constraint="h200|h100|a100"       # 9 chunks
./embed.sh dinov3 100000 --constraint="h200|h100|a100"          # 14 chunks
./embed.sh c-radio 90000 --constraint="h200|h100|a100"          # 15 chunks
./submit.sh --time=0:30:00 --constraint="h200|h100|a100" --dependency=afterok:<finalize> -- regression_<encoder>.yml \
  regression_<encoder>_single_frame.yml                          # fit, then eval + export; per encoder
cd /hpc/projects/comp.micro/virtual_staining/models/dynaformer/Andrea_mlbi/code
PYTHONPATH=. .venv/bin/python -m reproduce.report               # results/tables.md, results/figure4.png
PYTHONPATH=. .venv/bin/python ../../VisCy/applications/dynaformer/scripts/leger_benchmark_stats.py \
  ../../runs/cellcycle_leger                                     # paired intervals, landmark counts
```

- **Embedding caches** live in `$DATA/embeddings/<encoder>.npy` `(1,315,985, 8, D)` float16, one row per frame of
  `index.csv` (splits in order train, test, test_drugged; tracks by name; frames in time order), 8 rotations/flips per
  frame. `embed.sh` plans chunks of whole tracks sized for a 30 min job, and `run_finalize.slurm` checks every chunk
  and every row before writing `<encoder>.json`; the data module reads no cache without it, nor one written against
  a different `index.csv`.
- **Runs** in `runs/cellcycle_leger/<leaf>/`. `run_eval.slurm` scores the best checkpoint on `val`, `test` and
  `test_drugged` (on CPU, float32) into `eval/best_regression/`, then `python -m dynaformer.leger_export` writes the
  harness's `Andrea_mlbi/code/results/predictions/dynaformer_<encoder>_{temporal,single_frame}_<split>.npz`.

## Results

Trained and scored 2026-10-10: 5 encoders × {temporal, single-frame}, one seed each. Every fit ran its 15,000 steps
(30 validations); the scores below are of the checkpoint with the best validation r.

| encoder | val r, temporal (step) | val r, single-frame (step) | embedding, GPU-min | fit, GPU-min (temporal / single) |
|---|---|---|---|---|
| MorphEm | 0.962 (6k) | 0.725 (14.5k) | 21 | 5 / 3 |
| CELL-DINO | 0.964 (8k) | 0.742 (13.5k) | 70 | 51 / 50 |
| DINOv3 | **0.965** (8k) | 0.711 (15k) | 120 | 57 / 56 |
| C-RADIOv4-H | 0.960 (8k) | 0.733 (14.5k) | 120 | 199 / 212 |
| Untrained ViT | 0.947 (2k) | 0.460 (14.5k) | 20 | 34 / 34 |

The fits are bound by reading the cache, not by the GPU: MorphEm's 16 GB cache trained 15k steps in one 5-minute
allocation, C-RADIO's 81 GB one needed eight allocations of up to 30 minutes. Trainable parameters: 7.5–8.7 M (T
7.1 M; the input projection grows with the embedding width), single-frame 0.4 M.

Scores are the harness's (`Andrea_mlbi/code/reproduce`): per-track L1 per channel and the multivariate ΔDTW, each
mean ± std over tracks as in the paper's tables. Cdt1 is the paper's FUCCI1, Geminin its FUCCI2 (the stored order is
the reverse). *Track r*, not in the paper, is each track's Pearson r averaged over the two channels and then over
tracks; for every run it equals the engine's own track r to 1e-6, which checks the export. Rows:

- *paper* — the printed values (Tables 1-3, Figure 4 of Leger et al.), for which no predictions exist;
- *released* — the authors' released checkpoints, re-scored by the harness. `bf_mlp` and `bf_mamba` reproduce their
  Regular L1 within 0.006; `bf_transformer` does not (see `Andrea_mlbi/code/README.md`);
- *August UNeXt2* — the dynaformer of `../examples/leger` (UNeXt2 encoder trained end to end on the 64 px
  brightfield; `dynaformer_temporal` / `dynaformer_single_frame` in the harness tables);
- *ours* — this folder (`dynaformer_<encoder>_{temporal,single_frame}` in the harness tables).

Bold: best brightfield value per column.

### Regular (`test`: 358 tracks, one movie)

| model | L1 Cdt1 | L1 Geminin | ΔDTW | track r |
|---|---|---|---|---|
| *paper:* single frame (MLP) | 0.193 ± 0.066 | 0.146 ± 0.045 | 3.735 ± 0.863 | – |
| causal CNN | 0.157 ± 0.078 | 0.122 ± 0.049 | 2.468 ± 0.917 | – |
| Mamba | 0.112 ± 0.072 | 0.091 ± 0.049 | 1.444 ± 0.898 | – |
| transformer | 0.066 ± 0.038 | 0.062 ± 0.037 | 1.285 ± 0.553 | – |
| *released:* `bf_mlp` | 0.192 ± 0.063 | 0.151 ± 0.053 | 4.027 ± 0.877 | 0.738 ± 0.134 |
| `bf_mamba` | 0.118 ± 0.069 | 0.094 ± 0.043 | 1.330 ± 0.563 | 0.908 ± 0.079 |
| `bf_transformer` | 0.129 ± 0.040 | 0.091 ± 0.034 | 2.150 ± 0.526 | 0.942 ± 0.053 |
| `h2b_transformer` (H2B input) | 0.096 ± 0.044 | 0.081 ± 0.033 | 1.805 ± 0.690 | 0.952 ± 0.033 |
| *August UNeXt2:* temporal | 0.073 ± 0.049 | 0.065 ± 0.036 | 1.291 ± 0.575 | 0.959 ± 0.053 |
| single-frame | 0.216 ± 0.062 | 0.164 ± 0.048 | 4.511 ± 0.928 | 0.686 ± 0.145 |
| *ours, temporal:* MorphEm | **0.060** ± 0.038 | 0.058 ± 0.030 | 1.175 ± 0.679 | **0.970** ± 0.039 |
| CELL-DINO | 0.067 ± 0.043 | 0.051 ± 0.028 | **1.112** ± 0.676 | 0.965 ± 0.043 |
| DINOv3 | 0.068 ± 0.043 | 0.051 ± 0.026 | 1.120 ± 0.600 | 0.966 ± 0.049 |
| C-RADIOv4-H | 0.070 ± 0.046 | **0.050** ± 0.025 | 1.119 ± 0.563 | 0.964 ± 0.057 |
| Untrained ViT | 0.090 ± 0.055 | 0.056 ± 0.034 | 1.466 ± 0.930 | 0.947 ± 0.082 |
| *ours, single-frame:* MorphEm | 0.221 ± 0.053 | 0.163 ± 0.042 | 3.951 ± 0.769 | 0.728 ± 0.135 |
| CELL-DINO | 0.217 ± 0.057 | 0.159 ± 0.043 | 3.908 ± 0.743 | 0.736 ± 0.139 |
| DINOv3 | 0.236 ± 0.051 | 0.177 ± 0.041 | 4.184 ± 0.710 | 0.683 ± 0.132 |
| C-RADIOv4-H | 0.225 ± 0.054 | 0.167 ± 0.042 | 3.967 ± 0.733 | 0.714 ± 0.129 |
| Untrained ViT | 0.296 ± 0.043 | 0.240 ± 0.043 | 4.847 ± 0.815 | 0.486 ± 0.178 |

### Drug (`test_drugged`: 73 tracks, two movies, 10 nM Palbociclib)

| model | L1 Cdt1 | L1 Geminin | ΔDTW | track r |
|---|---|---|---|---|
| *paper:* single frame (MLP) | 0.239 ± 0.082 | 0.182 ± 0.056 | 5.329 ± 1.147 | – |
| causal CNN | 0.252 ± 0.113 | 0.161 ± 0.059 | 4.323 ± 1.302 | – |
| Mamba | 0.485 ± 0.090 | 0.259 ± 0.045 | 3.563 ± 1.918 | – |
| transformer | **0.147** ± 0.056 | 0.139 ± 0.048 | **3.022** ± 0.985 | – |
| *released:* `bf_mlp` | 0.214 ± 0.076 | 0.141 ± 0.042 | 5.383 ± 1.162 | 0.672 ± 0.119 |
| `bf_mamba` | 0.509 ± 0.081 | 0.266 ± 0.046 | 3.534 ± 1.874 | 0.349 ± 0.118 |
| `bf_transformer` | **0.147** ± 0.044 | **0.109** ± 0.035 | 3.718 ± 0.752 | **0.849** ± 0.095 |
| `h2b_transformer` (H2B input) | 0.154 ± 0.031 | 0.123 ± 0.029 | 4.273 ± 1.113 | 0.808 ± 0.076 |
| *August UNeXt2:* temporal | 0.168 ± 0.072 | 0.160 ± 0.058 | 4.683 ± 1.667 | 0.729 ± 0.149 |
| single-frame | 0.235 ± 0.078 | 0.171 ± 0.046 | 6.140 ± 1.124 | 0.614 ± 0.124 |
| *ours, temporal:* MorphEm | 0.213 ± 0.090 | 0.149 ± 0.065 | 4.696 ± 2.200 | 0.664 ± 0.202 |
| CELL-DINO | 0.240 ± 0.100 | 0.144 ± 0.064 | 4.577 ± 2.061 | 0.655 ± 0.176 |
| DINOv3 | 0.191 ± 0.077 | 0.124 ± 0.060 | 4.442 ± 1.938 | 0.716 ± 0.180 |
| C-RADIOv4-H | 0.161 ± 0.064 | 0.122 ± 0.060 | 4.172 ± 1.831 | 0.762 ± 0.150 |
| Untrained ViT | 0.280 ± 0.053 | 0.210 ± 0.047 | 7.196 ± 2.022 | 0.447 ± 0.183 |
| *ours, single-frame:* MorphEm | 0.277 ± 0.083 | 0.192 ± 0.061 | 5.470 ± 0.877 | 0.573 ± 0.139 |
| CELL-DINO | 0.298 ± 0.085 | 0.197 ± 0.057 | 5.409 ± 0.879 | 0.553 ± 0.139 |
| DINOv3 | 0.273 ± 0.077 | 0.195 ± 0.055 | 5.593 ± 0.852 | 0.559 ± 0.140 |
| C-RADIOv4-H | 0.272 ± 0.082 | 0.191 ± 0.057 | 5.383 ± 0.849 | 0.574 ± 0.139 |
| Untrained ViT | 0.261 ± 0.059 | 0.242 ± 0.058 | 6.537 ± 1.056 | 0.445 ± 0.154 |

### Paired differences

Per-track L1 (mean of the two channels) of our temporal run minus the reference's, over the same tracks; 95% bootstrap
interval over tracks (10,000 resamples, seed 0; `scripts/leger_benchmark_stats.py`, full list in
`runs/cellcycle_leger/leger_benchmark_stats.md`). Negative: ours is better.

| ours, temporal | − August UNeXt2 | − released `bf_mamba` | − released `bf_transformer` | − our single-frame |
|---|---|---|---|---|
| *Regular:* MorphEm | −0.010 [−0.013, −0.006] | −0.047 [−0.052, −0.042] | −0.051 [−0.055, −0.047] | −0.133 [−0.137, −0.128] |
| CELL-DINO | −0.010 [−0.014, −0.006] | −0.047 [−0.053, −0.042] | −0.051 [−0.056, −0.047] | −0.129 [−0.134, −0.124] |
| DINOv3 | −0.009 [−0.013, −0.006] | −0.046 [−0.052, −0.041] | −0.051 [−0.054, −0.047] | −0.147 [−0.151, −0.143] |
| C-RADIOv4-H | −0.009 [−0.012, −0.005] | −0.046 [−0.052, −0.041] | −0.050 [−0.054, −0.047] | −0.136 [−0.141, −0.131] |
| Untrained ViT | +0.005 [+0.001, +0.009] | −0.033 [−0.038, −0.027] | −0.037 [−0.041, −0.032] | −0.195 [−0.200, −0.190] |
| *Drug:* MorphEm | +0.017 [+0.001, +0.032] | −0.206 [−0.220, −0.192] | +0.053 [+0.039, +0.068] | −0.054 [−0.068, −0.040] |
| CELL-DINO | +0.028 [+0.012, +0.043] | −0.195 [−0.210, −0.180] | +0.064 [+0.047, +0.081] | −0.055 [−0.070, −0.042] |
| DINOv3 | −0.007 [−0.022, +0.007] | −0.230 [−0.242, −0.217] | +0.029 [+0.017, +0.041] | −0.077 [−0.093, −0.061] |
| C-RADIOv4-H | −0.023 [−0.037, −0.009] | −0.246 [−0.258, −0.233] | +0.013 [+0.002, +0.025] | −0.090 [−0.106, −0.075] |
| Untrained ViT | +0.081 [+0.067, +0.093] | −0.142 [−0.155, −0.130] | +0.117 [+0.105, +0.128] | −0.006 [−0.021, +0.007] |

### Biological checkpoints (Regular)

Appendix A.5 landmarks: Δt is the mean absolute timing error of the detected transitions, *found* the number of tracks
whose predicted S/G2 is detected at all, F1 per phase over frames, R² of predicted against true transition time
(Figure 4; `Andrea_mlbi/code/results/figure4.png`).

| model | Δt G1/S [min]¹ | Δt S/G2 [min] | S/G2 found | F1 G1¹ / S¹ / G2 | R² G1/S¹ | R² S/G2 |
|---|---|---|---|---|---|---|
| *paper:* single frame (MLP) | 191.3 | 117.9 | – | 0.71 / 0.64 / 0.87 | 0.276 | 0.542 |
| causal CNN | 146.8 | 113.3 | – | 0.78 / 0.64 / 0.86 | – | – |
| Mamba | 111.4 | 102.4 | – | 0.83 / 0.72 / 0.89 | 0.436 | 0.619 |
| transformer | 60.1 | 57.2 | – | 0.90 / 0.85 / **0.93** | 0.632 | 0.803 |
| *released:* `bf_mlp` | 33.0 | 165.7 | 231 | 0.31 / 0.86 / 0.56 | 0.051 | 0.551 |
| `bf_mamba` | 21.2 | 124.6 | 358 | 0.65 / 0.91 / 0.85 | 0.381 | 0.530 |
| `bf_transformer` | 23.2 | 113.1 | 293 | 0.56 / 0.90 / 0.72 | 0.279 | 0.812 |
| *August UNeXt2:* temporal | 19.0 | 63.8 | 356 | 0.72 / 0.95 / 0.91 | 0.438 | 0.808 |
| single-frame | 32.7 | 164.9 | 209 | 0.15 / 0.85 / 0.52 | 0.048 | 0.583 |
| *ours, temporal:* MorphEm | 17.4 | **52.4** | 358 | 0.73 / 0.96 / **0.93** | 0.535 | **0.848** |
| CELL-DINO | 19.0 | 60.4 | 358 | 0.72 / 0.95 / 0.92 | 0.490 | 0.799 |
| DINOv3 | 17.0 | 62.0 | 357 | 0.75 / 0.95 / 0.91 | 0.571 | 0.803 |
| C-RADIOv4-H | 18.7 | 59.8 | 358 | 0.72 / 0.95 / 0.91 | 0.483 | 0.824 |
| Untrained ViT | 19.6 | 77.7 | 358 | 0.73 / 0.94 / 0.88 | 0.518 | 0.776 |
| *ours, single-frame:* MorphEm | 34.2 | (261.4)² | 18 | 0.03 / 0.81 / 0.03 | 0.011 | (0.545)² |
| CELL-DINO | 34.2 | (252.7)² | 24 | 0.02 / 0.81 / 0.07 | 0.025 | (0.644)² |
| DINOv3 | 34.5 | (245.6)² | 8 | 0.01 / 0.81 / 0.02 | 0.003 | (0.944)² |
| C-RADIOv4-H | 33.9 | (206.6)² | 35 | 0.04 / 0.81 / 0.10 | 0.017 | (0.564)² |
| Untrained ViT | 34.6 | – | 0 | 0.00 / 0.81 / 0.00 | –³ | – |

¹ Not interpretable on the released labels (`Andrea_mlbi/code/README.md`): A.5 places the 5% threshold in linear
fluorescence, but the labels are log2-compressed and min-max scaled per track, so the rising Geminin crosses it within
the first hours of every track — 28-39 distinct detection frames over 358 tracks for every temporal run. The small
Δt G1/S values reflect that clustering, not accuracy; F1 G1 and S inherit it. ² Over only the tracks where S/G2 is
found. ³ Every G1/S detection lands on the same frame.

### Findings

1. **On Regular, T on any of the four pretrained encoders has lower error than every published brightfield model.**
   Mean per-track L1 over the two channels is 0.059–0.060, against 0.064 for the paper's printed transformer, 0.069
   for the August UNeXt2 (−0.009 to −0.010 paired, every interval within [−0.014, −0.005]) and 0.106 / 0.110 for the
   released Mamba / transformer (−0.046 to −0.051). ΔDTW is 1.11–1.17 against 1.285 printed, 1.291 August and 1.330
   released Mamba (C-RADIO −0.21 [−0.26, −0.16]); track r 0.964–0.970 against 0.959 (August) and 0.908 (released
   Mamba). The gain is in Geminin (0.050–0.058 against the printed 0.062); in Cdt1 only MorphEm is below the printed
   transformer (0.060; the others 0.067–0.070 against 0.066). The printed rows have no predictions to pair with, so
   the comparisons with them carry no interval, and each printed row is one seed of five.
2. **The four pretrained encoders tie on Regular, and validation does not separate them either.** C-RADIO minus each
   other encoder is +0.0002 to +0.0010, every interval spanning zero (all within [−0.0022, +0.0040]); validation r spans
   0.960–0.965. Chosen on validation, the only choice that leaves the test splits untouched, the model would be DINOv3.
3. **Temporal context does most of the work, and single frames of a frozen foundation model trail a fine-tuned
   ResNet-18.** Removing T's layers raises L1 by 0.129–0.147 on Regular and drops track r from 0.964–0.970 to
   0.683–0.736. Single-frame, the frozen features are behind the released single-frame MLP (ResNet-18 fine-tuned on the
   64 px crops, L1 0.172): +0.016 to +0.035 on Regular and +0.055 to +0.070 on Drug, every interval above zero. Their
   trajectories are also too noisy for the S/G2 detector, which fires in 8–35 of 358 tracks (released MLP 231, August
   single-frame 209), so their S/G2 timing is not comparable.
4. **On Drug, C-RADIO generalizes best of ours and beats the August UNeXt2, but the paper's transformer still
   leads.** L1 is 0.141 (C-RADIO), 0.157 (DINOv3), 0.181 (MorphEm), 0.192 (CELL-DINO); C-RADIO minus DINOv3 is
   −0.016 [−0.027, −0.004], minus MorphEm −0.040 [−0.054, −0.025], minus CELL-DINO −0.051 [−0.066, −0.036].
   C-RADIO beats the August UNeXt2 (−0.023) and the released Mamba, which collapses on Drug as in the paper (−0.246);
   it trails the released transformer (+0.013 [+0.002, +0.025]) and is level with the printed one in L1 (0.141
   against 0.143) but not in ΔDTW (4.17 against 3.02; against the released Mamba +0.64 [+0.15, +1.11], the released
   transformer +0.45 [+0.09, +0.84]). The Drug ranking is not the validation ranking (C-RADIO is last on validation,
   0.960), so it could not have been chosen in advance, and it rests on 73 tracks from two movies.
5. **For MorphEm, longer training traded Drug for Regular.** The 200-step check run (still inside the 500-step
   warm-up; `runs/cellcycle_leger/check_200_steps`) scored L1 0.077 on Regular — 0.018 worse than the selected 6k-step
   checkpoint ([+0.015, +0.022]) but already below both released sequence models (0.106, 0.110). On Drug it was
   better: L1 0.161 against 0.181 (−0.020 [−0.034, −0.006]) and track r 0.788 against 0.664 (+0.124 [+0.081,
   +0.169]). Checkpoints are selected on a Regular movie, so nothing in the selection protects Drug.
6. **Without pretraining, T still does well on Regular, but not on Drug.** On the untrained ViT, T reaches L1 0.073 and
   track r 0.947 on Regular, from single frames that are the worst of all (0.268, track r 0.486): 0.005 behind the
   August UNeXt2 ([+0.001, +0.009]) and 0.014 behind C-RADIO ([+0.011, +0.017]), still ahead of both released sequence
   models. On Drug it falls to 0.245, 0.104 behind C-RADIO ([+0.089, +0.116]), and T adds nothing over its single
   frames (−0.006 [−0.021, +0.007]). T makes up for weak frame features on Regular, not on Drug.
7. **S/G2 timing matches the printed transformer's.** The temporal runs on pretrained encoders place S/G2 52–62 min
   from the truth on average (MorphEm 52.4), against 57.2 printed, 63.8 for the August UNeXt2 and 113.1 / 124.6 for
   the released transformer / Mamba, with R² 0.80–0.85 (printed 0.803) and F1 G2 0.91–0.93 (printed 0.93). G1/S
   cannot be compared (footnote 1).

### Caveats

- **One seed per configuration.** The intervals resample tracks: they cover which tracks were scored, not seeds, and
  tracks of one movie (Regular) or two (Drug) are not independent, so a movie-level interval would be wider. The paper
  reports one seed of five, chosen before evaluation, and shows the spread across seeds only as box plots (Figure 3a).
- **Less training data and a selected checkpoint.** Movie 0608 (311 of the 5,188 training tracks) is held out to pick
  the checkpoint (best validation r: 6k–8k of 15k steps for the pretrained temporal runs, 2k for the untrained ViT).
  The paper trains on all 8 training movies for 150 epochs and keeps the last epoch (no early stopping); its released
  runs log a validation L1 whose split is not documented.
- **What is trained differs.** Here E is frozen (the 64 px brightfield resized to 224) and only T and f train, with the
  NIH3T3 recipe untuned for this data (AdamW 3e-4, 15k steps of ~6,144 frames, L1 + MSE). The paper fine-tunes an
  ImageNet ResNet-18 (11.2 M parameters) end to end on the 64 px crops with ≈1 M-parameter heads (AdamW 1e-4, batch 8,
  L1, 150 epochs); the August UNeXt2 trains everything from scratch.
- **Bidirectional.** T attends to the whole track, like the paper's transformer and the August UNeXt2. The paper's
  Mamba and causal CNN only see the past and are the models for online use, so the comparison with them also measures
  what seeing the future is worth.
- **Brightfield colour window.** DINOv3 and C-RADIO see z-scores through (−6, 4); the phase window (−4, 8) was not
  tried.
- **The released transformer does not reproduce its paper row** (Regular L1 0.129 / 0.091, ΔDTW 2.150, against 0.066 /
  0.062, 1.285), so it and the printed transformer are separate rows; only the released one has an interval.
- **Labels are per-track min-max scaled** (log2 of a softplus, then min-max): L1 and ΔDTW are relative to each track's
  own range, and the G1/S landmark is degenerate on them (footnote 1).
