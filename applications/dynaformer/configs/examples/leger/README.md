# Dynaformer on the Leger et al. cell-cycle dataset

Brightfield 64×64 movies of tracked RPE1 cells in, two continuous FUCCI channels per frame out.
Trained 2026-08-04 on the dataset from [Zenodo 14987478](https://zenodo.org/records/14987478); the
reference implementation and its released checkpoints live in `../../../../../Andrea_mlbi/`.

## What was run

Two runs, identical in encoder, head, optimizer, data and budget — differing only in whether
information crosses time:

| leaf | temporal module | params | best val L1 | epoch |
|---|---|---|---|---|
| `fit_unext2_bf.yml` | RoPE transformer, 4 layers | 41.0 M | **0.0534** | 8 |
| `fit_unext2_bf_single_frame.yml` | `n_layers: 0` → LayerNorm only | 28.4 M | 0.1681 | 11 |

`n_layers: 0` is a genuine ablation, not an approximation: perturbing frame 5 changes frame 0 by
exactly `0.000e+00`, against `9.9e-02` with 4 layers.

The encoder (ConvNeXtV2-tiny via `UNeXt2FrameEncoder`) trains **from scratch**, which is the
like-for-like match to the reference study's from-scratch ResNet-18 — so the comparison isolates
the temporal module rather than pretraining. Loss is pure L1 (`mse_weight: 0`), as they used.

```sh
sbatch run_fit.slurm fit_unext2_bf.yml
sbatch run_fit.slurm fit_unext2_bf_single_frame.yml
# then, per run:
python ../../../scripts/export_leger_predictions.py --ckpt <best>.ckpt --config <leaf>.yml --run <name>
cd ../../../../../Andrea_mlbi/code && python -m reproduce.report
```

## Results — Regular test split (358 tracks)

Scored by `Andrea_mlbi/code/reproduce/metrics.py`, the same code that reproduces two of the four
released checkpoints to six significant figures.

| model | L1 FUCCI₁ | L1 FUCCI₂ | Δ_DTW |
|---|---|---|---|
| Leger, **as printed in the paper** | 0.066 | 0.062 | 1.285 |
| Leger transformer, **from their released weights** | 0.129 | 0.091 | 2.150 |
| Leger Mamba, from released weights | 0.118 | 0.094 | 1.330 |
| Leger single-frame MLP, from released weights | 0.192 | 0.151 | 4.027 |
| **dynaformer temporal** | **0.073 ± 0.049** | **0.065 ± 0.036** | **1.291 ± 0.575** |
| dynaformer single-frame | 0.216 ± 0.062 | 0.164 ± 0.048 | 4.511 ± 0.928 |

**Dynaformer reproduces the paper's headline result that the authors' own released checkpoint does
not.** Δ_DTW 1.291 against a published 1.285, and L1 0.073/0.065 against 0.066/0.062 — while their
released transformer weights give 0.129/0.091 and Δ_DTW 2.150. This is independent support for the
conclusion in `Andrea_mlbi/code/README.md` that the uploaded transformer checkpoint is not the
model behind the paper's table.

**The ablation is the cleaner claim.** Four RoPE layers, nothing else changed, cut L1 by ~3× and
Δ_DTW by 3.5×. The paper's central argument, measured within one codebase rather than across two.

Landmark metrics agree where they can be trusted. On the S/G2 transition — the one that survives
the released labels' irreversible min-max scaling (see `Andrea_mlbi/code/README.md`) — dynaformer
gets R² 0.808 against the paper's 0.803, and Δt 63.8 min against 57.2; the released checkpoint gives
113.1 min. Phase F1 is 0.72 / 0.95 / 0.91, the best of any model scored here. The G1/S landmark is
not reproducible from this data for any model and should be ignored.

## Where dynaformer is worse — Drug split (73 tracks, out of distribution)

| model | L1 FUCCI₁ | L1 FUCCI₂ | Δ_DTW |
|---|---|---|---|
| Leger transformer, from released weights | 0.147 | 0.109 | 3.718 |
| **dynaformer temporal** | 0.168 ± 0.072 | 0.160 ± 0.058 | 4.683 ± 1.667 |
| dynaformer single-frame | 0.235 ± 0.078 | 0.171 ± 0.046 | 6.140 ± 1.124 |

On Palbociclib-treated cells, whose G1 is roughly doubled, dynaformer generalizes **worse** than the
reference transformer despite being better in distribution. Its validation loss also stopped
improving at epoch 8 of 24, i.e. it overfits earlier. Better in-distribution accuracy here comes
with weaker out-of-distribution behaviour, and quoting only the Regular row would misrepresent that.

## Caveats that favour dynaformer

Both are deliberate, and both should be stated whenever these numbers are quoted:

1. **Less training data, cleaner protocol.** Dynaformer trains on 7 of the 8 training movies,
   holding out `0608` (311 tracks, 6 %) for validation. Leger had no validation split
   (`input_val: null`) and monitored on the test set.
2. **Checkpoint selection.** Dynaformer is evaluated at its best-validation epoch; the released
   Leger weights are final-epoch (`epoch=149`).

Sample budget *is* matched: ~2.8 M frames/epoch × 24 epochs ≈ 66 M frames, against their
150 × ~375 k ≈ 56 M. Epoch counts are not comparable between the two; frames seen is.

## Notes

- No image decoder — this dataset has no image target, only the two per-frame scalars. `decoder:
  null` makes `engine.py` skip the image loss, and `TrackTrajectoryMetrics` is dropped from the
  callbacks because it requires `norm_meta` and only logs its image readout when a decoder exists.
  Checkpoint selection falls back to `loss/validate`.
- `run_fit.slurm` runs *from* the run directory rather than overriding `trainer.logger` on the
  command line. Overriding the logger from the CLI wedged startup — no output, 4 s of CPU in
  14 min, GPU never allocated.
- 4 h 12 m per run on one H100 at 24 epochs, GPU-bound at ~97 % utilisation.
