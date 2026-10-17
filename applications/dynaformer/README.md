# Dynaformer

**Decoupled spatio-temporal virtual staining.** A per-frame spatial encoder embeds every frame of a movie
into one token, a transformer integrates across time, and the temporally-informed features drive a
virtual-staining decoder plus auxiliary per-frame heads.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

```
movie (B, T, C_in, Z, Y, X)
  │
  ├─ FrameEncoder     per-frame; time folded into the batch (B·T)
  │     └─→ FrameFeatures(token: (B·T, d_enc), spatial: patch tokens | multi-scale maps)
  │
  ├─ TemporalModule   bidirectional RoPE transformer over the T frame tokens
  │     └─→ (B, T, d_model)
  │
  ├─ ImageDecoder(spatial, cond=temporal token)     ← zero-init conditioning
  │     └─→ (B, T, C_out, Z_out, Y, X)
  │
  └─ Heads(temporal tokens)                          → per-frame scalars / class logits
```

Spatial and temporal processing are separate modules with a narrow interface, so the encoder, the temporal
module, the decoder and the heads are independent configuration choices.

## Two properties that carry the results

1. **RoPE, no learned positional embedding.** The temporal transformer accepts any sequence length, so the
   model trains on variable-length clips (`T` between `min_frames` and `window_size`) and runs whole-track
   inference in a single forward pass.
2. **Zero-init temporal conditioning.** The temporally-mixed token reaches the image decoder through a
   zero-initialized projection. At step 0 the decoder is exactly the per-frame decoder and only learns
   temporal corrections. In the originating study this single edge moved the image-head readout from
   0.303 to 0.581 per-track Pearson r.

## Installation

```sh
uv sync --all-packages --all-extras
```

## Usage

```sh
cd applications/dynaformer/configs/examples

# UNeXt2 encoder + UNeXt2 decoder (no external download)
uv run python -m dynaformer fit -c fucci/fit_unext2.yml

# Pretrained MorphEm ViT encoder + temporally-conditioned MAE decoder (reference model)
uv run python -m dynaformer fit -c fucci/fit_morphem_tcond.yml

uv run python -m dynaformer predict -c fucci/predict.yml
```

Both leaves need `data.init_args.data_path` (HCS OME-Zarr plate) and `data.init_args.tracks_path`
(directory of per-FOV tracking CSVs, ultrack column layout) filled in.

## Components

| Piece | Class | Notes |
|---|---|---|
| Frame encoder | `dynaformer.models.MorphEmFrameEncoder` | pretrained `CaicedoLab/MorphEm` ViT-S; CLS + 196 patch tokens |
| Frame encoder | `dynaformer.models.UNeXt2FrameEncoder` | `UNeXt2Stem` + timm ConvNeXt-V2; multi-scale maps + bottleneck token |
| Temporal | `dynaformer.models.RoPETransformerEncoder` | bidirectional, pre-norm, rotary position |
| Decoder | `dynaformer.models.MAEImageDecoder` | shallow ViT over patch tokens, unpatchify |
| Decoder | `dynaformer.models.UNeXt2ImageDecoder` | `UNeXt2Decoder` + `PixelToVoxelHead`, skip connections |
| Composition | `dynaformer.models.SpatioTemporalModel` | folds time into batch, wires conditioning |
| Engine | `dynaformer.DynaformerModule` | image loss + auxiliary heads, discriminative fine-tuning |
| Evaluation | `dynaformer.TrackTrajectoryMetrics` | whole-track Pearson / CCC / MAE / dominance accuracy |

Data loading lives in `viscy_data.track_window` (`TrackWindowDataModule`), the frame encoder's pretrained
backbone in `viscy_models.foundation` (`MorphEmModel`), and the length-bucketed batch sampler in
`viscy_data.sampler` (`LengthBucketedBatchSampler`) — all shared, so other applications can use them.

## Cell-cycle reference configuration

The reference leaf reproduces the setup from a study predicting the PIP-FUCCI cell-cycle readout
`gem = mKO / (mKO + mVenus)` from phase-contrast movies of tracked NIH3T3 nuclei: MorphEm ViT-S encoder
fine-tuned at `1e-5` while new parameters train at `3e-4`, a 4-layer × 8-head RoPE transformer at
`d_model=384`, a 4-block MAE decoder at `dec_dim=256` with temporal conditioning on, an L1+MSE image loss
and an equally weighted per-frame regression head, `window_size=64`, and always-on Gaussian noise
(`σ=0.05`) on the normalized input channel only.

That study's dataset used pre-cropped per-track stores on a mount that is not available here. This
application re-sources the same sampling from an HCS plate plus tracking CSVs, which is the layout the rest
of the monorepo uses. **The reference numbers have not been reproduced in this codebase** — the configs
encode the published hyperparameters, but a retrain is required once the tracked-cell data is mounted.

## Temporal embeddings of foundation models (cell cycle)

The second model family puts a pretrained foundation model in front of the temporal transformer and reads the
temporally integrated features out two ways:

```
phase movie [p_t]
  │
  ├─ E: FoundationFrameEncoder, frozen           e_t = [global token | mean of the central 4×4 patch tokens]
  │     MorphEm · CELL-DINO · DINOv3 · C-RADIOv4-H   (cached: run once per track point, 8 rotations/flips)
  │
  ├─ T: RoPETransformerEncoder over the track    [ê_t] = T([e_t])
  │
  ├─ f: MLPRegressionHead(ê_t)                   → Geminin-mKO and PIP-mVenus median in the nucleus
  └─ g: ConditionedUNet(p_t, ê_t)                → the two reporter images (zero-init, centre-gated FiLM)
```

| Piece | Class |
|---|---|
| E | `dynaformer.models.FoundationFrameEncoder` (`morphem`, `cell-dino`, `dinov3`, `c-radio`; `vit-untrained`, optionally `trainable`) |
| T, f, g together | `dynaformer.models.TemporalEmbeddingModel` |
| g | `dynaformer.models.ConditionedUNet`: UNeXt2 (as VSCyto2D) with zero-initialized FiLM at the bottleneck and every decoder stage |
| Data | `dynaformer.cellcycle_data.CellCycleDataModule` |
| Engine | `dynaformer.cellcycle_engine.TemporalEmbeddingModule`: regression, staining or both (`regression_weight`, `stain_weight`) |
| Scores | `dynaformer.cellcycle_metrics`: frame-, track- and PIP-FUCCI-phase-level |
| Cache | `python -m dynaformer.embed` |

Design choices:

- **E is frozen and cached.** Its embedding of every track point is computed once, under all 8 rotations/flips of
  the square; a training clip draws one of them and applies the same rotation to its images, which is exactly
  augmenting the movie. Training never runs E, so even the 653 M-parameter C-RADIOv4-H costs nothing per step.
- **Controls without a foundation model.** `vit-untrained` is MorphEm's ViT-S/16 with seeded random weights, frozen
  and cached like the others, so pretraining is the only difference from `morphem`. The same network and weights with
  `trainable: true` (`recipes/encoders/vit-scratch.yml`) are trained with T, f and g. A trained E cannot be cached:
  the data module's `phase_frames` loads every frame's centred crop (the crop the cache encodes, in the clip's view)
  and E runs inside every step, its blocks under activation checkpointing so 2048 frames fit in about 17 GB.
- **Images only where needed.** A batch carries the embeddings of every frame (temporal context, dense regression
  targets) but images of only `images_per_batch` frames, so the UNet's cost per step does not grow with the clip.
- **g is conditioned without breaking the per-frame UNet.** FiLM (`h·(1+γ)+β`) with zero-initialized projections:
  a fresh conditioned g is exactly the unconditioned UNet. ê_t describes the tracked cell at the crop centre, so the
  modulation is weighted by a Gaussian around the centre (`cond_sigma`, 48 px; the nucleus is about 64 px across).
- **Both readouts are scored on the same quantity.** The stain is read out as the median of the predicted image in
  the true nucleus mask, in the units the regression head predicts, so f and g compare directly.

Data: `Bill/5-cellcycle` (`/hpc/projects/virtual_staining/datasets/huang-lab/Bill/5-cellcycle`, see its README): a
spatial split of the 20x QPM PIP-FUCCI NIH3T3 tracks, reporter medians in every tracked nucleus, 256 px crops.

```sh
cd applications/dynaformer/configs/cellcycle
for e in morphem cell-dino dinov3 c-radio vit-untrained; do sbatch run_embed.slurm $e; done   # the caches
./submit.sh --time=0:30:00 -- regression_c-radio.yml stain_c-radio.yml multitask_c-radio.yml
python ../../scripts/summarize_cellcycle.py --split test                                # all runs' scores
```

`experiments/` has every `{regression,stain,multitask}_{morphem,cell-dino,dinov3,c-radio}.yml`, the single-frame
ablations `regression_<encoder>_single_frame.yml` and `stain_c-radio_single_frame.yml` (T without layers),
`stain_c-radio_ungated.yml` (conditioning on every pixel), `stain_unconditioned.yml` (the plain per-frame UNet), and
the same four regression/stain/multitask leaves for the two controls, `*_vit-untrained.yml` (frozen) and
`*_vit-scratch.yml` (trained from scratch).
Results of the 27 runs are in `configs/cellcycle/README.md`: temporal integration doubles track-level correlation
and lifts PIP-FUCCI phase accuracy from about 0.70 (always-G1 scores 0.69) to 0.86-0.88 for every pretrained
encoder, while MorphEm's ViT-S without pretraining reaches 0.65 frozen and 0.79 trained from scratch (0.67 when
trained through the staining loss alone, below the phase-only UNet's 0.76; 0.83 in the stain readout when trained
through both losses). `submit.sh` queues each fit (requeued automatically when its allocation ends) and then its
evaluation, which writes every frame's predictions and the scores of each best checkpoint on the validation and test
splits to `runs/cellcycle/<leaf>/eval/`.

## Testing

```sh
uv run pytest applications/dynaformer
```

Tests build a synthetic timelapse HCS OME-Zarr plus tracking CSVs, so they run on CPU with no external
data. The MorphEm path downloads weights from HuggingFace and is marked `slow` (deselected by default);
run it with `uv run pytest applications/dynaformer -m slow`.
