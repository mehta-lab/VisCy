# Dynaformer — notes for Claude Code sessions

## Two properties that carry the results

Both are cheap to break and neither shows up in a shape assertion or a loss curve.
`tests/test_models.py` pins both — if you change the decoders or the temporal module, keep those tests passing.

### 1. Zero-initialized temporal conditioning

`ZeroInitConditioning` projects the temporally-mixed token into the decoder with weights and bias
initialized to zero, so a freshly built conditioned model is **bit-identical** to the unconditioned
per-frame model. Training then only ever adds temporal corrections on top of something that already
works.

In the study this application generalizes, adding exactly this edge moved the image readout from
0.303 to 0.581 per-track Pearson r. A decoder that reconstructs each frame from per-frame features
alone produces a trajectory that jitters between neighboring frames — the images look fine, the
trajectory is useless.

Do not "simplify" the zero-init to a standard initialization, and do not add a `LayerNorm` or
non-zero bias downstream of it.

### 2. Rotary position embedding, no learned positional parameter

`RoPETransformerEncoder` has no length-dependent weights. That is what lets the model train on
variable-length clips and run a whole track in one forward pass at inference. Swapping in a learned
positional embedding would silently pin the model to one clip length and break
`TrackTrajectoryMetrics`, which feeds whole tracks.

Attention is **bidirectional** — no causal mask. Each frame may use future frames.

## Shape conventions

The model consumes `(B, T, C, Z, Y, X)` and returns `(B, T, C_out, Z_out, Y, X)`. Time is folded
into the batch for the encoder and decoder (`B*T`) and unfolded around the temporal module. Frame
encoders and image decoders never see the time axis.

## The encoder/decoder pairing has a hard constraint

`UNeXt2ImageDecoder` runs `len(feature_channels) - 1` stages at stride 2, and `PixelToVoxelHead`
upsamples by 4. A matching `UNeXt2FrameEncoder` shrinks by `stem_stride * 2 ** (n - 1)`. The two
agree **only when the stem stride is 4**, so the decoder rejects any other value rather than
returning an image of the wrong size. `yx_patch_size` on the encoder must equal the data module's
`final_yx_patch_size`, and must be divisible by 32.

## Data: one augmentation draw per clip

`TrackWindowDataModule.on_after_batch_transfer` folds time into the channel axis before running
transforms, so each `Batched*` transform draws its random parameters **once per clip**. Every frame
of a clip therefore receives the same flip, rotation and crop, which is what keeps the clip
spatially registered. Reshaping to `(B*T, C, ...)` instead would flip frames independently and
destroy that correspondence without failing any shape check.

The object mask is carried through spatial transforms automatically via
`ForegroundMaskSupport.patch_spatial_transforms` — configs list only the image channels.

`norm_meta` is deliberately kept in the batch after transforms: the trajectory readout is nonlinear
(`a / (a + b)`), so it has to be computed in physical units, which means undoing normalization.

## Checkpoint selection

Monitor `val/track_pearson/image` (mode `max`), not `loss/validate`. The image loss rewards per-frame
reconstruction, which a temporally blind model can win while producing a jittering trajectory.
Whenever a fit config sets that monitor, it must also register `TrackTrajectoryMetrics` — otherwise
`ModelCheckpoint` watches a metric nothing produces. `test_training_integration.py` checks this.

Remember that `base:` composition **replaces** lists rather than merging them, so a leaf that
re-lists `trainer.callbacks` must repeat every callback it still wants.

## The pretrained backbone needs a compatibility shim

`CaicedoLab/MorphEm` ships remote code written for transformers 4.x. Under transformers 5 the plain
`AutoModel.from_pretrained` path fails on `all_tied_weights_keys`, which `PreTrainedModel.post_init`
populates and the published code never calls. `viscy_models.foundation.morphem._load_backbone`
resolves the class first and supplies the empty mapping. If the checkpoint is ever republished, the
shim becomes a no-op — an instance attribute from a real `post_init` takes precedence.

Tests touching MorphEm are marked `slow` (they download weights) and are deselected by the root
`addopts`. CPU tests default to the UNeXt2 encoder.

## What has not been verified

No training run has been done in this codebase — the tracked-cell dataset the reference
hyperparameters came from is not mounted here. The configs encode published values; the numbers
themselves have not been reproduced. Say so rather than implying otherwise.

## Cell-cycle models (`TemporalEmbeddingModel`)

`tests/test_cellcycle.py` pins the invariants below; keep it passing.

- **A clip's embeddings and its images share one view.** The embedding cache holds E's output under the 8
  rotations/flips of `cellcycle_data.d4_view`; the data module draws one view per clip and applies it to the clip's
  images too. `embed.py` (torch, GPU) and the data loader (NumPy) both call `d4_view`, and the test checks they agree.
  Changing the view convention in one place silently decouples images from embeddings.
- **FiLM in `ConditionedUNet` is zero-initialized**, like `ZeroInitConditioning`: a fresh conditioned UNet equals
  the unconditioned one bit for bit. Do not initialize the projections otherwise.
- **Every training track must reach the model.** Training draws one clip per track and epoch, its length rounded
  down to a multiple of `length_quantum`, and keeps partial batches. The first version bucketed clips by exact length
  with `drop_last=True`: most lengths had fewer tracks than a batch holds, so 70% of the training tracks (everything
  shorter than the 64-frame window) were silently never trained on, and the models fit almost only cells present for
  the whole movie. Nothing failed; single-frame models just generalized worse than a linear probe on the same
  embeddings. `test_every_training_track_gives_one_clip_and_every_clip_is_batched` pins this at the real scale.
- **A trained encoder sees exactly what the cache encodes.** `phase_frames` loads every frame's centred crop in the
  clip's view, the input `embed.py` caches; `test_phase_frames_are_what_the_cache_encodes` compares the two. A
  trainable `FoundationFrameEncoder` keeps its weights in checkpoints (a frozen one is dropped and rebuilt), and
  `vit-untrained` builds the same seeded weights frozen or trainable — **on one CPU type**. Torch's seeded init draws
  different weights on the Intel GPU nodes than on the AMD nodes, so the NIH3T3 `vit-untrained` cache
  (embedded on an AMD node) and the `vit-scratch` runs (started on Intel nodes) are two draws of the same
  initialization, not one. Anything that must reuse random weights across jobs loads them from a file, as
  `leger_embed` does with `vit-untrained.weights.pt`.
- **A Leger et al. cache is only as good as its row alignment.** `leger_embed` writes the RPE1 caches in chunks
  against `index.csv` and records its fingerprint; `LegerDataModule` refuses a cache without its finalized `.json` or
  written against another index. The engine takes reporter names, scales and phase thresholds from the data module,
  so the RPE1 targets (Geminin, Cdt1) skip the PIP-FUCCI phase scores. `tests/test_cellcycle_leger.py` pins both, and
  that the Leger leaves train exactly the NIH3T3 regression model.
- **The input LayerNorm removes constant shifts of an embedding**, so a perturbation test must perturb with noise.
- **The stain readout is the median of the prediction inside the true nucleus mask**, in the regression head's
  units; validation and test score both readouts with the same `score_frames`.
- **PIP-FUCCI phases**: truth uses the strict `pipfucci_qc` rule (unclear frames left out); predictions a forced
  choice with the same S / G2/M cut (PIP at its off level), so a perfect prediction scores 1.
- **Run on the VAST environment** (`/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer`, built from the same lock with
  `UV_PROJECT_ENVIRONMENT=... uv sync --all-packages --all-extras --frozen`). Importing torch from the comp.micro
  `.venv` takes minutes on compute nodes (file-system metadata latency); the SLURM scripts default to the VAST copy.
- **The cluster schedules short jobs first** (backfill). `run_fit.slurm` requeues itself through Lightning's SLURM
  hooks (`--signal=SIGUSR1@180 --requeue`), so a 1-2 h allocation is enough for any run; evaluation is a separate
  dependent job so a requeue can never land in the middle of it.
