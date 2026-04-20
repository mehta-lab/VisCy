# `UNeXt2` vs `FullyConvolutionalMAE`: one paper architecture, two PyTorch models

Reconciling the Cytoland paper
([Liu et al., *Nat. Mach. Intell.* 2025, doi:10.1038/s42256-025-01046-2](https://doi.org/10.1038/s42256-025-01046-2))
with the two independent Python classes that claim to implement its
"UNeXt2" architecture. Needed while planning FCMAE-pretrained finetune
runs on `dynacell-models`, where the naming otherwise misleads.

## TL;DR

- The paper (Fig 1b ↔ 1c) describes **one** architecture — "UNeXt2" —
  trained twice: first self-supervised via FCMAE masking, then supervised
  with the pretrained encoder transferred in.
- The code has **two independent Python classes** claiming to implement
  that architecture: `viscy_models.unet.unext2.UNeXt2` (timm-backed) and
  `viscy_models.unet.fcmae.FullyConvolutionalMAE` (custom masked
  re-implementation). They have **incompatible state_dicts** AND
  **structurally different models** — verified below by parameter count.
- The split predates the packaging refactor and predates the `UNeXt2`
  rename. The supervised path started as `viscy/unet/networks/Unet21D.py`
  in August 2023, and the masked FCMAE path was added as
  `viscy/unet/networks/fcmae.py` in April 2024. The key reason for the
  second implementation was masked pre-training: `timm.models.convnext`
  did not expose the per-block masking hooks needed by FCMAE, so Ziwen
  Liu (paper lead author) wrote a standalone masked ConvNeXtV2 encoder.
  Some of the larger architectural divergence we see today is current
  implementation reality, not necessarily the original motivation.
- In the paper's published workflow,
  **`FcmaeUNet(architecture="fcmae")` is used for BOTH the self-supervised
  pretrain AND the supervised finetune** (the `pretraining` boolean
  toggles masking in `forward`). The timm-backed `UNeXt2` class is
  **never** used with FCMAE-pretrained weights.
- The checkpoint matters. The published and current fine-tuning script
  `/hpc/mydata/alex.kalinin/vs_test/finetune_3d.py` loads
  `/hpc/projects/virtual_staining/models/mehta-lab/VSCyto3D/fcmae.ckpt`,
  and that checkpoint **does** load into the current
  `FullyConvolutionalMAE`/`FcmaeUNet` path. The other checkpoint explored
  during planning,
  `/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v1/.../last.ckpt`,
  does **not** load into the current packaged FCMAE class because its
  stem tensor shapes differ.
- **Setting `pretraining=False` on the FCMAE model does not produce the
  same PyTorch model as `UNeXt2`.** They differ in stem (LayerNorm or
  not), head (trainable Conv3d or pure PixelShuffle), num_blocks (6 vs 8),
  total parameter count (32.4M vs 32.1M), and block forward numerics.
  They are the same *conceptual* architecture from the paper's pen-and-
  paper diagram, not the same PyTorch hypothesis class.
- So the currently-running dynacell `unext2.yml` job (timm-backed
  `UNeXt2`) is a valid "from-scratch ConvNeXtV2-tiny baseline" but is
  **not** the apples-to-apples random-init control for a FCMAE-pretrained
  finetune. For a clean comparison, both runs must be
  `FullyConvolutionalMAE(pretraining=False)`.

## What the paper says (Fig 1b ↔ 1c)

One architecture, called **UNeXt2** =
*3D projection stem + 2D encoder + 2D decoder + 3D head*.
Trained twice:

- **1b (FCMAE pretrain):** masked input, reconstruction loss on masked
  regions.
- **1c (virtual-staining supervised):** same net, pretrained encoder
  weights copied in, decoder trained from scratch, phase→fluor regression.

Unambiguous — it's the *same* network, two training regimes.

## What the code actually has

Two independent classes under `packages/viscy-models/src/viscy_models/unet/`:

| | `unext2.py::UNeXt2` | `fcmae.py::FullyConvolutionalMAE` |
|---|---|---|
| Encoder impl | `timm.create_model("convnextv2_tiny", features_only=True)` with `stem_0 → nn.Identity()`, separate `UNeXt2Stem` prepended | Custom `MaskedMultiscaleEncoder` built from `MaskedConvNeXtV2Block` + `MaskedAdaptiveProjection` — from-scratch re-implementation of ConvNeXtV2 with masking hooks in every block |
| Stem params | `stem.weight`, `stem_1.weight` | `encoder.stem.conv3d.*`, `encoder.stem.conv2d.*`, `encoder.stem.norm.*` |
| Block params | `encoder_stages.stages_0.blocks.0.conv_dw.weight`, `.norm.weight` | `encoder.stages.0.blocks.0.dwconv.weight`, `.layernorm.weight` |
| Masking hook | none — inference only | `unmasked: BoolTensor \| None` kwarg threaded through every block's `forward` |
| State_dict interchange | — | **Not compatible.** No adapter exists in the codebase. |

## Why `pretraining=False` does **not** collapse the gap

The natural intuition is that `FullyConvolutionalMAE(pretraining=False)`
with `mask_ratio=0.0, unmasked=None` degenerates to a plain ConvNeXtV2
forward pass and should therefore be structurally equivalent to `UNeXt2`
(both wrap ConvNeXtV2-tiny). Probing both classes at matching config
(`backbone=convnextv2_tiny, in_stack_depth=15, stem_kernel_size=[5,4,4],
decoder_conv_blocks=2, in_channels=1, out_channels=1, drop_path_rate=0.1`)
shows that is not the case:

```
UNeXt2                     total params: 32,426,277   num_blocks: 6
FullyConvolutionalMAE(p=F) total params: 32,148,528   num_blocks: 8
  delta: -277,749 (-0.86%)

UNeXt2 children                      FCMAE(p=F) children
  encoder_stages: 27,860,256           encoder:  27,857,856   (stem folded in)
  stem:                2,592           decoder:   4,290,672
  decoder:         4,561,616           head:             0
  head:                1,813           (no separate stem module)

UNeXt2 stem has LayerNorm?   False
FCMAE encoder.stem has norm? True
```

Concrete structural differences that survive `unmasked=None`:

1. **Stem normalization.** `MaskedAdaptiveProjection` applies
   `nn.LayerNorm(out_channels)` after the 3D→channels projection.
   `UNeXt2Stem` is just `Conv3d + reshape` with no normalization. The
   first activations handed to stage 0 have different statistics in the
   two classes.

2. **Head is structurally different.** `UNeXt2.head` is
   `PixelToVoxelHead` = `UpSample(pixelshuffle) + Conv3d + icnr_init +
   PixelShuffle` (1,813 trainable params).
   `FullyConvolutionalMAE.head` defaults to `PixelToVoxelShuffleHead` =
   a pure `UpSample(pixelshuffle)` (**0 trainable params**) and pushes
   all channel math into the decoder's last stage. Not the same output
   pathway. `FullyConvolutionalMAE(head_conv=True, ...)` would select
   `PixelToVoxelHead` but with different channel wiring than `UNeXt2`.

3. **`num_blocks` differs (6 vs 8).** Consumed by
   `DynacellUNet._make_divisible_pad` / `VSUNet._make_divisible_pad` to
   require input spatial dims divisible by `2**num_blocks`. UNeXt2 needs
   multiples of 64; FCMAE needs multiples of 256. A YX patch size that
   validates for one will not necessarily validate for the other.

4. **Block forward numerics diverge.** `MaskedConvNeXtV2Block.forward` is
   `shortcut → dwconv → masked_patchify(x, unmasked=None) (flatten to
   BLC) → LayerNorm on channels-last → GlobalResponseNormMlp(unsqueeze→
   squeeze) → masked_unpatchify (reshape back to BCHW) → drop_path +
   shortcut`. Timm's `ConvNeXtV2Block.forward` is `shortcut → conv_dw →
   norm (as LayerNorm2d in channels-first, or permute-for-channels-last
   if `use_conv_mlp`) → mlp → gamma-scale (LayerScale when
   `ls_init_value` is set) → drop_path + shortcut`. The masked block
   always pays the patchify↔unpatchify reshape even in the no-mask case;
   timm stays channels-first throughout; the LayerScale `gamma`
   parameter is present in timm and absent in the masked block. Given
   identical parameter tensors the two forward passes would not produce
   bit-identical outputs.

5. **Parameter count delta of 277,749 is structural, not initialization
   noise.** Sources: the stem LayerNorm (+2 params), the head/decoder
   partition difference (UNeXt2 head 1,813 + decoder 4,561,616 = 4,563,429
   vs FCMAE head 0 + decoder 4,290,672 = 4,290,672, delta 272,757 in the
   decoder-plus-head block), and the block-level presence/absence of the
   LayerScale `gamma` parameter.

Conclusion: these are the same *conceptual* architecture from Fig 1 but
not the same PyTorch hypothesis class. Training one from scratch does
not yield an equivalent starting point to training the other from
scratch — different parameter sets, different normalization pathways,
different forward numerics.

## Archaeology: why two on pre-refactor `main`

History on `origin/main` (all commits by Ziwen Liu, paper's lead author):

| SHA | Date | PR | Change |
|---|---|---|---|
| `b4ec13c` | 2023-08-30 | #37 | `viscy/unet/networks/Unet21D.py` introduced — supervised ConvNeXt-backed virtual-staining model with custom 3D stem and 3D head. This is the ancestor of today's `UNeXt2` class. |
| **`0536d29`** | **2024-04-08** | **#67** | **`viscy/unet/networks/fcmae.py` added as a new file**, commit titled "Masked autoencoder pre-training for virtual staining models". Squashed commit text explicitly shows the new masked encoder work: `draft fcmae encoder` → `add stem to the encoder` → `wip: masked stem layernorm` → `wip: patchify masked features for linear` → `use mlp from timm`. This was a new implementation, not a refactor of `Unet21D.py`. |
| `9a0fe64` | 2024-06-11 | #84 | `viscy/unet/networks/Unet21D.py` → `viscy/unet/networks/unext2.py`; class lineage rebranded to `UNeXt2`. `fcmae.py` remained a separate file. |

**Why a standalone class instead of reusing Unet21D / UNeXt2?**
`timm.models.convnext.ConvNeXtBlock` has no per-block mask argument —
its `forward` computes `dwconv → norm → mlp → residual` with no hooks
for zeroing out masked activations or for sparse-gradient propagation.
FCMAE requires all three: masked dwconv input,
`masked_patchify`/`masked_unpatchify` around the pointwise MLP (so the
MLP only runs on visible patches and GRN statistics aren't polluted by
masked zeros), and drop-path/shortcut that skip the masked regions. The
clean path was to write `MaskedConvNeXtV2Block` from scratch with those
hooks baked in; monkey-patching timm's ConvNeXtBlock would have been
fragile across timm upgrades.

**Why didn't the two codepaths converge later?**
There is no evidence that state_dict compatibility between the two
classes was ever a goal. The paper and the published scripts use the
FCMAE-side class for FCMAE pre-train and FCMAE-initialized finetune, and
use the supervised/timm side for scratch supervised baselines. So the
code never needed a translation layer to support the published workflow.
That explains the persistent key mismatch: `UNeXt2` inherits timm-style
naming (`stages_N`, `conv_dw`, `norm`), whereas the masked path uses its
own naming (`stages.N`, `dwconv`, `layernorm`). No adapter or
equivalence tests were added because the two state_dicts were not
expected to cross in production.

## How the paper's own workflow handles the split

The published fine-tuning path as currently exercised by
`/hpc/mydata/alex.kalinin/vs_test/finetune_3d.py` uses
**`FcmaeUNet` for both regimes**:

```python
unet = FcmaeUNet(model_config=dict(
    in_channels=1, out_channels=2,
    encoder_blocks=[3, 3, 9, 3], encoder_drop_path_rate=0.1,
    dims=[96, 192, 384, 768], decoder_conv_blocks=2,
    stem_kernel_size=(5, 4, 4), in_stack_depth=15,
    pretraining=False,        # supervised mode, no masking in forward
))

if encoder_only:
    encoder_weights = {
        k.split("model.encoder.")[1]: v
        for k, v in pretrained["state_dict"].items()
        if "encoder" in k
    }
    unet.model.encoder.load_state_dict(encoder_weights)   # same class, trivial load
```

`FcmaeUNet` wraps `FullyConvolutionalMAE`. The `pretraining` flag inside
`model_config` toggles masking in `forward`:
- `pretraining=True`  → masked input + reconstruction loss (Fig 1b regime)
- `pretraining=False` → no masking + supervised regression loss (Fig 1c regime)

Weight transfer between the two regimes is **trivial** because both
sides are `FullyConvolutionalMAE` — identical parameter names throughout.
No key translation, no adapter needed.

On pre-refactor `main`, the encoder-only transfer lived in *user code*,
inside the fine-tune script, not in the library. The
`encoder_only` / `_load_encoder_weights` helper on
`cytoland.engine.FcmaeUNet` was added later on the modular branch to
formalize that same pattern.

## Implications for our benchmarks

The two Python classes serve distinct roles:

- `FullyConvolutionalMAE` (via `FcmaeUNet`) — the FCMAE pretrain ⇄
  finetune codepath. This is what the paper's Fig 1b/1c workflow uses,
  on both sides.
- `UNeXt2` — from-scratch supervised training *without* FCMAE
  pretraining. Used for baselines / ablations that skip FCMAE entirely.

**"UNeXt2" in the paper refers to the conceptual architecture, not the
Python class of the same name.** The Python class `UNeXt2` has never
been used with FCMAE-pretrained weights in any checked-in script or
benchmark — not on main, not on this branch, not in the published
artifacts.

Dynacell's currently-running from-scratch job
(`benchmarks/virtual_staining/train/er/ipsc_confocal/unext2.yml`, SLURM
31122607) uses `DynacellUNet(architecture="UNeXt2")` — the timm-backed
class. That's a valid "from-scratch baseline with a timm ConvNeXtV2-tiny
encoder," but it trains a structurally different model (stem without
LayerNorm, Conv3d-backed head, 277k extra params, num_blocks=6) from
the FCMAE codepath. It is **not** the apples-to-apples random-init
control for an FCMAE-pretrained-init finetune: it's a different
hypothesis class that happens to share the paper's conceptual name. A
paper-faithful comparison requires both runs to use
`FullyConvolutionalMAE(pretraining=False)`.

### Recommended benchmark layout for dynacell

Do **not** treat the current `unext2.yml` leaf as the random-init control
for an FCMAE-pretrained run. Keep it, but label it honestly as the
timm-backed supervised UNeXt2 baseline.

For the FCMAE question, add a separate pair of leaves that use the same
class on both sides:

- `fcmae_vscyto3d_scratch`
- `fcmae_vscyto3d_pretrained`

Those two leaves should be identical except for encoder initialization:

- same `FullyConvolutionalMAE(pretraining=False)` / `FcmaeUNet`-style model
- same decoder config
- same LR / batch / crops / epochs
- only `encoder_only + ckpt_path` differs

Use the compatible checkpoint from the latest fine-tuning script:

- `/hpc/projects/virtual_staining/models/mehta-lab/VSCyto3D/fcmae.ckpt`

Do **not** use the incompatible checkpoint:

- `/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v1/lightning_logs/pretrain-neuro-aic-hek-200ep_maxsize_fry1_resume4/checkpoints/last.ckpt`

### Alternative paths

1. **Use `FullyConvolutionalMAE(pretraining=False)` for both the
   random-init and FCMAE-pretrained-init leaves** (retire the
   timm-backed `unext2.yml` leaf, or re-frame it as a separate
   baseline). Paper-faithful. The only axis of comparison between the
   two new leaves is the encoder init.
2. **Keep the existing timm-backed `unext2.yml` as an informal baseline**,
   add a `FullyConvolutionalMAE(pretraining=False)` FCMAE-finetune leaf
   on the side. Comparison has an architecture asterisk — same paper
   concept, structurally different PyTorch models (param count, stem,
   head, num_blocks).
3. **Unify the two classes in `viscy-models`** (replace `UNeXt2`'s timm
   encoder with a shared backbone that supports optional masking, or
   make the timm encoder's state_dict transformable to FCMAE naming via
   a one-shot adapter). Clean but a separate `viscy-models` PR.
