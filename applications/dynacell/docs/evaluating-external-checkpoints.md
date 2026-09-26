# Evaluating an external model on the DynaCell benchmark

How to score a **third-party model** (yours or a collaborator's) on the two DynaCell
virtual-staining test sets (iPSC confocal + A549 mantis) and get the same metrics the
paper matrix reports — pixel, feature (FID/KID/precision-recall + deep embeddings), and
optional instance-AP.

**Scope of this guide:** the model is a **PyTorch** `nn.Module` (not TF/JAX), likely
**not** a Lightning module, and you are **not** adding new data — you predict on the
existing DynaCell test stores and run metrics. If your model can't be expressed as a
torch module on `(B, 1, Z, Y, X)` grids, jump to [Appendix B](#appendix-b--pure-torch-no-lightning-at-all).
To **fine-tune on DynaCell train data first**, see [§8](#8-fine-tuning-on-dynacell-train-data).

---

## 0. Mental model: two steps, two contracts

The eval pipeline **scores prediction zarrs — it never runs your model.** So every route
is two steps:

```
your .ckpt ──(predict)──▶ prediction.zarr ──(dynacell evaluate)──▶ metrics
```

You only ever have to satisfy two contracts:

1. **Prediction-zarr contract** (what `dynacell evaluate` reads): an HCS OME-Zarr whose
   position names match the GT store, with a channel named `<target>_prediction`.
2. **Predict-writer contract** (only if you reuse `HCSPredictionWriter`, recommended):
   your `predict_step` returns a `(B, C, Z, Y, X)` tensor; the DataModule supplies the
   `batch["index"]` FOV triplet. Reusing the writer guarantees contract #1 for free
   (correct position names, channel naming, pixel-size, and 2.5D Z-blending).

GT path, cell-segmentation path, GT channel name, and pixel spacing are **all supplied
automatically** by the dataset manifest — you never hand-copy them (see Step 2).

---

## 1. Prerequisites

### Recreate the environment

VisCy is a `uv` workspace. Build the venv from the repo root with **all workspace
packages and extras** — that pulls the dynacell `eval` + `eval_gpu` extras (cellpose≥4.2 with dinov3 for cpdino,
`cubic` from git, `torch-fidelity` w/ MIND, and the CUDA-13
`cupy-cuda13x`/`cucim-cu13` GPU stack that matches torch 2.12 cu13):

```sh
git clone https://github.com/mehta-lab/VisCy.git && cd VisCy
git checkout dynacell-paths-module     # branch carrying the dynacell eval pipeline
uv venv -p 3.13                        # 3.11 / 3.12 also fine
uv sync --all-packages --all-extras    # local install of all workspace packages + extras
```

Run **everything below via `uv run …`** so it uses this `.venv`. `cubic` (GPU) is a hard
dep of the eval pipeline — no CPU fallback.

### Other prerequisites

- **DINOv3 is a gated HF model.** Feature metrics load it from the team HF cache
  (`HF_HUB_CACHE` defaults to the shared project cache; override with
  `DYNACELL_SHARED_HF_CACHE`). You need a HF token with access, or set
  `compute_feature_metrics=false` for pixel-only.
- **Instance-AP (`compute_instance_ap=true`) with the `cpdino` backend** needs a separate
  `cpdino-eval` venv — the cpdino Cellpose-DINO stack doesn't co-install cleanly with the
  main eval `.venv`. Pixel + feature metrics run in the `.venv` above; see §5.
- GT feature caches and cell-segmentation stores for **both** test sets are already
  warmed and the manifests are repointed to them — nothing to precompute.

---

## 2. The two test sets (all values come from the manifest)

You never set these by hand; they are here for reference. Eval resolves them from the
manifest via the `dataset_ref`.

### iPSC confocal — `dataset: aics-hipsc`, spacing `[0.290, 0.108, 0.108]` (z,y,x µm)

| organelle | `dataset_ref.target` | GT channel | test store |
|---|---|---|---|
| ER | `sec61b` | `Structure` | `ipsc/dataset_v4/test_cropped/SEC61B.zarr` |
| Mito | `tomm20` | `Structure` | `ipsc/dataset_v4/test_cropped/TOMM20.zarr` |
| Nucleus | `nucleus` | `Nuclei` | `ipsc/dataset_v4/test_cropped/cell.zarr` |
| Membrane | `membrane` | `Membrane` | `ipsc/dataset_v4/test_cropped/cell.zarr` |

### A549 mantis — `dataset: a549-mantis-<gene>-<cond>`, spacing `[0.174, 0.1494, 0.1494]`

12 cells = 4 organelles × `<cond>` ∈ {`mock`, `denv`, `zikv`}. Test stores at
`a549/mantis/test/` (regenerated, 640×960 raw). **Note the key asymmetry**: A549
nucleus/membrane are keyed by **gene** (`h2b`/`caax`), not by organelle.

| organelle | `dataset_ref.target` | GT channel | test store (`_<cond>`) |
|---|---|---|---|
| ER | `sec61b` | `Structure` | `SEC61B_<cond>.zarr` |
| Mito | `tomm20` | `Structure` | `TOMM20_<cond>.zarr` |
| Nucleus | `h2b` | `Nuclei` | `dual_nucl_memb_<cond>.zarr` |
| Membrane | `caax` | `Membrane` | `dual_nucl_memb_<cond>.zarr` |

The `(dataset, target)` pairs above are the single source of truth used by **both** the
predict driver (via the resolver) and the eval driver (via `dataset_ref`).

---

## 3. Step 1 — Predict (thin Lightning shim, ~15 lines)

Wrap your torch module in a minimal `LightningModule` so you inherit `HCSDataModule`
(reads Phase3D, tiles, normalizes, emits the FOV `index`) and `HCSPredictionWriter`
(writes the contract-correct zarr). **Your model does not need to be Lightning** — the
shim only wraps its `forward`.

```python
# predict_external.py  — run:  uv run python predict_external.py
from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from viscy_data.hcs import HCSDataModule
from viscy_transforms import NormalizeSampled
from viscy_utils.callbacks import HCSPredictionWriter
from viscy_utils.trainer import VisCyTrainer

from dynacell.data.resolver import dataset_ref_from_dict, resolve_dataset_ref

# ─── EDIT THESE FOUR THINGS ────────────────────────────────────────────────
OUT_ROOT = Path("/hpc/projects/.../my_model/predictions")   # your own dir, NOT the campaign tree
MODEL_ORGANELLES = ["nucleus", "membrane"]                   # which organelle(s) your model predicts
Z_WINDOW = 15                                                # axial context your model expects (1 for 2D)
FULL_IMAGE = True                                            # False -> tile fixed-size nets (see note)

def load_model() -> torch.nn.Module:
    """Load YOUR torch model with weights, in eval mode."""
    model = MyNet(...)                                       # <- your architecture
    model.load_state_dict(torch.load("/path/to/your.ckpt")["state_dict"])
    return model.eval()
# ───────────────────────────────────────────────────────────────────────────

class TorchShim(LightningModule):
    """Adapts a plain torch nn.Module to the predict-writer contract."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        source = batch["source"]                            # (B, 1, Z, Y, X)
        if FULL_IMAGE:
            pred = self.model(source)                       # FCN handles the full FOV
        else:
            # fixed-size net: overlap-averaging sliding window over 512² tiles
            from dynacell.engine import _sliding_window_inference
            pred = _sliding_window_inference(self.model, source, (Z_WINDOW, 512, 512), overlap_size=32)
        return pred                                         # MUST be (B, C, Z, Y, X), C == #targets

# organelle -> the (dataset, target) cells to predict on both test sets
CELLS = {
    "er":       [("aics-hipsc", "sec61b")] + [(f"a549-mantis-sec61b-{c}", "sec61b") for c in ("mock", "denv", "zikv")],
    "mito":     [("aics-hipsc", "tomm20")] + [(f"a549-mantis-tomm20-{c}", "tomm20") for c in ("mock", "denv", "zikv")],
    "nucleus":  [("aics-hipsc", "nucleus")] + [(f"a549-mantis-h2b-{c}", "h2b") for c in ("mock", "denv", "zikv")],
    "membrane": [("aics-hipsc", "membrane")] + [(f"a549-mantis-caax-{c}", "caax") for c in ("mock", "denv", "zikv")],
}

shim = TorchShim(load_model())
for organelle in MODEL_ORGANELLES:
    for dataset, target in CELLS[organelle]:
        r = resolve_dataset_ref(dataset_ref_from_dict({"dataset": dataset, "target": target}))
        out = OUT_ROOT / f"{dataset}__{target}.zarr"        # one pred store per cell
        dm = HCSDataModule(
            data_path=str(r.data_path_test),                # the GT test store (has Phase3D)
            source_channel=r.source_channel,                # "Phase3D"
            target_channel=[r.target_channel],              # "Structure"/"Nuclei"/"Membrane" -> "<name>_prediction"
            z_window_size=Z_WINDOW,
            batch_size=1,
            num_workers=8,
            normalizations=[NormalizeSampled([r.source_channel], level="fov_statistics",
                                             subtrahend="mean", divisor="std")],  # ← must match YOUR training!
        )
        trainer = VisCyTrainer(callbacks=[HCSPredictionWriter(str(out))])
        trainer.predict(model=shim, datamodule=dm, return_predictions=False)
        print(f"wrote {out}  (channel {r.target_channel}_prediction)")
```

**Correctness knobs you MUST get right:**

- **Normalization** — the single biggest footgun. `NormalizeSampled(fov_statistics, …)`
  reads precomputed per-FOV stats already stored in the test zarrs. **Set `subtrahend`/
  `divisor` (and the transform itself) to exactly what your model trained with.** If your
  model expects a different normalization, replace this transform. Wrong normalization
  silently produces garbage metrics.
- **`z_window_size`** — `1` for a 2D model, `k` for 2.5D, your model's full axial input
  for 3D. Determines the Z extent fed per sample.
- **`FULL_IMAGE`** — the A549 test FOVs are **640×960**. If your net is fully
  convolutional, `True` runs the whole FOV. If it's fixed-size (e.g. 512²), set `False`
  to tile (as the campaign's ViT models do); iPSC (512²) is a single tile either way.
- **Output channel name** is `<target_channel>_prediction`, which is exactly what eval's
  `pred_channel_name` defaults to — leave `target_channel` matching the GT channel.
- **Positions align automatically** because you predict *from* the GT store, so the
  writer creates the same FOV names eval matches against.

> Proven `data:` params for the standard architectures live in
> `configs/benchmarks/virtual_staining/_internal/shared/model/model_overlays/*_predict.yml`
> — copy the `z_window_size` / `normalizations` / `predict_method` from the overlay
> closest to your model's dimensionality if unsure.

For SLURM instead of a local GPU, wrap the same logic behind
`tools/submit_benchmark_job.py` with your own predict leaf; for a one-off, the script
above on an interactive GPU is simplest.

### If your model is generative (flow-matching / diffusion)

A flow-matching fine-tune (e.g. on an EvolutionaryScale **Katamari** backbone) is **not a
single-forward regressor** — it produces a sample by integrating an ODE from noise,
conditioned on the phase input. Two consequences:

1. **`predict_step` runs the sampler, not `self.model(source)`.** Return the final sample
   as `(B, C, Z, Y, X)`:

   ```python
   @torch.no_grad()
   def predict_step(self, batch, batch_idx, dataloader_idx=0):
       source = batch["source"]                          # (B, 1, Z, Y, X) conditioning
       pred = self.model.generate(source, num_steps=100)  # your ODE sampler → (B, C, Z, Y, X)
       return pred
   ```

   The repo already ships this pattern: **`DynacellFlowMatching`**
   (`applications/dynacell/src/dynacell/engine.py:480`, wrapping `CELLDiff3DVS` in
   `celldiff_wrapper.py`) dispatches `predict_step` to
   `model.generate` / `generate_sliding_window` / `generate_iterative`. Mirror it — your
   Katamari model just needs to expose a `generate(cond, num_steps)`-style sampler, or you
   inline your sampling loop in `predict_step`.

2. **Everything downstream is identical.** The benchmark **already scores CellDiff, a
   flow-matching model**, on the same metrics — so once you write a conforming
   `prediction.zarr`, **Step 2 (eval) is unchanged.** No pipeline edits.

**Latent / conditioned / non-Lightning samplers → use [Appendix B](#appendix-b--pure-torch-no-lightning-at-all), not the shim.**
If the model isn't a plain phase→image module — it samples in a latent space, conditions
on more than the phase (an identity/target-channel embedding), decodes embeddings to
pixels, or runs under a non-Lightning trainer — don't force it through `HCSDataModule` /
`predict_step`. Run its **native sampler in its own environment** to produce per-FOV pixel
images, then write a conforming `prediction.zarr` (Appendix B) and evaluate. Keep these
aligned so the numbers match the matrix:

- **Predict the exact DynaCell test FOVs and preserve position names** — the pred store's
  `row/col/fov` must equal the GT store's (eval matches by name).
- **Map each organelle to its conditioning and output channel** — condition on the target
  identity for that organelle and write the sample as its GT channel
  (`Nuclei`/`Membrane`/`Structure`) + `_prediction`. For an **identity-conditioned** model,
  confirm the DynaCell target proteins (H2B→nucleus, CAAX→membrane, SEC61B→ER,
  TOMM20→mito) are in the model's conditioning vocabulary — otherwise the run is *zero-shot*
  on those identities, a valid but distinct claim to state explicitly.
- **A 2D model must write Z=48, not Z=1** (verified against the eval code). The pipeline
  assumes `pred.Z == GT.Z == seg.Z == 48` — the in-focus plane is computed on the GT and
  applied to the prediction — so a `Z=1` store raises `ValueError` in *every* metric family
  (`pipeline.py` pixel loop, CP `regionprops`, `focus.py:slab_mip` empty-slice reduction).
  **Replicate your single predicted plane across all 48 Z.** Then the focus-reduced metrics
  — **deep-feature FID/KID/precision-recall and instance-AP** — recover exactly your plane
  and are correct (all planes identical → the focus-plane MIP equals your prediction). ⚠️
  But **pixel metrics (SSIM/PCC/FSC) and CP-regionprops stay full-3D volumetric** — they
  never consult the focus plane — so a flat-replicated pred scored against the real 48-plane
  GT gives *unfair* per-pixel/morphometry numbers. For comparable pixel/CP numbers on a 2D
  model you must either **predict per-Z to fill a true volume** (≈48× the sampling cost) or
  **add focus-plane reduction to the pixel/CP path** (a small pipeline change). This is the
  one place a 2D generative model needs a real decision, not just a writer tweak.
- **Seed the sampler** and record `num_steps` / guidance scale for reproducibility.

**Generative-specific gotchas:**

- **Seed the sampler** for reproducibility (`torch.manual_seed(...)`) and record
  `num_steps` — it trades quality vs wall-time.
- **Sampling is slow** (CellDiff runs ~hours/FOV at 100 steps). Run predict on SLURM via
  `tools/submit_benchmark_job.py`, and use `--resume-predict` if a wall-time kill leaves a
  partially-written store. Resume relies on the writer's completion markers (which record
  the checkpoint and a hash of the prediction settings), so a store written before markers
  existed, or with other weights or settings, is refused: predict it into a new output
  store instead.
- **640×960 A549 exceeds a single flow-matching patch** — use an overlap-anchored tiler
  (`generate_iterative`), not one full-FOV pass (same constraint CellDiff/ViT hit; iPSC
  512² is a single patch, so `FULL_IMAGE`-equivalent there).
- **Reading the metrics:** FID/KID/precision-recall (distribution-level) are the
  meaningful generative-quality metrics; per-pixel SSIM/PCC penalize a stochastic sample's
  non-registration. The benchmark reports all and scores CellDiff on all, so your numbers
  stay directly comparable to the matrix.

---

## 4. Step 2 — Evaluate (manifest-driven, one call per cell)

Point eval at your prediction zarr; the `target=` + `predict_set=` **groups** splice
`gt_path`, `cell_segmentation_path`, `gt_channel_name`, `pred_channel_name`, and
`pixel_metrics.spacing` from the manifest. You supply only `io.pred_path` and
`save.save_dir`.

**iPSC, e.g. nucleus:**
```sh
uv run dynacell evaluate \
  target=nucleus predict_set=ipsc_confocal \
  io.pred_path=/hpc/.../my_model/predictions/aics-hipsc__nucleus.zarr \
  compute_feature_metrics=true \
  save.save_dir=/hpc/.../my_model/eval/ipsc_nucleus
```

**A549 ER, mock:**
```sh
uv run dynacell evaluate \
  target=er_sec61b predict_set=a549_mantis_sec61b_mock \
  io.pred_path=/hpc/.../my_model/predictions/a549-mantis-sec61b-mock__sec61b.zarr \
  compute_feature_metrics=true \
  save.save_dir=/hpc/.../my_model/eval/a549_er_mock
```

**A549 nucleus/membrane need the gene-key override** (the target group defaults to the
organelle name, but the A549 manifest keys by gene):
```sh
uv run dynacell evaluate \
  target=nucleus predict_set=a549_mantis_h2b_mock \
  benchmark.dataset_ref.target=h2b \
  io.pred_path=/hpc/.../my_model/predictions/a549-mantis-h2b-mock__h2b.zarr \
  compute_feature_metrics=true \
  save.save_dir=/hpc/.../my_model/eval/a549_nucleus_mock
```

**Group name reference:**

| organelle | `target=` | iPSC `predict_set=` | A549 `predict_set=` (`_<cond>`) | A549 extra override |
|---|---|---|---|---|
| ER | `er_sec61b` | `ipsc_confocal` | `a549_mantis_sec61b_<cond>` | — |
| Mito | `mito_tomm20` | `ipsc_confocal` | `a549_mantis_tomm20_<cond>` | — |
| Nucleus | `nucleus` | `ipsc_confocal` | `a549_mantis_h2b_<cond>` | `benchmark.dataset_ref.target=h2b` |
| Membrane | `membrane` | `ipsc_confocal` | `a549_mantis_caax_<cond>` | `benchmark.dataset_ref.target=caax` |

A tiny bash loop over your `MODEL_ORGANELLES` × these rows evaluates every cell.
Do **not** use `dynacell evaluate-grouped` / the grouped-eval generators for this — those
walk the campaign's on-disk prediction tree (and the walker fix isn't on `main` yet). One
`dynacell evaluate` call per cell is the robust path for an external model.

---

## 5. What each metric family needs

| Family | Extra input | Flag | Env |
|---|---|---|---|
| **Pixel** (SSIM, PCC, spectral, FSC…) | pred + GT + spacing (auto) | default | repo `.venv` |
| **Feature** (FID/KID/precision-recall + DINOv3/DynaCLR/CELL-DINO/MorphEm embeddings) | labeled cell-seg store (auto from manifest) | `compute_feature_metrics=true` | repo `.venv` + HF token for DINOv3 |
| **Instance-AP** (whole-cell / nucleus mAP) | runs its own segmentation | `compute_instance_ap=true` + `segmentation.backend=cpdino` | needs the **`cpdino-eval`** venv (repo cubic can't run cpdino) |

Notes:
- Feature crops come from the **fixed** `cell_segmentation_path`, so they're independent
  of the instance-AP segmentation — the two segmentations are separate.
- Focus-2D is **on by default** (`slice_selection=focus`, `feature_metrics.focus_slab`)
  and needs a `Phase3D` channel for the focus plane — present in every test store, so
  leave it on. Disable those knobs only if you point at a store without phase.
- No cell-seg store and don't want features? `compute_feature_metrics=false` gives
  pixel + binary-mask metrics only.

---

## 6. Outputs

Each `dynacell evaluate` writes to its `save.save_dir`: per-FOV and aggregate metric CSVs,
plus (when `compute_feature_metrics=true`) the deep-embedding NPZ files. Collect the
`save_dir`s across cells to assemble your comparison table.

---

## 7. Working alongside the active A549 campaign

The A549 regen→retrain→re-eval campaign is mid-flight (re-predict draining, re-eval
pending), so:

- **Write your predictions and evals to your own directory** — never into
  `a549/predictions/`, `a549/evaluations*_with_embeddings/`, or the per-train-dir
  prediction tree. Standalone `io.pred_path` + `save.save_dir` keep you fully insulated.
- **Both test sets are safe to read now**: the manifests are repointed to the
  regenerated 640×960 raw A549 stores + combined `dual_nucl_memb` store, and GT
  caches/seg stores are warmed. iPSC has been stable throughout.
- The pred-side eval cache is geometry-blind — since you write to fresh `io.pred_path`s
  under your own dir, you won't hit stale-cache aliasing.

---

## 8. Fine-tuning on DynaCell train data

To fine-tune your model on DynaCell before benchmarking, use the **train** stores. They are
**physically separate** from the test stores (§2) — built from a fixed split by the
assembler — so training on a full train store never touches an eval FOV. **Never point
training at a `test` / `test_cropped` store**, or the benchmark numbers are invalid.

### Train stores

Resolve them from the same manifest resolver as predict/eval — `data_path_train` instead of
`data_path_test`:

```python
from dynacell.data.resolver import dataset_ref_from_dict, resolve_dataset_ref
r = resolve_dataset_ref(dataset_ref_from_dict({"dataset": "aics-hipsc", "target": "nucleus"}))
r.data_path_train    # train store (disjoint from r.data_path_test)
r.source_channel     # "Phase3D"
r.target_channel     # "Nuclei" / "Membrane" / "Structure" — paired GT in the SAME store
```

| test set | organelle | `(dataset, target)` | train store |
|---|---|---|---|
| iPSC | ER / Mito / Nuc / Memb | `(aics-hipsc, {sec61b,tomm20,nucleus,membrane})` | `ipsc/dataset_v4/train/{SEC61B,TOMM20,cell,cell}.zarr` |
| A549 | ER | `(a549-mantis-sec61b-<cond>, sec61b)` | `a549/mantis/train/SEC61B_all.zarr` |
| A549 | Mito | `(a549-mantis-tomm20-<cond>, tomm20)` | `a549/mantis/train/TOMM20_all.zarr` |
| A549 | Nuc / Memb | `(a549-mantis-{h2b,caax}-<cond>, {h2b,caax})` | `a549/mantis/train/dual_nucl_memb_all.zarr` |

Each store holds **paired** `(Phase3D source, target fluorescence)` per FOV — the training
signal. Notes:

- **A549 train is condition-pooled** — one `_all.zarr` per target combines mock + ZIKV +
  DENV train FOVs, with the condition dropped from the FOV name (`fov0000…`). Any A549
  `<cond>` in the ref resolves to the same pooled train store; test stays per-condition.
- **iPSC** ships a dedicated `train/` vs `test_cropped/` split — the whole `train/` store
  *is* the train split.
- **Geometry differs across sets** — A549 640×960 @ 0.174/0.1494 µm, iPSC 512² @
  0.290/0.108 µm. If you pool both, resample/normalize to a common convention (DynaCell
  models train per-set; matching the target set's voxel size is safest).
- **The split is metadata-fixed** — each manifest target carries `splits/<...>.yaml`
  (`dynacell.data.manifests.load_splits()` → `.train`/`.test`); the assembler already
  applied it to separate the stores, so you don't filter FOVs yourself — that file is just
  the record of what is train vs test.

### Loading for training

- **Lightning trainer**: `HCSDataModule(data_path=r.data_path_train,
  source_channel=r.source_channel, target_channel=[r.target_channel], …)` in fit mode
  yields paired `source`/`target` batches (what the DynaCell models use).
- **Non-Lightning trainer** (e.g. a Zarr-based pipeline): open `r.data_path_train` with
  `iohub.open_ome_zarr` and read the `source_channel` + `target_channel` arrays per
  position directly. Apply your model's own normalization — the stores also carry
  `viscy preprocess` fov-statistics in `.zattrs` if you want to match DynaCell's.
- **Identity-conditioned models**: the `target_channel` per store *is* the protein you're
  learning to stain — H2B (nucleus), CAAX (membrane), SEC61B (ER), TOMM20 (mito). Use these
  as the conditioning identities so fine-tuning and eval (§3) agree.

---

## Appendix A — reusing an existing architecture

If your checkpoint *is* one of the DynaCell engine architectures
(`DynacellUNet`/`DynacellGAN`/celldiff), skip the shim: author a predict leaf
(`base:` the matching `model_overlays/<arch>_predict.yml`), set
`model.init_args.ckpt_path`, and run `tools/submit_benchmark_job.py <leaf> --ckpt best`.
Then evaluate as in Step 2.

## Appendix B — pure torch, no Lightning at all

If your model can't fit `HCSDataModule`'s I/O (different input channels, an external
sampler, non-grid), skip the whole predict stack and satisfy only the prediction-zarr
contract directly with iohub:

```python
from iohub import open_ome_zarr
from dynacell.data.resolver import dataset_ref_from_dict, resolve_dataset_ref

r = resolve_dataset_ref(dataset_ref_from_dict({"dataset": "aics-hipsc", "target": "nucleus"}))
with open_ome_zarr(str(r.data_path_test), mode="r") as gt, \
     open_ome_zarr("pred.zarr", layout="hcs", mode="w-",
                   channel_names=[f"{r.target_channel}_prediction"]) as out:
    src_idx = gt.channel_names.index(r.source_channel)      # Phase3D
    for pos_name, gt_pos in gt.positions():                 # e.g. "0/0/fov0000"
        row, col, fov = pos_name.split("/")
        phase = gt_pos["0"][:, src_idx:src_idx + 1]         # (T, 1, Z, Y, X)
        pred_tczyx = run_your_model(phase)                  # (T, 1, Z, Y, X), Z/Y/X == GT
        out.create_position(row, col, fov).create_image("0", pred_tczyx)
```

Then run Step 2 unchanged — position names mirror the GT store, and the channel is named
`<target>_prediction`, so eval matches by name. (Set `io.pred_channel_name` explicitly if
you name it anything else.)

> **2D model?** `pred_tczyx` must be `Z=48` (match the GT), not `Z=1` — a single-Z store
> fails every metric. Replicate your predicted plane across Z (`np.broadcast_to(plane,
> (T, 1, 48, Y, X))`). Feature + instance-AP metrics then recover your plane correctly;
> pixel/CP metrics stay volumetric — see the Z-handling bullet in §3.
```

---

## Key source references

- Prediction-writer contract: `packages/viscy-utils/src/viscy_utils/callbacks/prediction_writer.py`
  (`on_predict_start` names `<target>_prediction`; `write_sample` maps the `batch["index"]` FOV triplet).
- `predict_step` shape contract: `applications/dynacell/src/dynacell/engine.py:402`
  (`DynacellUNet.predict_step` — pad → forward → center-crop; `_sliding_window_inference` helper).
- Manifest → eval splice: `applications/dynacell/src/dynacell/evaluation/_ref_hook.py`
  (`_RESOLVED_FIELDS`: `io.gt_path`, `io.cell_segmentation_path`, `io.gt_channel_name`,
  `io.gt_cache_dir`, `pred_channel_name`, `pixel_metrics.spacing`).
- Resolver API: `applications/dynacell/src/dynacell/data/resolver.py`
  (`resolve_dataset_ref`, `dataset_ref_from_dict`, `ResolvedDataset`).
- Eval config surface: `applications/dynacell/src/dynacell/evaluation/_configs/eval.yaml`
  (`io.*`, `segmentation.*`, `compute_feature_metrics`, `compute_instance_ap`, `focus.*`).
- Test-set manifests: `applications/dynacell/src/dynacell/_manifests/{aics-hipsc,a549-mantis-*}/manifest.yaml`.
- Target/predict-set groups: `configs/benchmarks/virtual_staining/_internal/shared/eval/target/`,
  `src/dynacell/evaluation/_configs/predict_set/`.
