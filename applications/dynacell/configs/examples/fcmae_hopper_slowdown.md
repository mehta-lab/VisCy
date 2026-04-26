# FCMAE VSCyto3D training on Hopper — ~10× slowdown vs pre-Hopper GPUs

## Summary

Pretrained FCMAE VSCyto3D finetunes for iPSC SEC61B (job 31297032) and TOMM20
(job 31375338) train at roughly ~75 s/step on H200, vs ~5 s/step on L40S under
the same config + pretrained checkpoint. The slowdown is consistent across
Hopper GPUs (H100 and H200) and is **not** fixed by switching precision
(fp16-mixed → bf16-mixed).

## Verdict (2026-04-25) — MS-DSSIM is the dominant slow term

Direct per-step measurements from a Lightning-config probe that drops
the MS-DSSIM term from the loss (T2: `MixedLoss(l1_alpha=1.0,
ms_dssim_alpha=0.0)`), 4 GPUs on H100 (gpu-f-6), `SEC61B_test48.zarr`,
otherwise identical to the prod config:

| Step | data_wait_ms | compute_ms |
| ---: | ---:         | ---:       |
| 0    | 8198         | 16397 *(init / cuDNN autotune)* |
| 1    | 583.7        | 929.3      |
| 2    | 594.7        | 903.7      |
| 3    | 700.0        | 891.2      |
| 4    | 609.2        | 2958.2     |
| 5    | 583.3        | 1416.1     |
| 6    | 560.0        | 935.7      |
| 7    | 555.2        | 1257.7     |
| 8    | 555.8        | 1241.8     |
| 9    | 548.1        | 1870.6     |

Steady-state floor on rank 0 is ~0.9 s/step compute. The same prod config
**with** MS-DSSIM measured **45.5 s/step compute** on identical N=4 H100
hardware (Probe C job 31451180 and T5 job 31451960, 8 and 4 consecutive
steady-state STEP_TIMER lines respectively) — a **~50× delta** attributable
to the MS-DSSIM term.

A symmetric MS-DSSIM-only probe (T4) was queued but cancelled when its
StartTime slipped 8 hours; the deduction is closed by subtraction:
L1-only is fast, L1+MS-DSSIM is slow, therefore MS-DSSIM is the slow
term.

### Fix landed (2026-04-25) — bf16 SSIM helper, 11× speedup confirmed

The fix replaces monai's `compute_ssim_and_cs` with a precision-aware
helper `_compute_ssim_and_cs_bf16` (commit `e42c49a`) and drops the
now-redundant `@torch.amp.custom_fwd(cast_inputs=fp32)` decorator from
`MixedLoss` and `SpotlightLoss` (commit `3a7fa05`). The helper runs the
5 Gaussian-mean convolutions in bf16 (with squared products computed in
fp32 before casting, to preserve squaring precision) and promotes only
the variance subtractions and C₁/C₂-guarded divisions to fp32.

T6 sanity probe (job 31453564, gpu-f-2 H100, N=4, MS-DSSIM enabled,
`SEC61B_test48.zarr`, otherwise identical to T5/Probe C):

| Step | data_wait_ms | compute_ms |
| ---: | ---:         | ---:       |
| 0    | 11894.6      | 15674.9 *(init / cuDNN autotune)* |
| 1    | 542.1        | 4139.3     |
| 2    | 535.8        | 4140.2     |
| 3    | 538.7        | 4143.2     |
| 4    | 548.5        | 4155.6     |
| 5    | 542.4        | 4164.7     |
| 6    | 541.1        | 4166.4     |

Steady-state mean compute_ms: **4151 ms** over 6 consecutive lines
(std ~10 ms). Versus the **45.5 s/step** baseline on identical N=4 H100
hardware with fp32 MS-DSSIM (Probe C `31451180`, T5 `31451960`):
**~10.97× speedup**.

The remaining ~4 s/step is the actual MS-DSSIM compute cost on Hopper
in the bf16 regime — an order of magnitude faster than fp32, but still
meaningfully more than the 0.9 s/step L1-only floor (T2). Further
mitigation (reducing pyramid depth, MS-DSSIM frequency) would attack
that residual; not pursued here since 4 s/step is workable for FCMAE
finetune training.

Validated by tests in `packages/viscy-utils/tests/test_metrics.py`,
`test_mixed_loss.py`, and the extended `test_spotlight_loss.py`. The
contract holds at: per-pixel rtol=5e-2/atol=1e-1 (random), aggregate
rtol=1e-2/atol=1e-2 (random), aggregate rtol=2e-3/atol=5e-3
(correlated-pair), gradient cosine similarity ≥0.99, sign-flip
fraction <1% on `|grad_ref|>1e-3` voxels.

The earlier "Hopper kernel / DDP bucket-view stride mismatch" hypothesis
(see [Earlier diagnosis](#earlier-diagnosis-superseded) below) is wrong as
a cause — the stride warnings are real but not load-bearing for the
slowdown. A plain Lightning + synthetic-data + DDP probe on N=4 H100
(no MS-DSSIM, no real data pipeline) ran at 185.6 ms/step steady-state,
ruling out Lightning, DDP, cuDNN, and the data pipeline as the
bottleneck.

### Why MS-DSSIM is slow on Hopper

The MS-DSSIM 5-level pyramid runs in fp32 — at every level,
`compute_ssim_and_cs` (monai) executes 5 large-kernel convolutions
(μₓ, μᵧ, μₓₓ, μᵧᵧ, μₓᵧ) with a `(D=15, 11, 11)` kernel; the multiscale
wrapper then `avg_pool3d`-downsamples and repeats 5 times. The
conv-heavy core therefore runs in fp32 regardless of any outer
mixed-precision context.

What the **measurement** establishes: with that path skipped (T2),
compute drops from 45.5 s/step → ~0.9 s/step on identical N=4 H100
hardware. So fp32 MS-DSSIM is dominating on Hopper.

What is **inferred but not directly measured**: that the gap is
specifically explained by Hopper's bf16/fp16-vs-fp32 tensor-core
advantage being larger than Ampere's / Ada's. Plausible from
published kernel ratios, but we have not benchmarked the same monai
SSIM kernels across architectures. The cross-architecture explanation
should be treated as the leading theory, not an established fact.

### Where the fp32 cast comes from (two stacked sources)

1. **monai's `compute_ssim_and_cs`** (`monai/metrics/regression.py:402-403`)
   force-casts both inputs to fp32 unconditionally:

   ```python
   y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
   y     = convert_data_type(y,      output_type=torch.Tensor, dtype=torch.float)[0]
   ```

   This is what makes MS-DSSIM slow on Hopper. The 25 conv ops (5 stats ×
   5 pyramid levels) all run with fp32 weights and fp32 inputs.

2. **viscy's `MixedLoss.forward`** (`packages/viscy-utils/.../mixed_loss.py:43`)
   adds an outer `@torch.amp.custom_fwd(cast_inputs=torch.float32)`.
   Introduced in commit `b4ec13c` (PR
   [#37](https://github.com/mehta-lab/VisCy/pull/37), 2023-08-30), which
   focused on the pixelshuffle decoder — the cast came in alongside
   without a justifying note in the PR review. The mechanic of
   `custom_fwd(cast_inputs=...)` is "cast the inputs and run the
   forward with autocast disabled," so the decorator really does
   create an fp32 island when active. The pattern is the well-known
   "force MS-SSIM to fp32 to avoid NaN under autocast" workaround
   (cf. torchmetrics issue
   [#2281](https://github.com/Lightning-AI/torchmetrics/issues/2281):
   `σ² = E[X²] − μ²` subtraction produces tiny negative values from
   float deviations, the C₁/C₂ stability constants don't cover them,
   resulting in NaN; later fixed upstream). This is corroborated by
   the `clamp=True` flag in our `ms_ssim_25d`, documented as "for
   training stability when used in loss" — author was already fighting
   numerical instability.

   Today this outer cast is **largely redundant**: the conv-heavy core
   inside monai is already pinned to fp32, so removing the `@custom_fwd`
   decorator only affects `F.l1_loss`, `F.mse_loss`, and the outer
   `F.avg_pool3d` downsamplings — none of which are the slow path.
   Removing it alone does **not** unblock Hopper.

### Mitigation options (revised)

The bottleneck is monai's internal fp32 cast on the SSIM convs, not
viscy's outer decorator. Practical fallback order:

1. **Local-patch monai's `compute_ssim_and_cs` with a mixed-precision
   variant** — keep variance-sensitive math in fp32, run convs/pooling
   in bf16:
   - convs (μₓ, μᵧ, μₓₓ, μᵧᵧ, μₓᵧ) and `avg_pool3d` between levels:
     **bf16** (kernels and inputs both)
   - `mu_xx − mu_x*mu_x`, `mu_yy − mu_y*mu_y`, `mu_xy − mu_x*mu_y`:
     **fp32**
   - C₁/C₂-guarded divisions for `contrast_sensitivity` and `ssim`:
     **fp32**

   bf16 keeps fp32's 8-bit exponent (vs fp16's 5-bit), so it is the
   right candidate for SSIM's near-equal subtraction; fp16 should be
   avoided. Validate numerical equivalence against the current fp32
   path on a representative batch before training.

2. **Reduce MS-DSSIM frequency** — e.g. apply the MS-DSSIM term every
   N steps and L1-only on the others. No precision changes; degrades
   loss signal but doesn't risk numerical regression.

3. **Drop MS-DSSIM entirely** on Hopper finetune runs. Largest
   behavior change; should be backed by parity training runs against
   the L1+MS-DSSIM baseline.

### Fastest confirmation experiment

Patch `compute_ssim_and_cs` so that:

- it does **not** immediately cast `y_pred` and `y` to fp32,
- the convs run under autocast (bf16 on Hopper),
- only the variance subtraction and C₁/C₂-guarded divisions are
  explicitly promoted to fp32.

Re-run the T2-style sanity probe with MS-DSSIM **enabled** and this
patched path. If step time collapses from ~45 s toward the L1-only
~0.9 s regime, that load-bearing identification is confirmed.

> **Note:** dropping only the `@torch.amp.custom_fwd` decorator from
> `MixedLoss.forward` (without touching monai's internal cast) will
> **not** restore Hopper throughput — the 25 fp32 convs in the pyramid
> remain. This was an earlier mitigation suggestion that I retracted
> after reading monai's source.

## Throughput measurements

All rows below use the same `fcmae.ckpt`-warm-started FCMAE VSCyto3D model,
`ddp_find_unused_parameters_true`, 4 GPUs, `z=15, yx=256`, `num_samples=4`,
`mmap_preload=true`, `scratch_dir=/dev/shm`.

### Pretrained finetune sanity probes across architectures

All probes use `SEC61B_test48.zarr` (48 FOVs), pretrained FCMAE VSCyto3D
warm-start, 4 GPUs, fp16-mixed unless noted, 3 epochs.

| GPU      | Arch         | Compute cap | Precision  | Node     | RAM    | /dev/shm | s/step | Source |
| ---      | ---          | ---         | ---        | ---      | ---:   | ---:     | ---:   | --- |
| A40      | Ampere       | sm_86       | fp16-mixed | gpu-c-1  | 2.0 TB | 1002 GB  | **2.80** | sanity 31406782 (86 steps / 241 s) |
| A6000    | Ampere       | sm_86       | fp16-mixed | gpu-b-3  | 0.5 TB | 252 GB   | **3.56** | sanity 31406785 (86 steps / 307 s) |
| L40S     | Ada Lovelace | sm_89       | fp16-mixed | gpu-g-2  | 1.16 TB| —        | **5.1**  | earlier sanity (80 steps / 355 s) |
| A100-40  | Ampere       | sm_80       | fp16-mixed | gpu-a-3  | 2.04 TB| —        | —      | sanity 31406783 NCCL BROADCAST timeout in DDP setup |
| A100-80  | Ampere       | sm_80       | fp16-mixed | gpu-d-2  | 2.0 TB | —        | —      | sanity 31406784 NCCL BROADCAST timeout in DDP setup |
| H100     | Hopper       | sm_90       | fp16-mixed | gpu-f-3  | 2.0 TB | —        | **47.6** | sanity 31400433 (20 steps / 951 s) |
| H200     | Hopper       | sm_90       | bf16-mixed | gpu-h-3  | 2.06 TB| —        | **65.8** | sanity 31400431 (10 steps / 658 s) |
| H200     | Hopper       | sm_90       | fp16-mixed | gpu-h-5  | 2.06 TB| —        | **~75**  | prod 31297032 (SEC61B, OOM after 60 h) |

**The architecture split is sharp:** every pre-Hopper GPU runs at 2.8–5.1
s/step. Every Hopper run we have measured (H100 fp16, H200 fp16, H200 bf16,
scratch H100 fp16, scratch H100 bf16) lands in 46–75 s/step — a **13–27×
slowdown** across the Hopper boundary regardless of precision, warm-start,
or which Hopper SKU.

The two A100 attempts both NCCL-timed out during a 32 M-element BROADCAST
in DDP setup before any training step. That number matches the FCMAE
encoder param count (32.1 M), so the symptom is consistent with rank 0
being slow to load `fcmae.ckpt` (or otherwise blocked on rank-0-only I/O)
while ranks 1–3 sat at the collective and the 30-min watchdog fired. This
is an I/O coordination problem on those A100 nodes' shared-storage path,
not a hardware fault — and not informative for the Hopper-vs-Ampere
question. (Separate follow-up: rerun A100 sanity with rank 0 staging
`fcmae.ckpt` before the DDP barrier, or measure raw read bandwidth from
gpu-a-3 / gpu-d-2 to `/hpc/projects/virtual_staining`.)

### Scratch-vs-pretrained × fp16-vs-bf16 controlled matrix (H100)

To rule out the warm-start `ckpt_path` + `encoder_only: true` path and
precision as the cause, we ran the same sanity harness with random init
(no `ckpt_path`, no `encoder_only`) across both precisions on identical
H100 hardware.

| Init       | Precision   | GPU  | s/step | Source |
| ---        | ---         | ---  | ---:   | --- |
| Scratch    | fp16-mixed  | H100 | 46.57  | sanity job 31402627 (step 9→19 / 466 s) |
| Scratch    | bf16-mixed  | H100 | 46.33  | sanity job 31402692 (step 9→19 / 463 s) |
| Pretrained | fp16-mixed  | H100 | 47.60  | sanity job 31400433 (step 9→29 / 951 s) |
| Pretrained | bf16-mixed  | H200 | 65.80  | sanity job 31400431 (step 9→19 / 658 s) |

All four Hopper runs cluster in 46–66 s/step vs L40S 5.1 s/step. The
slowdown is invariant to:

1. Precision (fp16 ↔ bf16: 46.57 vs 46.33 on scratch — no difference).
2. Warm-start (scratch ↔ pretrained on fp16: 46.57 vs 47.60 — no
   difference).
3. Hopper generation (H100 vs H200: both in the same band).

**No config knob fixes this.** The slowdown is intrinsic to the FCMAE
ConvNeXt graph hitting a slow Hopper kernel path.

### Scratch FCMAE (prod runs, same architecture)

| Dataset | GPU | Arch | Node | s/step | Notes |
| --- | --- | --- | --- | ---: | --- |
| SEC61B  | A40  | Ampere       | gpu-c-1 | 2.99 | 30 089 steps / 89 965 s (run 20260421-112347) |
| TOMM20  | L40S | Ada Lovelace | gpu-g-2 | 4.91 | 15 599 steps / 76 641 s (run 20260422-060655) |
| TOMM20  | H200 | Hopper       | gpu-h-2 | —    | failed after 4 s (missing ckpt path) |
| SEC61B  | H200 | Hopper       | gpu-h-1 | —    | failed after 112 s (find_unused_parameters) |
| SEC61B  | H200 | Hopper       | gpu-h-2 | —    | failed after ~46 min (missing resume ckpt) |

**No FCMAE run (scratch or pretrained) has successfully reached steady-state
training throughput on Hopper.** The "scratch ran fine" runs were all on
pre-Hopper hardware (A40, L40S, A100 attempts).

<a id="earlier-diagnosis-superseded"></a>
## Earlier diagnosis (superseded by 2026-04-25 verdict above)

The notes below were the working hypothesis before the L1-only probe
showed compute time collapses 50× when MS-DSSIM is removed. They are
left for the record — the cuDNN/DDP-stride warnings are real, but they
are not the cause of the slowdown.

- `py-spy dump` on live H200 rank 0 (prod SEC61B, job 31297032) pinned the
  MainThread inside `_engine_run_backward` (`torch/autograd/graph.py:865`)
  across 3 consecutive snapshots. DataLoader workers (`pt_data_worker`),
  pin-memory loop, and wandb threads were all idle. → **bottleneck is
  GPU-side backward(), not data loading.** (Consistent with verdict —
  MS-DSSIM has a heavy backward.)
- Hopper stderr consistently emits DDP warnings that don't appear on L40S:
  - `AccumulateGrad node's stream does not match the stream of the node that
    produced the incoming gradient` (pointing at DDP + stream ordering).
  - `Grad strides do not match bucket view strides ... grad.sizes() = [240, 960, 1, 1],
    strides() = [960, 1, 960, 960] vs bucket_view ... [960, 1, 1, 1]`
    (pointing at a specific layer whose weight grad memory format breaks DDP's
    bucket view contract on Hopper).
- Switching `precision: bf16-mixed` on H200 changed throughput from ~75 → 65.8
  s/step — basically the same order of magnitude. **Precision is not the
  cause.**

Earlier working hypothesis (now wrong): a specific kernel/layer (likely a
pointwise 1×1 conv in the FCMAE ConvNeXt encoder given the `[240, 960, 1, 1]`
weight shape) hits a slow Hopper path, and DDP can't fuse its grads cleanly
due to the stride mismatch. **The L1-only probe falsified this** — with
MS-DSSIM removed, that same encoder graph runs at ~0.9 s/step on H100,
so the encoder is not the slow path.

## OOM after ~20 h (separate, unresolved issue)

These same 4-GPU jobs have also hit host-RAM OOM after **~20 hours of
successful training**, even on datasets with ample nominal headroom (ER
SEC61B is 80 GB compressed / 199 GB uncompressed on a 512 GB allocation).
Because the kill happens deep into training and not at peak, this is a **slow
host-RAM leak**, not a peak-sizing problem. Bumping `--mem=640G` just buys
runway — it does not address the leak.

Likely suspects (not yet instrumented):
- Persistent DataLoader workers drifting via torch multiprocessing ref-count
  leaks on forked COW pages.
- `mmap_preload` + `/dev/shm` state not reclaimed across epochs.
- A zarr chunk / pin-memory cache growing unbounded.

Open TODO: log per-rank RSS at each epoch boundary in a production run and
correlate with memory pressure signals to pin the actual source.

## Recommendation

1. **Done (commits `e42c49a` + `3a7fa05`):** local bf16 SSIM helper +
   redundant decorator removal landed. Hopper FCMAE compute is now
   ~4.15 s/step (T6 measurement) vs ~45.5 s/step before — within an
   order of magnitude of L40S throughput.
2. Future training runs (fresh starts, intentional checkpoint
   migrations) can now target Hopper directly. Active prod runs on
   A40/L40S (jobs 31415937, 31446584) should not be precision-flipped
   mid-resume.
3. Separately, add RSS instrumentation and investigate the 20 h host-RAM
   leak; do not treat the `--mem=640G` bump as a fix.
4. **Exclude A100 nodes for FCMAE training** via
   `--constraint='h100|h200|a40|a6000|l40s'`. Two FCMAE scratch jobs
   on A100 nodes (gpu-a-1, gpu-a-2) hit a ~30-min NCCL watchdog
   timeout on the SeqNum-13 BROADCAST of the 32,148,528-element
   encoder weights during DDP setup — same I/O coordination problem
   that killed the earlier A100 sanity attempts. The bf16 fix doesn't
   address this; it's an A100-shared-storage issue.

## Final findings (2026-04-26) — bf16 fix shipped, 4-organelle benchmark in flight

### Solution (one paragraph)

The viscy-utils SSIM helper (`packages/viscy-utils/src/viscy_utils/
evaluation/metrics.py`) replaces monai's fp32-pinned
`compute_ssim_and_cs` with a precision-aware variant that runs the 5
uniform-window mean convolutions in bf16 with squared products
computed in fp32 *before* casting to bf16, and promotes only the
variance subtractions and C₁/C₂-guarded divisions back to fp32.
Redundant `@torch.amp.custom_fwd(cast_inputs=fp32)` decorators on
`MixedLoss.forward` and `SpotlightLoss.forward` were removed in the
same PR. Numerical contract is documented and tested with a 4-tier
tolerance (per-pixel random, aggregate random, correlated-pair,
gradient cosine + sign-flip) at ≥2× margin over measured drift. PR
[#412](https://github.com/mehta-lab/VisCy/pull/412) merged as squash
commit `48f4878`.

### Live 8-job FCMAE matrix (4 organelles × {scratch, pretrained})

In flight on `dynacell-models` HEAD (post-bf16-fix, plus the new
nucleus + membrane configs at `787fed9` with the augmentation-key
override at `1c3034a`):

| Run                                | Hardware           | global_step | Median s/step | p10 s/step |
| ---                                | ---                | ---:        | ---:          | ---:       |
| Nucleus pretrained (NEW)           | H200 gpu-h-8       | 1,679       | 5.37          | 5.36       |
| Nucleus scratch (NEW)              | H200 gpu-h-4       | 1,069       | 5.73          | 5.45       |
| Membrane pretrained (NEW)          | H100 gpu-f-6       | 1,609       | 5.08          | 4.97       |
| Membrane scratch (NEW)             | H100 gpu-f-4       | 1,329       | 5.76          | 5.07       |
| ER (SEC61B) scratch (NEW)          | H200 gpu-h-1       | 299         | 5.56          | 5.46       |
| Mito (TOMM20) scratch (NEW)        | H200 gpu-h-7       | 469         | 4.77          | 4.75       |
| ER (SEC61B) pretrained (BASELINE)  | L40S gpu-g-2       | 30,779      | 4.76          | 4.75       |
| Mito (TOMM20) pretrained (BASELINE)| A40 gpu-c-1        | 35,619      | 2.40          | 2.36       |

Numbers come from wandb `loss/train_step` step×timestamp deltas with
a `dt < 60s` filter (excludes ckpt-save / val / epoch-boundary
insertions). All 8 jobs request `--mem=1024G` except the L40S baseline
at 768G (CPU/RAM is not the driver). **All numbers will be re-pulled
once the new Hopper jobs reach gstep ≥10k** — they are still in epoch
0 (gstep < 1.7k) and warmup effects (DataLoader cache, mmap_preload,
prefetch) likely inflate the per-step time. Steady-state from
gstep ≥30k on A40/L40S is reliable.

### What we can claim now

- **Catastrophic Hopper slowdown is fixed.** Pre-fix Hopper measured
  45–75 s/step; post-fix Hopper sits in 4.77–5.76 s/step — a 9–14×
  recovery, within the same order of magnitude as L40S/A40.
- **Hopper is now competitive but not visibly faster than L40S.** The
  bf16 fix unblocks the catastrophic kernel cost but doesn't expose
  a Hopper-specific tensor-core advantage in the end-to-end loop.
- **A40 (gpu-c-1) is still the fastest measured path** at 2.40 s/step
  steady-state. Likely drivers (not yet measured): node-local I/O for
  zarr reads, /dev/shm topology, PCIe + CPU-worker layout. Same
  shared-FS data path as Hopper, so the differentiator is on the
  compute-node side.

### What remains open

- **Hopper steady-state vs warmup.** Will re-measure once new jobs
  reach gstep ≥10k. Hopper might converge to ~A40 throughput once
  caches warm, or may stay ~5 s/step (would suggest a stable
  data-pipeline ceiling).
- **A40 vs Hopper compute path.** No profiler / `iostat` /
  `nvidia-smi dmon` data on either side. If Hopper steady-state stays
  above ~3 s/step, a 5-min profiling probe on both nodes during
  steady-state would isolate the bottleneck (data I/O vs GPU compute
  vs DDP collective).
- **20-hour host-RAM leak** still unresolved (separate thread, see
  above). Independent of the bf16 fix.
- **NCCL BROADCAST hang on A100** is a known I/O coordination issue
  on those specific nodes' shared-storage path. Mitigation in place
  (constraint OR-list); root cause not investigated.
