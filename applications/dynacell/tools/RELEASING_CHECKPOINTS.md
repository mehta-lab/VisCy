# Releasing / updating the DynaCell public checkpoint zoo

`assemble_release_checkpoints.py` builds the `models/` tree of the public S3
mirror (`/hpc/projects/virtual_staining/dynacell_v1`) from the trained
checkpoints on `/hpc/projects/comp.micro`. It is **idempotent** — re-running it
overwrites only the leaves whose pinned checkpoint changed, which is how a
retrained model is published (see *Updating after a retrain* below).

## Public layout

```
dynacell_v1/models/
├── checkpoints.csv                                    # manifest (written on --execute)
└── {ipsc,a549,joint}/{nucleus,membrane,er,mito}/{model_slug}/
        epoch=NNN-step=MMMM.ckpt                        # original filename preserved
        config.yaml                                     # fit config (architecture)
```

- **Axis order** `train_set / organelle / model` mirrors the source tree and the
  `paths.py` grammar.
- **One checkpoint per leaf** → glob `*.ckpt` for the unambiguous file; the
  `epoch=NNN-step=MMMM` filename is preserved verbatim so provenance survives
  even if files and docs diverge. `last.ckpt` / `best_ep*.ckpt` pins are resolved
  back to their canonical `epoch=` sibling before copy.
- `model_slug` ∈ `fnet3d`, `unext2`, `vscyto3d`, `unetvit3d`, `celldiff`,
  `pix2pix3d` (paper-facing; see `../CLAUDE.md` for the code↔paper table).

## Source of truth

The pinned checkpoint per `(train_set, organelle, model)` is read from the
**canonical benchmark predict configs** (`ckpt_path:` in
`configs/benchmarks/virtual_staining/{er,membrane,mito,nucleus}/**/predict__*.yml`),
never a hand-maintained table — so the release always matches what the evals
consumed. `_dual_nucl_memb/` (ablations) and `_internal/` (generated leaves) are
excluded. Name normalization (source → public):

| Axis | Source | Public |
|---|---|---|
| train_set | `ipsc` / `a549_mantis` / `joint_ipsc_confocal_a549_mantis` | `ipsc` / `a549` / `joint` |
| organelle | `nucl` / `memb` / `sec61b` / `tomm20` | `nucleus` / `membrane` / `er` / `mito` |
| model | `fnet3d_paper` · `fcmae_vscyto3d_scratch` · `fcmae_vscyto3d_pretrained[_ws8500]` · `unetvit3d` · `celldiff_r2` · `pix2pix3d_unetvit[_modernized_…]` | `fnet3d` · `unext2` · `vscyto3d` · `unetvit3d` · `celldiff` · `pix2pix3d` |

The bare `celldiff` (pre-`_r2`) dir is deliberately never published.

## Running

```sh
# dry run (default): print coverage matrix + copy plan + pending/not-trained, no writes
uv run python applications/dynacell/tools/assemble_release_checkpoints.py

# execute: copy resolved ckpts + config.yaml into <dest>/models/ and write checkpoints.csv
uv run python applications/dynacell/tools/assemble_release_checkpoints.py --execute
```

`--dest` overrides the release root; `--manifest` overrides the CSV path.
`--models` selects which architectures to publish — it defaults to the **paper
set** `fnet3d,unext2,vscyto3d,unetvit3d,celldiff` (the release `models/README.md`
list). **Pix2Pix3D is excluded by default** (internal baseline, not in the
paper); add it back with `--models fnet3d,unext2,vscyto3d,unetvit3d,celldiff,pix2pix3d`.

### Per-cell status

| Status | Meaning | Action |
|---|---|---|
| `resolved` | pinned checkpoint exists | copied |
| `pending` | model dir exists but the pinned epoch is gone (mid-retrain / awaiting re-pin) | **not** copied, listed |
| `not_trained` | model dir absent (stale config for a model never trained) | dropped |

As of 2026-09-30, with the default paper set: `56 resolved · 0 pending · 0
not_trained` (35.3 GB). UNetViT3D covers only the ER/Mito rows for A549 and
Joint (plus all four iPSC organelles), which is why it has 8 cells rather than
12. With `--models …,pix2pix3d` (full zoo): `68 · 0 · 0` (53.4 GB).

## Provenance: every published target is raw fluorescence

The manifest's `provenance` column is `raw` for every row. v1 A549 and Joint
ER/Mito models were trained against *deconvolved* GFP: the `Structure` channel
of `a549/mantis_v1/train/*_all.zarr` is deconvolved (see that dir's
`CHANNEL_PROVENANCE.md`). Those checkpoints were replaced by retrains on the
2026-07-06 rebuilt `a549/mantis/train/*_all.zarr`, whose `Structure` channel is
raw (+93 camera floor) next to a separate `Structure_deconvolved`. All 20
currently pinned A549/Joint ER/Mito checkpoints were traced on 2026-09-30 to
fits that start fresh on the rebuilt store, including every `--ckpt_path`
resume hop. The evidence is in
`applications/dynacell/experiments/2026-09-30_s3-release-refresh/CHECKPOINTS.md`.

To re-verify a checkpoint's training data, check its lineage, not `config.yaml`:

- The checkpoint stores model hparams only (no datamodule hparams).
- The run dir's `config.yaml` is overwritten by every later fit. The S3-mirror
  copy next to the known-deconv joint/er/vscyto3d `epoch=111` reads
  `a549/mantis/`, although that checkpoint was trained on `mantis_v1`.
- Use the `resolved/fit_*.yml` whose timestamp precedes the checkpoint, and
  follow any `Restoring states from the checkpoint path` line in the run's
  `slurm/*.err` back to a fresh start. May (deconv) and July (raw) fits shared
  one `dirpath`, and the May checkpoints sit in `checkpoints/preflip_deconv/`.

## Updating after a retrain

1. **Confirm each new run is complete**, not just alive — cross-check wandb final
   `epoch` vs configured `max_epochs`, `sacct` state/exit code, and the resolved
   fit YAML (see root `CLAUDE.md § Job monitoring`). A `finished` wandb state
   alone is not enough.
2. **Re-evaluate** the models so their `predict__*.yml` `ckpt_path` is re-pinned
   to the new `epoch=NNN-step=MMMM.ckpt` (or edit `ckpt_path` by hand). This
   tool reads only the config — it does not pick "latest" itself.
3. **Dry-run** the assembler and confirm the new epochs match the re-evaluated
   runs.
4. **`--execute`**: overwrites exactly the changed leaves and rewrites
   `checkpoints.csv`. Unchanged leaves are byte-identical re-copies (safe).
5. Update `models/README.md`, the
   [Confluence "Checkpoints used for evaluations" page][ckpt-page], and re-sync
   `models/` to S3.

[ckpt-page]: https://czbiohub.atlassian.net/wiki/spaces/MUG/pages/5485396007

## Navigation

- Up: [tools](./README.md) · [applications/dynacell](../README.md)
