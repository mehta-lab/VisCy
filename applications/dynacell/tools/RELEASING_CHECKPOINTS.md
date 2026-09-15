# Releasing / updating the DynaCell public checkpoint zoo

`assemble_release_checkpoints.py` builds the `models/` tree of the public S3
mirror (`/hpc/projects/virtual_staining/dynacell_v1`) from the trained
checkpoints on `/hpc/projects/comp.micro`. It is **idempotent** — re-running it
overwrites only the leaves whose pinned checkpoint changed, which is how the
recurring **deconv → raw fluorescence** update (see below) is applied.

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
| `not_trained` | model dir absent (stale config for a model never trained, e.g. Joint ER/Mito Pix2Pix3D) | dropped |

With the default paper set: `43 resolved · 9 pending · 0 not_trained`
(~27.9 GB). With `--models …,pix2pix3d` (full zoo): `50 · 9 · 2` — the 59
evaluated cells + 2 stale Pix2Pix3D configs (Joint ER/Mito, never trained).

## The deconv → raw fluorescence update (recurring)

In v1, **A549 and Joint ER/Mito** targets are *deconvolved* GFP; everything else
(all membrane/nucleus, all iPSC) is *raw*. See
`project_channel_raw_vs_deconv_provenance`. The manifest tags every affected
cell `provenance = deconv->raw`; these are the "half" that the raw-fluorescence
retrain replaces. When those runs finish:

1. **Confirm each raw run is complete**, not just alive — cross-check wandb final
   `epoch` vs configured `max_epochs`, `sacct` state/exit code, and the resolved
   fit YAML (see root `CLAUDE.md § Job monitoring`). A `finished` wandb state
   alone is not enough.
2. **Re-evaluate** the raw models so their `predict__*.yml` `ckpt_path` is
   re-pinned to the new raw `epoch=NNN-step=MMMM.ckpt` (or edit `ckpt_path` by
   hand). This tool reads only the config — it does not pick "latest" itself.
3. **Dry-run** the assembler: the `deconv->raw` cells that were `pending` flip to
   `resolved`. Confirm the new epochs match the re-evaluated runs.
4. **`--execute`**: overwrites exactly those ER/Mito leaves and rewrites
   `checkpoints.csv`. Raw/unaffected leaves are byte-identical re-copies (safe).
5. **Propagate the provenance flip**: change the affected rows from
   `deconv->raw` to `raw` here (`DECONV_TRAIN`/`DECONV_ORG` in the script — or
   drop them once nothing is deconv), update `models/README.md`, the
   [Confluence "Checkpoints used for evaluations" page][ckpt-page], and re-sync
   `models/` to S3.

Until then, the currently-published `deconv->raw` leaves ship the v1 deconv
checkpoints (flagged in the manifest); the 9 `pending` deterministic ER/Mito
cells whose pins were already pruned are simply absent from the release until
step 2 re-pins them.

[ckpt-page]: https://czbiohub.atlassian.net/wiki/spaces/MUG/pages/5485396007

## Navigation

- Up: [tools](./README.md) · [applications/dynacell](../README.md)
