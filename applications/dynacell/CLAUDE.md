# dynacell — Claude Code reference

## Model name conventions

Code names (used in YAML config keys, prediction zarr filenames, eval pipeline keys, W&B run names) differ from the paper names. When writing/reading anything that crosses the code/paper boundary (figures, tables, Confluence pages, manuscripts), translate:

| Code name (config / zarr / W&B) | Paper / display name |
| --- | --- |
| `fcmae_vscyto3d_scratch` | **UNeXt2** |
| `fcmae_vscyto3d_pretrained` | **VSCyto3D** (FCMAE-pretrained is the canonical VSCyto3D variant) |
| `unext2` | UNeXt2 (legacy zarr prefix; superseded by `fcmae_vscyto3d_scratch`) |
| `vscyto3d` | VSCyto3D (display key in Dihan's eval pipeline; sources `*_fcmae_vscyto3d_pretrained` predictions) |
| `unetvit3d` | UNetViT3D |
| `fnet3d_paper` | FNet3D |
| `celldiff` | CELL-Diff (variants: `iterative`, `sliding_window`, `denoise`/Mean Predictor) |

Eval-pipeline directory naming (`/hpc/projects/virtual_staining/training/dynacell/{ipsc,a549}/evaluations/eval_<model>_<organelle>[_<plate>]`) uses the **paper key** (`unext2`, `vscyto3d`, `fnet3d`, `unetvit3d`, `celldiff_*`), not the config key. So `eval_unext2_membrane` maps to the `fcmae_vscyto3d_scratch` predictions, `eval_vscyto3d_membrane` maps to `fcmae_vscyto3d_pretrained`.

## Prediction zarr naming convention

Set by `trainer.callbacks[…HCSPredictionWriter].init_args.output_store` in each leaf of `applications/dynacell/configs/benchmarks/virtual_staining/<organelle>/<model>/<train_set>/predict__*.yml`. The infix between model name and the optional plate condition flags the **training set** of the source model:

| Trained on | Test set | Filename |
| --- | --- | --- |
| iPSC | iPSC | `<org>_<model>.zarr` |
| iPSC | A549 plate | `<org>_<model>_<cond>.zarr` |
| A549 | iPSC | `<org>_<model>_a549trained.zarr` |
| A549 | A549 plate | `<org>_<model>_a549trained_<cond>.zarr` |
| Joint (iPSC + A549) | iPSC | `<org>_<model>_jointtrained.zarr` |
| Joint (iPSC + A549) | A549 plate | `<org>_<model>_jointtrained_<cond>.zarr` |

Where `<org>` is `nucl` / `memb` / `sec61b` / `tomm20`, `<model>` is the **code name** from the table above (e.g. `fcmae_vscyto3d_scratch`, `fnet3d_paper`), and `<cond>` is `mock` / `denv` / `zikv`. The (no-infix) iPSC-trained naming is historical baggage from before joint/A549 training existed; don't add a `_ipsctrained` infix retroactively. Output dirs: iPSC test predictions land under `ipsc/predictions/`, A549 plate predictions under `a549/predictions/`, regardless of training set.

Caveat: Dihan's earlier ER + Mito iPSC-trained zarrs use a legacy `<gene>_<model>__<gene>_<cond>.zarr` shape (e.g. `sec61b_fcmae_vscyto3d_scratch__sec61b_mock.zarr`, double-underscore + redundant gene prefix). New leaves should follow the table above; do not propagate the legacy form.
