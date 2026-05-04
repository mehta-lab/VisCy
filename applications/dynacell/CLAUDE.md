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
