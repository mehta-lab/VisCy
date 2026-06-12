# DynaCell HuggingFace Demo

Single-page Gradio Space that predicts fluorescence (membrane, nuclei, ER,
mitochondria) from label-free phase-contrast 3-D microscopy of live A549 cells.
Three stacked sections, each with its own Timepoint + Z-slice sliders:

1. **Data** — pick a demo organelle dataset and browse Phase | experimental
   fluorescence by timepoint and Z.
2. **Regression** — deterministic models **FNet3D** and **VSCyto3D**: predict +
   Spectral PCC.
3. **Generative** — **CELL-Diff** ODE trajectory: an ODE-step slider scrubs
   noise → prediction, with that step's Spectral PCC.

Inference runs on the single selected timepoint only. Data comes from the demo
dataset repo (no user upload).

## Layout

```
hf_demo/
  hf_space/                 # the deployed Space
    app.py                  # Gradio UI + section callbacks + renderers
    predict_runner.py       # checkpoint download, prediction, ODE trajectory
    requirements.txt
    README.md               # Space card
    config_templates/       # per-model predict YAMLs (celldiff/fnet3d/vscyto3d)
  cards/                    # README cards for the model + dataset repos
  upload_checkpoints.py     # publish checkpoints (run on HPC)
  upload_hf_space.py        # push hf_space/ to the Space
  mirror_to_biohub.py       # one-time mirror from dihan-zheng/*
```

Hosting, access model, ZeroGPU configuration, and the deploy/smoke-test
workflow: see the `hf-dynacell` Claude Code skill.
