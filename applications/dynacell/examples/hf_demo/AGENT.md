# DynaCell HuggingFace Demo

Gradio Space that predicts fluorescence (membrane, nuclei, ER, mitochondria)
from label-free phase-contrast 3-D microscopy of live A549 cells, using three
models — **CELL-Diff** (flow-matching), **FNet3D** (3-D U-Net), and **VSCyto3D**
(FCMAE-pretrained U-Net). Tab 1 runs prediction + Spectral PCC; Tab 2 animates
the CELL-Diff ODE denoising trajectory.

## Layout

```
hf_demo/
  hf_space/                 # the deployed Space
    app.py                  # Gradio UI + tab callbacks
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
