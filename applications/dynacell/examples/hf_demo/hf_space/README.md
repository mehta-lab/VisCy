---
title: DynaCell Virtual Staining Demo
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
python_version: "3.12"
suggested_hardware: zero-a10g
models:
  - biohub/dynacell-checkpoints
datasets:
  - biohub/dynacell-demo-data
---

# DynaCell Virtual Staining Demo

Predict fluorescence (membrane, nuclei, ER, mitochondria) from label-free phase 3-D
microscopy of live A549 cells with three DynaCell baselines:

- **FNet3D**: 3-D U-Net (deterministic regression)
- **VSCyto3D**: FCMAE-pretrained UNeXt2 (deterministic regression)
- **CELL-Diff**: flow-matching diffusion model; scrub its ODE trajectory from noise to prediction

The page has three sections: **Data** (browse phase and experimental fluorescence),
**Regression** (FNet3D and VSCyto3D with Spectral PCC against the experimental channel) and
**Generative** (CELL-Diff). Inference runs on the selected timepoint only.

## Quick start

1. Select an organelle and click **Load Demo Data**.
2. Run the regression models, or generate the CELL-Diff trajectory.

## Data and checkpoints

Demo data ([`biohub/dynacell-demo-data`](https://huggingface.co/datasets/biohub/dynacell-demo-data))
is one held-out test field of view per marker, cropped from the DynaCell v1 release
(`s3://dynacell/v1/data/biohub-a549/test/`). The checkpoints
([`biohub/dynacell-checkpoints`](https://huggingface.co/biohub/dynacell-checkpoints)) are the
released A549-trained models (`s3://dynacell/v1/models/a549/`).

Code: [VisCy `applications/dynacell`](https://github.com/mehta-lab/VisCy/tree/dynacell-models/applications/dynacell).
