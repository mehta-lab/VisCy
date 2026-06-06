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

Predict fluorescence channels (membrane, nuclei, or organelle structure) from phase-contrast OME-Zarr using three models:

- **CELL-Diff** — flow-matching diffusion model
- **FNet3D** — 3-D U-Net (FNet architecture)
- **VSCyto3D** — masked-autoencoder pretrained U-Net

## Quick start

1. Select an organelle from the dropdown.
2. Click **Load Demo Data** to fetch the matching A549-cell demo dataset directly into the Space — no download/upload needed.
3. Run predictions in **Tab 1** or generate the CELL-Diff ODE trajectory in **Tab 2**.

## Using your own data

The input must be an OME-Zarr HCS store zipped into a single `.zip` file, with layout:

```
your_data.zarr/
  0/0/fov0000/0      # array shape (T, C, Z, Y, X)
                     # C[0] = Phase3D, Z = 16, YX = 512×512
```

Use [iohub](https://github.com/czbiohub-sf/iohub) to create compatible zarr stores.
