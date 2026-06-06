---
license: bsd-3-clause
library_name: viscy
pipeline_tag: image-to-image
tags:
  - virtual-staining
  - microscopy
  - fluorescence-prediction
  - cell-biology
  - phase-contrast
  - ome-zarr
---

# DynaCell Virtual Staining Checkpoints

Model checkpoints for the **DynaCell** virtual-staining demo: predict fluorescence
channels (membrane, nuclei, ER, mitochondria) from label-free **phase-contrast**
3-D microscopy of live A549 cells. Trained and run with
[VisCy](https://github.com/mehta-lab/VisCy).

Live demo: [`biohub/dynacell`](https://huggingface.co/spaces/biohub/dynacell)

## Models

Three architectures, each trained per organelle (12 checkpoints total):

| Model | Architecture | Notes |
| --- | --- | --- |
| **CELL-Diff** | Flow-matching diffusion (`dynacell.engine.DynacellFlowMatching`) | Iterative ODE generation; supports denoising-trajectory visualization |
| **FNet3D** | 3-D U-Net (`DynacellUNet`, `FNet3D`) | Deterministic regression |
| **VSCyto3D** | FCMAE-pretrained 3-D U-Net (`DynacellUNet`, `fcmae`) | Masked-autoencoder pretrained encoder |

## Targets

All models take a single `Phase3D` input channel (`z_window_size=16`,
`512×512` YX) and predict one fluorescence target:

| Organelle marker | Predicted channel |
| --- | --- |
| CAAX | Membrane |
| H2B | Nuclei / chromatin |
| SEC61B | ER (structure) |
| TOMM20 | Mitochondria (structure) |

## Files

`{model}_{organelle}.ckpt`, e.g. `celldiff_caax.ckpt`, `fnet3d_h2b.ckpt`,
`vscyto3d_sec61b.ckpt` — `model ∈ {celldiff, fnet3d, vscyto3d}`,
`organelle ∈ {caax, h2b, sec61b, tomm20}`.

## Usage

The demo Space downloads checkpoints with `huggingface_hub` and runs prediction
through VisCy:

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("biohub/dynacell-checkpoints", "vscyto3d_caax.ckpt")
# load via dynacell.engine.DynacellUNet / DynacellFlowMatching (see VisCy)
```

Input data must be an OME-Zarr HCS store with `Phase3D` as the source channel.
See the [demo Space](https://huggingface.co/spaces/biohub/dynacell) and
[VisCy](https://github.com/mehta-lab/VisCy) for the full prediction config.

## Training data

Live A549 cells imaged on the mantis microscope (phase contrast + paired
fluorescence ground truth).

## License

BSD 3-Clause — © CZ Biohub SF.

## Citation

Please cite the DynaCell study and VisCy. <!-- TODO: add final paper reference / DOI -->

```bibtex
@software{viscy,
  title  = {VisCy: computer vision for virtual staining of cells},
  author = {CZ Biohub SF and contributors},
  url    = {https://github.com/mehta-lab/VisCy}
}
```
