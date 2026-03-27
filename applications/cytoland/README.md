# Cytoland

Robust virtual staining of landmark organelles from label-free microscopy.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

> **Paper:** [Robust virtual staining in Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01046-2)

## Installation

```bash
# From the VisCy monorepo root
uv pip install -e "applications/cytoland"
```

## Usage

Training and prediction use the shared `viscy` CLI provided by `viscy-utils`:

```bash
# Training (pick a model-specific config)
uv run --package cytoland viscy fit -c examples/configs/vscyto3d/finetune.yml

# Training with Spotlight loss
uv run --package cytoland viscy fit -c examples/configs/vscyto3d/train_spotlight.yml

# Prediction
uv run --package cytoland viscy predict -c examples/configs/vscyto3d/predict.yml
```

The YAML config determines which model and data module to use via `class_path`:

```yaml
model:
  class_path: cytoland.engine.VSUNet
data:
  class_path: viscy_data.hcs.HCSDataModule
```

## Models

| Model | Input | Output | Architecture |
|-------|-------|--------|-------------|
| VSCyto3D | Phase3D | Nuclei + Membrane | FCMAE / UNeXt2 |
| VSCyto2D | Phase2D | Nuclei + Membrane | UNeXt2 |
| VSNeuromast | DIC | Multiple fluorescent markers | UNeXt2 |
| FNet3D | Transmitted light | Fluorescence | Unet3d (Ounkomol et al. 2018) |

> **FNet3D note:** All spatial dimensions (Z, Y, X) must be divisible by `2^depth`
> (default depth=4 requires divisibility by 16). See `examples/configs/fnet3d/fit.yml`.

## References

<details>
<summary>Liu, Hirata-Miyasaki et al., 2025</summary>

```bibtex
@article{liu2025robust,
    title = {Robust virtual staining of landmark organelles},
    author = {Liu, Ziwen and Hirata-Miyasaki, Eduardo and Pradeep, Soorya and others},
    journal = {Nature Machine Intelligence},
    year = {2025},
    doi = {10.1038/s42256-025-01046-2},
}
```
</details>
