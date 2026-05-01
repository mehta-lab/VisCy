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

## Tutorials and demos

Scripts and tutorials live under [`examples/`](./examples/):

| Folder | What it demonstrates |
|--------|----------------------|
| [`examples/VS_model_inference/`](./examples/VS_model_inference/) | Python API inference demos for VSCyto2D, VSCyto3D, VSNeuromast, and TTA-augmented sliding-window prediction |
| [`examples/vcp_tutorials/`](./examples/vcp_tutorials/) | Virtual Cell Platform quick-start and organism-specific walkthroughs (HEK293T, neuromast) |
| [`examples/dl-course-exercise/`](./examples/dl-course-exercise/) | Image-translation course exercise (training from scratch + evaluation) — used at DL@MBL and DL@Janelia |
| [`examples/configs/`](./examples/configs/) | YAML configs for `viscy fit` / `viscy predict` across models (VSCyto2D/3D, VSNeuromast, FNet3D, dynacell) |

All demo scripts are written as jupytext-style percent-cell `.py` files.
Regenerate paired `.ipynb` notebooks with `jupytext --to ipynb solution.py`
if you prefer the notebook UI.

## Models

| Model | Input | Output | Architecture |
|-------|-------|--------|-------------|
| VSCyto3D | Phase3D | Nuclei + Membrane | FCMAE / UNeXt2 |
| VSCyto2D | Phase2D | Nuclei + Membrane | UNeXt2 |
| VSNeuromast | DIC | Multiple fluorescent markers | UNeXt2 |
| FNet3D | Transmitted light | Fluorescence | Unet3d (Ounkomol et al. 2018) |

> **FNet3D note:** All spatial dimensions (Z, Y, X) must be divisible by `2^depth`
> (default depth=4 requires divisibility by 16). See `examples/configs/fnet3d/fit.yml`.

> **Benchmark note:** FNet3D and SEC61B benchmarks now launch from
> [`applications/dynacell/`](../dynacell/README.md). Cytoland copies are
> transitional legacy — see `examples/configs/dynacell/` and `examples/configs/fnet3d/`.

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
