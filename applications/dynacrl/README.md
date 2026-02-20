# DynaCLR

Self-supervised contrastive learning for robust representations of cell and organelle dynamics from time-lapse microscopy.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

> **Preprint:** [DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization](https://arxiv.org/abs/2410.11281)

![DynaCLR schematic](https://github.com/mehta-lab/VisCy/blob/e5318d88e2bb5d404d3bae8d633b8cc07b1fbd61/docs/figures/DynaCLR_schematic_v2.png?raw=true)

<details>
<summary>Hirata-Miyasaki et al., 2025</summary>

```bibtex
@misc{hiratamiyasaki2025dynaclr,
    title = {DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization},
    author = {Hirata-Miyasaki, Eduardo and Pradeep, Soorya and Liu, Ziwen and Imran, Alishba and Theodoro, Taylla Milena and Ivanov, Ivan E. and Khadka, Sudip and Lee, See-Chi and Grunberg, Michelle and Woosley, Hunter and Bhave, Madhura and Arias, Carolina and Mehta, Shalin B.},
    year = {2025},
    eprint = {2410.11281},
    archivePrefix = {arXiv},
    url = {https://arxiv.org/abs/2410.11281},
}
```
</details>

## Installation

```bash
# From the VisCy monorepo root
uv pip install -e "applications/dynaclr"

# With evaluation extras (PHATE, UMAP, etc.)
uv pip install -e "applications/dynaclr[eval]"
```

## Usage

Training and prediction use the shared `viscy` CLI provided by `viscy-utils`:

```bash
# Training
uv run --package dynaclr viscy fit -c examples/configs/fit.yml

# Prediction (embedding extraction)
uv run --package dynaclr viscy predict -c examples/configs/predict.yml

# On SLURM (see examples/configs/fit_slurm.sh and predict_slurm.sh)
sbatch examples/configs/fit_slurm.sh
```

The YAML config determines which model and data module to use via `class_path`:

```yaml
model:
  class_path: dynaclr.engine.ContrastiveModule
data:
  class_path: viscy_data.triplet.TripletDataModule
```

DynaCLR also provides evaluation-specific commands:

```bash
# Train a linear classifier on cell embeddings
uv run --package dynaclr dynaclr train-linear-classifier --help

# Apply a trained classifier to new embeddings
uv run --package dynaclr dynaclr apply-linear-classifier --help
```

## Examples

| Example | Description |
|---------|-------------|
| [Quick start](examples/quickstart/) | Get started with model inference |
| [Infection analysis](examples/demos/infection_analysis/) | Compare ImageNet vs DynaCLR embeddings for cell infection |
| [Embedding explorer](examples/demos/embedding_explorer/) | Interactive web-based embedding visualization |
| [Classical sampling](examples/data_preparation/classical_sampling/) | Generate pseudo-tracks for classical triplet sampling |
| [Configs](examples/configs/) | Training, prediction, and ONNX export configs |

## Datasets and Models

- [Test datasets](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/)
- [Pre-trained models](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_models/)
