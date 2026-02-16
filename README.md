# VisCy

[![Python package index](https://img.shields.io/pypi/v/viscy.svg)](https://pypi.org/project/viscy)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/viscy.svg)](https://pypistats.org/packages/viscy)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/mehta-lab/VisCy)](https://github.com/mehta-lab/VisCy/graphs/contributors)
![GitHub Repo stars](https://img.shields.io/github/stars/mehta-lab/VisCy)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15022186.svg)](https://doi.org/10.5281/zenodo.15022186)

VisCy (blend of `vision` and `cyto`) is a deep learning pipeline for training and deploying computer vision models for image-based phenotyping at single-cell resolution.

## Packages

VisCy is organized as a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo:

| Package | Description | Install |
|---------|-------------|---------|
| [viscy-transforms](./packages/viscy-transforms/) | GPU-accelerated image transforms for microscopy | `pip install viscy-transforms` |
| [viscy-models](./packages/viscy-models/) | Neural network architectures (UNet, contrastive, VAE) | `pip install viscy-models` |

More packages coming soon: `viscy-data`, `viscy-airtable`.

## Installation

Install individual packages:

```sh
pip install viscy-transforms
pip install viscy-models
```

Or install from source with all development dependencies:

```sh
git clone https://github.com/mehta-lab/VisCy.git
cd VisCy
uv sync
```

## Cytoland (Robust Virtual Staining)

### Demo [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/chanzuckerberg/Cytoland)

Try the 2D virtual staining demo of cell nuclei and membrane from label-free images on
[Hugging Face](https://huggingface.co/spaces/chanzuckerberg/Cytoland).

<p align="center">
<a href="https://huggingface.co/spaces/chanzuckerberg/Cytoland" target="_blank" rel="noopener noreferrer">
<img src="https://github.com/mehta-lab/VisCy/blob/7d3bed92e91fb44611a45be5350320d65ffcc111/docs/figures/vs_hf_demo.gif?raw=true" alt="Virtual Staining App Demo" height="300px" />
</a>
</p>

### Cytoland @ Virtual Cells Platform

Cytoland models are accessible via the Chan Zuckerberg Initiative's Virtual Cells Platform:

- [Model card](https://virtualcellmodels.cziscience.com/model/01961244-1970-7851-a4b9-fdbfa2fba9b2)
- [Quick-start (VSCyto2D)](https://virtualcellmodels.cziscience.com/quickstart/cytoland-quickstart)
- CLI tutorials: [VSCyto3D](https://virtualcellmodels.cziscience.com/tutorial/cytoland-tutorial) | [VSNeuromast](https://virtualcellmodels.cziscience.com/tutorial/cytoland-neuromast)

### Gallery

Below are some examples of virtually stained images (click to play videos).

| VSCyto3D | VSNeuromast | VSCyto2D |
|:---:|:---:|:---:|
| [![HEK293T](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_1.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755) | [![Neuromast](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_3.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/4cef8333-895c-486c-b260-167debb7fd64) | [![A549](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_5.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/287737dd-6b74-4ce3-8ee5-25fbf8be0018) |

### References

The Cytoland models and training protocols are reported in [Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01046-2).

<details>
<summary>Liu, Hirata-Miyasaki et al., 2025</summary>

```bibtex
@article{liu_robust_2025,
    title = {Robust virtual staining of landmark organelles with {Cytoland}},
    journal = {Nature Machine Intelligence},
    author = {Liu, Ziwen and Hirata-Miyasaki, Eduardo and Pradeep, Soorya and others},
    year = {2025},
    doi = {10.1038/s42256-025-01046-2},
}
```
</details>

## DynaCLR (Embedding Cell Dynamics)

DynaCLR is a self-supervised method for learning robust representations of cell and organelle dynamics from time-lapse microscopy using contrastive learning.

- [Preprint on arXiv](https://arxiv.org/abs/2410.11281)
- [Demo dataset and checkpoints](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_demo/)

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

## Development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## License

[BSD-3-Clause](./LICENSE)
