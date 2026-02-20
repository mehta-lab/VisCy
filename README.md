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
| [viscy-data](./packages/viscy-data/) | Data loading and Lightning DataModules for microscopy | `pip install viscy-data` |
| [viscy-models](./packages/viscy-models/) | Neural network architectures (UNet, contrastive, VAE) | `pip install viscy-models` |
| [viscy-transforms](./packages/viscy-transforms/) | GPU-accelerated image transforms for microscopy | `pip install viscy-transforms` |
| [viscy-utils](./packages/viscy-utils/) | Shared ML infrastructure for microscopy | `pip install viscy-utils` |

## Applications

| Application | Description | Install |
|-------------|-------------|---------|
| [DynaCLR](./applications/dynaclr/) | Self-supervised contrastive learning for cellular dynamics | `uv pip install -e "applications/dynaclr"` |

## Installation

Install individual packages (e.g.):

```sh
pip install viscy-models
```

Or install from source with all development dependencies:

```sh
git clone https://github.com/mehta-lab/VisCy.git
cd VisCy
uv sync
```

## Development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## License

[BSD-3-Clause](./LICENSE)
