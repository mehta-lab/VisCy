# VisCy

|             |                                                                                                                                                              |
| :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Status**  | [![Docs][badge-docs]][link-docs] [![Tests][badge-tests]][link-tests]                                                                                          |
| **Package** | [![PyPI][badge-pypi]][link-pypi] [![Downloads][badge-downloads]][link-downloads]                                                                              |
|  **Meta**   | [![SPEC 0][badge-spec0]][link-spec0] [![Contributors][badge-contributors]][link-contributors] [![Stars][badge-stars]][link-repo]                              |
|  **Cite**   | [![DOI][badge-doi]][link-doi]                                                                                                                                 |

[badge-docs]: https://github.com/mehta-lab/VisCy/actions/workflows/docs.yml/badge.svg
[badge-tests]: https://github.com/mehta-lab/VisCy/actions/workflows/test.yml/badge.svg
[badge-pypi]: https://img.shields.io/pypi/v/viscy.svg
[badge-downloads]: https://img.shields.io/pypi/dm/viscy.svg
[badge-spec0]: https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[badge-contributors]: https://img.shields.io/github/contributors-anon/mehta-lab/VisCy
[badge-stars]: https://img.shields.io/github/stars/mehta-lab/VisCy
[badge-doi]: https://zenodo.org/badge/DOI/10.5281/zenodo.15022186.svg

[link-docs]: https://mehta-lab.github.io/VisCy/stable/
[link-tests]: https://github.com/mehta-lab/VisCy/actions/workflows/test.yml
[link-pypi]: https://pypi.org/project/viscy
[link-downloads]: https://pypistats.org/packages/viscy
[link-spec0]: https://scientific-python.org/specs/spec-0000/
[link-contributors]: https://github.com/mehta-lab/VisCy/graphs/contributors
[link-repo]: https://github.com/mehta-lab/VisCy
[link-doi]: https://doi.org/10.5281/zenodo.15022186

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
| [Cytoland](./applications/cytoland/) | Robust virtual staining of organelles from label-free images | `uv pip install -e "applications/cytoland"` |
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

## Documentation

Full documentation is hosted at <https://mehta-lab.github.io/VisCy/stable/>.

## Development

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

## License

[BSD-3-Clause](./LICENSE)
