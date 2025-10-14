# VisCy

[![Python package index](https://img.shields.io/pypi/v/viscy.svg)](https://pypi.org/project/viscy)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/viscy.svg)](https://pypistats.org/packages/viscy)
[![Total downloads](https://pepy.tech/badge/viscy)](https://pepy.tech/project/viscy)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/mehta-lab/VisCy)](https://github.com/mehta-lab/VisCy/graphs/contributors)
![GitHub Repo stars](https://img.shields.io/github/stars/mehta-lab/VisCy)
![GitHub forks](https://img.shields.io/github/forks/mehta-lab/VisCy)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15022186.svg)](https://doi.org/10.5281/zenodo.15022186)

VisCy (blend of `vision` and `cyto`) is a deep learning pipeline for training and deploying computer vision models for image-based phenotyping at single-cell resolution.

This repository provides a pipeline for the following.

- Image translation
  - Robust virtual staining of landmark organelles with Cytoland
- Image representation learning
  - Self-supervised learning of the cell state and organelle phenotypes with DynaCLR
- Semantic segmentation
  - Supervised learning of of cell state (e.g. state of infection)

> **Note:**
VisCy is under active development.
While we strive to maintain stability,
the main branch may occasionally be updated with backward-incompatible changes
which are subsequently shipped in releases following [semantic versioning](https://semver.org/).
Please choose a stable release from PyPI for production use.

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

Cytoland models are accessible via the Chan Zuckerberg Initiative's Virtual Cells Platform.
Notebooks are available as pre-rendered pages or on Colab:

- [Model card](https://virtualcellmodels.cziscience.com/model/01961244-1970-7851-a4b9-fdbfa2fba9b2)
- [Quick-start (VSCyto2D)](https://virtualcellmodels.cziscience.com/quickstart/cytoland-quickstart)
- CLI tutorials:
  - [VSCyto3D](https://virtualcellmodels.cziscience.com/tutorial/cytoland-tutorial)
  - [VSNeuromast](https://virtualcellmodels.cziscience.com/tutorial/cytoland-neuromast)

### Tutorials

- [Virtual staining exercise](https://github.com/mehta-lab/VisCy/blob/main/examples/virtual_staining/dlmbl_exercise/solution.ipynb):
Notebook illustrating how to use VisCy to train, predict and evaluate the VSCyto2D model. This notebook was developed for the [DL@MBL2024](https://github.com/dlmbl/DL-MBL-2024) course and uses UNeXt2 architecture.

- [Image translation demo](https://github.com/mehta-lab/VisCy/blob/main/examples/virtual_staining/img2img_translation/solution.ipynb): Fluorescence images can be predicted from label-free images. Can we predict label-free image from fluorescence? Find out using this notebook.

- [Training Virtual Staining Models via CLI](https://github.com/mehta-lab/VisCy/wiki/virtual-staining-instructions):
Instructions for how to train and run inference on VisCy's virtual staining models (*VSCyto3D*, *VSCyto2D* and *VSNeuromast*).

### Gallery

Below are some examples of virtually stained images (click to play videos).
See the full gallery [here](https://github.com/mehta-lab/VisCy/wiki/Gallery).

| VSCyto3D | VSNeuromast | VSCyto2D |
|:---:|:---:|:---:|
| [![HEK293T](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_1.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755) | [![Neuromast](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_3.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/4cef8333-895c-486c-b260-167debb7fd64) | [![A549](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_5.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/287737dd-6b74-4ce3-8ee5-25fbf8be0018) |

### References

The Cytoland models and training protocols are reported in our recent [paper on robust virtual staining in Nature Machine Intelligence]([https://www.biorxiv.org/content/10.1101/2024.05.31.596901](https://www.nature.com/articles/s42256-025-01046-2)).

This package evolved from the [TensorFlow version of virtual staining pipeline](https://github.com/mehta-lab/microDL), which we reported in [this paper in 2020 in eLife](https://elifesciences.org/articles/55502).

<details>
  <summary>Liu, Hirata-Miyasaki et al., 2025</summary>

  <pre><code>
  @article{liu_robust_2025,
      title = {Robust virtual staining of landmark organelles with {Cytoland}},
      copyright = {2025 The Author(s)},
      issn = {2522-5839},
      url = {https://www.nature.com/articles/s42256-025-01046-2},
      doi = {10.1038/s42256-025-01046-2},
      abstract = {Correlative live-cell imaging of landmark organelles—such as nuclei, nucleoli, cell membranes, nuclear envelope and lipid droplets—is critical for systems cell biology and drug discovery. However, achieving this with molecular labels alone remains challenging. Virtual staining of multiple organelles and cell states from label-free images with deep neural networks is an emerging solution. Virtual staining frees the light spectrum for imaging molecular sensors, photomanipulation or other tasks. Current methods for virtual staining of landmark organelles often fail in the presence of nuisance variations in imaging, culture conditions and cell types. Here we address this with Cytoland, a collection of models for robust virtual staining of landmark organelles across diverse imaging parameters, cell states and types. These models were trained with self-supervised and supervised pre-training using a flexible convolutional architecture (UNeXt2) and augmentations inspired by image formation of light microscopes. Cytoland models enable virtual staining of nuclei and membranes across multiple cell types—including human cell lines, zebrafish neuromasts, induced pluripotent stem cells (iPSCs) and iPSC-derived neurons—under a range of imaging conditions. We assess models using intensity, segmentation and application-specific measurements obtained from virtually and experimentally stained nuclei and membranes. These models rescue missing labels, correct non-uniform labelling and mitigate photobleaching. We share multiple pre-trained models, open-source software (VisCy) for training, inference and deployment, and the datasets.},
      language = {en},
      urldate = {2025-06-23},
      journal = {Nature Machine Intelligence},
      author = {Liu, Ziwen and Hirata-Miyasaki, Eduardo and Pradeep, Soorya and Rahm, Johanna V. and Foley, Christian and Chandler, Talon and Ivanov, Ivan E. and Woosley, Hunter O. and Lee, See-Chi and Khadka, Sudip and Lao, Tiger and Balasubramanian, Akilandeswari and Marreiros, Rita and Liu, Chad and Januel, Camille and Leonetti, Manuel D. and Aviner, Ranen and Arias, Carolina and Jacobo, Adrian and Mehta, Shalin B.},
      month = jun,
      year = {2025},
      note = {Publisher: Nature Publishing Group},
      pages = {1--15},
      }
  </code></pre>
</details>

<details>
  <summary>Guo, Yeh, Folkesson et al., 2020</summary>

  <pre><code>
  @article {10.7554/eLife.55502,
      article_type = {journal},
      title = {Revealing architectural order with quantitative label-free imaging and deep learning},
      author = {Guo, Syuan-Ming and Yeh, Li-Hao and Folkesson, Jenny and Ivanov, Ivan E and Krishnan, Anitha P and Keefe, Matthew G and Hashemi, Ezzat and Shin, David and Chhun, Bryant B and Cho, Nathan H and Leonetti, Manuel D and Han, May H and Nowakowski, Tomasz J and Mehta, Shalin B},
      editor = {Forstmann, Birte and Malhotra, Vivek and Van Valen, David},
      volume = 9,
      year = 2020,
      month = {jul},
      pub_date = {2020-07-27},
      pages = {e55502},
      citation = {eLife 2020;9:e55502},
      doi = {10.7554/eLife.55502},
      url = {https://doi.org/10.7554/eLife.55502},
      keywords = {label-free imaging, inverse algorithms, deep learning, human tissue, polarization, phase},
      journal = {eLife},
      issn = {2050-084X},
      publisher = {eLife Sciences Publications, Ltd},
      }
  </code></pre>
</details>

### Library of Virtual Staining (VS) Models

The robust virtual staining models (i.e *VSCyto2D*, *VSCyto3D*, *VSNeuromast*), and fine-tuned models can be found [here](https://github.com/mehta-lab/VisCy/wiki/Library-of-virtual-staining-(VS)-Models)

## DynaCLR (Embedding Cell Dynamics via Contrastive Learning of Representations)

DynaCLR is a self-supervised method for learning robust and temporally-regularized representations of cell and organelle dynamics from time-lapse microscopy using contrastive learning. It supports diverse downstream biological tasks -- including cell state classification with efficient human annotations, knowledge distillation across fluorescence and label-free imaging channels, and alignment of cell state dynamics.

### Preprint

[DynaCLR on arXiv](https://arxiv.org/abs/2410.11281):

![DynaCLR schematic](https://github.com/mehta-lab/VisCy/blob/e5318d88e2bb5d404d3bae8d633b8cc07b1fbd61/docs/figures/DynaCLR_schematic_v2.png?raw=true)

### Demo

- [DynaCLR demos](examples/DynaCLR/README.md)

- Example test dataset, model checkpoint, and predictions can be found
[here](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_demo/).

- See tutorial on exploration of learned embeddings with napari-iohub
[here](https://github.com/czbiohub-sf/napari-iohub/wiki/View-tracked-cells-and-their-associated-predictions/).

## Installation

1. We recommend using a new Conda/virtual environment.

    ```sh
    conda create --name viscy python=3.11
    # OR specify a custom path since the dependencies are large:
    # conda create --prefix /path/to/conda/envs/viscy python=3.11
    ```

2. Install a released version of VisCy from PyPI:

    ```sh
    pip install viscy
    ```

    If evaluating virtually stained images for segmentation tasks,
    install additional dependencies:

    ```sh
    pip install "viscy[metrics]"
    ```

    Visualizing the model architecture requires `visual` dependencies:

    ```sh
    pip install "viscy[visual]"
    ```

3. Verify installation by accessing the CLI help message:

    ```sh
    viscy --help
    ```

For development installation, see [the contributing guide](https://github.com/mehta-lab/VisCy/blob/main/CONTRIBUTING.md).

## Additional Notes

The pipeline is built using the [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) framework.
The [iohub](https://github.com/czbiohub-sf/iohub) library is used
for reading and writing data in [OME-Zarr](https://www.nature.com/articles/s41592-021-01326-w) format.

The full functionality is tested on Linux `x86_64` with NVIDIA Ampere/Hopper GPUs (CUDA 12.6).
Some features (e.g. mixed precision and distributed training) may not be available with other setups,
see [PyTorch documentation](https://pytorch.org) for details.
