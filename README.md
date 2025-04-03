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
  - Robust virtual staining of landmark organelles
- Image representation learning
  - Self-supervised learning of the cell state and organelle phenotypes
- Semantic segmentation
  - Supervised learning of of cell state (e.g. state of infection)

> **Note:**
> VisCy has been extensively tested for the image translation task. The code for other tasks is under active development. Frequent breaking changes are expected in the main branch as we unify the codebase for above tasks. If you are looking for a well-tested version for virtual staining, please use release `0.2.1` from PyPI.

## Virtual staining

### Demos

- [Interactive 2D virtual staining demo of cell nuclei and membrane from labelfree data 🤗](https://compmicro-czb-virtualstaining.hf.space)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/compmicro-czb/VirtualStaining)

<p align="center">
<a href="https://compmicro-czb-virtualstaining.hf.space" target="_blank" rel="noopener noreferrer">
<img src="https://github.com/mehta-lab/VisCy/blob/7d3bed92e91fb44611a45be5350320d65ffcc111/docs/figures/vs_hf_demo.gif?raw=true" alt="Virtual Staining App Demo" height="300px" />
</a>
</p>

- [Virtual staining exercise](https://github.com/mehta-lab/VisCy/blob/46beba4ecc8c4f312fda0b04d5229631a41b6cb5/examples/virtual_staining/dlmbl_exercise/solution.ipynb):
Notebook illustrating how to use VisCy to train, predict and evaluate the VSCyto2D model. This notebook was developed for the [DL@MBL2024](https://github.com/dlmbl/DL-MBL-2024) course and uses UNeXt2 architecture.

- [Image translation demo](https://github.com/mehta-lab/VisCy/blob/92215bc1387316f3af49c83c321b9d134d871116/examples/virtual_staining/img2img_translation/solution.ipynb): Fluorescence images can be predicted from label-free images. Can we predict label-free image from fluorescence? Find out using this notebook.

- [Training Virtual Staining Models via CLI](https://github.com/mehta-lab/VisCy/wiki/virtual-staining-instructions):
Instructions for how to train and run inference on ViSCy's virtual staining models (*VSCyto3D*, *VSCyto2D* and *VSNeuromast*).

### Gallery

Below are some examples of virtually stained images (click to play videos).
See the full gallery [here](https://github.com/mehta-lab/VisCy/wiki/Gallery).

| VSCyto3D | VSNeuromast | VSCyto2D |
|:---:|:---:|:---:|
| [![HEK293T](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_1.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755) | [![Neuromast](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_3.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/4cef8333-895c-486c-b260-167debb7fd64) | [![A549](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/figures/svideo_5.png?raw=true)](https://github.com/mehta-lab/VisCy/assets/67518483/287737dd-6b74-4ce3-8ee5-25fbf8be0018) |

### Reference

The virtual staining models and training protocols are reported in our recent [preprint on robust virtual staining](https://www.biorxiv.org/content/10.1101/2024.05.31.596901).

This package evolved from the [TensorFlow version of virtual staining pipeline](https://github.com/mehta-lab/microDL), which we reported in [this paper in 2020](https://elifesciences.org/articles/55502).

<details>
  <summary>Liu, Hirata-Miyasaki et al., 2024</summary>

  <pre><code>
  @article {Liu2024.05.31.596901,
          author = {Liu, Ziwen and Hirata-Miyasaki, Eduardo and Pradeep, Soorya and Rahm, Johanna and Foley, Christian and Chandler, Talon and Ivanov, Ivan and Woosley, Hunter and Lao, Tiger and Balasubramanian, Akilandeswari and Liu, Chad and Leonetti, Manu and Arias, Carolina and Jacobo, Adrian and Mehta, Shalin B.},
          title = {Robust virtual staining of landmark organelles},
          elocation-id = {2024.05.31.596901},
          year = {2024},
          doi = {10.1101/2024.05.31.596901},
          publisher = {Cold Spring Harbor Laboratory},
          URL = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596901},
          eprint = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596901.full.pdf},
          journal = {bioRxiv}
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

### Library of virtual staining (VS) models

The robust virtual staining models (i.e *VSCyto2D*, *VSCyto3D*, *VSNeuromast*), and fine-tuned models can be found [here](https://github.com/mehta-lab/VisCy/wiki/Library-of-virtual-staining-(VS)-Models)

### Pipeline

A full illustration of the virtual staining pipeline can be found [here](https://github.com/mehta-lab/VisCy/blob/dde3e27482e58a30f7c202e56d89378031180c75/docs/virtual_staining.md).

## Image representation learning

We are currently developing self-supervised representation learning to map cell state dynamics in response to perturbations,
with focus on cell and organelle remodeling due to viral infection.

See our recent work on temporally regularized contrastive sampling method
for representation learning on [arXiv](https://arxiv.org/abs/2410.11281).

<details>
 <summary> Pradeep, Imran, Liu et al., 2024 </summary>

  <pre><code>
@misc{pradeep_contrastive_2024,
      title={Contrastive learning of cell state dynamics in response to perturbations},
      author={Soorya Pradeep and Alishba Imran and Ziwen Liu and Taylla Milena Theodoro and Eduardo Hirata-Miyasaki and Ivan Ivanov and Madhura Bhave and Sudip Khadka and Hunter Woosley and Carolina Arias and Shalin B. Mehta},
      year={2024},
      eprint={2410.11281},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.11281},
}
    </code></pre>
  </details>

### Workflow demo

- Example test dataset, model checkpoint, and predictions can be found
[here](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_demo/).

- See tutorial on exploration of learned embeddings with napari-iohub
[here](https://github.com/czbiohub-sf/napari-iohub/wiki/View-tracked-cells-and-their-associated-predictions/).

![DynaCLR schematic](https://github.com/mehta-lab/VisCy/blob/9eaab7eca50d684d8a473ad9da089aeab0e8f6a0/docs/figures/dynaCLR_schematic.png?raw=true)

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
