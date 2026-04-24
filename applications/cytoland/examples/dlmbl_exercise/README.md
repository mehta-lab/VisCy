# Exercise 6: Image translation - Part 1

This demo script was developed for the DL@MBL 2024 course by Eduardo Hirata-Miyasaki, Ziwen Liu and Shalin Mehta, with many inputs and bugfixes by [Morgan Schwartz](https://github.com/msschwartz21), [Caroline Malin-Mayor](https://github.com/cmalinmayor), and [Peter Park](https://github.com/peterhpark).


# Image translation (Virtual Staining)

Written by Eduardo Hirata-Miyasaki, Ziwen Liu, and Shalin Mehta, CZ Biohub San Francisco.

## Overview

In this exercise, we will predict fluorescence images of nuclei and plasma membrane markers from quantitative phase images of cells, i.e., we will _virtually stain_ the nuclei and plasma membrane visible in the phase image.
This is an example of an image translation task. We will apply spatial and intensity augmentations to train robust models and evaluate their performance. Finally, we will explore the opposite process of predicting a phase image from a fluorescence membrane label.

[![HEK293T](https://raw.githubusercontent.com/mehta-lab/VisCy/main/docs/figures/svideo_1.png)](https://github.com/mehta-lab/VisCy/assets/67518483/d53a81eb-eb37-44f3-b522-8bd7bddc7755)
(Click on image to play video)

## Goals

### Part 1: Learn to use iohub (I/O library), VisCy dataloaders, and TensorBoard.

  - Use a OME-Zarr dataset of 34 FOVs of adenocarcinomic human alveolar basal epithelial cells (A549),
  each FOV has 3 channels (phase, nuclei, and cell membrane).
  The nuclei were stained with DAPI and the cell membrane with Cellmask.
  - Explore OME-Zarr using [iohub](https://czbiohub-sf.github.io/iohub/main/index.html)
  and the high-content-screen (HCS) format.
  - Use [MONAI](https://monai.io/) to implement data augmentations.

### Part 2: Train and evaluate the model to translate phase into fluorescence, and vice versa.
  - Train a 2D UNeXt2 model to predict nuclei and membrane from phase images.
  - Compare the performance of the trained model and a pre-trained model.
  - Evaluate the model using pixel-level and instance-level metrics.


Checkout [VisCy](https://github.com/mehta-lab/VisCy/tree/main/examples/demos),
our deep learning pipeline for training and deploying computer vision models
for image-based phenotyping including the robust virtual staining of landmark organelles.
VisCy exploits recent advances in data and metadata formats
([OME-zarr](https://www.nature.com/articles/s41592-021-01326-w)) and DL frameworks,
[PyTorch Lightning](https://lightning.ai/) and [MONAI](https://monai.io/).

## Setup

From the exercise folder, run:

```bash
cd applications/cytoland/examples/dlmbl_exercise
bash setup.sh
```

The script will:

- Install [`uv`](https://docs.astral.sh/uv/) if it isn't already on your PATH.
- Create a Python 3.11 virtual environment at `./.venv`.
- Install `cytoland` (editable) plus the tutorial extras:
  `cellpose`, `torchview`, `jupyter`, `ipykernel`, `ipywidgets`, `jupytext`.
- Register the venv as a Jupyter kernel named **`06_image_translation`**
  (display name: *Python (06_image_translation)*).
- Download the training / test OME-Zarr datasets and the VSCyto2D
  pretrained checkpoint into `~/data/06_image_translation/`.

Everything is self-contained inside this folder — no conda required.

## Use VSCode

Install VSCode and the Python + Jupyter extensions, then open
[`solution.py`](solution.py) and pick the **Python (06_image_translation)**
kernel from the top-right kernel selector. The script uses
[cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py), so
you can execute each `# %%` block interactively.

## Use Jupyter Notebook

Generate a notebook from the solution script and launch Jupyter:

```bash
./.venv/bin/jupytext --to ipynb solution.py
./.venv/bin/jupyter notebook solution.ipynb
```

Pick **Python (06_image_translation)** as the kernel.

If the kernel is missing (e.g. you reinstalled the venv), re-register it:

```bash
./.venv/bin/python -m ipykernel install --user \
    --name 06_image_translation \
    --display-name "Python (06_image_translation)"
```

### References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2024) Robust Virtual Staining of Cellular Landmarks](https://www.biorxiv.org/content/10.1101/2024.05.31.596901v2.full.pdf)
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. eLife](https://elifesciences.org/articles/55502)
