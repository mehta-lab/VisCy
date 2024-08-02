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

Make sure that you are inside of the `image_translation` folder by using the `cd` command to change directories if needed.

Make sure that you can use conda to switch environments.

```bash
conda init
```

**Close your shell, and login again.** 

Run the setup script to create the environment for this exercise and download the dataset.
```bash
sh setup.sh
```
Activate your environment
```bash
conda activate 06_image_translation
```

## Use vscode

Install vscode, install jupyter extension inside vscode, and setup [cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py). Open [solution.py](solution.py) and run the script interactively.

## Use Jupyter Notebook

The matching exercise and solution notebooks can be found [here](https://github.com/dlmbl/image_translation/tree/28e0e515b4a8ad3f392a69c8341e105f730d204f) on the course repository.

Launch a jupyter environment

```
jupyter notebook
```

...and continue with the instructions in the notebook.

If `06_image_translation` is not available as a kernel in jupyter, run:

```
python -m ipykernel install --user --name=06_image_translation
```

### References

- [Liu, Z. and Hirata-Miyasaki, E. et al. (2024) Robust Virtual Staining of Cellular Landmarks](https://www.biorxiv.org/content/10.1101/2024.05.31.596901v2.full.pdf)
- [Guo et al. (2020) Revealing architectural order with quantitative label-free imaging and deep learning. eLife](https://elifesciences.org/articles/55502)
