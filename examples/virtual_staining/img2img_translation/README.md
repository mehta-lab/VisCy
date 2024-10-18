# Demo: Image to Image translation (Phase to fluorescence and fluorescence to phase)

# Overview

In this exercise, you will:
- Train a phase to fluorescence model to virtually stain cell nuclei and membrane
- Train a model with the opposite task going from fluorescence to phase


## Setup

**Close your shell, and login again.** 

Run the setup script to create the environment for this exercise and download the dataset.
```bash
source setup.sh
```

Activate your environment
```bash
conda activate img2img
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

If `img2img` is not available as a kernel in jupyter, run:

```
python -m ipykernel install --user --name=img2img
```
