# Demo: Virtual staining of phase contrast data

# Overview:

Generalization to Zernike phase contrast images. This demo showcases the use of VSCyto3D model with and without augmentations on Zernike phase contrast data.

## Setup

Run the setup script to create the environment for this exercise and download the dataset.
```bash
source setup.sh
```

Activate your environment
```bash
conda activate vs_Phc
```

## Use vscode

Install vscode, install jupyter extension inside vscode, and setup [cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py). Open [solution.py](solution.py) and run the script interactively.

## Use Jupyter Notebook

Launch a jupyter environment

```
jupyter notebook
```

...and continue with the instructions in the notebook.

If `vs_Phc` is not available as a kernel in jupyter, run:

```
python -m ipykernel install --user --name=vs_Phc
```
