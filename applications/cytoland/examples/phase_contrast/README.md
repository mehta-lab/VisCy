# Demo: Virtual staining of phase contrast data

# Overview:

Generalization to Zernike phase contrast images. This demo showcases the use of VSCyto3D model with and without augmentations on Zernike phase contrast data.

## Setup

Run the setup script from this examples folder to create the environment and download the dataset:
```bash
cd applications/cytoland/examples/phase_contrast
source setup.sh
```
The script resolves the cytoland package relative to its own location, so it works regardless of the current working directory.

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
