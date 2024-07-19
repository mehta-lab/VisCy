# Exercise 6: Image translation - Part 1

This demo script was developed for the DL@MBL 2024 course by Eduardo Hirata-Miyasaki, Ziwen Liu and Shalin Mehta, with many inputs and bugfixes by [Morgan Schwartz](https://github.com/msschwartz21), [Caroline Malin-Mayor](https://github.com/cmalinmayor), and [Peter Park](https://github.com/peterhpark).  




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
