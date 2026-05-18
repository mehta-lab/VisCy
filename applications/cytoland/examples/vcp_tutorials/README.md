# Virtual Cell Platform Tutorials

This directory contains tutorial scripts for the Virtual Cell Platform.
Jupyter notebooks can be generated from these Python scripts as described below.

- [Quick Start](quick_start.py):
get started with model inference in Python with a A549 cell dataset.
- [CLI inference and visualization](hek293t.py):
run inference from CLI on a HEK293T cell dataset and visualize the results.
- [Virtual staining _in vivo_](neuromast.py):
compare virtual staining and fluorescence in a time-lapse dataset of the zebrafish neuromast.

## Development

The development happens on the Python scripts,
which are converted to Jupyter notebooks with:

```sh
# TODO: change the file name at the end to be the script to convert
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update quick_start.py
```
