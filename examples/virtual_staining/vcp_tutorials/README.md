# Virtual Cell Platform Tutorials

This directory contains tutorial notebooks for the Virtual Cell Platform,
available in both Python scripts and Jupyter notebooks.

- [Quick Start](quick_start.ipynb):
get started with model inference in Python with a A549 cell dataset.
- [CLI inference and visualization](hek293t.ipynb):
run inference from CLI on a HEK293T cell dataset and visualize the results.
- [Virtual staining _in vivo_](neuromast.ipynb):
compare virtual staining and fluorescence in a time-lapse dataset of the zebrafish neuromast.

## Development

The development happens on the Python scripts,
which are converted to Jupyter notebooks with:

```sh
# TODO: change the file name at the end to be the script to convert
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update quick_start.py
```
