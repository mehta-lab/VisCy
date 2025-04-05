# Virtual Cell Platform Tutorials

This directory contains tutorial notebooks for the Virtual Cell Platform,
available in both Python scripts and Jupyter notebooks.

- [Quick Start](quick_start.ipynb): get started with model inference in Python.

## Development

The development happens on the Python scripts,
which are converted to Jupyter notebooks with:

```sh
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update quick_start.py
```
