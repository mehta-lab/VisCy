# DynaCLR Quick Start

Get started with model inference in Python with an A549 cell dataset.

- [quickstart.ipynb](quickstart.ipynb) — Jupyter notebook
- [quickstart.py](quickstart.py) — Python script

## Development

The development happens on the Python scripts,
which are converted to Jupyter notebooks with:

```sh
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update quickstart.py
```
