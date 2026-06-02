# Get started

## Install

=== ":material-language-python: pip"

    Install a single package:

    ```sh
    pip install viscy-data
    ```

    …or the umbrella package, which pulls in every `viscy-*` package at once:

    ```sh
    pip install viscy
    ```

=== ":material-package-variant-closed: uv"

    Add a package to your project:

    ```sh
    uv add viscy-data
    ```

    Or grab the whole stack:

    ```sh
    uv add viscy
    ```

=== ":material-microscope: From source"

    For development, clone the monorepo and sync the full workspace:

    ```sh
    git clone https://github.com/mehta-lab/VisCy.git
    cd VisCy
    uv venv -p 3.13            # (1)!
    uv sync --all-packages --all-extras   # (2)!
    ```

    1. VisCy targets Python ≥ 3.12. 3.13 is a safe default.
    2. `--all-packages` installs every workspace package in editable mode;
       `--all-extras` adds optional dependencies (e.g. `triplet`, `livecell`).

    Run anything inside the environment with `uv run <command>`.

!!! tip "On HPC Systems"

    Symlink the uv cache out of your home directory before syncing as home quotas can
    fill up fast:

    ```sh
    mkdir -p /hpc/mydata/first.last/.cache/uv && ln -s /hpc/mydata/first.last/.cache/uv ~/.cache/uv
    ```

## Verify the install

```python
import viscy_data, viscy_models, viscy_transforms, viscy_utils  # (1)!

print(viscy_models.__version__)  # (2)!
```

1. The import names use underscores (`viscy_data`), while the PyPI distribution
   names use hyphens (`viscy-data`).
2. Versions are derived from git tags via `uv-dynamic-versioning`.

## Build the docs locally

These docs are built with [Zensical](https://zensical.org/). The dev server
live-reloads as you edit:

```sh
uv sync --all-packages --group doc   # (1)!
uv run python docs/_gen_versions.py  # (2)!
uv run zensical serve
```

1. The `doc` dependency group lives only on the root project — it pulls in
   `zensical`, `mkdocstrings-python`, and `mike`. No subpackage carries doc deps.
2. Regenerates the package version table in `docs/packages/index.md`.


!!! note "Looking for an application?"

    Cytoland (virtual staining) and DynaCLR (contrastive learning) build on these
    packages, and their documentation will be coming soon.
