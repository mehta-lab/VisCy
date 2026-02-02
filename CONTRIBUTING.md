# Contributing guide

Thanks for your interest in contributing to VisCy!

Please see the following steps for our workflow.

## Getting started

Please read the [README](./README.md) for an overview of the project
and how you can install and use the package.

## Issues

We use [issues](https://github.com/mehta-lab/VisCy/issues) to track
bug reports, feature requests, and provide user support.

Before opening a new issue, please first search existing issues (including closed ones)
to see if there is an existing discussion about it.

## Making changes

Any change made to the `main` branch needs to be proposed in a
[pull request](https://github.com/mehta-lab/VisCy/pulls) (PR).

If there is an issue that can be addressed by the PR, please reference it.
If there is not a relevant issue, please either open an issue first,
or describe the bug fixed or feature implemented in the PR.

### Setting up development environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and is organized as a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo.

#### Install uv

See [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

#### Clone the repository

If you have push permission to the repository:

```sh
git clone https://github.com/mehta-lab/VisCy.git
```

Otherwise, you can follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/mehta-lab/VisCy/fork) the repository.

#### Install dependencies

First, create a virtual environment with a supported Python version (3.11-3.13):

```sh
cd VisCy/
uv venv -p 3.13  # or 3.11 or 3.12
```

This makes a virtual environment in `.venv/` where the dependencies will be installed.

Then sync dependencies:

```sh
uv sync
```

> **Note**: `uv sync` installs the [`dev` group by default](https://docs.astral.sh/uv/concepts/projects/sync/#syncing-development-dependencies),
> which includes all development dependencies. See [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) for more details.

#### Repository structure

VisCy is organized as a workspace monorepo:

```shell
viscy/
├── pyproject.toml          # Root workspace configuration
├── packages/
│   └── viscy-transforms/   # Image transforms subpackage
│       ├── pyproject.toml
│       └── src/
│           └── viscy_transforms/
└── src/
    └── viscy/              # Umbrella package (re-exports from subpackages)
```

Each package in `packages/` is an independent Python package that can be:

- Developed in isolation
- Published to PyPI separately
- Installed independently by users

```python
# Import directly from subpackages
from viscy_transforms import NormalizeSampled
```

Then make the changes and [track them with Git](https://docs.github.com/en/get-started/using-git/about-git#example-contribute-to-an-existing-repository).

### Testing

If you made code changes, make sure that there are also tests for them!
Local test runs and coverage check can be invoked by:

```sh
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/viscy-transforms/

# Run with coverage
uv run pytest --cov=viscy_transforms
```

### Code style

We use [prek](https://github.com/j178/prek) (a faster [pre-commit](https://pre-commit.com/) runner)
to automatically format and lint code prior to each commit.
To minimize test errors when submitting pull requests, install the hooks:

```bash
uvx prek install
```

> `uvx` runs tools in isolated, cached environments—no binaries added to your PATH
> and no dependencies installed in your project venv.

To run manually:

```bash
uvx prek run              # run on staged files only
uvx prek run --all-files  # run on all files
```

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uvx ruff check .          # lint
uvx ruff check --fix .    # lint and auto-fix
uvx ruff format .         # format
```

When executed within the project root directory, ruff automatically uses
the [project settings](./pyproject.toml).

## Useful links

### uv documentation

- [uv Overview](https://docs.astral.sh/uv/)
- [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/) - Sync dependencies and packages
- [uv Workspaces](https://docs.astral.sh/uv/concepts/workspaces/) - Monorepo management
- [uv add](https://docs.astral.sh/uv/concepts/projects/dependencies/) - Adding dependencies
- [uv run](https://docs.astral.sh/uv/concepts/projects/run/) - Running commands in the environment
- [Dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)

### Related tools

- [ruff](https://docs.astral.sh/ruff/) - Fast Python linter and formatter
- [pytest](https://docs.pytest.org/) - Testing framework
- [prek](https://github.com/j178/prek) - Fast pre-commit runner
