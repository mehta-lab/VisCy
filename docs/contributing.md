# Contributing

Thanks for your interest in contributing to VisCy! This page covers the whole
workflow — setup, code, tests, docs, and releases.[^repo]

[^repo]:
    This mirrors the repo's
    [`CONTRIBUTING.md`](https://github.com/mehta-lab/VisCy/blob/main/CONTRIBUTING.md);
    edit both if you change the process.

## Set up your environment

VisCy uses [uv](https://docs.astral.sh/uv/) and is a
[uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo.

**1. Install uv** — see the [installation docs](https://docs.astral.sh/uv/getting-started/installation/).

!!! warning "On HPC"

    Your home directory fills up fast. Symlink the uv cache to scratch storage
    *before* the first sync:

    ```sh
    mkdir -p /hpc/mydata/first.last/.cache/uv && ln -s /hpc/mydata/first.last/.cache/uv ~/.cache/uv
    ```

**2. Get the code:**

=== ":material-source-branch: Push access"

    ```sh
    git clone https://github.com/mehta-lab/VisCy.git
    ```

=== ":material-source-fork: Fork"

    [Fork the repo](https://github.com/mehta-lab/VisCy/fork), then clone your fork.
    See [forking a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

**3. Create the environment and sync:**

```sh
cd VisCy/
uv venv -p 3.13                 # (1)!
uv sync --all-packages          # (2)!
```

1. VisCy targets Python ≥ 3.12.
2. Installs every workspace package plus the default `dev` dependency group.

## Report an issue

We track bugs, feature requests, and support in
[issues](https://github.com/mehta-lab/VisCy/issues). Search existing issues
(including closed ones) before opening a new one.

## Make changes

Every change to `main` goes through a
[pull request](https://github.com/mehta-lab/VisCy/pulls). Reference the issue it
addresses, or describe the fix or feature in the PR.

!!! tip "Packages vs. applications"

    Shared code belongs in `packages/`. Applications consume packages and never
    import each other — the dependency graph flows `applications/ → packages/`.

### Repository structure

```text
viscy/
├─ pyproject.toml          # root workspace config (ruff, pytest, uv)
├─ packages/               # independently published packages
│  ├─ viscy-data/
│  ├─ viscy-models/
│  ├─ viscy-transforms/
│  └─ viscy-utils/
├─ applications/           # self-contained research apps
├─ docs/                   # this documentation site
└─ src/viscy/              # umbrella package (re-exports)
```

Each `packages/` member is developed in isolation, published to PyPI separately,
and installed on its own.

### Test

Add tests for every code change.

```sh
uv run pytest                          # everything
uv run pytest packages/viscy-data/     # one package
uv run pytest --cov=viscy_data         # with coverage
```

### Code style

We use [prek](https://github.com/j178/prek) to format and lint on commit. Install
the hooks once:

```sh
uvx prek install
uvx prek run --all-files   # run manually
```

[ruff](https://docs.astral.sh/ruff/) handles linting and formatting; docstrings
follow the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).

!!! warning "Ruff config lives only in the root"

    All `[tool.ruff.*]` settings belong in the root `pyproject.toml`. Ruff does not
    inherit config — a `[tool.ruff.*]` section in a subpackage silently overrides
    the entire root config.

## Documentation

[Zensical](https://zensical.org/) builds one site for the whole monorepo, from
`zensical.toml` and `docs/` at the repo root. Doc tools live in the root `doc`
group.

=== ":material-eye: Serve (live reload)"

    ```sh
    uv sync --all-packages --group doc    # (1)!
    uv run python docs/_gen_versions.py   # (2)!
    uv run zensical serve                 # http://localhost:8000
    ```

    1. `--all-packages` matters: mkdocstrings imports the packages to read docstrings.
    2. Rewrites the package version table in `docs/packages/index.md`.

=== ":material-hammer: Static build"

    ```sh
    uv run zensical build --clean   # output in site/ (git-ignored)
    ```

Authoring notes:

- Markdown lives in `docs/`; the `nav` table in `zensical.toml` sets page order.
- `::: viscy_data` renders a package's API from its docstrings. A template override
  (`docs/_templates/python/material/module.html.jinja`) hides the package's
  top-level docstring — write overview prose in the Markdown page instead.
- `zensical.toml` enables only the Markdown extensions in use. Adding new syntax
  (math, keyboard keys, …)? Enable its extension there, or it renders as text.

## Release a package

Versions come from git tags via
[`uv-dynamic-versioning`](https://github.com/ninoseki/uv-dynamic-versioning); the
tag **prefix** picks the package.

| Package | Tag prefix | Example |
|---|---|---|
| `viscy-data` | `viscy-data-` | `viscy-data-v0.2.1` |
| `viscy-models` | `viscy-models-` | `viscy-models-v0.4.0` |
| `viscy-transforms` | `viscy-transforms-` | `viscy-transforms-v0.1.3` |
| `viscy-utils` | `viscy-utils-` | `viscy-utils-v0.3.0` |
| `viscy` (umbrella) | none | `v0.6.0` |

```sh
git checkout main && git pull && git fetch --tags
git tag viscy-data-v0.2.1                          # (1)!
uv build --package viscy-data --out-dir dist/      # (2)!
ls dist/                                           # viscy_data-0.2.1-...whl  (3)!
git push origin viscy-data-v0.2.1                  # (4)!
```

1. One tag per package you ship.
2. Version is derived from the tag.
3. Verify the wheel name matches the tag before pushing.
4. Nothing is public until you push the tag.

!!! note "Docs deploy automatically"

    CI publishes the site with [`mike`](https://github.com/squidfunk/mike): merging
    to `main` updates the `dev` build; a `vX.Y.Z` umbrella tag publishes a release
    and moves `stable`.

## Before you open a PR

- [ ] Tests cover the change and `uv run pytest` passes
- [ ] `uvx prek run --all-files` is clean
- [ ] Docs build (`uv run zensical build --clean`) with no warnings
- [ ] New doc pages are listed in the `nav` table
- [ ] The PR references an issue or describes the change
