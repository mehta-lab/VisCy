# CLAUDE.md

Project-specific instructions for Claude Code sessions in this repository.

## Repository Structure

VisCy is a **uv workspace monorepo**. Sub-packages live under `packages/`:

```
pyproject.toml              # Root config (ruff, pytest, uv workspace)
packages/
  viscy-data/               # Data loading and Lightning DataModules
  viscy-models/             # Neural network architectures
  viscy-transforms/         # Image transforms
src/viscy/                  # Umbrella package (re-exports)
```

## Code Style

- **Ruff config is centralized in the root `pyproject.toml` only.**
  Sub-packages must NOT have their own `[tool.ruff.*]` sections.
  Ruff does not inherit config — any `[tool.ruff.*]` in a sub-package
  silently overrides the entire root config (including `lint.select`,
  `per-file-ignores`, etc.).
- Docstrings use **numpy style** (`convention = "numpy"`).
- Lint rules: `D, E, F, I, NPY, PD, W`.
- `D` rules are ignored in `**/tests/**` and notebooks.
- Format: double quotes, spaces, 120 char line length.

## Testing

```sh
uv run pytest                          # all tests
uv run pytest packages/viscy-data/     # single package (data)
uv run pytest packages/viscy-models/   # single package (models)
```

## Common Commands

```sh
uvx ruff check packages/       # lint
uvx ruff check --fix packages/  # lint + auto-fix
uvx ruff format packages/       # format
```
