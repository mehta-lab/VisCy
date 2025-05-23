# Contributing to viscy

## Development installation

Clone or fork the repository,
then make an editable installation with all the development dependencies:

```sh
# in project root directory (parent folder of pyproject.toml)
pip install -e ".[dev]"
```

## CI requirements

Lint with Ruff:

```sh
ruff check viscy
```

Format the code with Black:

```sh
black viscy
```

Run tests with `pytest`:

```sh
pytest -v
```
