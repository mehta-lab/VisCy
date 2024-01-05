# Contributing to viscy

## Development installation

Clone or fork the repository,
then make an editable installation with all the optional dependencies:

```sh
# in project root directory (parent folder of pyproject.toml)
pip install -e ".[dev,visual,metrics]"
```

## Testing

Run tests with `pytest`:

```sh
pytest -v
```
