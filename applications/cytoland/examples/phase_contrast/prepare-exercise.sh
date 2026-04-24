# Run ruff format on .py files
# ruff format solution.py

# Convert .py to ipynb

# "cell_metadata_filter": "all" preserve cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py
jupyter nbconvert solution.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags task --to notebook --output solution.ipynb
