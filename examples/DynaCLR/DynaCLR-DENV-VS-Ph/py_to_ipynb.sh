# Run black on .py files
black demo_infection.py

# Convert .py to ipynb
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update demo_infection.py