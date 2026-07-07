# examples

Runnable, user-facing entry points into DynaCell: an exploratory notebook and the interactive Hugging Face
Space. Generic fit/predict config stubs live separately under [`../configs/examples/`](../configs/README.md).

## Contents

- **notebooks/** — `get-to-know-dynacell-dataset.ipynb`, a guided tour of the DynaCell OME-Zarr dataset
  (linked from the top-level [README](../README.md)).
- **hf_demo/** — the DynaCell virtual-staining Hugging Face Space (Gradio, ZeroGPU) and its deploy tooling:
  - `hf_space/` — the deployed app (`app.py`, `predict_runner.py`, `config_templates/`, `requirements.txt`) and
    its own [Space README](hf_demo/hf_space/README.md) (HF card front-matter).
  - `upload_checkpoints.py`, `upload_hf_space.py` — push checkpoints / the Space to HF.
  - `cards/` — HF repo cards for the checkpoints and demo-data repos.
  - `AGENT.md` — how the demo is structured and deployed (see also the `hf-dynacell` skill).

## Navigation

- Up: [applications/dynacell](../README.md)
