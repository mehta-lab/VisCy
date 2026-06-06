---
name: hf-dynacell
description: >-
  Develop, deploy, and maintain the DynaCell virtual-staining HuggingFace demo
  hosted at biohub/dynacell (ZeroGPU). Covers hosting in the biohub Dynacell
  resource group, the private-artifacts + HF_TOKEN access model, ZeroGPU
  configuration, and the deploy/smoke-test workflow. Use when the user asks to
  "push the dynacell space", "update the dynacell demo", "upload checkpoints",
  "deploy the hf space", mentions biohub/dynacell, or works in
  applications/dynacell/examples/hf_demo/.
---

# DynaCell HuggingFace Demo — maintenance

Operational reference for the demo. Content + file layout live in
`applications/dynacell/examples/hf_demo/AGENT.md`; this skill holds hosting,
access, ZeroGPU, and deploy knowledge.

## Hosting — biohub org, Dynacell resource group

Three repos under **`biohub`**, all in the **Dynacell** resource group
(`id 6a234bb4507cbbbb04456767`):

| Repo | Type | Visibility |
| --- | --- | --- |
| `biohub/dynacell` | Space (Gradio, ZeroGPU) | private (→ public at launch) |
| `biohub/dynacell-checkpoints` | model | **private** |
| `biohub/dynacell-demo-data` | dataset | **private** |

RG members who can push: `edyoshikun`, `shalinmehta` (admin); `alxndrkalinin`,
`dihan-zheng` (write). Migrated from `dihan-zheng/*` (byte-for-byte mirror).

Moving an *existing* repo into a resource group needs org-admin. As org-write
members, create repos directly in the group with
`create_repo(..., resource_group_id="6a234bb4507cbbbb04456767")` — the upload
scripts do this.

## Access model — public demo, private artifacts

Decision (2026-06): checkpoints and demo data stay **private** even when the
Space is public. The Space downloads them server-side via an `HF_TOKEN`
**secret** (Space → Settings → Variables and secrets) — a **fine-grained,
read-only** token scoped to exactly those two repos. Public visitors get only
outputs (plots, PCC, GIFs); raw artifacts stay gated. If the token is revoked or
loses RG read access, runtime downloads break (rotate via "Replace").

## ZeroGPU configuration (validated 2026-06)

- Hardware `zero-a10g` (RTX Pro 6000 Blackwell, 48 GB). Enterprise quota
  60 GPU-min/day, shared.
- **`python_version: "3.12"` required** in the Space README — the default base
  image is 3.10, but VisCy needs `>=3.12` (else
  `BUILD_ERROR: viscy-data requires ... not in '>=3.12'`). ZeroGPU supports
  Python 3.12.12 / 3.10.13, PyTorch 2.8→latest, Gradio SDK only.
- `requirements.txt` includes `spaces`; `@spaces.GPU(duration=120)` decorates
  `run_prediction` and `compute_trajectory` (`predict_runner.py`).
- A subprocess inside `@spaces.GPU` inherits the GPU, so Tab 1's `dynacell
  predict` subprocess works on ZeroGPU. `gr.Progress` args work under the
  decorator. Keep `import torch` lazy (no CUDA at import).
- Tuning: CELL-Diff at 50–100 ODE steps can approach the 120 s ceiling on a cold
  call; raise `duration` or use a dynamic-duration callable if timeouts appear.

## Code pointers

`CHECKPOINT_REPO` (`predict_runner.py`), `_DEMO_REPO` (`app.py`), and the README
`models:`/`datasets:` fields all point at `biohub/*`.

## Deploy

1. `hf auth login` (token with write to the Dynacell RG).
2. Edit `hf_space/`.
3. Push: `uv run python applications/dynacell/examples/hf_demo/upload_hf_space.py`.
4. Watch: `HfApi().get_space_runtime("biohub/dynacell").stage`. Build/run logs at
   `GET /api/spaces/biohub/dynacell/logs/{build,run}` are **SSE streams** — read
   with `stream=True` + a line cap; a plain GET hangs.

Re-publish checkpoints from HPC: edit the path map in `upload_checkpoints.py`
and run it where `/hpc/projects/...` is mounted.

## Smoke test

Private Space needs a token: `Client("biohub/dynacell", token=get_token())`,
then `view_api()` and `predict(..., api_name="/run_demo")`. Last validated:
`/load_demo_data` OK; `/run_demo` (vscyto3d) → PCC 0.95 in 35 s;
`/run_trajectory_demo` (celldiff, 10 steps) → GIF in 74 s.

## Launch checklist

- Optionally flip the two artifact repos + Space public; if public, remove the
  `HF_TOKEN` secret.
- Fill citation/DOI TODOs in `cards/*.md` and the uploaded READMEs.
- Re-pin `requirements.txt` VisCy refs off `@dynacell-models` once merged/renamed.
