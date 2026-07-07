# configs

Hydra/LightningCLI YAML configs for the DynaCell benchmark: reusable fragments, generic examples, and the
runnable benchmark leaves that drive every fit/predict/eval in the paper.

## Contents

- **[benchmarks/](benchmarks/README.md)** — the campaign configs. Runnable leaves composed from shared axes, one
  per `(organelle, train_set, model)` fit / `(…, predict_set)` predict / eval. This is the tree you run.
- **recipes/** — reusable composition fragments consumed by other configs, grouped by axis:
  `data/` (the `hcs_phase_fluor_3d` DataModule), `models/` (per-architecture model init args), `topology/`
  (single- vs multi-GPU / GAN DDP), `trainer/` (`fit`/`predict` trainer blocks), `modes/` (`spotlight`).
- **examples/** — generic `fit.yml` + `predict.yml` pair per model family (`celldiff/`, `fnet3d/`, `unetvit3d/`,
  `unext2/`) with `#TODO` data-path placeholders. Starting points for a one-off run; see the top-level
  [Quickstart](../README.md#quickstart).
- **evaluations/** — legacy per-model shell scripts (`run_eval_*.sh`) that invoke `dynacell evaluate`
  directly, grouped by model (`celldiff/`, `fnet3d/`, `unetvit3d/`, `unext2/`, `vscyto3d/`). Predate the
  composed `benchmarks/*/eval__*.yaml` leaves; kept for reference.
- **movies/** — a small crop demo (`2024_11_21_A549_TOMM20_DENV_crop/`): predict configs that render a
  cropped A549 TOMM20/DENV time-lapse with iPSC- and joint-trained regression models.

## Navigation

- Up: [applications/dynacell](../README.md)
- Subdirectories: [benchmarks/](benchmarks/README.md)
