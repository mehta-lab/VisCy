# configs/benchmarks

Runnable benchmark campaigns. Each task subfolder holds composed leaves that are submitted end-to-end via
`tools/submit_benchmark_job.py` (or run directly with `uv run dynacell <sub> -c <leaf>`).

## Contents

- **[virtual_staining/](virtual_staining/README.md)** — the DynaCell virtual-staining benchmark: fit / predict /
  eval leaves for every `(organelle, model, train_set, predict_set)` combination in the paper, plus the
  `_internal/{shared,leaf}` composition system that generates them.

## Navigation

- Up: [configs](../README.md)
- Subdirectories: [virtual_staining/](virtual_staining/README.md)
