# tests

Pytest suite for the `dynacell` package and its `tools/`. Integration-first: tests import and run the real
code paths (config composition, resolver, eval pipeline, reporting) rather than isolated stubs. Run with
`uv run pytest applications/dynacell/`.

## Grouping

- **Config composition & CLI** — `test_benchmark_config_composition.py`, `test_evaluate_compose.py`,
  `test_cli_routing.py`, `test_hydra_ref_hook.py`, `test_preprocess_config.py`.
- **Data layer** — `test_data_manifests.py`, `test_dataset_ref.py`, `test_paths.py`.
- **Engine / training** — `test_engine.py`, `test_training_integration.py`, `test_lazy_init.py`.
- **Evaluation pipeline** — `test_evaluation_pipeline*.py` (serial + parallel CPU/GPU),
  `test_evaluation_metrics.py`, `test_evaluation_plot_metrics.py`, `test_evaluation_cache.py`, `test_pipeline_cache.py`,
  `test_evaluation_extractors.py`, `test_evaluation_grouped.py`, `test_evaluation_cross_store_nuclei.py`,
  `test_evaluation_precompute_cli.py`, `test_focus.py`, `test_runtime.py`, `test_cross_condition_probe.py`,
  `test_pixel_metrics_parity.py`.
- **Reporting** — `test_reporting_tables.py`, `test_reporting_tables_extended.py`, `test_reporting_figures.py`.
- **Preprocess** — `test_preprocess_zarr_utils.py`.
- **Tools** — `test_submit_benchmark_job.py`, `test_submit_benchmark_batch.py`.

## Fixtures & helpers

- `conftest.py`, `_eval_fixtures.py` — shared fixtures (import via pytest, not directly).
- `generate_celldiff_trajectory.py` — helper to synthesize a CELL-Diff trajectory fixture.
- `data/` — golden artifacts: `pixel_metrics_golden.npz` (+ its `_generate_*.py` regenerator) pinning pixel-metric
  parity.

## Navigation

- Up: [applications/dynacell](../README.md)
