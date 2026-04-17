# LEGACY — Dihan's pre-schema CellDiff configs

**Reference-only.** `base:` paths were patched post-move from
`../../../configs/recipes/...` to `../../../../configs/recipes/...` so the
equivalence test in `tests/test_benchmark_config_composition.py` can still
compose them. The patched files are not intended to be launched directly —
use the migrated leaves under `configs/benchmarks/virtual_staining/` via
`submit_benchmark_job.py`.

## Why kept

These are the source-of-truth hyperparameter reference for the migrated
benchmark leaves under `configs/benchmarks/virtual_staining/train/` and
`.../predict/`. The equivalence test
(`tests/test_benchmark_config_composition.py`) asserts that each migrated
leaf composes to the same values these files compose to. Delete this tree
only after:

1. One successful end-to-end `submit_benchmark_job.py` run against a
   migrated leaf (fit or predict), verified on wandb/disk; and
2. 2026-06-30 at the earliest.

Whoever deletes this should note both conditions in the commit message.

## Rerunning these configs

Copy them back out to the original location or fix the `base:` paths
manually. They are preserved exactly as they were when they worked.
