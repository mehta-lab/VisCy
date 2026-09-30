"""Repo-wide pytest setup, loaded before every package's own conftest."""

import os

# Under pytest-xdist every worker is a separate process, and torch/OpenMP size
# their thread pools to all visible cores by default, so `-n N` runs ~N x cores
# threads. On 11 cores that made the suite 10x slower (1242 s vs 117 s) and
# timed out subprocess tests. xdist sets PYTEST_XDIST_WORKER before this module
# is imported, which is before any test module imports torch. Assigned, not
# setdefault: an inherited limit (a job script exporting OMP_NUM_THREADS=16)
# would otherwise oversubscribe the cores the same way.
if "PYTEST_XDIST_WORKER" in os.environ:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
