"""Force iohub to use the zarr-python implementation instead of ``zarrs``.

iohub (>=0.3.6) selects its zarr backend by *implementation name* via
:mod:`iohub.core.registry`, whose module-level default is still
``"zarrs-python"`` — the Rust ``zarrs`` pipeline. ``zarrs`` 0.2.3 (the only
version iohub's ``zarrs>=0.2.3`` floor accepts) builds a rank-6 read offset
for rank-5 sharded arrays, raising ``RuntimeError: incompatible offset ...
for region with start ...`` on every sharded zarr-v3 read, and deadlocks the
spawned ranks of a DDP run.

iohub 0.3.6 exposes :func:`iohub.core.registry.set_default_implementation`,
the supported switch for choosing the pure-Python ``zarr-python``
implementation (whose codec pipeline is zarr-python's native
``BatchedCodecPipeline``). We call it once at import time so the choice is in
place before any iohub store is opened. ``set_default_implementation`` is not
re-exported at the iohub top level, hence the fully qualified import.
"""

from iohub.core.registry import set_default_implementation

_ZARR_PYTHON = "zarr-python"


def use_zarr_python_codec() -> None:
    """Pin iohub's default implementation to ``zarr-python``.

    Idempotent. Sets iohub's default implementation name so every store open
    uses zarr-python's native ``BatchedCodecPipeline`` rather than the broken
    ``zarrs`` Rust pipeline.
    """
    set_default_implementation(_ZARR_PYTHON)


# Apply on import so the override is in place before any iohub store is opened.
use_zarr_python_codec()
