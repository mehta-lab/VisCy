"""Force iohub to use the zarr-python codec pipeline instead of ``zarrs``.

iohub selects its zarr-v3 codec pipeline via
:func:`iohub.core.config._default_codec_pipeline`, which prefers the Rust
``zarrs`` backend (``zarrs.ZarrsCodecPipeline``) whenever ``zarrs`` is
importable. ``zarrs`` 0.2.3 builds a rank-6 read offset for rank-5 sharded
arrays, raising ``RuntimeError: incompatible offset ... for region with
start ...`` on every sharded zarr-v3 read, and deadlocks the spawned ranks
of a DDP run. We pin the codec pipeline to zarr-python's native
``BatchedCodecPipeline`` to avoid both failure modes.

iohub re-applies its codec choice on every ``ZarrPythonImplementation``
construction (i.e. on every store open), so a one-time
``zarr.config.set(...)`` is clobbered by the next ``open_ome_zarr`` call.
Overriding iohub's own ``ZarrConfig`` field default is the only override
that survives, because iohub then re-applies the native pipeline.
"""

from iohub.core.config import ZarrConfig

_NATIVE_CODEC_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"


def use_zarr_python_codec() -> None:
    """Pin iohub's default codec pipeline to zarr-python's native one.

    Idempotent. Overrides the ``codec_pipeline`` field default on
    :class:`iohub.core.config.ZarrConfig` so every iohub store open uses
    ``BatchedCodecPipeline`` rather than ``zarrs.ZarrsCodecPipeline``.
    """
    field = ZarrConfig.model_fields["codec_pipeline"]
    if field.default_factory is not None and field.default_factory() == _NATIVE_CODEC_PIPELINE:
        return
    field.default_factory = lambda: _NATIVE_CODEC_PIPELINE
    ZarrConfig.model_rebuild(force=True)


# Apply on import so the override is in place before any iohub store is opened.
use_zarr_python_codec()
