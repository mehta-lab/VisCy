"""Bundled dataset manifests — the default registry for the DynaCell resolver.

This package ships canonical manifest YAMLs (mirrored from
``dynacell-paper/_configs/datasets/``) so the resolver works out-of-the-box
on any clone. Auto-discovered via the ``dynacell.manifest_roots`` entry
point declared in ``applications/dynacell/pyproject.toml``.

VisCy is the source of truth for manifest *content* (this directory).
``dynacell-paper`` is the source of truth for manifest *authoring* — when
a new dataset is preprocessed there, the change is mirrored back here and
``tests/test_manifest_sync.py`` enforces the parity.

Override at runtime with ``DYNACELL_MANIFEST_ROOTS=/path/to/other/registry``
(env var) or by passing ``cli_roots=`` to ``discover_manifest_roots``.
"""
