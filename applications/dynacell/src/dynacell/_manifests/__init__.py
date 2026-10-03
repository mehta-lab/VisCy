"""Bundled dataset manifests — the default registry for the DynaCell resolver.

This package ships the canonical manifest YAMLs so the resolver works
out-of-the-box on any clone. Auto-discovered via the
``dynacell.manifest_roots`` entry point declared in
``applications/dynacell/pyproject.toml``.

**This package is the source of truth for dataset manifests** — both
content and authoring. ``dynacell-paper`` deleted its own
``_configs/datasets/`` tree when it consumed the migrated preprocessing
code (Phase 13), so there is no second copy and no parity to enforce.
Edit these YAMLs directly.

Override at runtime with ``DYNACELL_MANIFEST_ROOTS=/path/to/other/registry``
(env var) or by passing ``cli_roots=`` to ``discover_manifest_roots``.
"""
