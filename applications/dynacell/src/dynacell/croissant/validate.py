"""Croissant JSON-LD validation.

Wraps ``mlcroissant.Dataset(jsonld=...)`` so a malformed doc raises
loudly. Imports ``mlcroissant`` at module top — a missing dep produces
``ModuleNotFoundError`` that propagates with the standard "module not
installed" message; the package README documents
``uv sync --extra croissant`` as the install path.
"""

import copy
from pathlib import Path
from typing import Any

import mlcroissant as mlc


def validate_croissant(jsonld: dict[str, Any] | Path) -> None:
    """Validate a Croissant JSON-LD doc; raise on any issue.

    Parameters
    ----------
    jsonld
        Either an in-memory dict or a ``Path`` (or path-like) to a
        JSON file. Raw JSON strings are not supported — parse them
        with :func:`json.loads` first and pass the resulting dict.

    Raises
    ------
    Exception
        Any error mlcroissant surfaces during validation. We do not
        catch — callers see the raw mlcroissant error so the cause is
        visible without a layer of translation.

    Notes
    -----
    ``mlcroissant.Dataset`` mutates the input dict (adds ``@base`` to
    the JSON-LD context). We deep-copy the input first so the caller's
    dict survives validation unchanged — important for the
    in-sync-with-committed-JSON tests.
    """
    if isinstance(jsonld, dict):
        mlc.Dataset(jsonld=copy.deepcopy(jsonld))
    else:
        mlc.Dataset(jsonld=Path(jsonld))
