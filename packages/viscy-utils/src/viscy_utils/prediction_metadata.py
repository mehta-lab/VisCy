"""Prediction completion metadata shared by writers and submission tools."""

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from iohub.ngff import ImageArray, Position

__all__ = [
    "PREDICTION_COMPLETE_KEY",
    "checkpoint_sha256_12",
    "clear_completion",
    "completion_marker",
    "mark_complete",
    "prediction_complete",
    "prediction_run",
    "same_run",
    "tzyx_shape",
]

# Position attribute mapping each prediction channel to the marker recorded when
# the channel was fully predicted: the source TZYX it came from plus the run
# identity from ``prediction_run``. A channel that is missing, or recorded with
# a different marker, is incomplete and must be recomputed.
PREDICTION_COMPLETE_KEY = "viscy_prediction_complete"

# Marker fields kept for provenance only; two markers that differ solely in
# these still describe the same prediction (a checkpoint may be moved).
_PROVENANCE_KEYS = frozenset({"checkpoint_path"})


def tzyx_shape(array: ImageArray) -> list[int]:
    """Return the TZYX extent of a 5D OME-Zarr array.

    Parameters
    ----------
    array : ImageArray
        Array whose ``(T, C, Z, Y, X)`` shape to project.

    Returns
    -------
    list of int
        ``[T, Z, Y, X]``, the source identity a completion marker records.
    """
    return [array.frames, array.slices, array.height, array.width]


def checkpoint_sha256_12(path: str | os.PathLike) -> str:
    """Return the first 12 hex chars of the sha256 of the file at *path*.

    On repeated calls for the same checkpoint, reads the digest from a
    ``<path>.sha256`` sidecar file, avoiding a multi-GB re-read. The sidecar
    records the size and ``st_mtime_ns`` of the file it hashed and is only
    trusted when both still match exactly: a checkpoint replaced in place
    changes at least one of them even when it was copied with a preserved or
    older mtime, so the digest never certifies stale weights (a sidecar that
    is merely newer than the checkpoint proves nothing). Writes the sidecar
    after a fresh hash; silently tolerates read-only parent directories and
    NFS flakes by falling back to recompute.

    Parameters
    ----------
    path : str or PathLike
        Checkpoint file to hash.

    Returns
    -------
    str
        First 12 hex characters of the file's sha256 digest.
    """
    ckpt = Path(path)
    sidecar = ckpt.with_suffix(ckpt.suffix + ".sha256")
    stat = ckpt.stat()
    try:
        recorded = json.loads(sidecar.read_text())
        if (
            isinstance(recorded, dict)
            and recorded.get("size") == stat.st_size
            and recorded.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(recorded.get("sha256"), str)
            and len(recorded["sha256"]) == 64
        ):
            return recorded["sha256"][:12]
    except (OSError, ValueError):
        pass
    hasher = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    try:
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps({"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}) + "\n")
        tmp.replace(sidecar)
    except OSError:
        pass
    return digest[:12]


def prediction_run(
    *,
    array_key: str,
    z_window_size: int,
    z_reduction: str,
    checkpoint_path: str | os.PathLike | None,
) -> dict[str, Any]:
    """Describe what determines a run's voxels: the weights and the depth handling.

    Call once per run; the checkpoint is hashed here, not per FOV.

    Parameters
    ----------
    array_key : str
        Name of the source (and output) array level that is predicted.
    z_window_size : int
        Depth of the model's input window.
    z_reduction : str
        How overlapping depth windows are combined by the writer.
    checkpoint_path : str or PathLike or None
        Checkpoint the model predicts with; ``None`` when the run has none.

    Returns
    -------
    dict
        JSON-serializable identity, including ``checkpoint_sha256_12`` of the
        file's content and its path for provenance.
    """
    return {
        "array_key": array_key,
        "z_window_size": int(z_window_size),
        "z_reduction": z_reduction,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256_12": None if checkpoint_path is None else checkpoint_sha256_12(checkpoint_path),
    }


def completion_marker(source_shape: list[int], run: dict[str, Any]) -> dict[str, Any]:
    """Return the marker a run records for a channel predicted from ``source_shape``.

    Parameters
    ----------
    source_shape : list of int
        TZYX shape of the source array, from :func:`tzyx_shape`.
    run : dict
        Run identity from :func:`prediction_run`.

    Returns
    -------
    dict
        ``{"source_shape": [T, Z, Y, X], **run}``.
    """
    return {"source_shape": list(source_shape), **run}


def _identity(marker: Any) -> Any:
    """Strip provenance-only fields so markers compare on what was predicted."""
    if isinstance(marker, dict):
        return {key: value for key, value in marker.items() if key not in _PROVENANCE_KEYS}
    return marker


def same_run(marker: Any, run: dict[str, Any]) -> bool:
    """Return whether a recorded marker came from ``run``, whatever its source shape.

    Parameters
    ----------
    marker : Any
        Recorded marker value; markers from other layouts never match.
    run : dict
        Run identity from :func:`prediction_run`.

    Returns
    -------
    bool
        True when the marker's run fields equal ``run`` up to provenance.
    """
    if not isinstance(marker, dict):
        return False
    recorded = {key: value for key, value in marker.items() if key != "source_shape"}
    return _identity(recorded) == _identity(run)


def mark_complete(position: Position, channels: Iterable[str], marker: dict[str, Any]) -> None:
    """Record ``marker`` as the completion of ``channels``, keeping other channels' markers.

    Parameters
    ----------
    position : Position
        Output position whose completion attribute to update.
    channels : iterable of str
        Prediction channels that were fully written.
    marker : dict
        Marker from :func:`completion_marker`.
    """
    completed = dict(position.zattrs.get(PREDICTION_COMPLETE_KEY, {}))
    for channel in channels:
        completed[channel] = marker
    position.zattrs[PREDICTION_COMPLETE_KEY] = completed


def clear_completion(position: Position, channels: Iterable[str]) -> None:
    """Drop ``channels`` from the completion attribute, keeping other channels' markers.

    Parameters
    ----------
    position : Position
        Output position whose completion attribute to update.
    channels : iterable of str
        Prediction channels about to be rewritten.
    """
    completed = dict(position.zattrs.get(PREDICTION_COMPLETE_KEY, {}))
    for channel in channels:
        completed.pop(channel, None)
    position.zattrs[PREDICTION_COMPLETE_KEY] = completed


def prediction_complete(position: Position, channels: Iterable[str], marker: dict[str, Any]) -> bool:
    """Return whether every channel of ``position`` was completed with ``marker``.

    Parameters
    ----------
    position : Position
        Output position to inspect.
    channels : iterable of str
        Prediction channels that must all be complete.
    marker : dict
        Marker the current run would record; see :func:`completion_marker`.

    Returns
    -------
    bool
        True only when each channel's recorded marker equals ``marker`` up to
        provenance fields.
    """
    completed = position.zattrs.get(PREDICTION_COMPLETE_KEY, {})
    expected = _identity(marker)
    return all(_identity(completed.get(channel)) == expected for channel in channels)
