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
    "checkpoint_signature",
    "completion_marker",
    "mark_complete",
    "mark_started",
    "outruns",
    "prediction_complete",
    "prediction_run",
    "same_marker",
    "same_run",
    "started_marker",
    "tzyx_shape",
]

# Position attribute mapping each prediction channel to its marker: the run
# identity from ``prediction_run`` while a run is writing the channel
# (``started_marker``), and that identity plus the source TZYX once every
# window was written (``completion_marker``). A channel with no entry of its
# own was never recorded by any run -- written before markers existed -- so
# nothing can vouch for it, even when other channels of the FOV are marked.
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


def outruns(array: ImageArray, source_shape: list[int]) -> bool:
    """Return whether an output array holds more frames or depth slices than its source.

    Arrays only grow and no run writes beyond its source's extent, so the
    excess would keep stale voxels under a fresh completion marker.

    Parameters
    ----------
    array : ImageArray
        Existing output array.
    source_shape : list of int
        TZYX shape of the source, from :func:`tzyx_shape`.

    Returns
    -------
    bool
        True when ``array`` exceeds the source in T or Z.
    """
    return array.frames > source_shape[0] or array.slices > source_shape[1]


_SIGNATURE_FIELDS = ("size", "mtime_ns", "ctime_ns", "ino")


def checkpoint_signature(path: str | os.PathLike) -> tuple[int, int, int, int]:
    """Return the size, ``st_mtime_ns``, ``st_ctime_ns`` and inode of the file at *path*.

    Overwriting a file or replacing it by rename changes at least one of these:
    a copy can preserve size and mtime, but the kernel sets ctime itself and a
    rename swaps the inode. Equal signatures therefore mean the same bytes for
    the hash memo in :func:`checkpoint_sha256_12` and for the writer's check
    that a checkpoint did not change between loading its weights and hashing
    it. The one blind spot is a same-size, mtime-preserving overwrite within
    the same clock tick as the previous change; identity-critical callers hash
    the bytes instead (:func:`prediction_run`).

    Parameters
    ----------
    path : str or PathLike
        Checkpoint file to describe.

    Returns
    -------
    tuple of int
        ``(size, st_mtime_ns, st_ctime_ns, st_ino)``.
    """
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _sha256_hex(path: Path) -> str:
    """Return the full sha256 hex digest of the file at *path*, read in 1 MiB chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checkpoint_sha256_12(path: str | os.PathLike, *, memoize: bool = True) -> str:
    """Return the first 12 hex chars of the sha256 of the file at *path*.

    On repeated calls for the same checkpoint, reads the digest from a
    ``<path>.sha256`` sidecar file, avoiding a multi-GB re-read. The sidecar
    records the :func:`checkpoint_signature` of the file it hashed and is only
    trusted when every field still matches exactly: a checkpoint overwritten
    or replaced changes at least one of them even when it was copied with a
    preserved or older mtime, so the digest never certifies stale weights (a
    sidecar that is merely newer than the checkpoint proves nothing). Writes
    the sidecar after a fresh hash; silently tolerates read-only parent
    directories and NFS flakes by falling back to recompute.

    Parameters
    ----------
    path : str or PathLike
        Checkpoint file to hash.
    memoize : bool, optional
        Consult and maintain the sidecar (default). Pass False to hash the
        bytes as they are now and leave the checkpoint's directory untouched:
        file metadata cannot prove byte equality, so a digest that certifies
        which weights a run loaded must come from the bytes themselves.

    Returns
    -------
    str
        First 12 hex characters of the file's sha256 digest.
    """
    ckpt = Path(path)
    if not memoize:
        return _sha256_hex(ckpt)[:12]
    sidecar = ckpt.with_suffix(ckpt.suffix + ".sha256")
    signature = dict(zip(_SIGNATURE_FIELDS, checkpoint_signature(ckpt), strict=True))
    try:
        recorded = json.loads(sidecar.read_text())
        if (
            isinstance(recorded, dict)
            and all(recorded.get(field) == value for field, value in signature.items())
            and isinstance(recorded.get("sha256"), str)
            and len(recorded["sha256"]) == 64
        ):
            return recorded["sha256"][:12]
    except (OSError, ValueError):
        pass
    digest = _sha256_hex(ckpt)
    try:
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps({"sha256": digest, **signature}) + "\n")
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
    settings_sha256_12: str | None = None,
) -> dict[str, Any]:
    """Describe what determines a run's voxels: the weights, the depth handling and the other settings.

    Call once per run; the checkpoint's bytes are hashed here, not per FOV,
    and never taken from a ``<ckpt>.sha256`` sidecar: the marker certifies
    which weights produced the voxels, and a memo keyed on file metadata
    cannot prove that a same-size copy did not replace them.

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
    settings_sha256_12 : str or None, optional
        Hash of the remaining settings that shape the predicted voxels (model
        inference arguments, input normalization, precision), computed by the
        submitter from the resolved config; ``None`` when the run records none.

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
        "checkpoint_sha256_12": None
        if checkpoint_path is None
        else checkpoint_sha256_12(checkpoint_path, memoize=False),
        "settings_sha256_12": settings_sha256_12,
    }


def started_marker(run: dict[str, Any]) -> dict[str, Any]:
    """Return the marker a run records while it is writing a channel: its identity alone.

    Parameters
    ----------
    run : dict
        Run identity from :func:`prediction_run`.

    Returns
    -------
    dict
        ``dict(run)``; :func:`completion_marker` adds ``source_shape`` once the
        channel is fully written.
    """
    return dict(run)


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


def same_marker(recorded: Any, marker: Any) -> bool:
    """Return whether a recorded marker equals ``marker`` up to provenance fields.

    Parameters
    ----------
    recorded : Any
        Recorded marker value, or None when the channel has no entry.
    marker : Any
        Marker from :func:`started_marker` or :func:`completion_marker`.

    Returns
    -------
    bool
        True when both describe the same prediction state.
    """
    return _identity(recorded) == _identity(marker)


def same_run(marker: Any, run: dict[str, Any]) -> bool:
    """Return whether a recorded marker came from ``run``, started or complete, whatever its source shape.

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


def mark_started(position: Position, channels: Iterable[str], run: dict[str, Any]) -> None:
    """Record that ``run`` is writing ``channels``, keeping other channels' markers.

    Replaces any completion of ``channels``, whose voxels are about to change,
    so an interrupted run never leaves an old completion behind; unlike a
    channel with no entry, a started channel is known to be this run's
    unfinished work rather than an unverifiable legacy prediction.

    Parameters
    ----------
    position : Position
        Output position whose completion attribute to update.
    channels : iterable of str
        Prediction channels about to be (re)written.
    run : dict
        Run identity from :func:`prediction_run`.
    """
    completed = dict(position.zattrs.get(PREDICTION_COMPLETE_KEY, {}))
    for channel in channels:
        completed[channel] = started_marker(run)
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
    return all(same_marker(completed.get(channel), marker) for channel in channels)
