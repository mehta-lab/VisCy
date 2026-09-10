"""Prediction completion metadata shared by writers and submission tools."""

from collections.abc import Iterable
from typing import Any

from iohub.ngff import ImageArray, Position

__all__ = ["PREDICTION_COMPLETE_KEY", "clear_completion", "mark_complete", "prediction_complete", "tzyx_shape"]

# Position attribute mapping each prediction channel to the marker recorded when
# the channel was fully predicted. A channel that is missing, or recorded with a
# different marker, is incomplete and must be recomputed.
PREDICTION_COMPLETE_KEY = "viscy_prediction_complete"


def tzyx_shape(array: ImageArray) -> list[int]:
    """Return the TZYX extent of a 5D OME-Zarr array.

    Parameters
    ----------
    array : ImageArray
        Array whose ``(T, C, Z, Y, X)`` shape to project.

    Returns
    -------
    list of int
        ``[T, Z, Y, X]``, the identity a completion marker records.
    """
    return [array.frames, array.slices, array.height, array.width]


def mark_complete(position: Position, channels: Iterable[str], marker: Any) -> None:
    """Record ``marker`` as the completion of ``channels``, keeping other channels' markers.

    Parameters
    ----------
    position : Position
        Output position whose completion attribute to update.
    channels : iterable of str
        Prediction channels that were fully written.
    marker : Any
        JSON-serializable description of what was predicted.
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


def prediction_complete(position: Position, channels: Iterable[str], marker: Any) -> bool:
    """Return whether every channel of ``position`` was completed with ``marker``.

    Parameters
    ----------
    position : Position
        Output position to inspect.
    channels : iterable of str
        Prediction channels that must all be complete.
    marker : Any
        Marker the current run would record; see :func:`mark_complete`.

    Returns
    -------
    bool
        True only when each channel's recorded marker equals ``marker``.
    """
    completed = position.zattrs.get(PREDICTION_COMPLETE_KEY, {})
    return all(completed.get(channel) == marker for channel in channels)
