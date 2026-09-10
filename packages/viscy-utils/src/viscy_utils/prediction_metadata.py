"""Prediction completion metadata shared by writers and submission tools."""

from iohub.ngff import ImageArray

__all__ = ["PREDICTION_COMPLETE_KEY", "tzyx_shape"]

# Position attribute mapping each prediction channel to the TZYX shape of the
# source it was fully predicted from. A channel that is missing, or recorded
# against a different source shape, is incomplete and must be recomputed.
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
