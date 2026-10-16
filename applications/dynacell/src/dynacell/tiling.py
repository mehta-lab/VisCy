"""Patch tiling shared by every tiled-inference path in this application.

Lives outside :mod:`dynacell.engine` and :mod:`dynacell.celldiff_wrapper` because
both consume it and ``engine`` already imports ``celldiff_wrapper`` — a tiler in
either module would either cycle or tie a model-agnostic helper to one backbone.
"""

from monai.utils import ensure_tuple_rep


def window_starts(spatial: tuple[int, ...], patch: tuple[int, ...], overlap: int | tuple[int, ...]) -> list[list[int]]:
    """Per-dimension tile start indices covering ``spatial`` with ``patch`` windows.

    Windows step by ``patch - overlap``; the last start in each dimension is
    snapped to the edge (``size - patch``) so coverage is complete, which means it
    may overlap its predecessor by more than ``overlap`` when the extent is not a
    multiple of the stride. ``overlap=0`` gives the non-overlapping partition.

    Shared by every tiled-inference path (``engine._sliding_window_inference`` and
    the three ``CELLDiff3DVS.*sliding_window`` / ``generate_iterative`` methods) so
    they cannot drift apart.

    Parameters
    ----------
    spatial : tuple of int
        Extent of each tiled dimension.
    patch : tuple of int
        Window size per dimension; must have the same rank as ``spatial``.
    overlap : int or tuple of int
        Overlap per dimension, or one value broadcast to every dimension.

    Returns
    -------
    list of list of int
        Start indices per dimension, in ``spatial`` order.

    Raises
    ------
    ValueError
        If ``patch`` or ``overlap`` rank does not match ``spatial``, if any
        dimension is smaller than its patch, or an overlap is outside ``[0, patch)``.
    """
    starts_per_dim: list[list[int]] = []
    for i, (size, p, ov) in enumerate(zip(spatial, patch, ensure_tuple_rep(overlap, len(spatial)), strict=True)):
        if size < p:
            raise ValueError(f"spatial dim {i} size {size} must be >= patch size {p}")
        if not 0 <= ov < p:
            raise ValueError(f"overlap at dim {i} must satisfy 0 <= overlap < patch (got {ov} vs {p})")
        last = size - p
        # range() stops before ``last``, so appending it is the edge snap; when
        # size == patch (last == 0) the range is empty and this is just [0].
        starts_per_dim.append([*range(0, last, p - ov), last])
    return starts_per_dim
