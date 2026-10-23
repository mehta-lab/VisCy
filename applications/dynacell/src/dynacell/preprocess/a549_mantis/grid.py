"""Temporal subsampling grid for A549 mantis assembly.

Implements the 2-h sampling grid on window [5, 23] hpi with ±1.5 h tail
tolerance. Grid ticks are fixed (default odd hpi from 5) and each tick
resolves to a native frame index by aligning with the plate's native
T-axis, which starts at ``hpi_start`` with step ``native_delta_t_min``.

If the next tick past the last in-window frame falls within
``tail_tol_h`` of imaging end, snap it to the last native frame — this
lets short-window plates (e.g. 10_31) contribute one more sample.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridFrame:
    """One selected frame from a plate's native time axis."""

    native_idx: int
    """Index into the plate's native T-axis."""

    hpi: float
    """Actual hours-post-infection of the selected native frame."""

    tick_hpi: float
    """Nominal tick hpi on the target grid — equals ``hpi`` unless snapped."""


def _native_idx_for_hpi(tick_hpi: float, hpi_start: float, native_dt_h: float) -> int | None:
    """Return the native frame index that lands exactly on ``tick_hpi``.

    Returns None if ``tick_hpi`` does not fall on a native frame.
    """
    native_idx_f = (tick_hpi - hpi_start) / native_dt_h
    native_idx = int(round(native_idx_f))
    if abs(native_idx - native_idx_f) > 1e-6:
        return None
    return native_idx


def build_grid(
    native_delta_t_min: float,
    native_t: int,
    hpi_start: float,
    stride_h: float = 2.0,
    window: tuple[float, float] = (5.0, 23.0),
    tail_tol_h: float = 1.5,
) -> list[GridFrame]:
    """Select native frames for the target grid.

    Parameters
    ----------
    native_delta_t_min : float
        Native inter-frame interval in minutes.
    native_t : int
        Native T-axis length (number of frames).
    hpi_start : float
        hpi of native frame 0 (infection clock).
    stride_h : float
        Grid stride in hours (default 2.0).
    window : tuple of float
        (low, high) inclusive hpi bounds. Ticks start at ``low`` and step
        ``stride_h``. All in-window ticks are included if a native frame
        exists; the first out-of-window tick is snapped to the last
        native frame if it falls within ``tail_tol_h`` past imaging end.
    tail_tol_h : float
        Tail-snap tolerance in hours.

    Returns
    -------
    list of GridFrame
        One entry per output frame, in temporal order.
    """
    native_dt_h = native_delta_t_min / 60.0
    imaging_end_hpi = hpi_start + (native_t - 1) * native_dt_h
    win_low, win_high = window

    frames: list[GridFrame] = []
    tick_hpi = win_low
    while tick_hpi <= win_high + 1e-9:
        if tick_hpi < hpi_start - 1e-9:
            tick_hpi += stride_h
            continue
        if tick_hpi > imaging_end_hpi + 1e-9:
            break
        native_idx = _native_idx_for_hpi(tick_hpi, hpi_start, native_dt_h)
        if native_idx is None or not (0 <= native_idx < native_t):
            raise ValueError(
                f"tick hpi={tick_hpi} does not align to native frame "
                f"(hpi_start={hpi_start}, native_dt_min={native_delta_t_min}). "
                "Grid phase mismatch — check plate metadata."
            )
        frames.append(
            GridFrame(
                native_idx=native_idx,
                hpi=hpi_start + native_idx * native_dt_h,
                tick_hpi=tick_hpi,
            )
        )
        tick_hpi += stride_h

    next_tick_hpi = frames[-1].tick_hpi + stride_h if frames else win_low
    if (
        frames
        and next_tick_hpi <= win_high + 1e-9
        and next_tick_hpi > imaging_end_hpi
        and next_tick_hpi - imaging_end_hpi <= tail_tol_h + 1e-9
    ):
        snap_idx = native_t - 1
        frames.append(
            GridFrame(
                native_idx=snap_idx,
                hpi=imaging_end_hpi,
                tick_hpi=next_tick_hpi,
            )
        )

    return frames
