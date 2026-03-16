"""Statistical helpers for significance testing and plot annotations."""

import matplotlib.pyplot as plt


def get_significance_stars(p: float) -> str:
    """
    Convert a p-value to a significance star string.

    Returns '***', '**', '*', or 'ns'.
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    p_val: float,
    fontsize: int = 10,
) -> None:
    """
    Draw a significance bracket between two bars on *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
    x1, x2 : float  Bar x-positions.
    y : float        Height at which the horizontal bar starts.
    h : float        Additional height for the bracket tip.
    p_val : float    p-value to convert to stars.
    fontsize : int   Font size for the star label.
    """
    stars = get_significance_stars(p_val)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text(
        (x1 + x2) / 2,
        y + h,
        f"{stars}\np={p_val:.2e}",
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
