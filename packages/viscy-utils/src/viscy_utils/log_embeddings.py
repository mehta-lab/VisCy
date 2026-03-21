"""Utilities for logging embedding visualizations during training."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def pca_pairplot(
    embeddings: np.ndarray,
    meta: list[dict],
    color_keys: list[str] | None = None,
    n_components: int = 8,
    title: str = "",
    dpi: int = 150,
) -> "Figure":
    """Compute PCA and return a row of pairplots, one per color key.

    Axes are fixed to the PCA score range (standardized) so plots are
    comparable across epochs. When multiple ``color_keys`` are given the
    pairplots are concatenated horizontally into a single figure.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of shape ``(N, D)``.
    meta : list[dict]
        Per-sample metadata dicts. Length must match ``embeddings``.
    color_keys : list[str] or None
        Keys in ``meta`` to color points by. One pairplot per key.
        If None or empty, a single uncolored pairplot is produced.
    n_components : int
        Number of PCA components to show. Default: 8.
    title : str
        Figure suptitle.
    dpi : int
        Figure resolution. Default: 150.

    Returns
    -------
    matplotlib.figure.Figure
        Concatenated pairplot figure. Caller is responsible for closing it.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca_coords = PCA(n_components=n_components).fit_transform(embeddings)

    # Standardize so PC axes are in units of std — fixes axis scale across epochs
    stds = pca_coords.std(axis=0).clip(min=1e-8)
    pca_coords = pca_coords / stds

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(pca_coords, columns=pc_cols)

    # Add all color key columns up front
    keys = color_keys if color_keys else [None]
    for key in keys:
        if key is not None:
            df[key] = [meta[i].get(key) if i < len(meta) else None for i in range(len(pca_coords))]

    # Fixed axis limits: ±3 std (scores already standardized)
    axis_lim = (-3.5, 3.5)

    n_keys = len(keys)
    panel_size = 2
    fig, all_axes = plt.subplots(
        n_components,
        n_components * n_keys,
        figsize=(panel_size * n_components * n_keys, panel_size * n_components),
        dpi=dpi,
    )
    # Ensure all_axes is always 2-D
    if n_components == 1:
        all_axes = np.array([[all_axes]])

    cmap = plt.get_cmap("tab10")

    for key_idx, color_key in enumerate(keys):
        col_offset = key_idx * n_components

        unique_labels = sorted(df[color_key].dropna().unique()) if color_key else []
        label_to_color = {lbl: cmap(idx % 10) for idx, lbl in enumerate(unique_labels)}

        for row, pc_y in enumerate(pc_cols):
            for col, pc_x in enumerate(pc_cols):
                ax = all_axes[row, col_offset + col]

                if row == col:
                    if color_key and unique_labels:
                        for lbl in unique_labels:
                            subset = df[df[color_key] == lbl][pc_x]
                            ax.hist(subset, bins=20, alpha=0.5, color=label_to_color[lbl], label=str(lbl))
                    else:
                        ax.hist(df[pc_x], bins=20, alpha=0.7)
                    ax.set_xlim(*axis_lim)
                else:
                    if color_key and unique_labels:
                        for lbl in unique_labels:
                            subset = df[df[color_key] == lbl]
                            ax.scatter(subset[pc_x], subset[pc_y], s=4, alpha=0.5, color=label_to_color[lbl])
                    else:
                        ax.scatter(df[pc_x], df[pc_y], s=4, alpha=0.5)
                    ax.set_xlim(*axis_lim)
                    ax.set_ylim(*axis_lim)

                if row == n_components - 1:
                    ax.set_xlabel(pc_x, fontsize=7)
                else:
                    ax.set_xticklabels([])
                if col == col_offset:
                    ax.set_ylabel(pc_y, fontsize=7)
                else:
                    ax.set_yticklabels([])

                # Label each panel group at the top
                if row == 0 and col == 0 and color_key:
                    ax.set_title(f"color: {color_key}", fontsize=8, loc="left")

        if color_key and unique_labels:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl], label=str(lbl))
                for lbl in unique_labels
            ]
            # Place legend at top of each panel group
            legend_x = (col_offset + n_components / 2) / (n_components * n_keys)
            fig.legend(
                handles=handles,
                title=color_key,
                loc="upper center",
                bbox_to_anchor=(legend_x, 1.0),
                fontsize=6,
                ncol=min(len(unique_labels), 5),
            )

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig
