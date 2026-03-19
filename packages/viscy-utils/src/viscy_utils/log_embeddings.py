"""Utilities for logging embedding visualizations during training."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def pca_pairplot(
    embeddings: np.ndarray,
    meta: list[dict],
    color_key: str | None = None,
    n_components: int = 8,
    title: str = "",
    dpi: int = 150,
) -> "Figure":
    """Compute PCA and return a pairplot figure colored by a metadata key.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of shape ``(N, D)``.
    meta : list[dict]
        Per-sample metadata dicts. Length must match ``embeddings``.
    color_key : str or None
        Key in ``meta`` to color points by. If None, no coloring.
    n_components : int
        Number of PCA components to show. Default: 8.
    title : str
        Figure suptitle.
    dpi : int
        Figure resolution. Default: 150.

    Returns
    -------
    matplotlib.figure.Figure
        Pairplot figure. Caller is responsible for closing it.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca_coords = PCA(n_components=n_components).fit_transform(embeddings)

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(pca_coords, columns=pc_cols)
    if color_key is not None:
        df[color_key] = [meta[i].get(color_key) if i < len(meta) else None for i in range(len(pca_coords))]

    unique_labels = sorted(df[color_key].dropna().unique()) if color_key else []
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    label_to_color = {lbl: cmap(idx) for idx, lbl in enumerate(unique_labels)}

    fig, axes = plt.subplots(n_components, n_components, figsize=(2 * n_components, 2 * n_components), dpi=dpi)

    for row, pc_y in enumerate(pc_cols):
        for col, pc_x in enumerate(pc_cols):
            ax = axes[row, col]
            if row == col:
                if color_key and unique_labels:
                    for lbl in unique_labels:
                        subset = df[df[color_key] == lbl][pc_x]
                        ax.hist(subset, bins=20, alpha=0.5, color=label_to_color[lbl], label=str(lbl))
                else:
                    ax.hist(df[pc_x], bins=20, alpha=0.7)
            else:
                if color_key and unique_labels:
                    for lbl in unique_labels:
                        subset = df[df[color_key] == lbl]
                        ax.scatter(subset[pc_x], subset[pc_y], s=4, alpha=0.5, color=label_to_color[lbl])
                else:
                    ax.scatter(df[pc_x], df[pc_y], s=4, alpha=0.5)
            if row == n_components - 1:
                ax.set_xlabel(pc_x, fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(pc_y, fontsize=7)
            else:
                ax.set_yticklabels([])

    if color_key and unique_labels:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl], label=str(lbl))
            for lbl in unique_labels
        ]
        fig.legend(handles=handles, title=color_key, loc="upper right", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
