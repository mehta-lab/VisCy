"""
Scatter plots of PHATE / PCA embeddings coloured by infection or remodeling status.

Produces:
  - PHATE embedding coloured by predicted_infection (per-well subplots + combined)
  - PHATE embedding coloured by remodeling_status  (per-well subplots + combined)

Input:
  embedding_csv – organelle_embeddings_with_predictions.csv
    Columns: fov_name, track_id, t, well, PHATE1, PHATE2, PCA1, PCA2,
             predicted_infection, remodeling_status
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %% paths
embedding_csv = Path("organelle_embeddings_with_predictions.csv")
output_dir = Path("figures/")

# %% configuration
INFECTION_PALETTE = {
    "uninfected": "steelblue",
    "infected": "orange",
}
REMODELING_PALETTE = {
    "control": "green",
    "remodeled": "red",
}
POINT_SIZE = 4
ALPHA = 0.5

output_dir.mkdir(parents=True, exist_ok=True)

# %% load data
df = pd.read_csv(embedding_csv)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
print(f"Wells: {df['well'].unique()}")
print(f"Infection labels: {df['predicted_infection'].unique()}")
print(f"Remodeling labels: {df['remodeling_status'].unique()}")


# %% generic embedding plot function
def plot_embeddings_by_condition(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    palette: dict,
    condition_list: list,
    n_cols: int,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot embedding scatter subplots, one per condition in condition_list.

    Parameters
    ----------
    df            : Full DataFrame.
    x_col, y_col  : Column names for embedding coordinates.
    hue_col       : Column used for colouring points.
    palette       : Dict mapping hue_col values to colours.
    condition_list: List of well names (one subplot each).
    n_cols        : Number of subplot columns.
    title         : Figure suptitle.
    output_path   : SVG save path.
    """
    n = len(condition_list)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13)

    for idx, well in enumerate(condition_list):
        row_i = idx // n_cols
        col_i = idx % n_cols
        ax = axes[row_i][col_i]

        subset = df[df["well"] == well]
        if subset.empty:
            ax.text(0.5, 0.5, f"No data\n{well}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(well)
            ax.axis("off")
            continue

        for label, color in palette.items():
            mask = subset[hue_col].astype(str).str.lower() == label.lower()
            pts = subset[mask]
            ax.scatter(pts[x_col], pts[y_col], c=color, s=POINT_SIZE,
                       alpha=ALPHA, label=label, rasterized=True)

        ax.set_title(well, fontsize=9)
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=c, label=lbl) for lbl, c in palette.items()
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9,
               bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# %% all-wells combined plot function
def plot_all_wells_combined(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    palette: dict,
    output_path: Path,
    title: str = "",
) -> None:
    """
    Single scatter plot with all wells overlaid, coloured by hue_col.

    Parameters
    ----------
    df          : Full DataFrame.
    x_col, y_col: Embedding coordinate columns.
    hue_col     : Column used for colouring.
    palette     : Dict mapping hue_col values to colours.
    output_path : SVG save path.
    title       : Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in palette.items():
        mask = df[hue_col].astype(str).str.lower() == label.lower()
        pts = df[mask]
        ax.scatter(pts[x_col], pts[y_col], c=color, s=POINT_SIZE,
                   alpha=ALPHA, label=label, rasterized=True)

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(title or f"All wells — {hue_col}", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# %% collect wells
all_wells = sorted(df["well"].dropna().unique().tolist())
n_cols = min(4, len(all_wells))

# %% PHATE coloured by predicted_infection — per-well subplots
plot_embeddings_by_condition(
    df=df,
    x_col="PHATE1",
    y_col="PHATE2",
    hue_col="predicted_infection",
    palette=INFECTION_PALETTE,
    condition_list=all_wells,
    n_cols=n_cols,
    title="PHATE embedding — predicted infection status (per well)",
    output_path=output_dir / "PHATE_infection_per_well.svg",
)

# %% PHATE coloured by remodeling_status — per-well subplots
plot_embeddings_by_condition(
    df=df,
    x_col="PHATE1",
    y_col="PHATE2",
    hue_col="remodeling_status",
    palette=REMODELING_PALETTE,
    condition_list=all_wells,
    n_cols=n_cols,
    title="PHATE embedding — remodeling status (per well)",
    output_path=output_dir / "PHATE_remodeling_per_well.svg",
)

# %% PHATE coloured by predicted_infection — all wells combined
plot_all_wells_combined(
    df=df,
    x_col="PHATE1",
    y_col="PHATE2",
    hue_col="predicted_infection",
    palette=INFECTION_PALETTE,
    output_path=output_dir / "PHATE_infection_combined.svg",
    title="PHATE embedding — infection status (all wells)",
)

# %% PHATE coloured by remodeling_status — all wells combined
plot_all_wells_combined(
    df=df,
    x_col="PHATE1",
    y_col="PHATE2",
    hue_col="remodeling_status",
    palette=REMODELING_PALETTE,
    output_path=output_dir / "PHATE_remodeling_combined.svg",
    title="PHATE embedding — remodeling status (all wells)",
)
