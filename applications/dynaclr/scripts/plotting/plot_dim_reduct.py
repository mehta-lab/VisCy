# %%
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

# %% Configuration
ZARR_DIR = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_01_24_A549_G3BP1_DENV/4-phenotyping/predictions/"
    "DynaCLR-2D-BagOfChannels-timeaware/v3"
)

OUTPUT_DIR = ZARR_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_KEY = "X_phate"  # or "X_pca"
COMPONENTS = (0, 1)  # 0-indexed
POINT_SIZE = 1.0

# %%

for zarr_path in ZARR_DIR.glob("timeaware_*ckpt.zarr"):
    adata = ad.read_zarr(zarr_path)

    emb = adata.obsm[EMBEDDING_KEY]
    ci, cj = COMPONENTS[0], COMPONENTS[1]
    x, y = emb[:, ci], emb[:, cj]

    predict_cols = sorted([c for c in adata.obs.columns if c.startswith("predicted_")])
    print(f"Prediction columns: {predict_cols}")

    ncols = len(predict_cols)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5 * ncols, 5),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    shuffle_idx = np.random.RandomState(42).permutation(len(x))

    for ax, col in zip(axes, predict_cols):
        categories = adata.obs[col].astype("category")
        cat_codes = categories.cat.codes.values
        unique_cats = categories.cat.categories.tolist()
        colors = ["#1b69a1", "#d9534f"]

        for i, cat in enumerate(unique_cats):
            mask = cat_codes == i
            order = np.argsort(shuffle_idx[mask])
            ax.scatter(
                x[mask][order],
                y[mask][order],
                s=POINT_SIZE,
                c=colors[i % len(colors)],
                label=cat,
                alpha=0.5,
                rasterized=True,
            )
        ax.legend(markerscale=5, fontsize=8, loc="best", framealpha=0.8)
        title = col.replace("predicted_", "").replace("_", " ").title()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{EMBEDDING_KEY} {COMPONENTS[0]}")
        ax.set_ylabel(f"{EMBEDDING_KEY} {COMPONENTS[1]}")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)

    fig.suptitle(
        f"Comparison for linear classifiers for: {zarr_path.stem}",
        fontsize=12,
        fontweight="bold",
    )
    plt.show()
    fig.savefig(OUTPUT_DIR / f"plots_{EMBEDDING_KEY}_{zarr_path.stem}.pdf")
    # plt.close(fig)

# %%
