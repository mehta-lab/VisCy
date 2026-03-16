# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import anndata
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA

from utils.embedding_utils import convert_to_dataframe

# %% dataset configuration
# Add more dataset dicts to this list to include additional replicates or conditions.
# Each entry must have:
#   name             : unique identifier used in output filenames
#   zarr_path        : Path to the AnnData zarr embedding store
#   infection_model_path : Path to a pre-trained joblib classifier
#   condition_key    : obs column that identifies DENV / ZIKV / uninfected wells
datasets = [
    {
        "name": "medium_organelles_rep1",
        "zarr_path": Path("medium_organelles_DynaCLR_embeddings.zarr"),
        "infection_model_path": Path("models/sensor_model.joblib"),
        "condition_key": "condition",
    },
    # Add more replicates here, e.g.:
    # {
    #     "name": "medium_organelles_rep2",
    #     "zarr_path": Path("medium_organelles_rep2_DynaCLR_embeddings.zarr"),
    #     "infection_model_path": Path("models/sensor_model.joblib"),
    #     "condition_key": "condition",
    # },
]

output_dir = Path("mmd_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Time mapping: list of t-index values that correspond to hours post-infection
timepoints = [6, 12, 18, 24]     # hours post infection
time_interval_hours = 0.5        # hours per t-index step
# t-index → HPI: hpi = t_index * time_interval_hours

# Class labels used by the infection classifier
INFECTED_CLASS = 2
UNINFECTED_CLASS = 1


# %% MMD implementation
def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute the RBF-kernel Maximum Mean Discrepancy between sets X and Y.

    Uses the unbiased estimator:
        MMD^2(X,Y) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    where k(a,b) = exp(-gamma * ||a-b||^2).

    Parameters
    ----------
    X, Y:
        Feature matrices of shape (n_samples, n_features).
    gamma:
        RBF bandwidth parameter.

    Returns
    -------
    float
        MMD value (non-negative).
    """
    def rbf_kernel(A, B, g):
        # ||a - b||^2 via the (a-b)·(a-b) identity
        AA = np.sum(A ** 2, axis=1, keepdims=True)
        BB = np.sum(B ** 2, axis=1, keepdims=True)
        AB = A @ B.T
        sq_dist = AA + BB.T - 2 * AB
        return np.exp(-g * np.maximum(sq_dist, 0))

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    n, m = len(X), len(Y)
    # Unbiased: zero out diagonal of K_XX and K_YY
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd2 = (K_XX.sum() / (n * (n - 1))
            - 2 * K_XY.mean()
            + K_YY.sum() / (m * (m - 1)))
    return float(np.sqrt(np.maximum(mmd2, 0)))


# %% helper – reduce embeddings to a manageable dimension before MMD
def reduce_embeddings(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA-reduce *X* to *n_components* if it has more features."""
    if X.shape[1] <= n_components:
        return X
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


# %% main processing loop
all_mmd_records = []

for ds_config in datasets:
    ds_name = ds_config["name"]
    zarr_path = ds_config["zarr_path"]
    infection_model_path = ds_config["infection_model_path"]
    condition_key = ds_config["condition_key"]

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"  zarr:  {zarr_path}")

    # Load embeddings
    adata = anndata.read_zarr(zarr_path)
    df = convert_to_dataframe(adata)

    obs_cols = list(adata.obs.columns)
    feature_cols = [c for c in df.columns if c not in obs_cols]

    # Load infection classifier and predict
    clf = joblib.load(infection_model_path)
    df["predicted_state"] = clf.predict(df[feature_cols].values)

    # Identify unique organelles (wells) and timepoints
    if condition_key in df.columns:
        organelles = df[condition_key].unique().tolist()
    elif "fov_name" in df.columns:
        organelles = df["fov_name"].unique().tolist()
    else:
        organelles = ["all"]

    # Convert HPI to t-indices
    t_indices = [int(hpi / time_interval_hours) for hpi in timepoints]

    mmd_records = []

    for organelle in organelles:
        if condition_key in df.columns:
            org_df = df[df[condition_key] == organelle]
        else:
            org_df = df

        for hpi, t_idx in zip(timepoints, t_indices):
            t_col = "t" if "t" in org_df.columns else None
            if t_col is not None:
                t_df = org_df[org_df[t_col] == t_idx]
            else:
                t_df = org_df  # use all timepoints if t not present

            uninfected = t_df[t_df["predicted_state"] == UNINFECTED_CLASS]
            infected = t_df[t_df["predicted_state"] == INFECTED_CLASS]

            if len(uninfected) < 5 or len(infected) < 5:
                print(f"  [SKIP] {organelle} t={t_idx} ({hpi}h): "
                      f"too few cells (uninf={len(uninfected)}, inf={len(infected)})")
                mmd_val = np.nan
            else:
                X = reduce_embeddings(uninfected[feature_cols].values)
                Y = reduce_embeddings(infected[feature_cols].values)
                mmd_val = compute_mmd(X, Y)
                print(f"  {organelle} | {hpi}h | MMD = {mmd_val:.4f} "
                      f"(n_uninf={len(uninfected)}, n_inf={len(infected)})")

            mmd_records.append(
                {
                    "dataset": ds_name,
                    "organelle": organelle,
                    "hpi": hpi,
                    "t_index": t_idx,
                    "MMD": mmd_val,
                    "n_uninfected": len(uninfected),
                    "n_infected": len(infected),
                }
            )

    ds_mmd_df = pd.DataFrame(mmd_records)
    all_mmd_records.append(ds_mmd_df)

    # Save per-dataset CSV
    ds_csv = output_dir / f"{ds_name}_mmd.csv"
    ds_mmd_df.to_csv(ds_csv, index=False)
    print(f"  Saved {ds_csv}")

    # %% heatmap (organelle × timepoint)
    pivot = ds_mmd_df.pivot(index="organelle", columns="hpi", values="MMD")
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}h" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hours Post Infection")
    ax.set_ylabel("Organelle / Condition")
    ax.set_title(f"MMD Heatmap – {ds_name}")
    plt.colorbar(im, ax=ax, label="MMD")
    plt.tight_layout()
    heatmap_svg = output_dir / f"{ds_name}_mmd_heatmap.svg"
    fig.savefig(heatmap_svg)
    plt.close(fig)
    print(f"  Saved heatmap → {heatmap_svg}")

    # %% line plot (MMD over time per organelle)
    fig, ax = plt.subplots(figsize=(8, 5))
    for organelle in ds_mmd_df["organelle"].unique():
        sub = ds_mmd_df[ds_mmd_df["organelle"] == organelle].sort_values("hpi")
        ax.plot(sub["hpi"], sub["MMD"], marker="o", label=organelle)
    ax.set_xlabel("Hours Post Infection")
    ax.set_ylabel("MMD (uninfected vs infected)")
    ax.set_title(f"Organelle Remodelling Score – {ds_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    lineplot_svg = output_dir / f"{ds_name}_mmd_lineplot.svg"
    fig.savefig(lineplot_svg)
    plt.close(fig)
    print(f"  Saved line plot → {lineplot_svg}")

# %% pool all datasets and save combined CSV
combined_df = pd.concat(all_mmd_records, ignore_index=True)
combined_csv = output_dir / "all_datasets_mmd.csv"
combined_df.to_csv(combined_csv, index=False)
print(f"\nCombined MMD results saved to {combined_csv}")
print(combined_df.head())
