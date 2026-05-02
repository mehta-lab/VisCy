"""Embedding analysis for microglia and ALFI datasets.

Microglia (unsupervised):
    PCA/UMAP colored by perturbation condition and per-track embedding
    displacement — proxy for morphological dynamics (Khurana et al. 2022,
    https://doi.org/10.1091/mbc.E21-11-0561).

ALFI HeLa (supervised):
    PCA/UMAP colored by cell cycle phase annotations (interphase vs mitosis)
    from the ALFI dataset (Dang et al. 2023,
    https://doi.org/10.1038/s41597-023-02540-1).

Usage
-----
python scripts/evaluation/microglia_alfi_analysis.py \\
    --microglia-embeddings /path/to/microglia/embeddings.zarr \\
    --alfi-embeddings /path/to/alfi/embeddings.zarr \\
    --output-dir /path/to/output/
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

ALFI_ANNOTATIONS = Path("/hpc/projects/organelle_phenotyping/datasets/annotations/ALFI/ALFI_combined_annotations.csv")

DIVISION_PALETTE = {
    "interphase": "cornflowerblue",
    "mitosis": "darkorange",
}


def compute_track_displacement_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """Compute per-track embedding displacement metrics.

    Parameters
    ----------
    adata : AnnData
        Embeddings with obs columns fov_name, track_id, t.
        adata.X contains raw embeddings (N x D).

    Returns
    -------
    pd.DataFrame
        One row per track with columns:
        fov_name, track_id, mean_step_size, total_path_length,
        net_displacement, track_length, and any available metadata columns.
    """
    embeddings = np.asarray(adata.X)
    obs = adata.obs.copy()
    obs["_idx"] = np.arange(len(obs))

    meta_cols = [c for c in ["perturbation", "marker", "experiment"] if c in obs.columns]
    records = []

    for (fov, tid), grp in obs.groupby(["fov_name", "track_id"], sort=False):
        grp = grp.sort_values("t")
        idxs = grp["_idx"].values
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        steps = np.linalg.norm(np.diff(embs, axis=0), axis=1)
        record = {
            "fov_name": fov,
            "track_id": tid,
            "mean_step_size": steps.mean(),
            "total_path_length": steps.sum(),
            "net_displacement": float(np.linalg.norm(embs[-1] - embs[0])),
            "track_length": len(idxs),
        }
        for col in meta_cols:
            record[col] = grp[col].iloc[0]
        records.append(record)

    return pd.DataFrame(records)


def _get_or_compute_pca(adata: ad.AnnData, features_scaled: np.ndarray) -> np.ndarray:
    if "X_pca" in adata.obsm:
        return adata.obsm["X_pca"]
    pca = PCA(n_components=32)
    return pca.fit_transform(features_scaled)


def _get_or_compute_umap(adata: ad.AnnData, features_scaled: np.ndarray) -> np.ndarray:
    if "X_umap" in adata.obsm:
        return adata.obsm["X_umap"]
    print("  Computing UMAP...")
    return UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(features_scaled)


def analyze_microglia(adata: ad.AnnData, output_dir: Path) -> None:
    """Run microglia displacement analysis and save plots."""
    print(f"Microglia: {adata.shape[0]:,} observations")

    features = np.asarray(adata.X)
    features_scaled = StandardScaler().fit_transform(features)
    pca_emb = _get_or_compute_pca(adata, features_scaled)
    umap_emb = _get_or_compute_umap(adata, features_scaled)

    track_metrics = compute_track_displacement_metrics(adata)
    print(f"  {len(track_metrics):,} tracks")

    obs = adata.obs.copy().merge(
        track_metrics[["fov_name", "track_id", "mean_step_size", "net_displacement"]],
        on=["fov_name", "track_id"],
        how="left",
    )

    perturbations = sorted(obs["perturbation"].unique()) if "perturbation" in obs.columns else []
    markers = sorted(obs["marker"].unique()) if "marker" in obs.columns else []
    palette_p = dict(zip(perturbations, sns.color_palette("tab10", len(perturbations))))
    palette_m = dict(zip(markers, sns.color_palette("Set2", len(markers))))

    plot_df = pd.DataFrame(
        {
            "PC1": pca_emb[:, 0],
            "PC2": pca_emb[:, 1],
            "UMAP1": umap_emb[:, 0],
            "UMAP2": umap_emb[:, 1],
            "perturbation": obs["perturbation"].values if "perturbation" in obs.columns else "unknown",
            "marker": obs["marker"].values if "marker" in obs.columns else "unknown",
            "mean_step_size": obs["mean_step_size"].values,
            "net_displacement": obs["net_displacement"].values,
        }
    )

    vmin = np.nanpercentile(plot_df["mean_step_size"], 5)
    vmax = np.nanpercentile(plot_df["mean_step_size"], 95)

    for reduction, x_col, y_col in [("pca", "PC1", "PC2"), ("umap", "UMAP1", "UMAP2")]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="perturbation",
            palette=palette_p,
            ax=axes[0],
            alpha=0.5,
            s=8,
            linewidth=0,
        )
        axes[0].set_title(f"{reduction.upper()} — perturbation")

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="marker",
            palette=palette_m,
            ax=axes[1],
            alpha=0.5,
            s=8,
            linewidth=0,
        )
        axes[1].set_title(f"{reduction.upper()} — channel/marker")

        sc = axes[2].scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=plot_df["mean_step_size"],
            cmap="plasma",
            alpha=0.5,
            s=8,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(sc, ax=axes[2], label="Mean embedding step size")
        axes[2].set_title(f"{reduction.upper()} — embedding displacement")
        axes[2].set_xlabel(x_col)
        axes[2].set_ylabel(y_col)

        plt.tight_layout()
        out = output_dir / f"microglia_{reduction}.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    # Displacement by perturbation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = sorted(track_metrics["perturbation"].unique()) if "perturbation" in track_metrics.columns else None

    sns.boxplot(data=track_metrics, x="perturbation", y="mean_step_size", ax=axes[0], order=order)
    axes[0].set_title("Mean embedding step size by perturbation")
    axes[0].set_ylabel("Mean step size in embedding space")
    axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=track_metrics, x="perturbation", y="net_displacement", ax=axes[1], order=order)
    axes[1].set_title("Net displacement (start→end) by perturbation")
    axes[1].set_ylabel("Net displacement in embedding space")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = output_dir / "microglia_displacement_by_perturbation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    summary = track_metrics.groupby("perturbation")[["mean_step_size", "net_displacement", "track_length"]].agg(
        ["median", "mean", "std", "count"]
    )
    print("\n## Microglia track displacement summary\n")
    print(summary.to_markdown())


def analyze_alfi(adata: ad.AnnData, output_dir: Path) -> None:
    """Run ALFI HeLa cell cycle analysis and save plots."""
    print(f"\nALFI total: {adata.shape[0]:,} observations")

    # Filter to HeLa (MI06)
    if "fov_name" in adata.obs.columns:
        hela_mask = adata.obs["fov_name"] == "MI06"
    elif "experiment" in adata.obs.columns:
        hela_mask = adata.obs["experiment"].str.contains("HeLa")
    else:
        raise RuntimeError("Cannot identify HeLa cells — no fov_name or experiment column in obs")

    adata_hela = adata[hela_mask].copy()
    print(f"  HeLa (MI06): {adata_hela.shape[0]:,} observations")

    # Join annotations
    annotations = pd.read_csv(ALFI_ANNOTATIONS)
    ann_indexed = annotations.set_index(["fov_name", "track_id", "t"])

    obs_hela = adata_hela.obs.copy()
    mi = pd.MultiIndex.from_arrays(
        [
            obs_hela["fov_name"],
            obs_hela["track_id"].astype(int),
            obs_hela["t"].astype(int),
        ],
        names=["fov_name", "track_id", "t"],
    )
    obs_hela["cell_division_state"] = ann_indexed.reindex(mi)["cell_division_state"].values
    obs_hela["cell_cycle_fine_state"] = ann_indexed.reindex(mi)["cell_cycle_fine_state"].values

    n_annotated = obs_hela["cell_division_state"].notna().sum()
    print(f"  Annotated: {n_annotated:,} / {len(obs_hela):,}")
    print(obs_hela["cell_division_state"].value_counts().to_string())

    features_hela = np.asarray(adata_hela.X)
    features_scaled = StandardScaler().fit_transform(features_hela)
    pca_emb = _get_or_compute_pca(adata_hela, features_scaled)
    umap_emb = _get_or_compute_umap(adata_hela, features_scaled)

    unannotated = obs_hela["cell_division_state"].isna()

    for reduction, emb in [("pca", pca_emb), ("umap", umap_emb)]:
        x_col, y_col = ("PC1", "PC2") if reduction == "pca" else ("UMAP1", "UMAP2")

        # Division state plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, fine in zip(axes, [False, True]):
            col = "cell_cycle_fine_state" if fine else "cell_division_state"
            states = obs_hela[col].dropna().unique()
            if fine:
                palette = dict(zip(sorted(states), sns.color_palette("tab10", len(states))))
            else:
                palette = DIVISION_PALETTE

            for state, color in palette.items():
                mask = obs_hela[col] == state
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=color,
                    label=state,
                    alpha=0.6,
                    s=10,
                    linewidth=0,
                )
            ax.scatter(
                emb[unannotated, 0],
                emb[unannotated, 1],
                c="lightgray",
                label="unannotated",
                alpha=0.3,
                s=6,
                linewidth=0,
            )
            title = "fine cell cycle state" if fine else "cell division state"
            ax.set_title(f"HeLa {reduction.upper()} — {title}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend(markerscale=2, bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        out = output_dir / f"alfi_hela_{reduction}_cell_cycle.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    # Displacement by cell cycle state
    track_metrics = compute_track_displacement_metrics(adata_hela)

    track_annotations = (
        annotations[annotations["fov_name"] == "MI06"]
        .groupby(["fov_name", "track_id"])["cell_division_state"]
        .agg(lambda x: x.dropna().mode().iloc[0] if x.dropna().shape[0] > 0 else pd.NA)
        .reset_index()
        .rename(columns={"cell_division_state": "dominant_state"})
    )
    track_metrics = track_metrics.merge(track_annotations, on=["fov_name", "track_id"], how="left")

    annotated = track_metrics.dropna(subset=["dominant_state"])
    if len(annotated) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            data=annotated,
            x="dominant_state",
            y="mean_step_size",
            palette=DIVISION_PALETTE,
            ax=ax,
            order=[s for s in DIVISION_PALETTE if s in annotated["dominant_state"].unique()],
        )
        ax.set_title("HeLa: embedding step size by cell cycle state")
        ax.set_xlabel("Dominant cell division state (per track)")
        ax.set_ylabel("Mean step size in embedding space")
        plt.tight_layout()
        out = output_dir / "alfi_hela_displacement_by_state.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

        summary = annotated.groupby("dominant_state")["mean_step_size"].describe()
        print("\n## ALFI HeLa displacement by state\n")
        print(summary.to_markdown())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--microglia-embeddings", type=Path, required=True, help="AnnData zarr from microglia inference"
    )
    parser.add_argument("--alfi-embeddings", type=Path, required=True, help="AnnData zarr from ALFI inference")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save PDF figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Microglia analysis ===")
    adata_micro = ad.read_zarr(args.microglia_embeddings)
    analyze_microglia(adata_micro, args.output_dir)

    print("\n=== ALFI analysis ===")
    adata_alfi = ad.read_zarr(args.alfi_embeddings)
    analyze_alfi(adata_alfi, args.output_dir)


if __name__ == "__main__":
    main()
