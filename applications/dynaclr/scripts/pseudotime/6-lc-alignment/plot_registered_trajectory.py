"""Qualitative plot: Phase3D embedding trajectory before vs after LC registration.

For each dataset in the recipe:

1. Load Phase3D embeddings + per-cell LC registration parquet.
2. Restrict to cells where the LC crossed τ (have a valid ``t_LC_star``).
3. PCA-project the Phase3D embeddings to 2D.
4. Plot two panels per dataset:
   - **Left**: raw frame index ``t`` colored by raw HPI (no registration).
   - **Right**: same PCA projection, but each point colored by
     ``t_reg = t - t_LC_star`` (LC-derived hours-post-onset).

If trajectories tighten after registration (cells follow a more consistent
path through PC1/PC2 vs registered HPI), the qualitative claim that the
LC-derived event time recovers a meaningful pseudotime is supported.

Output: ``<dataset>_trajectory.png`` per dataset and a combined
``trajectory_grid.png``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("plot_registered_trajectory")


def _load_config_with_recipes(config_path: Path) -> dict:
    """Merge ``base:`` recipe imports and resolve per-dataset filenames.

    Same convention as ``evaluate_registered_organelle.py``.
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        leaf = yaml.safe_load(f) or {}
    merged: dict = {}
    for rel in leaf.pop("base", []):
        with open((config_path.parent / rel).resolve()) as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update(leaf)
    org_dir = merged.get("organelle_embedding_dir")
    reg_dir = merged.get("registration_dir")
    for d in merged.get("datasets") or []:
        if "organelle_embedding_zarr" not in d and "organelle_embedding_filename" in d:
            d["organelle_embedding_zarr"] = str(Path(org_dir) / d["organelle_embedding_filename"])
        if "registration_parquet" not in d and "registration_filename" in d:
            d["registration_parquet"] = str(Path(reg_dir) / d["registration_filename"])
    return merged


def _plot_dataset(name: str, organelle_zarr: Path, registration_parquet: Path, out_png: Path) -> None:
    """PCA + scatter, colored by raw t vs registered t_reg."""
    _logger.info("[%s] loading embeddings", name)
    a = ad.read_zarr(organelle_zarr)
    obs = a.obs.copy().reset_index(drop=True)
    obs["track_id"] = obs["track_id"].astype(int)
    obs["t"] = obs["t"].astype(int)
    obs["_iloc"] = np.arange(len(obs))
    obs = obs.drop_duplicates(subset=["fov_name", "track_id", "t"]).reset_index(drop=True)
    X = a.X[obs["_iloc"].to_numpy()]
    if hasattr(X, "toarray"):
        X = X.toarray()
    obs = obs.drop(columns=["_iloc"])

    reg = pd.read_parquet(registration_parquet)
    reg["track_id"] = reg["track_id"].astype(int)
    reg = reg[["fov_name", "track_id", "t_LC_star", "crossed_tau"]]
    merged = obs.merge(reg, on=["fov_name", "track_id"], how="left")
    keep = merged["crossed_tau"].fillna(False).to_numpy().astype(bool)
    if keep.sum() == 0:
        _logger.warning("[%s] no cells crossed tau; skipping plot", name)
        return
    obs_keep = merged.loc[keep].reset_index(drop=True)
    X_keep = X[keep]

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_keep)
    t_reg = obs_keep["t"] - obs_keep["t_LC_star"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    sc1 = axes[0].scatter(Z[:, 0], Z[:, 1], c=obs_keep["t"], s=4, cmap="viridis", alpha=0.7)
    axes[0].set_title(f"{name} — Phase3D PCA, raw frame t")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.colorbar(sc1, ax=axes[0], label="t (frame)")

    sc2 = axes[1].scatter(Z[:, 0], Z[:, 1], c=t_reg, s=4, cmap="coolwarm", alpha=0.7)
    axes[1].set_title(f"{name} — Phase3D PCA, registered $t_{{reg}} = t - t_{{LC^\\star}}$")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.colorbar(sc2, ax=axes[1], label="$t_{reg}$ (frames since LC onset)")

    fig.suptitle(
        f"LC-derived registration of Phase3D embeddings — {name}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    _logger.info("[%s] wrote %s", name, out_png)


def main() -> None:
    """Render per-dataset PCA panels (raw t vs registered t_reg)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_with_recipes(args.config)
    for d in cfg["datasets"]:
        out_png = args.output_dir / f"{d['name']}_trajectory.png"
        _plot_dataset(
            name=d["name"],
            organelle_zarr=Path(d["organelle_embedding_zarr"]),
            registration_parquet=Path(d["registration_parquet"]),
            out_png=out_png,
        )


if __name__ == "__main__":
    main()
