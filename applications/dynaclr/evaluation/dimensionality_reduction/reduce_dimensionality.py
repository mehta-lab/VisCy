"""
CLI tool for computing dimensionality reductions on saved embeddings.

Decouples PCA, UMAP, and PHATE computation from the prediction step,
allowing users to run reductions on existing AnnData zarr files.

Usage
-----
dynaclr reduce-dimensionality -c reduce_config.yaml
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import click
import numpy as np
from numpy.typing import NDArray

from viscy_utils.cli_utils import format_markdown_table, load_config

from .config import (
    DimensionalityReductionConfig,
    PCAConfig,
    PHATEConfig,
    UMAPConfig,
)


def _run_pca(features: NDArray, cfg: PCAConfig) -> tuple[str, NDArray]:
    from viscy_utils.evaluation.dimensionality_reduction import compute_pca

    pca_features, _ = compute_pca(
        features,
        n_components=cfg.n_components,
        normalize_features=cfg.normalize_features,
    )
    return "X_pca", pca_features


def _run_umap(features: NDArray, cfg: UMAPConfig) -> tuple[str, NDArray]:
    from viscy_utils.evaluation.dimensionality_reduction import _fit_transform_umap

    _, umap_embedding = _fit_transform_umap(
        features,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        normalize=cfg.normalize,
    )
    return "X_umap", umap_embedding


def _run_phate(features: NDArray, cfg: PHATEConfig) -> tuple[str, NDArray]:
    from viscy_utils.evaluation.dimensionality_reduction import compute_phate

    _, phate_embedding = compute_phate(
        features,
        n_components=cfg.n_components,
        knn=cfg.knn,
        decay=cfg.decay,
        scale_embeddings=cfg.scale_embeddings,
        random_state=cfg.random_state,
    )
    return "X_phate", phate_embedding


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """Compute PCA, UMAP, and/or PHATE on saved embeddings."""
    click.echo("Loading configuration...")
    raw_config = load_config(config)
    cfg = DimensionalityReductionConfig(**raw_config)

    click.echo(f"Reading embeddings from {cfg.input_path}...")
    if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    adata = ad.read_zarr(cfg.input_path)
    features = np.asarray(adata.X)
    click.echo(f"  Loaded {features.shape[0]:,} samples x {features.shape[1]} features")

    # Check for existing keys
    methods_to_run = []
    key_map = {"pca": "X_pca", "umap": "X_umap", "phate": "X_phate"}
    for method_name, obsm_key in key_map.items():
        method_cfg = getattr(cfg, method_name)
        if method_cfg is not None:
            if obsm_key in adata.obsm and not cfg.overwrite_keys:
                raise click.ClickException(
                    f"Key '{obsm_key}' already exists in .obsm. Use overwrite_keys: true to replace."
                )
            methods_to_run.append((method_name, method_cfg, obsm_key))

    runner_map = {"pca": _run_pca, "umap": _run_umap, "phate": _run_phate}

    click.echo(f"Computing {len(methods_to_run)} reduction(s): {', '.join(name for name, _, _ in methods_to_run)}")

    results = {}
    with ThreadPoolExecutor(max_workers=len(methods_to_run)) as executor:
        futures = {}
        for method_name, method_cfg, obsm_key in methods_to_run:
            future = executor.submit(runner_map[method_name], features, method_cfg)
            futures[future] = method_name

        for future in as_completed(futures):
            method_name = futures[future]
            try:
                key, embedding = future.result()
                results[key] = embedding
                click.echo(f"  {method_name.upper()} done -> {key} ({embedding.shape[1]} components)")
            except Exception as e:
                click.echo(f"  {method_name.upper()} failed: {e}", err=True)

    for key, embedding in results.items():
        adata.obsm[key] = embedding

    output_path = cfg.output_path or cfg.input_path
    click.echo(f"Writing results to {output_path}...")
    adata.write_zarr(output_path)

    # Print summary
    summary_data = []
    for key, embedding in sorted(results.items()):
        summary_data.append(
            {
                "method": key,
                "components": embedding.shape[1],
                "samples": embedding.shape[0],
            }
        )
    click.echo("\n" + format_markdown_table(summary_data, title="Dimensionality Reduction Results"))
    click.echo(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
