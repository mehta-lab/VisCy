"""
Joint dimensionality reduction (PCA, UMAP, PHATE) across multiple AnnData zarr stores.

Concatenates embeddings from all stores, fits joint reductions,
then writes per-store slices back as X_*_combined.

Usage
-----
dynaclr reduce-combined -c multi-dataset-dim-reduction.yml
"""

import anndata as ad
import click
import numpy as np

from viscy_utils.cli_utils import format_markdown_table, load_config_section
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr

from .config import CombinedDimensionalityReductionConfig
from .reduce_dimensionality import _run_pca, _run_phate, _run_umap


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=str),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: str):
    """Compute joint PCA, UMAP, and/or PHATE across multiple AnnData zarr stores."""
    click.echo("Loading configuration...")
    raw_config = load_config_section(config, None, default_section="reduce_combined")
    cfg = CombinedDimensionalityReductionConfig(**raw_config)

    if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True

    resolved_paths = [str(p) for p in cfg.input_paths]
    dataset_names = list(cfg.datasets.keys()) if cfg.datasets else None

    # Determine which keys will be written
    methods_to_run: list[tuple[str, object]] = []
    if cfg.pca is not None:
        methods_to_run.append(("pca", cfg.pca))
    if cfg.umap is not None:
        methods_to_run.append(("umap", cfg.umap))
    if cfg.phate is not None:
        methods_to_run.append(("phate", cfg.phate))

    key_map = {"pca": "X_pca_combined", "umap": "X_umap_combined", "phate": "X_phate_combined"}
    keys_to_write = [key_map[name] for name, _ in methods_to_run]

    # Check for existing keys before loading data
    if not cfg.overwrite_keys:
        for path in resolved_paths:
            adata = ad.read_zarr(path)
            for key in keys_to_write:
                if key in adata.obsm:
                    raise click.ClickException(
                        f"Key '{key}' already exists in {path}. Use overwrite_keys: true to replace."
                    )

    # Load embeddings from all stores
    all_features = []
    all_lineage_ids = []
    sample_counts = []
    for path in resolved_paths:
        click.echo(f"Reading {path}...")
        adata = ad.read_zarr(path)
        features = np.asarray(adata.X)
        all_features.append(features)
        sample_counts.append(features.shape[0])
        if "lineage_id" in adata.obs.columns:
            all_lineage_ids.append(adata.obs["lineage_id"].to_numpy())
        click.echo(f"  {features.shape[0]:,} samples x {features.shape[1]} features")

    combined = np.concatenate(all_features, axis=0)
    combined_lineage_ids = np.concatenate(all_lineage_ids) if all_lineage_ids else None
    click.echo(f"Combined: {combined.shape[0]:,} samples x {combined.shape[1]} features")

    # Compute reductions on joint data
    results: dict[str, np.ndarray] = {}

    runner_map = {"pca": _run_pca, "umap": _run_umap, "phate": _run_phate}
    for method_name, method_cfg in methods_to_run:
        if method_name == "phate":
            _, embedding = _run_phate(combined, method_cfg, lineage_ids=combined_lineage_ids)
        else:
            _, embedding = runner_map[method_name](combined, method_cfg)
        out_key = key_map[method_name]
        results[out_key] = embedding
        click.echo(f"  {method_name.upper()} done -> {out_key} ({embedding.shape[1]} components)")

    # Slice and write back to each store
    offset = 0
    for i, path in enumerate(resolved_paths):
        n = sample_counts[i]
        store_obsm = {key: emb[offset : offset + n] for key, emb in results.items()}
        store_uns = {}
        for method_name, _ in methods_to_run:
            store_uns[f"{method_name}_combined_datasets"] = resolved_paths
            if dataset_names is not None:
                store_uns[f"{method_name}_combined_dataset_names"] = dataset_names
        offset += n

        click.echo(f"Writing to {path} ({n:,} rows)...")
        append_to_anndata_zarr(path, obsm=store_obsm, uns=store_uns)

    # Summary
    summary_data = []
    for key, embedding in sorted(results.items()):
        summary_data.append(
            {
                "method": key,
                "components": embedding.shape[1],
                "total_samples": embedding.shape[0],
                "stores": len(resolved_paths),
            }
        )
    click.echo("\n" + format_markdown_table(summary_data, title="Combined Dimensionality Reduction"))
    click.echo(f"Results written to {len(resolved_paths)} store(s)")


if __name__ == "__main__":
    main()
