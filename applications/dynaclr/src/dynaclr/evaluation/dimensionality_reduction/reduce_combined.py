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

from .config import CombinedDimensionalityReductionConfig, PHATEConfig
from .reduce_dimensionality import _run_pca, _run_phate, _run_umap


def _phate_per_store_fit_idx(
    sample_counts: list[int],
    lineage_ids: np.ndarray | None,
    cap: int | None,
    random_state: int,
) -> tuple[np.ndarray, list[dict]]:
    """Build the PHATE fit-set with a per-store lineage cap.

    For each store, draw up to ``cap`` whole lineages (random sample without
    replacement). Stores with fewer lineages contribute all of theirs. The
    returned indices are global row indices into the concatenated feature
    matrix; PHATE is later transformed on the full matrix.

    Parameters
    ----------
    sample_counts : list[int]
        Row count contributed by each store, in concatenation order.
    lineage_ids : np.ndarray or None
        Per-row lineage identifier (already store-prefixed so namespaces are
        disjoint). When None, falls back to per-store random row sampling.
    cap : int or None
        Maximum lineages drawn per store. ``None`` keeps every row.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    fit_idx : np.ndarray
        Global row indices used to fit PHATE.
    per_store_stats : list[dict]
        One row per store with ``store_idx``, ``n_lineages_total``,
        ``n_lineages_kept``, ``n_rows_total``, ``n_rows_kept``.
    """
    rng = np.random.default_rng(random_state)
    fit_indices: list[np.ndarray] = []
    per_store_stats: list[dict] = []
    offset = 0
    for store_idx, n_rows in enumerate(sample_counts):
        store_slice = slice(offset, offset + n_rows)
        if lineage_ids is None:
            # No lineage info: cap rows directly.
            if cap is None or n_rows <= cap:
                idx = np.arange(offset, offset + n_rows)
                kept_lineages = -1  # sentinel: unknown
                total_lineages = -1
            else:
                local = rng.choice(n_rows, size=cap, replace=False)
                idx = local + offset
                kept_lineages = -1
                total_lineages = -1
        else:
            store_lineages = lineage_ids[store_slice]
            unique_lineages = np.unique(store_lineages)
            total_lineages = len(unique_lineages)
            if cap is None or total_lineages <= cap:
                idx = np.arange(offset, offset + n_rows)
                kept_lineages = total_lineages
            else:
                chosen = rng.choice(unique_lineages, size=cap, replace=False)
                local_mask = np.isin(store_lineages, chosen)
                idx = np.where(local_mask)[0] + offset
                kept_lineages = cap

        fit_indices.append(idx)
        per_store_stats.append(
            {
                "store_idx": store_idx,
                "n_lineages_total": total_lineages,
                "n_lineages_kept": kept_lineages,
                "n_rows_total": n_rows,
                "n_rows_kept": int(idx.size),
            }
        )
        offset += n_rows

    return np.concatenate(fit_indices), per_store_stats


def _select_phate_input(
    combined: np.ndarray,
    results: dict[str, np.ndarray],
    phate_cfg: PHATEConfig,
) -> tuple[np.ndarray, str]:
    """Choose what PHATE fits and transforms on.

    When ``phate.n_pca`` is None, the recipe is signalling that PHATE should
    *not* run its own internal PCA — feed the already-PCA-reduced input
    directly. We require ``X_pca_combined`` to exist in ``results`` (PCA
    runs first when both are configured) and use it.

    When ``phate.n_pca`` is an int, fall back to the raw concatenated
    embeddings — PHATE will run its internal PCA on them.
    """
    if phate_cfg.n_pca is None:
        if "X_pca_combined" not in results:
            raise click.ClickException(
                "PHATE is configured with n_pca=null (skip internal PCA), "
                "but X_pca_combined is not available. Add `pca:` to the "
                "reduce_combined recipe so PCA runs before PHATE, or set "
                "phate.n_pca to an int to use PHATE's internal PCA "
                "(warning: hangs on scipy 1.17.1)."
            )
        return results["X_pca_combined"], "X_pca_combined"
    return combined, "raw .X (PHATE will run internal PCA)"


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

    # Load embeddings from all stores. Derive lineage IDs for PHATE
    # subsampling: a lineage is (path, fov_name, track_id), prefixed
    # with the path index so track IDs from different stores don't
    # collide.
    all_features = []
    all_lineage_ids: list[np.ndarray] = []
    sample_counts = []
    have_lineage_cols = True
    for store_idx, path in enumerate(resolved_paths):
        click.echo(f"Reading {path}...")
        adata = ad.read_zarr(path)
        features = np.asarray(adata.X)
        all_features.append(features)
        sample_counts.append(features.shape[0])
        if "lineage_id" in adata.obs.columns:
            all_lineage_ids.append(adata.obs["lineage_id"].to_numpy())
        elif {"fov_name", "track_id"}.issubset(adata.obs.columns):
            fov = adata.obs["fov_name"].astype(str).to_numpy()
            tid = adata.obs["track_id"].astype(str).to_numpy()
            # Prefix with store_idx to keep lineage namespaces disjoint
            # across stores in the concatenated array.
            lineage = np.array([f"{store_idx}|{f}|{t}" for f, t in zip(fov, tid)])
            all_lineage_ids.append(lineage)
        else:
            have_lineage_cols = False
        click.echo(f"  {features.shape[0]:,} samples x {features.shape[1]} features")

    combined = np.concatenate(all_features, axis=0)
    if have_lineage_cols and all_lineage_ids:
        combined_lineage_ids = np.concatenate(all_lineage_ids)
        n_lineages = int(np.unique(combined_lineage_ids).size)
        click.echo(f"Combined: {combined.shape[0]:,} samples x {combined.shape[1]} features, {n_lineages:,} lineages")
    else:
        combined_lineage_ids = None
        click.echo(
            f"Combined: {combined.shape[0]:,} samples x {combined.shape[1]} features "
            "(no lineage_id / fov_name+track_id; PHATE will use random subsampling)"
        )

    # Compute reductions on joint data
    #
    # Order matters: PCA runs first when both PCA and PHATE are requested,
    # so PHATE can be fit on the already-PCA-reduced X_pca_combined. This
    # avoids PHATE's internal PCA pre-reduction (sklearn -> scipy.linalg.lu),
    # which deadlocks silently on scipy 1.17.1 + sklearn 1.8.0. Pass
    # n_pca=None in the recipe to skip the internal PCA when feeding
    # PCA-reduced input.
    results: dict[str, np.ndarray] = {}

    runner_map = {"pca": _run_pca, "umap": _run_umap, "phate": _run_phate}
    for method_name, method_cfg in methods_to_run:
        if method_name == "phate":
            assert isinstance(method_cfg, PHATEConfig)
            phate_input, source_label = _select_phate_input(combined, results, method_cfg)
            click.echo(f"  PHATE fitting on {source_label} ({phate_input.shape[1]} dims)")

            fit_idx, per_store_stats = _phate_per_store_fit_idx(
                sample_counts=sample_counts,
                lineage_ids=combined_lineage_ids,
                cap=method_cfg.subsample,
                random_state=method_cfg.random_state,
            )
            click.echo(
                "\n"
                + format_markdown_table(per_store_stats, title=f"PHATE per-store fit cap (cap={method_cfg.subsample})")
            )
            _, embedding = _run_phate(
                phate_input,
                method_cfg,
                lineage_ids=combined_lineage_ids,
                fit_idx=fit_idx,
            )
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
