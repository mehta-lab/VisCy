"""Persist pooled control-MAD/PCA coordinates into embedding Zarr stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import click
import numpy as np
import pandas as pd

from dynaclr.evaluation.mmd.config import MMDPooledConfig
from dynaclr.evaluation.mmd.representation import prepare_mmd_representation
from viscy_utils.cli_utils import load_config_section
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr

DEFAULT_REPRESENTATION_KEY = "X_normalized_pca80"
DEFAULT_CONFIG_SECTION = "pooled_representation"


@dataclass(frozen=True)
class _StoreRows:
    """Row interval and source metadata for one input store."""

    path: Path
    start: int
    stop: int
    existing_obsm: frozenset[str]
    existing_uns: frozenset[str]


def _extract_embeddings(adata: ad.AnnData, embedding_key: str | None) -> np.ndarray:
    values = adata.X if embedding_key is None else adata.obsm[embedding_key]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _fit_mask(obs: pd.DataFrame, config: MMDPooledConfig) -> np.ndarray:
    """Build the QC fit cohort while retaining both biological conditions."""
    keep = np.ones(len(obs), dtype=bool)
    if not config.obs_filter:
        return keep
    for column, value in config.obs_filter.items():
        if column not in obs:
            raise KeyError(f"obs_filter column {column!r} not found")
        if column != config.representation.control_key:
            keep &= obs[column].to_numpy() == value
    return keep


def _apply_condition_aliases(obs: pd.DataFrame, config: MMDPooledConfig) -> pd.DataFrame:
    if not config.condition_aliases:
        return obs
    if config.group_by not in obs:
        raise KeyError(f"condition alias column {config.group_by!r} not found")
    aliases = {variant: canonical for canonical, variants in config.condition_aliases.items() for variant in variants}
    obs = obs.copy()
    obs[config.group_by] = obs[config.group_by].map(lambda value: aliases.get(value, value))
    return obs


def _pca_checksums(prepared) -> dict[str, str]:
    return {
        marker: hashlib.sha256(fit.mean.tobytes() + fit.components.tobytes()).hexdigest()
        for marker, fit in prepared.pca_fits.items()
    }


def load_pooled_representation_config(path: Path) -> MMDPooledConfig:
    """Load the pooled representation section from the canonical recipe.

    Root-level historical configs remain readable for backward compatibility,
    but new work should use the canonical multi-section recipe.
    """
    raw = load_config_section(path, None, default_section=DEFAULT_CONFIG_SECTION)
    return MMDPooledConfig(**raw)


def export_pooled_representation(
    config: MMDPooledConfig,
    *,
    obsm_key: str = DEFAULT_REPRESENTATION_KEY,
    uns_key: str | None = None,
    artifact_dir: Path | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Fit one pooled representation and selectively update every input Zarr.

    Only ``obsm[obsm_key]`` and ``uns[uns_key or obsm_key]`` are replaced.
    Source ``X``, ``obs``, and all unrelated AnnData slots remain untouched.
    """
    uns_key = uns_key or obsm_key
    paths = [Path(path) for path in config.input_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("input_paths contains duplicate Zarr stores")
    if obsm_key == DEFAULT_REPRESENTATION_KEY and (
        config.representation.normalization != "control_mad" or config.representation.pca_variance != 0.80
    ):
        raise ValueError(
            f"{DEFAULT_REPRESENTATION_KEY!r} is reserved for control_mad with pca_variance=0.80; "
            "use a different --obsm-key for another representation"
        )

    matrices: list[np.ndarray] = []
    observations: list[pd.DataFrame] = []
    stores: list[_StoreRows] = []
    n_features: int | None = None
    offset = 0
    for path in paths:
        adata = ad.read_zarr(path)
        values = _extract_embeddings(adata, config.embedding_key)
        if n_features is None:
            n_features = values.shape[1]
        elif values.shape[1] != n_features:
            raise ValueError(f"{path}: embedding width {values.shape[1]} differs from pooled width {n_features}")
        required = {
            config.representation.experiment_key,
            config.representation.marker_key,
            config.representation.control_key,
            config.representation.hpi_key,
        }
        missing = required - set(adata.obs.columns)
        if missing:
            raise KeyError(f"{path}: missing obs columns {sorted(missing)}")
        if not overwrite and (obsm_key in adata.obsm or uns_key in adata.uns):
            raise FileExistsError(f"{path}: {obsm_key!r} or {uns_key!r} already exists; enable overwrite")
        matrices.append(values)
        observations.append(adata.obs.reset_index(drop=True).copy())
        stores.append(
            _StoreRows(
                path=path,
                start=offset,
                stop=offset + len(adata),
                existing_obsm=frozenset(adata.obsm.keys()),
                existing_uns=frozenset(adata.uns.keys()),
            )
        )
        offset += len(adata)

    pooled_obs = _apply_condition_aliases(pd.concat(observations, ignore_index=True), config)
    fit_mask = _fit_mask(pooled_obs, config)
    artifacts = artifact_dir or Path(config.output_dir) / "representation"
    prepared = prepare_mmd_representation(
        np.concatenate(matrices, axis=0),
        pooled_obs,
        config.representation,
        artifact_dir=artifacts,
        fit_mask=fit_mask,
    )
    metadata = {
        "schema_version": 1,
        "representation": prepared.label,
        "source_embedding_key": config.embedding_key or "X",
        "normalization": config.representation.model_dump(mode="json"),
        "pooled_fit": True,
        "fit_input_paths": [str(path) for path in paths],
        "fit_cells": int(fit_mask.sum()),
        "total_cells": int(len(pooled_obs)),
        "padded_dimensions": int(prepared.features.shape[1]),
        "n_components_by_marker": prepared.n_components_by_marker,
        "pca_model_checksums": _pca_checksums(prepared),
        "artifact_dir": str(artifacts),
    }

    rows: list[dict] = []
    for store in stores:
        local_features = prepared.features[store.start : store.stop]
        local_markers = sorted(
            pooled_obs.iloc[store.start : store.stop][config.representation.marker_key].astype(str).unique()
        )
        local_metadata = {
            **metadata,
            "source_path": str(store.path),
            "markers": local_markers,
        }
        append_to_anndata_zarr(
            store.path,
            obsm={obsm_key: local_features},
            uns={uns_key: local_metadata},
        )
        rows.append(
            {
                "input_path": str(store.path),
                "n_cells": int(len(local_features)),
                "n_dimensions": int(local_features.shape[1]),
                "markers": ",".join(local_markers),
                "replaced_obsm": obsm_key in store.existing_obsm,
                "replaced_uns": uns_key in store.existing_uns,
            }
        )

    manifest = pd.DataFrame(rows)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(artifacts / "zarr_export_manifest.csv", index=False)
    (artifacts / "zarr_export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Canonical biological-state YAML with a pooled_representation section.",
)
@click.option("--obsm-key", default=DEFAULT_REPRESENTATION_KEY, show_default=True)
@click.option("--uns-key", default=None, help="Metadata key; defaults to --obsm-key.")
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
def main(
    config: Path,
    obsm_key: str,
    uns_key: str | None,
    artifact_dir: Path | None,
    overwrite: bool,
) -> None:
    """Write pooled control-normalized PCA coordinates into source Zarrs."""
    cfg = load_pooled_representation_config(config)
    manifest = export_pooled_representation(
        cfg,
        obsm_key=obsm_key,
        uns_key=uns_key,
        artifact_dir=artifact_dir,
        overwrite=overwrite,
    )
    click.echo(manifest.to_string(index=False))
    click.echo(f"Wrote obsm[{obsm_key!r}] and uns[{uns_key or obsm_key!r}] to {len(manifest)} stores.")


if __name__ == "__main__":
    main()
