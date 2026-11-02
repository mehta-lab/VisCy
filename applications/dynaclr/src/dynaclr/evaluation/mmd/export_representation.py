"""Persist pooled control-MAD/PCA coordinates into embedding Zarr stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem

from dynaclr.evaluation.mmd.config import MMDPooledConfig
from dynaclr.evaluation.mmd.representation import (
    _control_mad_normalize,
    prepare_mmd_representation,
    save_representation_artifacts,
)
from viscy_utils.cli_utils import load_config_section
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr

DEFAULT_REPRESENTATION_KEY = "X_normalized_pca80"
DEFAULT_NORMALIZED_KEY = "X_control_mad"
DEFAULT_CONFIG_SECTION = "pooled_representation"


@dataclass
class _StoreRows:
    """Row interval and source metadata for one input store."""

    path: Path
    start: int
    stop: int
    obs: pd.DataFrame
    source_descriptor: dict[str, Any]
    obs_checksum: str
    existing_obsm: frozenset[str]
    existing_uns: frozenset[str]
    cache_metadata: dict[str, Any] | None = None
    cache_reused: bool = False


def _sha256_bytes(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _json_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_checksum(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _source_array(group: zarr.Group, embedding_key: str | None):
    return group["X"] if embedding_key is None else group[f"obsm/{embedding_key}"]


def _storage_fingerprint(store_path: Path, embedding_key: str | None) -> str:
    """Hash source chunk names, sizes, and mtimes without reading their values."""
    relative = Path("X") if embedding_key is None else Path("obsm") / embedding_key
    source_path = store_path / relative
    digest = hashlib.sha256()
    for path in sorted(item for item in source_path.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(str(path.relative_to(source_path)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _source_descriptor(array, embedding_key: str | None, store_path: Path) -> dict[str, Any]:
    """Describe source structure and storage state without reading raw values."""
    chunks = getattr(array, "chunks", None)
    return {
        "embedding_key": embedding_key or "X",
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "chunks": None if chunks is None else [int(value) for value in chunks],
        "storage_fingerprint": _storage_fingerprint(store_path, embedding_key),
    }


def _normalization_settings(config: MMDPooledConfig) -> dict[str, Any]:
    representation = config.representation
    return {
        "normalization": representation.normalization,
        "control_key": representation.control_key,
        "control_values": representation.control_values,
        "experiment_key": representation.experiment_key,
        "marker_key": representation.marker_key,
        "hpi_key": representation.hpi_key,
        "smooth_sigma_timepoints": representation.smooth_sigma_timepoints,
        "mad_floor_quantile": representation.mad_floor_quantile,
        "condition_column": config.group_by,
        "condition_aliases": config.condition_aliases,
        "fit_obs_filter": {
            key: value for key, value in (config.obs_filter or {}).items() if key != representation.control_key
        },
    }


def _normalization_obs_columns(config: MMDPooledConfig) -> list[str]:
    representation = config.representation
    columns = [
        representation.experiment_key,
        representation.marker_key,
        representation.control_key,
        representation.hpi_key,
    ]
    columns.extend(key for key in (config.obs_filter or {}) if key != representation.control_key and key not in columns)
    return columns


def _read_mapping(group: zarr.Group, path: str) -> dict[str, Any] | None:
    if path not in group:
        return None
    value = read_elem(group[path])
    return dict(value) if isinstance(value, dict) else None


def _valid_normalized_cache(
    group: zarr.Group,
    store: _StoreRows,
    normalized_key: str,
    settings_checksum: str,
) -> bool:
    metadata = _read_mapping(group, f"uns/{normalized_key}")
    if f"obsm/{normalized_key}" not in group or metadata is None:
        return False
    cached = group[f"obsm/{normalized_key}"]
    cached_descriptor = metadata.get("source_descriptor", {})
    cached_chunks = cached_descriptor.get("chunks")
    source_chunks = store.source_descriptor.get("chunks")
    chunks_match = (tuple(cached_chunks) if cached_chunks is not None else ()) == (
        tuple(source_chunks) if source_chunks is not None else ()
    )
    descriptor_matches = (
        cached_descriptor.get("embedding_key") == store.source_descriptor["embedding_key"]
        and tuple(cached_descriptor.get("shape", ())) == tuple(store.source_descriptor["shape"])
        and cached_descriptor.get("dtype") == store.source_descriptor["dtype"]
        and chunks_match
        and cached_descriptor.get("storage_fingerprint") == store.source_descriptor["storage_fingerprint"]
    )
    valid = (
        int(metadata.get("schema_version", 0)) >= 2
        and metadata.get("source_path") == str(store.path)
        and metadata.get("normalization_settings_checksum") == settings_checksum
        and metadata.get("normalization_obs_checksum") == store.obs_checksum
        and descriptor_matches
        and tuple(cached.shape) == tuple(store.source_descriptor["shape"])
    )
    if valid:
        store.cache_metadata = metadata
    return bool(valid)


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


def _groups(obs: pd.DataFrame, config: MMDPooledConfig) -> set[tuple[str, str]]:
    representation = config.representation
    return set(
        zip(
            obs[representation.experiment_key].astype(str),
            obs[representation.marker_key].astype(str),
            strict=True,
        )
    )


def _recompute_closure(stores: list[_StoreRows], invalid: set[int], config: MMDPooledConfig) -> set[int]:
    """Include every store sharing an experiment-marker normalization fit."""
    selected = set(invalid)
    while True:
        selected_groups = set().union(*(_groups(stores[index].obs, config) for index in selected))
        expanded = selected | {
            index for index, store in enumerate(stores) if _groups(store.obs, config) & selected_groups
        }
        if expanded == selected:
            return selected
        selected = expanded


def _contract_checksum(contracts: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for contract in sorted(contracts, key=lambda item: (str(item["experiment"]), str(item["marker"]))):
        digest.update(str(contract["experiment"]).encode())
        digest.update(str(contract["marker"]).encode())
        for key in ("hpi_grid", "control_centers", "control_mad"):
            values = np.ascontiguousarray(contract[key])
            digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _pca_fit_manifest(prepared, obs: pd.DataFrame, stores: list[_StoreRows], config: MMDPooledConfig):
    representation = config.representation
    controls = set(map(str, representation.control_values))
    manifest_rows: list[dict[str, Any]] = []
    row_checksums: dict[str, str] = {}
    for marker, indices in prepared.pca_fit_rows.items():
        indices = np.asarray(indices, dtype=np.int64)
        row_checksums[marker] = hashlib.sha256(indices.tobytes()).hexdigest()
        selected = obs.iloc[indices].copy()
        selected["pca_class"] = np.where(
            selected[representation.control_key].astype(str).isin(controls),
            "control",
            "perturbed",
        )
        for store in stores:
            local = selected[(selected.index >= store.start) & (selected.index < store.stop)]
            if local.empty:
                continue
            for (experiment, pca_class), frame in local.groupby(
                [representation.experiment_key, "pca_class"], observed=True, sort=True
            ):
                manifest_rows.append(
                    {
                        "marker": marker,
                        "input_path": str(store.path),
                        "pooled_row_start": store.start,
                        "pooled_row_stop": store.stop,
                        "experiment": str(experiment),
                        "pca_class": str(pca_class),
                        "selected_cells": int(len(frame)),
                    }
                )
    return pd.DataFrame(manifest_rows), row_checksums


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
    normalized_key: str = DEFAULT_NORMALIZED_KEY,
    artifact_dir: Path | None = None,
    overwrite: bool = True,
    reuse_normalized: bool = True,
) -> pd.DataFrame:
    """Persist reusable control-MAD coordinates, then fit and save pooled PCA."""
    uns_key = uns_key or obsm_key
    paths = [Path(path) for path in config.input_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("input_paths contains duplicate Zarr stores")
    if normalized_key == obsm_key:
        raise ValueError("normalized_key and obsm_key must be different")
    if obsm_key == DEFAULT_REPRESENTATION_KEY and (
        config.representation.normalization != "control_mad" or config.representation.pca_variance != 0.80
    ):
        raise ValueError(
            f"{DEFAULT_REPRESENTATION_KEY!r} is reserved for control_mad with pca_variance=0.80; "
            "use a different --obsm-key for another representation"
        )

    required = set(_normalization_obs_columns(config))
    settings = _normalization_settings(config)
    settings_checksum = _json_checksum(settings)
    observations: list[pd.DataFrame] = []
    stores: list[_StoreRows] = []
    n_features: int | None = None
    offset = 0
    invalid: set[int] = set()
    for index, path in enumerate(paths):
        group = zarr.open_group(str(path), mode="r", use_consolidated=False)
        obs = read_elem(group["obs"]).reset_index(drop=True)
        missing = required - set(obs.columns)
        if missing:
            raise KeyError(f"{path}: missing obs columns {sorted(missing)}")
        source = _source_array(group, config.embedding_key)
        descriptor = _source_descriptor(source, config.embedding_key, path)
        if len(descriptor["shape"]) != 2:
            raise ValueError(f"{path}: source embedding must be two-dimensional")
        if n_features is None:
            n_features = descriptor["shape"][1]
        elif descriptor["shape"][1] != n_features:
            raise ValueError(f"{path}: embedding width {descriptor['shape'][1]} differs from pooled width {n_features}")
        if len(obs) != descriptor["shape"][0]:
            raise ValueError(f"{path}: obs rows and source embedding rows differ")
        existing_obsm = frozenset(group["obsm"].keys()) if "obsm" in group else frozenset()
        existing_uns = frozenset(group["uns"].keys()) if "uns" in group else frozenset()
        if not overwrite and (obsm_key in existing_obsm or uns_key in existing_uns):
            raise FileExistsError(f"{path}: {obsm_key!r} or {uns_key!r} already exists; enable overwrite")
        aliased_obs = _apply_condition_aliases(obs, config)
        obs_checksum = _frame_checksum(aliased_obs[_normalization_obs_columns(config)])
        store = _StoreRows(
            path=path,
            start=offset,
            stop=offset + len(obs),
            obs=aliased_obs,
            source_descriptor=descriptor,
            obs_checksum=obs_checksum,
            existing_obsm=existing_obsm,
            existing_uns=existing_uns,
        )
        stores.append(store)
        observations.append(aliased_obs)
        if not reuse_normalized or not _valid_normalized_cache(group, store, normalized_key, settings_checksum):
            invalid.add(index)
        offset += len(obs)

    pooled_obs = pd.concat(observations, ignore_index=True)
    fit_mask = _fit_mask(pooled_obs, config)
    artifacts = artifact_dir or Path(config.output_dir) / "representation"
    normalized_by_store: dict[int, np.ndarray] = {}
    normalization_contracts: list[dict[str, Any]] = []
    normalization_summaries: list[pd.DataFrame] = []

    recompute = _recompute_closure(stores, invalid, config) if invalid else set()
    if recompute:
        recompute_indices = sorted(recompute)
        raw_parts: list[np.ndarray] = []
        obs_parts: list[pd.DataFrame] = []
        fit_parts: list[np.ndarray] = []
        for index in recompute_indices:
            store = stores[index]
            group = zarr.open_group(str(store.path), mode="r", use_consolidated=False)
            raw_parts.append(np.asarray(_source_array(group, config.embedding_key)[:], dtype=np.float32))
            obs_parts.append(store.obs)
            fit_parts.append(fit_mask[store.start : store.stop])
        raw = np.concatenate(raw_parts, axis=0)
        local_obs = pd.concat(obs_parts, ignore_index=True)
        local_fit = np.concatenate(fit_parts)
        if config.representation.normalization == "control_mad":
            normalized, contracts, summary = _control_mad_normalize(raw, local_obs, config.representation, local_fit)
        else:
            normalized, contracts, summary = raw.copy(), (), pd.DataFrame()
        normalization_contracts.extend(contracts)
        normalization_summaries.append(summary)
        cursor = 0
        for index, raw_part in zip(recompute_indices, raw_parts, strict=True):
            store = stores[index]
            local = normalized[cursor : cursor + len(raw_part)]
            cursor += len(raw_part)
            store_groups = _groups(store.obs, config)
            local_contracts = [
                contract
                for contract in contracts
                if (str(contract["experiment"]), str(contract["marker"])) in store_groups
            ]
            local_summary = (
                summary[
                    summary.apply(
                        lambda row: (str(row["experiment"]), str(row["marker"])) in store_groups,
                        axis=1,
                    )
                ]
                if not summary.empty
                else summary.copy()
            )
            local_metadata = {
                "schema_version": 2,
                "representation": config.representation.normalization,
                "source_path": str(store.path),
                "source_embedding_key": config.embedding_key or "X",
                "source_descriptor": store.source_descriptor,
                "source_values_sha256": _sha256_bytes(raw_part),
                "normalization_settings": settings,
                "normalization_settings_checksum": settings_checksum,
                "normalization_obs_checksum": store.obs_checksum,
                "normalized_values_sha256": _sha256_bytes(local),
                "contract_checksum": _contract_checksum(local_contracts),
                "contracts": {f"contract_{position}": contract for position, contract in enumerate(local_contracts)},
                "normalization_summary_json": local_summary.to_json(orient="records"),
            }
            append_to_anndata_zarr(
                store.path,
                obsm={normalized_key: local},
                uns={normalized_key: local_metadata},
            )
            store.cache_metadata = local_metadata
            normalized_by_store[index] = local

    for index, store in enumerate(stores):
        if index not in normalized_by_store:
            group = zarr.open_group(str(store.path), mode="r", use_consolidated=False)
            normalized_by_store[index] = np.asarray(group[f"obsm/{normalized_key}"][:], dtype=np.float32)
            store.cache_reused = True

    for store in stores:
        if not store.cache_reused:
            continue
        cached_contracts = store.cache_metadata.get("contracts", {})
        normalization_contracts.extend(dict(cached_contracts).values())
        cached_summary = store.cache_metadata.get("normalization_summary_json")
        if cached_summary:
            normalization_summaries.append(pd.read_json(StringIO(str(cached_summary)), orient="records"))

    normalization_contracts = list(
        {
            (str(contract["experiment"]), str(contract["marker"])): contract for contract in normalization_contracts
        }.values()
    )

    pooled_normalized = np.concatenate([normalized_by_store[index] for index in range(len(stores))], axis=0)
    normalization_summary = (
        pd.concat(normalization_summaries, ignore_index=True) if normalization_summaries else pd.DataFrame()
    )
    if not normalization_summary.empty:
        normalization_summary = normalization_summary.drop_duplicates(
            subset=["experiment", "marker"], keep="last"
        ).reset_index(drop=True)
    prepared = prepare_mmd_representation(
        pooled_normalized,
        pooled_obs,
        config.representation,
        fit_mask=fit_mask,
        pre_normalized=pooled_normalized,
        normalization_contracts=tuple(normalization_contracts),
        normalization_summary=normalization_summary,
    )
    save_representation_artifacts(prepared, config.representation, artifacts)

    pca_manifest, pca_row_checksums = _pca_fit_manifest(prepared, pooled_obs, stores, config)
    pca_manifest.to_csv(artifacts / "pca_fit_manifest.csv", index=False)
    input_manifest = pd.DataFrame(
        [
            {
                "input_path": str(store.path),
                "pooled_row_start": store.start,
                "pooled_row_stop": store.stop,
                "n_cells": store.stop - store.start,
                "normalization_obs_checksum": store.obs_checksum,
                "source_values_sha256": store.cache_metadata["source_values_sha256"],
                "normalized_values_sha256": store.cache_metadata["normalized_values_sha256"],
                "normalization_cache_reused": store.cache_reused,
            }
            for store in stores
        ]
    )
    input_manifest.to_csv(artifacts / "input_store_manifest.csv", index=False)
    fit_mask_checksum = hashlib.sha256(np.packbits(fit_mask).tobytes()).hexdigest()
    fit_contract = {
        "ordered_input_paths": [str(path) for path in paths],
        "normalization_cache_checksums": [store.cache_metadata["normalized_values_sha256"] for store in stores],
        "fit_mask_checksum": fit_mask_checksum,
        "pca_fit_row_checksums": pca_row_checksums,
        "representation_config": config.representation.model_dump(mode="json"),
    }
    metadata = {
        "schema_version": 2,
        "representation": prepared.label,
        "source_embedding_key": config.embedding_key or "X",
        "intermediate_embedding_key": normalized_key,
        "condition_column": config.group_by,
        "condition_aliases": config.condition_aliases,
        "normalization": config.representation.model_dump(mode="json"),
        "normalization_settings_checksum": settings_checksum,
        "pooled_fit": True,
        "pca_fit_id": _json_checksum(fit_contract),
        "pca_algorithm": {
            "solver": "numpy.linalg.eigh",
            "covariance": "sample covariance (n - 1 denominator)",
            "centered": True,
            "whitened": False,
            "component_selection": "fewest components reaching cumulative explained variance target",
        },
        "pca_sampling": {
            "strata": [config.representation.experiment_key, "control_vs_perturbed"],
            "control_key": config.representation.control_key,
            "control_values": config.representation.control_values,
            "max_cells_per_dataset_class": config.representation.pca_max_cells_per_dataset_class,
            "random_seed": config.representation.random_seed,
            "marker_seed": "random_seed + uint32(sha256(marker)[:4], little-endian)",
            "without_replacement": True,
        },
        "fit_input_paths": [str(path) for path in paths],
        "fit_cells": int(fit_mask.sum()),
        "total_cells": int(len(pooled_obs)),
        "fit_mask_checksum": fit_mask_checksum,
        "pca_fit_row_checksums": pca_row_checksums,
        "padded_dimensions": int(prepared.features.shape[1]),
        "n_components_by_marker": prepared.n_components_by_marker,
        "pca_model_checksums": _pca_checksums(prepared),
        "artifact_dir": str(artifacts),
        "provenance_files": {
            "input_stores": "input_store_manifest.csv",
            "pca_fit_summary": "pca_fit_manifest.csv",
            "pca_fit_rows": "pca_fit_rows/<marker>.npz",
            "pca_models": "pca_models/<marker>.npz",
            "pca_scree": "marker_pca_scree.csv",
        },
    }

    rows: list[dict] = []
    for store in stores:
        local_markers = sorted(
            pooled_obs.iloc[store.start : store.stop][config.representation.marker_key].astype(str).unique()
        )
        local_component_counts = {prepared.n_components_by_marker[marker] for marker in local_markers}
        if len(local_component_counts) != 1:
            widths = {marker: prepared.n_components_by_marker[marker] for marker in local_markers}
            raise ValueError(
                f"{store.path}: one AnnData store contains markers with different "
                f"PCA80 widths: {widths}. Export one marker per store so obsm "
                "can retain the exact marker width."
            )
        local_dimensions = local_component_counts.pop()
        local_features = prepared.features[store.start : store.stop, :local_dimensions]
        local_metadata = {
            **metadata,
            "source_path": str(store.path),
            "markers": local_markers,
            "n_dimensions": local_dimensions,
            "normalized_values_sha256": store.cache_metadata["normalized_values_sha256"],
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
                "normalization_cache_reused": store.cache_reused,
                "replaced_obsm": obsm_key in store.existing_obsm,
                "replaced_uns": uns_key in store.existing_uns,
            }
        )

    manifest = pd.DataFrame(rows)
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
@click.option("--normalized-key", default=DEFAULT_NORMALIZED_KEY, show_default=True)
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("--reuse-normalized/--no-reuse-normalized", default=True, show_default=True)
def main(
    config: Path,
    obsm_key: str,
    uns_key: str | None,
    normalized_key: str,
    artifact_dir: Path | None,
    overwrite: bool,
    reuse_normalized: bool,
) -> None:
    """Run evaluation Step 0: reusable normalization followed by pooled PCA."""
    cfg = load_pooled_representation_config(config)
    manifest = export_pooled_representation(
        cfg,
        obsm_key=obsm_key,
        uns_key=uns_key,
        normalized_key=normalized_key,
        artifact_dir=artifact_dir,
        overwrite=overwrite,
        reuse_normalized=reuse_normalized,
    )
    click.echo(manifest.to_string(index=False))
    click.echo(
        f"Wrote/reused obsm[{normalized_key!r}], then wrote obsm[{obsm_key!r}] "
        f"and uns[{uns_key or obsm_key!r}] in {len(manifest)} stores."
    )


if __name__ == "__main__":
    main()
