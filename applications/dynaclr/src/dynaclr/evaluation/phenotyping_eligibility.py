"""Optional, channel-specific observation eligibility; never derives cutoffs.

Only explicit boolean False excludes an observation. Callers authorize their
supplied eligibility table; review of any upstream threshold remains upstream.
Run this file (or the module in the full environment) with --input, --eligibility
and optionally --output. Without --output only obs is read and JSON is printed.

Filtered stores retain X and raw obsm (X_backbone/X_projections by default).
Other obsm, obsp and uns are removed because fitted representations and graphs
must be recomputed. Original uns is archived as JSON in the audit sidecar.
X must contain raw embeddings; normalized-only X requires re-embedding.
"""

import json
from contextlib import closing
from hashlib import sha256
from numbers import Integral
from pathlib import Path

import click
import numpy as np
import pandas as pd

KEY_COLUMNS = ("dataset", "fov_name", "track_id", "t", "channel_name")
ELIGIBILITY_COLUMNS = (*KEY_COLUMNS, "eligible", "reason")
RAW_OBSM_KEYS = ("X_backbone", "X_projections")


def _validate_keys(frame: pd.DataFrame, label: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{label}: duplicate columns")
    missing = set(KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing key columns {sorted(missing)}")
    for column in KEY_COLUMNS:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{label}: null key in {column}")
        if column in {"track_id", "t"}:
            valid = values.map(lambda value: isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)))
        else:
            valid = values.map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not np.asarray(valid, dtype=bool).all():
            raise ValueError(f"{label}: invalid key type or empty value in {column}")
    duplicated = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, list(KEY_COLUMNS)].iloc[0].to_dict()
        raise ValueError(f"{label}: duplicate observation keys: {example}")


def validate_eligibility(eligibility: pd.DataFrame) -> None:
    """Reject missing fields, ambiguous keys, and non-boolean decisions."""
    _validate_keys(eligibility, "eligibility")
    missing = set(ELIGIBILITY_COLUMNS) - set(eligibility.columns)
    if missing:
        raise ValueError(f"eligibility: missing columns {sorted(missing)}")
    decisions = eligibility["eligible"].dropna()
    if not np.asarray(decisions.map(lambda value: isinstance(value, (bool, np.bool_))), dtype=bool).all():
        raise ValueError("eligibility: eligible must contain only booleans or null")
    if not np.asarray(eligibility["reason"].dropna().map(lambda value: isinstance(value, str)), dtype=bool).all():
        raise ValueError("eligibility: reason must contain only strings or null")


def eligibility_records_sha256(eligibility: pd.DataFrame) -> str:
    """Hash canonical schema records, independent of row order and pandas dtypes."""
    validate_eligibility(eligibility)
    records = []
    for row in eligibility.loc[:, list(ELIGIBILITY_COLUMNS)].itertuples(index=False, name=None):
        dataset, fov, track, time, channel, eligible, reason = row
        records.append(
            [
                dataset,
                fov,
                int(track),
                int(time),
                channel,
                None if pd.isna(eligible) else bool(eligible),
                None if pd.isna(reason) else reason,
            ]
        )
    records.sort(key=lambda row: tuple(row[:5]))
    payload = json.dumps(
        {"schema": list(ELIGIBILITY_COLUMNS), "records": records}, ensure_ascii=False, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def load_eligibility(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read parquet and hash its authoritative bytes as well as canonical records."""
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        eligibility = pd.read_parquet(stream)
    return eligibility, {
        "eligibility_path": str(path.resolve()),
        "eligibility_file_sha256": digest.hexdigest(),
        "eligibility_records_sha256": eligibility_records_sha256(eligibility),
    }


def resolve_eligibility_path(value, predict_flags: dict | None = None) -> Path | None:
    """Resolve the optional defaults/model eligibility field before job launch."""
    if "eligibility" in (predict_flags or {}):
        raise ValueError("Set eligibility in defaults or the model entry; predict_flags.eligibility is unsupported")
    if value is None or value == "":
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError("eligibility must be a parquet path or null")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Eligibility parquet not found: {path}")
    return path


def eligibility_cache_matches(output_path: str | Path, expected: dict | None) -> bool:
    """Inspect only eligibility provenance, failing closed on missing/invalid hashes.

    An unfiltered request matches only a store without an eligibility audit.
    Filtered requests require both authoritative file SHA and canonical record
    SHA, plus the explicit-False policy. Never load embeddings or change stores.
    """
    output_path = Path(output_path)
    if not (output_path / "uns" / "phenotyping_eligibility").exists():
        return expected is None
    if expected is None:
        return False

    import zarr
    from anndata.io import read_elem

    try:
        with closing(zarr.storage.LocalStore(str(output_path), read_only=True)) as store:
            group = zarr.open_group(store=store, mode="r", use_consolidated=False)
            audit = read_elem(group["uns"]["phenotyping_eligibility"])
        if isinstance(audit, str):
            audit = json.loads(audit)
    except (OSError, KeyError, ValueError, TypeError):
        return False
    return (
        isinstance(audit, dict)
        and audit.get("policy") == "exclude_explicit_false_only"
        and all(audit.get(key) == expected[key] for key in ("eligibility_file_sha256", "eligibility_records_sha256"))
    )


def require_eligibility_output_identity(output_path: str | Path, expected: dict | None) -> None:
    """Protect an existing store from a different eligibility identity, even on overwrite."""
    if Path(output_path).exists() and not eligibility_cache_matches(output_path, expected):
        raise ValueError(
            f"Eligibility identity conflicts with existing output: {output_path}. "
            "Choose a distinct run/output identity and an explicit checkpoint path to the original trained model. "
            "Existing raw and normalized embeddings were not changed."
        )


def filter_observations(
    obs: pd.DataFrame,
    eligibility: pd.DataFrame,
    dataset: str | None = None,
    channel_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Keep original rows/order/index except explicit False on the full key.

    Supply dataset/channel_name for per-reporter tables lacking these columns.
    If a supplied context disagrees with an existing column, raise. No channel
    is inferred from marker labels or from the eligibility table. Keys are exact:
    no FOV path cleanup or string-to-integer coercion is performed. Both inputs
    must have unique full keys, though the obs index may be duplicated.
    """
    validate_eligibility(eligibility)
    keys = obs.copy()
    for column, value in (("dataset", dataset), ("channel_name", channel_name)):
        if value is not None:
            if column in keys and not keys[column].eq(value).fillna(False).all():
                raise ValueError(f"obs: {column} conflicts with supplied context {value!r}")
            keys[column] = value
    _validate_keys(keys, "obs")
    obs_keys = pd.MultiIndex.from_frame(keys.loc[:, list(KEY_COLUMNS)])
    eligibility_keys = pd.MultiIndex.from_frame(eligibility.loc[:, list(KEY_COLUMNS)])
    locations = eligibility_keys.get_indexer(obs_keys)
    matched = locations >= 0
    decisions = pd.array([pd.NA] * len(obs), dtype="boolean")
    decisions[matched] = pd.array(eligibility["eligible"], dtype="boolean")[locations[matched]]
    excluded = ~np.asarray(decisions.fillna(True), dtype=bool)
    audit = {
        "policy": "exclude_explicit_false_only",
        "n_input": len(obs),
        "n_retained": int((~excluded).sum()),
        "n_excluded": int(excluded.sum()),
        "n_explicit_true": int(np.asarray(decisions.fillna(False)).sum()),
        "n_unknown": int((matched & pd.isna(decisions)).sum()),
        "n_missing": int((~matched).sum()),
        "n_matched": int(matched.sum()),
        "n_eligibility_records": len(eligibility),
        "n_unused_eligibility_records": int(len(eligibility) - matched.sum()),
        "excluded_reason_counts": eligibility.iloc[locations[excluded]]["reason"]
        .astype(object)
        .fillna("<unspecified>")
        .value_counts()
        .to_dict(),
        "eligibility_records_sha256": eligibility_records_sha256(eligibility),
    }
    return obs.iloc[np.flatnonzero(~excluded)].copy(), audit


class EligibilityTripletMixin:
    """Prediction-only mixin for TripletDataModule, avoiding a shared data change.

    Compose before TripletDataModule in the MRO. Alignment reads tracking CSVs
    and image metadata; filtering finishes before TripletDataset reads patches.
    The audit dictionary is updated in place. Optional writer metadata receives
    JSON so free-text reasons (including slashes) are safe AnnData metadata.
    """

    def __init__(
        self,
        *args,
        eligibility_table,
        eligibility_dataset,
        eligibility_channel,
        eligibility_audit,
        eligibility_metadata=None,
        **kwargs,
    ):
        self.eligibility_table = eligibility_table
        self.eligibility_dataset = eligibility_dataset
        self.eligibility_channel = eligibility_channel
        self.eligibility_audit = eligibility_audit
        self.eligibility_metadata = eligibility_metadata
        super().__init__(*args, **kwargs)

    def _align_tracks_tables_with_positions(self):
        if getattr(self, "_eligibility_aligned_tables", None) is not None:
            return self._eligibility_aligned_tables
        positions, tables = super()._align_tracks_tables_with_positions()
        observations = [
            table.assign(fov_name=position.zgroup.name.strip("/"))
            for position, table in zip(positions, tables, strict=True)
        ]
        if not observations:
            raise ValueError("Eligibility prediction found no tracking tables")
        obs = pd.concat(observations, ignore_index=True)
        filtered, audit = filter_observations(
            obs, self.eligibility_table, self.eligibility_dataset, self.eligibility_channel
        )
        self.eligibility_audit.update(audit)
        if self.eligibility_metadata is not None:
            self.eligibility_metadata["phenotyping_eligibility"] = json.dumps(self.eligibility_audit, sort_keys=True)
        if filtered.empty:
            raise ValueError("Eligibility excluded all prediction observations; no image patches were read")
        keep = np.zeros(len(obs), dtype=bool)
        keep[filtered.index.to_numpy()] = True
        output_positions, output, offset = [], [], 0
        for position, table in zip(positions, tables, strict=True):
            selected = table.iloc[np.flatnonzero(keep[offset : offset + len(table)])].copy()
            if not selected.empty:
                output_positions.append(position)
                output.append(selected)
            offset += len(table)
        self._eligibility_aligned_tables = output_positions, output
        return self._eligibility_aligned_tables


def read_embedding_obs(path: str | Path) -> pd.DataFrame:
    """Read only obs from a backed H5AD or an AnnData Zarr store."""
    # AnnData is optional for the DataFrame API and pre-embedding mixin.
    import anndata as ad
    import zarr
    from anndata.io import read_elem

    path = Path(path)
    if path.suffix == ".h5ad":
        backed = ad.read_h5ad(path, backed="r")
        try:
            return backed.obs.copy()
        finally:
            backed.file.close()
    with closing(zarr.storage.LocalStore(str(path), read_only=True)) as store:
        return read_elem(zarr.open_group(store=store, mode="r")["obs"])


def filter_embedding_store(
    input_path: str | Path,
    output_path: str | Path,
    eligibility_path: str | Path,
    *,
    dataset: str | None = None,
    channel_name: str | None = None,
    raw_obsm_keys: tuple[str, ...] = RAW_OBSM_KEYS,
) -> dict:
    """Write a new filtered raw store and a JSON audit; refuse existing outputs.

    X, layers, var and raw are subset by observation. Only the named raw obsm
    survive. All obsp and uns are cleared; provenance is replaced with the
    eligibility audit, and original uns is archived in the sidecar. Refit all
    control normalization, PCA, UMAP and other derived representations.
    """
    import anndata as ad

    input_path, output_path = Path(input_path), Path(output_path)
    audit_path = Path(str(output_path) + ".eligibility.json")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError(f"Output or audit already exists: {output_path}")
    eligibility, provenance = load_eligibility(eligibility_path)
    adata = ad.read_h5ad(input_path) if input_path.suffix == ".h5ad" else ad.read_zarr(input_path)
    # A positional index prevents ambiguous selection when obs_names repeat.
    obs = adata.obs.copy()
    obs.index = pd.RangeIndex(len(obs))
    filtered, audit = filter_observations(obs, eligibility, dataset, channel_name)
    audit.update(provenance)
    audit["input_path"] = str(input_path.resolve())
    audit["requires_representation_refit"] = True
    result = adata[filtered.index.to_numpy()].copy()
    audit["removed_obsm_keys"] = sorted(set(result.obsm) - set(raw_obsm_keys))
    audit["removed_uns_keys"] = sorted(result.uns)
    original_uns = dict(result.uns)
    for key in audit["removed_obsm_keys"]:
        del result.obsm[key]
    result.obsp.clear()
    result.uns.clear()
    # Store JSON to avoid AnnData's restrictions on nullable audit values/lists.
    result.uns["phenotyping_eligibility"] = json.dumps(audit, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".h5ad":
        result.write_h5ad(output_path)
    else:
        result.write_zarr(output_path)
    with audit_path.open("x") as stream:
        json.dump({**audit, "source_uns": original_uns}, stream, indent=2, default=_json_value)
        stream.write("\n")
    return audit


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--eligibility", "eligibility_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="New raw store; omitted = obs-only audit.",
)
@click.option("--dataset", default=None, help="Exact acquisition dataset name if absent from obs.")
@click.option("--channel-name", default=None, help="Exact image channel name if absent from obs.")
@click.option(
    "--raw-obsm-key", multiple=True, help="Additional raw obsm key to retain alongside X_backbone/X_projections."
)
def main(input_path, eligibility_path, output_path, dataset, channel_name, raw_obsm_key):
    """Audit eligibility or create a filtered raw embedding store on compute."""
    if output_path is None:
        eligibility, provenance = load_eligibility(eligibility_path)
        _, audit = filter_observations(read_embedding_obs(input_path), eligibility, dataset, channel_name)
        audit.update(provenance)
    else:
        audit = filter_embedding_store(
            input_path,
            output_path,
            eligibility_path,
            dataset=dataset,
            channel_name=channel_name,
            raw_obsm_keys=(*RAW_OBSM_KEYS, *raw_obsm_key),
        )
    click.echo(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
