"""Artifact cache for the dynacell evaluation pipeline.

Stores organelle masks and feature embeddings under an explicit cache
directory so successive eval runs against the same source dataset skip the
expensive segmentation and feature-extraction work.

Cache identity is rooted in the source plate/channel plus
``cell_segmentation_path`` when cell-level features are involved.
Per-artifact invalidation is driven by extra params recorded in the manifest
(e.g. spacing, patch_size, checkpoint hash). On the prediction side each cached
position also records the :func:`prediction_sources` entry it was built from.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
import zarr
from iohub.ngff import ImageArray, Position, open_ome_zarr

from viscy_utils.prediction_metadata import PREDICTION_COMPLETE_KEY, marker_identity

FeatureKind = Literal["cp", "dinov3", "dynaclr", "celldino", "morphem"]

# The manifest is read and written once per FOV, and its per-position ``sources``
# make it large; libyaml's C loader and dumper are an order of magnitude faster than
# the pure-Python ones. They are absent only from a PyYAML built without libyaml,
# which then falls back to the pure-Python classes.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

#: YAML 1.1 bool spellings that ``OmegaConf.save`` quotes.
_OMEGACONF_BOOLS = frozenset(
    "y Y yes Yes YES n N no No NO true True TRUE false False FALSE on On ON off Off OFF".split()
)


class _ManifestDumper(getattr(yaml, "CSafeDumper", yaml.SafeDumper)):
    """libyaml dumper that writes what ``OmegaConf.save`` wrote: no anchors, str quoted like OmegaConf.

    OmegaConf's loader reads an unquoted ``4e1234567890`` (a possible hex sha12 marker)
    as a float, so a str that parses as a bool, int or float is single-quoted, exactly
    as OmegaConf's own representer does. Entries sharing one object (``ctx.spacing``)
    are written out in full rather than as ``&id001`` aliases.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    try:
        float(data)
        quoted = True
    except ValueError:
        quoted = data in _OMEGACONF_BOOLS
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'" if quoted else None)


_ManifestDumper.add_representer(str, _represent_str)

CACHE_SCHEMA_VERSION = 1

_MASK_CHANNEL = "target_seg"


class StaleCacheError(RuntimeError):
    """Raised when cache identity or artifact params disagree with the current config."""


@dataclass(frozen=True)
class CachePaths:
    """Filesystem layout for one artifact cache directory."""

    root: Path
    manifest: Path
    masks_dir: Path
    features_dir: Path
    instance_masks_dir: Path

    def mask_plate(self, target_name: str, backend: str = "supermodel") -> Path:
        """Return the HCS OME-Zarr plate for masks of *target_name*.

        Non-default segmentation backends get a ``__{backend}`` filename infix so
        masks from different segmenters never collide in one cache dir; the
        default ``supermodel`` keeps the bare ``{target_name}.zarr`` path so
        pre-existing caches are unaffected.
        """
        stem = target_name if backend == "supermodel" else f"{target_name}__{backend}"
        return self.masks_dir / f"{stem}.zarr"

    def instance_mask_plate(self, target_name: str, backend: str) -> Path:
        """Return the HCS OME-Zarr plate for instance (uint16) labels.

        Instance-label caches always carry a ``__{backend}`` infix
        (``nucleus__cellpose.zarr`` / ``membrane__cellpose_watershed.zarr``) and
        live under a separate directory from the binary ``organelle_masks`` so
        the two never collide.
        """
        return self.instance_masks_dir / f"{target_name}__{backend}.zarr"

    def cp_features(self) -> Path:
        """Return the zarr group path for CP regionprops features."""
        return self.features_dir / "cp.zarr"

    def dinov3_features(self, model_name: str) -> Path:
        """Return the zarr group path for DINOv3 features of *model_name*."""
        return self.features_dir / "dinov3" / f"{feature_slug(model_name)}.zarr"

    def dynaclr_features(self, ckpt_sha12: str) -> Path:
        """Return the zarr group path for DynaCLR features keyed by *ckpt_sha12*."""
        return self.features_dir / "dynaclr" / f"{ckpt_sha12}.zarr"

    def celldino_features(self, weights_sha12: str) -> Path:
        """Return the zarr group path for CELL-DINO features keyed by *weights_sha12*."""
        return self.features_dir / "celldino" / f"{weights_sha12}.zarr"

    def morphem_features(self, model_name: str) -> Path:
        """Return the zarr group path for MorphEm features of *model_name*."""
        return self.features_dir / "morphem" / f"{feature_slug(model_name)}.zarr"


def cache_paths(cache_dir: Path | str) -> CachePaths:
    """Build a CachePaths rooted at *cache_dir* (does not create directories)."""
    root = Path(cache_dir)
    return CachePaths(
        root=root,
        manifest=root / "manifest.yaml",
        masks_dir=root / "organelle_masks",
        features_dir=root / "features",
        instance_masks_dir=root / "instance_masks",
    )


def load_manifest(paths: CachePaths) -> dict[str, Any]:
    """Load the manifest YAML, or return an empty skeleton if the file is absent."""
    if not paths.manifest.exists():
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "gt": None,
            "pred": None,
            "cell_segmentation": None,
            "artifacts": {},
        }
    with open(paths.manifest) as f:
        raw = yaml.load(f, Loader=_YAML_LOADER)
    if not isinstance(raw, dict):
        raise StaleCacheError(f"Manifest at {paths.manifest} is not a mapping")
    raw.setdefault("gt", None)
    raw.setdefault("pred", None)
    raw.setdefault("cell_segmentation", None)
    raw.setdefault("artifacts", {})
    return raw


def save_manifest(paths: CachePaths, manifest: dict[str, Any]) -> None:
    """Persist *manifest* as YAML under *paths.manifest*, creating parents."""
    paths.root.mkdir(parents=True, exist_ok=True)
    with open(paths.manifest, "w") as f:
        yaml.dump(manifest, f, Dumper=_ManifestDumper, sort_keys=False, allow_unicode=True)


def check_cache_identity(
    manifest: dict[str, Any],
    *,
    source: Literal["gt", "pred"] | None = None,
    plate_path: str | None = None,
    channel_name: str | None = None,
    cell_segmentation_path: str | None = None,
) -> None:
    """Raise if the manifest's cache identity disagrees with the current config.

    Parameters
    ----------
    manifest
        Loaded manifest dict (may be the empty skeleton from :func:`load_manifest`).
    source
        Which side to check (``"gt"`` or ``"pred"``); ``None`` skips per-side
        identity and only validates ``cell_segmentation_path``.
    plate_path
        Current ``io.gt_path`` (when ``source="gt"``) or ``io.pred_path``
        (when ``source="pred"``).
    channel_name
        Current ``io.gt_channel_name`` or ``io.pred_channel_name``, matching
        *source*.
    cell_segmentation_path
        Current ``io.cell_segmentation_path``. ``None`` skips the check.
    """
    version = manifest.get("cache_schema_version")
    if version is not None and version != CACHE_SCHEMA_VERSION:
        raise StaleCacheError(
            f"Cache schema version mismatch: manifest has {version}, current is {CACHE_SCHEMA_VERSION}. "
            "Delete the cache directory or bump cache_schema_version."
        )
    if source is not None:
        entry = manifest.get(source)
        if entry is not None and plate_path is not None and entry.get("plate_path") != plate_path:
            raise StaleCacheError(
                f"{source}.plate_path mismatch: manifest={entry.get('plate_path')!r}, config={plate_path!r}"
            )
        if entry is not None and channel_name is not None and entry.get("channel_name") != channel_name:
            raise StaleCacheError(
                f"{source}.channel_name mismatch: manifest={entry.get('channel_name')!r}, config={channel_name!r}"
            )
    seg_entry = manifest.get("cell_segmentation")
    if seg_entry is not None and cell_segmentation_path is not None:
        if seg_entry.get("plate_path") != cell_segmentation_path:
            raise StaleCacheError(
                f"cell_segmentation.plate_path mismatch: manifest={seg_entry.get('plate_path')!r}, "
                f"config={cell_segmentation_path!r}"
            )


def seed_cache_identity(
    manifest: dict[str, Any],
    *,
    source: Literal["gt", "pred"] | None = None,
    plate_path: str | None = None,
    channel_name: str | None = None,
    cell_segmentation_path: str | None = None,
) -> None:
    """Populate source identity manifest entries if absent.

    Called before the first artifact is written. Safe to call repeatedly;
    later calls with conflicting values should be preceded by
    :func:`check_cache_identity`. Pass *source* once per side — seeding both
    sides in a single call is no longer supported.
    """
    manifest["cache_schema_version"] = CACHE_SCHEMA_VERSION
    if source is not None:
        if (plate_path is None) != (channel_name is None):
            raise ValueError(f"plate_path and channel_name must be provided together for source={source!r}")
        if plate_path is not None and channel_name is not None and manifest.get(source) is None:
            manifest[source] = {"plate_path": plate_path, "channel_name": channel_name}
    if cell_segmentation_path is not None and manifest.get("cell_segmentation") is None:
        manifest["cell_segmentation"] = {"plate_path": cell_segmentation_path}


def diff_artifact_params(
    entry: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    numeric_keys: tuple[str, ...] = (),
) -> list[tuple[str, Any, Any]]:
    """Return per-key mismatches between a manifest entry and current params.

    Parameters
    ----------
    entry
        Manifest entry for the artifact, or ``None`` if no entry exists yet
        (returns an empty list — the caller decides whether to treat
        absence as miss).
    current
        Current-config values keyed by the same names as in *entry*.
    numeric_keys
        Keys in *current* whose values should be compared with
        :func:`numpy.allclose` instead of ``==``.

    Returns
    -------
    list of tuple
        ``(key, cached_value, current_value)`` for every disagreement;
        empty when *entry* is ``None`` or every key matches.
    """
    if entry is None:
        return []
    if not isinstance(entry, dict):
        # A malformed manifest entry (string/list/scalar where a mapping is
        # expected — hand-edit or partial-write corruption) must surface as
        # mismatches so the caller can soft-invalidate, not as an
        # AttributeError escaping through `entry.get(...)`.
        return [(key, entry, value) for key, value in current.items()]
    mismatches: list[tuple[str, Any, Any]] = []
    for key, value in current.items():
        cached_value = entry.get(key)
        if key in numeric_keys:
            # A malformed cached value (None, wrong dtype, wrong length) must
            # surface as a mismatch so the caller can soft-invalidate, not as
            # a TypeError/ValueError that escapes through diff_artifact_params.
            try:
                close = cached_value is not None and np.allclose(
                    np.asarray(cached_value, dtype=float),
                    np.asarray(value, dtype=float),
                    rtol=1e-9,
                    atol=0.0,
                )
            except (TypeError, ValueError):
                close = False
            if not close:
                mismatches.append((key, cached_value, value))
        elif cached_value != value:
            mismatches.append((key, cached_value, value))
    return mismatches


def built_at_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format (for manifest entries)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_sha256_12(obj: Any) -> str:
    """Return the first 12 hex chars of the sha256 of *obj* serialized as JSON.

    Keys are sorted, so representation-equivalent mappings hash alike; values JSON
    cannot encode are serialized with ``str``.
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _chunk_files(directory: Path) -> Iterator[os.DirEntry]:
    """Yield the chunk files under *directory*, depth first and lazily.

    Array and group metadata (``zarr.json``, ``.zarray``, ``.zattrs``) is skipped. A
    caller taking ``next(...)`` reads one directory batch per level instead of listing
    every chunk.
    """
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                yield from _chunk_files(Path(entry.path))
            elif entry.name != "zarr.json" and not entry.name.startswith("."):
                yield entry


def _stored_chunk(array_dir: Path, key: str, spatial: int) -> os.DirEntry | Path | None:
    """Return the chunk stored at *key*, else the first stored chunk sharing its ``(t, c)`` prefix.

    A chunk never written, or reset to the fill value (zarr deletes it), is absent. The
    prefix fallback exists only for ``/``-separated keys, whose ``(t, c)`` chunks share
    one directory subtree, read one directory batch per level. A ``.``-separated key (a
    flat zarr v2 layout) resolves only exactly; when that chunk is absent the position
    reads as unwritten (``written_ns=None``), which rebuilds a legacy entry over it once.
    Every evaluated non-v3 store uses ``/`` separators.
    """
    chunk = array_dir / key
    if chunk.is_file():
        return chunk
    parts = key.split("/")
    if len(parts) <= spatial:
        return None
    prefix = array_dir.joinpath(*parts[:-spatial])
    return next(_chunk_files(prefix), None) if prefix.is_dir() else None


def _first_channel_chunk_mtime_ns(plate_path: Path, array: ImageArray, channel_index: int) -> int | None:
    """Return the ``st_mtime_ns`` of the first stored chunk holding *channel_index*, earliest ``t`` first.

    Each candidate ``(t, c, 0, 0, 0)`` is located through the array's own chunk-key
    encoding (zarr v2 or v3, sharded or not), so no layout is parsed by hand, and
    :func:`_stored_chunk` falls back to the first stored chunk of that ``(t, c)``.
    Later timepoints are tried because a blank output (all fill values, e.g. an
    all-zero prediction) stores no chunk at all: at most one lookup per timepoint.

    Returns
    -------
    int or None
        ``None`` when no chunk of the channel is stored at any timepoint.
    """
    outer = array.shards if array.shards is not None else array.chunks
    spatial = array.ndim - 2
    channel_chunk = channel_index // outer[1]
    array_dir = plate_path / array.path
    for t_chunk in range(-(-array.shape[0] // outer[0])):
        key = array.native.metadata.encode_chunk_key((t_chunk, channel_chunk) + (0,) * spatial)
        found = _stored_chunk(array_dir, key, spatial)
        if found is not None:
            return found.stat().st_mtime_ns
    return None


def _position_source(plate_path: Path, position: Position, channel_name: str, archive_ns: int | None) -> dict[str, Any]:
    """Return one position's source; see :func:`prediction_sources`."""
    marker = position.zattrs.get(PREDICTION_COMPLETE_KEY, {}).get(channel_name)
    digest = None if marker is None else json_sha256_12(marker_identity(marker))
    if archive_ns is not None:
        written_ns = archive_ns
    else:
        array = position[position.metadata.multiscales[0].datasets[0].path]
        written_ns = _first_channel_chunk_mtime_ns(plate_path, array, position.get_channel_index(channel_name))
    return {"marker": digest, "written_ns": written_ns}


def _archive_mtime_ns(path: Path) -> int | None:
    """Return a packed ``.ozx`` archive's own mtime, which stands in for its chunks; ``None`` for a directory store."""
    return path.stat().st_mtime_ns if path.is_file() else None


def prediction_sources(
    plate_path: Path | str, channel_name: str, positions: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Return the source identity of every position (or of *positions*) of one prediction channel.

    A position's source is what a cached artifact built from it must still match:

    ``marker``
        sha256_12 of the channel's predict-writer marker
        (:data:`~viscy_utils.prediction_metadata.PREDICTION_COMPLETE_KEY`) with its
        provenance-only fields stripped
        (:func:`~viscy_utils.prediction_metadata.marker_identity`), or ``None`` when the
        position was written before markers existed. It changes when a re-predict
        uses another checkpoint or settings.
    ``written_ns``
        ``st_mtime_ns`` of the channel's first stored chunk, earliest timepoint first,
        or ``None`` when no chunk of the channel is stored (an unwritten or entirely
        blank output). It changes on every re-predict of the channel, including one
        that leaves the marker alone: a code-only fix, an input outside the settings
        hash, or a writer from before the markers.

    Metadata-file mtimes are deliberately not part of it: another channel's markers,
    or a focus estimate written into the store, rewrite the position's attributes
    without touching this channel's voxels. A packed ``.ozx`` archive stores no
    per-chunk files, so its own mtime stands in for ``written_ns``.

    Costs one attribute read, one array-metadata read and a ``stat`` per position (one
    per timepoint while the channel's earlier chunks are blank); the store is opened
    read-only, and the eval never writes to it.

    Parameters
    ----------
    plate_path : Path or str
        HCS prediction store; a missing store raises ``FileNotFoundError``.
    channel_name : str
        Prediction channel the eval scores.
    positions : iterable of str, optional
        Read only these positions, e.g. one a running predict finished after the
        store-wide snapshot; each must exist (``KeyError`` otherwise). ``None`` reads
        every position.

    Returns
    -------
    dict
        ``{position_name: {"marker": str | None, "written_ns": int | None}}``.
    """
    path = Path(plate_path)
    archive_ns = _archive_mtime_ns(path)
    with open_ome_zarr(path, mode="r") as plate:
        items = plate.positions() if positions is None else ((name, plate[name]) for name in positions)
        return {name: _position_source(path, position, channel_name, archive_ns) for name, position in items}


def source_predates(source: dict[str, Any], horizon_ns: int) -> bool:
    """Return whether a :func:`prediction_sources` entry was written no later than *horizon_ns*.

    Dates caches recorded before sources existed, which carry only a time: a manifest
    entry's ``built_at``, or a metrics sidecar's mtime. A position whose first stored
    chunk (``written_ns``) is no newer than that time is verified; one with no stored
    chunk cannot be dated and never is.

    Parameters
    ----------
    source : dict
        One position's ``{"marker", "written_ns"}``.
    horizon_ns : int
        The legacy record's time, in ns since the epoch.

    Returns
    -------
    bool
        ``written_ns is not None and written_ns <= horizon_ns``.
    """
    return source["written_ns"] is not None and source["written_ns"] <= horizon_ns


def prediction_sources_sha256_12(sources: dict[str, dict[str, Any]]) -> str:
    """Return the sha256_12 over the name-sorted ``(position, source)`` pairs of :func:`prediction_sources`.

    Parameters
    ----------
    sources : dict
        Output of :func:`prediction_sources`.

    Returns
    -------
    str
        First 12 hex characters of the digest.
    """
    return json_sha256_12(sorted(sources.items()))


def _read_position_channel0(plate_path: Path, pos_name: str, dtype: npt.DTypeLike) -> np.ndarray | None:
    """Read channel 0 of one position as ``(T, D, H, W)`` cast to *dtype*.

    Returns ``None`` when the plate file or the position is absent (a cache miss).
    """
    if not plate_path.exists():
        return None
    with open_ome_zarr(plate_path, mode="r") as plate:
        try:
            position = plate[pos_name]
        except KeyError:
            return None
        # copy=False: the read already materialized a fresh array and the on-disk
        # dtype normally matches, so the cast is a no-op the caller shouldn't pay
        # a second full-array copy for.
        return np.asarray(position.data[:, 0]).astype(dtype, copy=False)


def read_mask(paths: CachePaths, target_name: str, pos_name: str, backend: str = "supermodel") -> np.ndarray | None:
    """Read cached organelle masks for a single position.

    Returns
    -------
    numpy.ndarray | None
        Bool array of shape ``(T, D, H, W)``, or ``None`` if the plate or
        position is absent.
    """
    return _read_position_channel0(paths.mask_plate(target_name, backend), pos_name, bool)


def _is_position_malformed(plate_path: Path, pos_name: str) -> bool:
    """Detect the partial-write signature on disk.

    A crashed prior eval can leave a position whose group dir exists
    (``pos/zarr.json`` present, listed in the well's NGFF ``images``) but
    whose inner data array is missing its metadata (``pos/0/zarr.json``
    absent). iohub's ``Plate[pos_name]`` raises ``KeyError`` for this state
    and zarr v3 then poisons the parent well's lookup in the same session.
    """
    pos_dir = plate_path / pos_name
    return pos_dir.exists() and not (pos_dir / "0" / "zarr.json").exists()


def _rewrite_inner_array(pos_dir: Path, data: np.ndarray) -> None:
    """Replace just the ``0`` inner array of a position, leaving NGFF metadata alone.

    iohub can't fix the malformed state from inside the plate API because
    cleaning the position dir empties the well, and zarr v3 marks an empty
    group falsy — which iohub interprets as a missing well. The position
    group's own multiscales metadata is intact, so the cheapest recovery
    is to rewrite the inner array directly via zarr.
    """
    shutil.rmtree(pos_dir / "0", ignore_errors=True)
    pos_group = zarr.open_group(str(pos_dir), mode="r+")
    pos_group.create_array("0", data=data)


def _write_position_channel0(
    plate_path: Path, pos_name: str, arr: np.ndarray, dtype: npt.DTypeLike, channel_name: str
) -> None:
    """Write ``arr`` ``(T, D, H, W)`` as channel 0 of one position, creating the plate if needed.

    Casts to *dtype* only after the rank check, so the common mistake here — passing
    a 5-D ``(T, C, D, H, W)`` array — raises without first copying it.

    Repairs the partial-write signature in place (see :func:`_is_position_malformed`)
    rather than through the plate API, which cannot recover from that state.
    """
    if arr.ndim != 4:
        raise ValueError(f"array must be 4-D (T, D, H, W); got shape {arr.shape}")
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    data = arr.astype(dtype, copy=False)[:, None]  # (T, 1, D, H, W); write-only, so a view is fine
    if plate_path.exists() and _is_position_malformed(plate_path, pos_name):
        _rewrite_inner_array(plate_path / pos_name, data)
        return
    mode = "r+" if plate_path.exists() else "w"
    with open_ome_zarr(
        plate_path,
        mode=mode,
        layout="hcs",
        channel_names=[channel_name],
        version="0.5",
    ) as plate:
        row, col, fov = pos_name.split("/")
        try:
            position = plate[pos_name]
        except KeyError:
            position = plate.create_position(row, col, fov)
        try:
            del position["0"]
        except KeyError:
            pass
        position.create_image("0", data)


def write_mask(
    paths: CachePaths,
    target_name: str,
    pos_name: str,
    masks: np.ndarray,
    *,
    channel_name: str = _MASK_CHANNEL,
    backend: str = "supermodel",
) -> None:
    """Append masks for a single position to the ``{target_name}.zarr`` plate.

    Parameters
    ----------
    paths
        Cache paths.
    target_name
        Organelle name (used as the mask plate's filename stem).
    pos_name
        HCS position name in ``row/col/fov`` form.
    masks
        Bool array of shape ``(T, D, H, W)`` — one channel per timepoint.
    channel_name
        OME-Zarr channel label to write for this mask plate.
    backend
        Segmentation backend (selects the plate filename infix).
    """
    _write_position_channel0(paths.mask_plate(target_name, backend), pos_name, masks, bool, channel_name)


_INSTANCE_MASK_CHANNEL = "instance_seg"


def read_instance_mask(paths: CachePaths, target_name: str, pos_name: str, backend: str) -> np.ndarray | None:
    """Read cached instance labels for a single position.

    Returns
    -------
    numpy.ndarray | None
        uint16 array of shape ``(T, D, H, W)`` (2-D runs are stored with
        ``D=1``), or ``None`` if the plate or position is absent.
    """
    return _read_position_channel0(paths.instance_mask_plate(target_name, backend), pos_name, np.uint16)


def write_instance_mask(
    paths: CachePaths,
    target_name: str,
    pos_name: str,
    labels: np.ndarray,
    *,
    channel_name: str = _INSTANCE_MASK_CHANNEL,
    backend: str,
) -> None:
    """Append uint16 instance labels for a single position to the instance plate.

    Mirrors :func:`write_mask` but preserves integer labels (no bool coercion).

    Parameters
    ----------
    paths
        Cache paths.
    target_name
        Organelle name (mask plate filename stem, with the ``__{backend}`` infix).
    pos_name
        HCS position name in ``row/col/fov`` form.
    labels
        uint16 array of shape ``(T, D, H, W)`` (2-D runs use ``D=1``).
    channel_name
        OME-Zarr channel label to write for this instance plate.
    backend
        Segmentation backend (selects the plate filename infix).
    """
    _write_position_channel0(paths.instance_mask_plate(target_name, backend), pos_name, labels, np.uint16, channel_name)


def _features_group_path(
    paths: CachePaths,
    kind: FeatureKind,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
    weights_sha12: str | None = None,
) -> Path:
    """Resolve the zarr group path for a feature cache entry."""
    if kind == "cp":
        return paths.cp_features()
    if kind == "dinov3":
        if model_name is None:
            raise ValueError("model_name is required for kind='dinov3'")
        return paths.dinov3_features(model_name)
    if kind == "dynaclr":
        if ckpt_sha12 is None:
            raise ValueError("ckpt_sha12 is required for kind='dynaclr'")
        return paths.dynaclr_features(ckpt_sha12)
    if kind == "celldino":
        if weights_sha12 is None:
            raise ValueError("weights_sha12 is required for kind='celldino'")
        return paths.celldino_features(weights_sha12)
    if kind == "morphem":
        if model_name is None:
            raise ValueError("model_name is required for kind='morphem'")
        return paths.morphem_features(model_name)
    raise ValueError(f"Unknown feature kind: {kind!r}")


# Group-level attribute recording the artifact's per-cell feature dimension.
# Lets reads detect and drop stale entries left by a recipe change or an
# interrupted partial rebuild (a feature group holding arrays of mixed column
# counts) so the caller recomputes them at the current recipe instead of
# crashing later on an opaque pred-vs-GT dimension mismatch.
_FEATURE_DIM_ATTR = "feature_dim"


def read_features_from_group(group, pos_name: str, t: int) -> np.ndarray | None:
    """Read one ``(n_cells, feature_dim)`` array from an already-open feature group.

    Returns ``None`` (treated as a cache miss → recompute) when the stored
    array's feature dimension disagrees with the group's recorded
    :data:`_FEATURE_DIM_ATTR` — i.e. a stale entry from a different recipe or a
    partially-rebuilt cache. The zero-cell ``(0, 0)`` sentinel is exempt (it
    carries no column count). Groups written before this attribute existed have
    no recorded dim, so no entry is dropped (bootstrap-safe).
    """
    key = f"{pos_name}/t{t}"
    if key not in group:
        return None
    arr = np.asarray(group[key])
    expected = group.attrs.get(_FEATURE_DIM_ATTR)
    if expected is not None and arr.ndim == 2 and arr.shape[1] > 0 and arr.shape[1] != int(expected):
        return None
    return arr


def write_features_to_group(group, pos_name: str, t: int, features: np.ndarray) -> None:
    """Write one ``(n_cells, feature_dim)`` array to an already-open feature group.

    Records the artifact's feature dimension in :data:`_FEATURE_DIM_ATTR` from
    the first non-empty write (and updates it if a later write carries a
    different dim — the current recipe is authoritative, and stale entries from
    the old dim then fail the read-side check and get recomputed). The
    zero-cell ``(0, 0)`` sentinel never sets the attribute.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (n_cells, feature_dim); got shape {features.shape}")
    key = f"{pos_name}/t{t}"
    if key in group:
        del group[key]
    group.create_array(key, data=np.asarray(features))
    if features.shape[0] > 0 and features.shape[1] > 0:
        dim = int(features.shape[1])
        if group.attrs.get(_FEATURE_DIM_ATTR) != dim:
            group.attrs[_FEATURE_DIM_ATTR] = dim


@contextmanager
def open_features_group(
    paths: CachePaths,
    kind: FeatureKind,
    *,
    mode: Literal["r", "a"] = "a",
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
    weights_sha12: str | None = None,
):
    """Yield an open zarr group for one feature-cache artifact.

    Use this for per-FOV batch reads/writes so the underlying store is opened
    once per FOV instead of once per timepoint.
    """
    group_path = _features_group_path(
        paths, kind, model_name=model_name, ckpt_sha12=ckpt_sha12, weights_sha12=weights_sha12
    )
    if mode == "r" and not group_path.exists():
        yield None
        return
    group_path.parent.mkdir(parents=True, exist_ok=True)
    yield zarr.open_group(str(group_path), mode=mode)


def read_features(
    paths: CachePaths,
    kind: FeatureKind,
    pos_name: str,
    t: int,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
    weights_sha12: str | None = None,
) -> np.ndarray | None:
    """Read cached target-side features for one (position, timepoint).

    Returns ``None`` if the group or the specific key is absent. Prefer
    :func:`open_features_group` + :func:`read_features_from_group` for
    per-FOV batch reads.
    """
    with open_features_group(
        paths, kind, mode="r", model_name=model_name, ckpt_sha12=ckpt_sha12, weights_sha12=weights_sha12
    ) as group:
        if group is None:
            return None
        return read_features_from_group(group, pos_name, t)


def write_features(
    paths: CachePaths,
    kind: FeatureKind,
    pos_name: str,
    t: int,
    features: np.ndarray,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
    weights_sha12: str | None = None,
) -> None:
    """Write target-side features for one (position, timepoint).

    Overwrites any existing entry at the same key. Prefer
    :func:`open_features_group` + :func:`write_features_to_group` for
    per-FOV batch writes.
    """
    with open_features_group(
        paths, kind, mode="a", model_name=model_name, ckpt_sha12=ckpt_sha12, weights_sha12=weights_sha12
    ) as group:
        write_features_to_group(group, pos_name, t, features)


def encoder_config_sha256_12(encoder_cfg: dict[str, Any]) -> str:
    """Return the first 12 hex chars of the sha256 of a JSON-serialized config.

    Keys are sorted so representation-equivalent configs produce the same hash.
    """
    return json_sha256_12(encoder_cfg)


def feature_slug(name: str) -> str:
    """Replace path separators in *name* so it is safe as a filename stem."""
    return name.replace("/", "__").replace(" ", "_")
