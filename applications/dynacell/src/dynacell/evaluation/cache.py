"""Artifact cache for the dynacell evaluation pipeline.

Stores organelle masks and feature embeddings under an explicit cache
directory so successive eval runs against the same source dataset skip the
expensive segmentation and feature-extraction work.

Cache identity is rooted in the source plate/channel plus
``cell_segmentation_path`` when cell-level features are involved.
Per-artifact invalidation is driven by extra params recorded in the manifest
(e.g. spacing, patch_size, checkpoint hash).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

FeatureKind = Literal["cp", "dinov3", "dynaclr", "celldino"]

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

    def mask_plate(self, target_name: str) -> Path:
        """Return the HCS OME-Zarr plate for masks of *target_name*."""
        return self.masks_dir / f"{target_name}.zarr"

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


def cache_paths(cache_dir: Path | str) -> CachePaths:
    """Build a CachePaths rooted at *cache_dir* (does not create directories)."""
    root = Path(cache_dir)
    return CachePaths(
        root=root,
        manifest=root / "manifest.yaml",
        masks_dir=root / "organelle_masks",
        features_dir=root / "features",
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
    raw = OmegaConf.to_container(OmegaConf.load(paths.manifest), resolve=True)
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
    OmegaConf.save(OmegaConf.create(manifest), paths.manifest)


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


def read_mask(paths: CachePaths, target_name: str, pos_name: str) -> np.ndarray | None:
    """Read cached organelle masks for a single position.

    Returns
    -------
    numpy.ndarray | None
        Bool array of shape ``(T, D, H, W)``, or ``None`` if the plate or
        position is absent.
    """
    plate_path = paths.mask_plate(target_name)
    if not plate_path.exists():
        return None
    with open_ome_zarr(plate_path, mode="r") as plate:
        try:
            position = plate[pos_name]
        except KeyError:
            return None
        data = np.asarray(position.data[:, 0]).astype(bool)
    return data


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


def write_mask(
    paths: CachePaths,
    target_name: str,
    pos_name: str,
    masks: np.ndarray,
    *,
    channel_name: str = _MASK_CHANNEL,
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
    """
    if masks.ndim != 4:
        raise ValueError(f"masks must be 4-D (T, D, H, W); got shape {masks.shape}")
    plate_path = paths.mask_plate(target_name)
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    data = masks.astype(bool)[:, None]  # (T, 1, D, H, W)
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
    raise ValueError(f"Unknown feature kind: {kind!r}")


def read_features_from_group(group, pos_name: str, t: int) -> np.ndarray | None:
    """Read one ``(n_cells, feature_dim)`` array from an already-open feature group."""
    key = f"{pos_name}/t{t}"
    if key not in group:
        return None
    return np.asarray(group[key])


def write_features_to_group(group, pos_name: str, t: int, features: np.ndarray) -> None:
    """Write one ``(n_cells, feature_dim)`` array to an already-open feature group."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (n_cells, feature_dim); got shape {features.shape}")
    key = f"{pos_name}/t{t}"
    if key in group:
        del group[key]
    group.create_array(key, data=np.asarray(features))


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


def ckpt_sha256_12(path: Path | str) -> str:
    """Return the first 12 hex chars of the sha256 of the file at *path*.

    On repeated calls for the same checkpoint, reads the digest from a
    ``<path>.sha256`` sidecar file when present and newer than the
    checkpoint, avoiding a multi-GB re-read. Writes the sidecar after a
    fresh hash; silently tolerates read-only parent directories and NFS
    flakes by falling back to recompute.
    """
    ckpt = Path(path)
    sidecar = ckpt.with_suffix(ckpt.suffix + ".sha256")
    try:
        if sidecar.stat().st_mtime >= ckpt.stat().st_mtime:
            digest = sidecar.read_text().strip()
            if len(digest) >= 12 and all(c in "0123456789abcdef" for c in digest[:12]):
                return digest[:12]
    except OSError:
        pass
    hasher = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    try:
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(digest + "\n")
        tmp.replace(sidecar)
    except OSError:
        pass
    return digest[:12]


def encoder_config_sha256_12(encoder_cfg: dict[str, Any]) -> str:
    """Return the first 12 hex chars of the sha256 of a JSON-serialized config.

    Keys are sorted so representation-equivalent configs produce the same hash.
    """
    payload = json.dumps(encoder_cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def feature_slug(name: str) -> str:
    """Replace path separators in *name* so it is safe as a filename stem."""
    return name.replace("/", "__").replace(" ", "_")
