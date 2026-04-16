"""GT artifact cache for the dynacell evaluation pipeline.

Stores target-side organelle masks and feature embeddings under an explicit
cache directory so successive eval runs against the same GT dataset skip
the expensive segmentation and feature-extraction work.

Cache identity is the tuple
``(cache_schema_version, gt_plate_path, gt_channel_name, cell_segmentation_path)``.
Per-artifact invalidation is driven by extra params recorded in the manifest
(e.g. spacing, patch_size, checkpoint hash).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

CACHE_SCHEMA_VERSION = 1

_MASK_CHANNEL = "target_seg"


class StaleCacheError(RuntimeError):
    """Raised when cache identity or artifact params disagree with the current config."""


@dataclass(frozen=True)
class CachePaths:
    """Filesystem layout for one GT cache directory."""

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
        slug = _safe_slug(model_name)
        return self.features_dir / "dinov3" / f"{slug}.zarr"

    def dynaclr_features(self, ckpt_sha12: str) -> Path:
        """Return the zarr group path for DynaCLR features keyed by *ckpt_sha12*."""
        return self.features_dir / "dynaclr" / f"{ckpt_sha12}.zarr"


def cache_paths(gt_cache_dir: Path | str) -> CachePaths:
    """Build a CachePaths rooted at *gt_cache_dir* (does not create directories)."""
    root = Path(gt_cache_dir)
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
            "cell_segmentation": None,
            "artifacts": {},
        }
    raw = OmegaConf.to_container(OmegaConf.load(paths.manifest), resolve=True)
    if not isinstance(raw, dict):
        raise StaleCacheError(f"Manifest at {paths.manifest} is not a mapping")
    raw.setdefault("artifacts", {})
    return raw


def save_manifest(paths: CachePaths, manifest: dict[str, Any]) -> None:
    """Persist *manifest* as YAML under *paths.manifest*, creating parents."""
    paths.root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(manifest), paths.manifest)


def check_cache_identity(
    manifest: dict[str, Any],
    *,
    gt_plate_path: str,
    gt_channel_name: str,
    cell_segmentation_path: str | None,
) -> None:
    """Raise if the manifest's cache identity disagrees with the current config.

    Parameters
    ----------
    manifest
        Loaded manifest dict (may be the empty skeleton from :func:`load_manifest`).
    gt_plate_path
        Current ``io.gt_path``.
    gt_channel_name
        Current ``io.gt_channel_name``.
    cell_segmentation_path
        Current ``io.cell_segmentation_path``. ``None`` skips the check.
    """
    version = manifest.get("cache_schema_version")
    if version is not None and version != CACHE_SCHEMA_VERSION:
        raise StaleCacheError(
            f"Cache schema version mismatch: manifest has {version}, current is {CACHE_SCHEMA_VERSION}. "
            "Delete the cache directory or bump cache_schema_version."
        )
    gt_entry = manifest.get("gt")
    if gt_entry is not None:
        if gt_entry.get("plate_path") != gt_plate_path:
            raise StaleCacheError(
                f"gt.plate_path mismatch: manifest={gt_entry.get('plate_path')!r}, config={gt_plate_path!r}"
            )
        if gt_entry.get("channel_name") != gt_channel_name:
            raise StaleCacheError(
                f"gt.channel_name mismatch: manifest={gt_entry.get('channel_name')!r}, config={gt_channel_name!r}"
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
    gt_plate_path: str,
    gt_channel_name: str,
    cell_segmentation_path: str | None,
) -> None:
    """Populate the ``gt`` / ``cell_segmentation`` manifest entries if absent.

    Called before the first artifact is written. Safe to call repeatedly;
    later calls with conflicting values should be preceded by
    :func:`check_cache_identity`.
    """
    manifest["cache_schema_version"] = CACHE_SCHEMA_VERSION
    if manifest.get("gt") is None:
        manifest["gt"] = {"plate_path": gt_plate_path, "channel_name": gt_channel_name}
    if cell_segmentation_path is not None and manifest.get("cell_segmentation") is None:
        manifest["cell_segmentation"] = {"plate_path": cell_segmentation_path}


def check_artifact_params(
    entry: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    artifact_label: str,
    numeric_keys: tuple[str, ...] = (),
) -> None:
    """Raise if a per-artifact manifest entry disagrees with *current* params.

    Parameters
    ----------
    entry
        Manifest entry for the artifact, or ``None`` if no entry exists yet
        (in which case this function is a no-op — the caller decides whether
        to treat absence as miss or miss+error).
    current
        Current-config values keyed by the same names as in *entry*.
    artifact_label
        Human-readable label for the error message (e.g. ``"cp_features"``).
    numeric_keys
        Keys in *current* whose values should be compared with
        :func:`numpy.allclose` instead of ``==``.
    """
    if entry is None:
        return
    for key, value in current.items():
        cached_value = entry.get(key)
        if key in numeric_keys:
            if cached_value is None or not np.allclose(
                np.asarray(cached_value, dtype=float),
                np.asarray(value, dtype=float),
                rtol=1e-9,
                atol=0.0,
            ):
                raise StaleCacheError(f"{artifact_label}: {key} mismatch: cached={cached_value!r}, current={value!r}")
        elif cached_value != value:
            raise StaleCacheError(f"{artifact_label}: {key} mismatch: cached={cached_value!r}, current={value!r}")


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


def write_mask(
    paths: CachePaths,
    target_name: str,
    pos_name: str,
    masks: np.ndarray,
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
    """
    if masks.ndim != 4:
        raise ValueError(f"masks must be 4-D (T, D, H, W); got shape {masks.shape}")
    plate_path = paths.mask_plate(target_name)
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if plate_path.exists() else "w"
    data = masks.astype(bool)[:, None]  # (T, 1, D, H, W)
    with open_ome_zarr(
        plate_path,
        mode=mode,
        layout="hcs",
        channel_names=[_MASK_CHANNEL],
        version="0.5",
    ) as plate:
        row, col, fov = pos_name.split("/")
        if pos_name in plate:
            del plate[pos_name]
        position = plate.create_position(row, col, fov)
        position.create_image("0", data)


def _features_group_path(
    paths: CachePaths,
    kind: str,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
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
    raise ValueError(f"Unknown feature kind: {kind!r}")


def read_features(
    paths: CachePaths,
    kind: str,
    pos_name: str,
    t: int,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
) -> np.ndarray | None:
    """Read cached target-side features for one (position, timepoint).

    Returns ``None`` if the group or the specific key is absent. An empty
    array ``(0, feature_dim)`` signals "zero cells at this timepoint" (not
    absence).
    """
    group_path = _features_group_path(paths, kind, model_name=model_name, ckpt_sha12=ckpt_sha12)
    if not group_path.exists():
        return None
    store = zarr.open_group(str(group_path), mode="r")
    key = f"{pos_name}/t{t}"
    if key not in store:
        return None
    return np.asarray(store[key])


def write_features(
    paths: CachePaths,
    kind: str,
    pos_name: str,
    t: int,
    features: np.ndarray,
    *,
    model_name: str | None = None,
    ckpt_sha12: str | None = None,
) -> None:
    """Write target-side features for one (position, timepoint).

    Overwrites any existing entry at the same key.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (n_cells, feature_dim); got shape {features.shape}")
    group_path = _features_group_path(paths, kind, model_name=model_name, ckpt_sha12=ckpt_sha12)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(group_path), mode="a")
    key = f"{pos_name}/t{t}"
    if key in store:
        del store[key]
    store.create_array(key, data=np.asarray(features))


def ckpt_sha256_12(path: Path | str) -> str:
    """Return the first 12 hex chars of the sha256 of the file at *path*."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def encoder_config_sha256_12(encoder_cfg: dict[str, Any]) -> str:
    """Return the first 12 hex chars of the sha256 of a JSON-serialized config.

    Keys are sorted so representation-equivalent configs produce the same hash.
    """
    payload = json.dumps(encoder_cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _safe_slug(name: str) -> str:
    """Replace path separators in *name* so it is safe as a filename stem."""
    return name.replace("/", "__").replace(" ", "_")
