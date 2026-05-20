"""Pipeline-level helpers for evaluation artifact caches.

Sits between :mod:`dynacell.evaluation.cache` (filesystem layout + raw
read/write) and :mod:`dynacell.evaluation.pipeline` (per-FOV orchestration).
Each per-FOV helper loads target- or prediction-side artifacts from cache
when present, otherwise computes and writes them — while honoring the per-artifact
``force_recompute`` flags and the ``io.require_complete_cache`` contract.
"""

from __future__ import annotations

import contextlib
import fcntl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation.cache import (
    CachePaths,
    FeatureKind,
    StaleCacheError,
    built_at_now,
    cache_paths,
    check_artifact_params,
    check_cache_identity,
    ckpt_sha256_12,
    encoder_config_sha256_12,
    feature_slug,
    load_manifest,
    open_features_group,
    read_features_from_group,
    read_mask,
    save_manifest,
    seed_cache_identity,
    write_features_to_group,
    write_mask,
)
from dynacell.evaluation.metrics import (
    cp_regionprops,
    deep_features,
)

_MASK_CHANNEL_BY_SIDE = {"gt": "target_seg", "pred": "prediction_seg"}

_OTHER_SIDE: dict[Literal["gt", "pred"], Literal["gt", "pred"]] = {"gt": "pred", "pred": "gt"}

# Per-side ``io.*`` field names: (cache_dir, plate_path, channel_name).
_SIDE_IO_KEYS: dict[Literal["gt", "pred"], tuple[str, str, str]] = {
    "gt": ("gt_cache_dir", "gt_path", "gt_channel_name"),
    "pred": ("pred_cache_dir", "pred_path", "pred_channel_name"),
}


@dataclass
class _CacheContext:
    """Per-eval-run cache state passed into FOV-level helpers."""

    paths: CachePaths | None
    manifest: dict[str, Any]
    force: dict[str, bool]
    require_complete: bool
    side: Literal["gt", "pred"]
    target_name: str
    spacing: list[float]
    patch_size: int
    dinov3_model_name: str | None = None
    dynaclr_ckpt_sha12: str | None = None
    dynaclr_encoder_sha12: str | None = None
    celldino_weights_sha12: str | None = None
    _manifest_dirty: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        """Whether cache read/write is active for this run."""
        return self.paths is not None

    @property
    def label_prefix(self) -> str:
        """Manifest artifact-label prefix for the pred side (empty for GT)."""
        return "pred_" if self.side == "pred" else ""

    @property
    def source_tag(self) -> dict[str, str]:
        """Manifest-entry tag distinguishing prediction artifacts; empty for GT."""
        return {"source": "prediction"} if self.side == "pred" else {}

    def mark_manifest_dirty(self) -> None:
        """Record that the manifest has unsaved changes (next flush will persist them)."""
        self._manifest_dirty = True

    def consume_manifest_dirty(self) -> bool:
        """Return ``True`` if there are pending writes and clear the dirty flag."""
        if self._manifest_dirty:
            self._manifest_dirty = False
            return True
        return False


def _resolve_force(force: DictConfig) -> dict[str, bool]:
    """Flatten ``force_recompute`` into per-artifact bools, honoring ``.all``."""
    all_flag = bool(force.all)
    return {
        "gt_masks": all_flag or bool(force.gt_masks),
        "gt_cp": all_flag or bool(force.gt_cp),
        "gt_dinov3": all_flag or bool(force.gt_dinov3),
        "gt_dynaclr": all_flag or bool(force.gt_dynaclr),
        "gt_celldino": all_flag or bool(force.gt_celldino),
        "pred_masks": all_flag or bool(force.pred_masks),
        "pred_cp": all_flag or bool(force.pred_cp),
        "pred_dinov3": all_flag or bool(force.pred_dinov3),
        "pred_dynaclr": all_flag or bool(force.pred_dynaclr),
        "pred_celldino": all_flag or bool(force.pred_celldino),
        "final_metrics": all_flag or bool(force.final_metrics),
    }


def init_cache_context(
    config: DictConfig,
    *,
    side: Literal["gt", "pred"],
    dinov3_model_name: str | None = None,
    dynaclr_ckpt_path: str | None = None,
    dynaclr_encoder_cfg: dict[str, Any] | None = None,
    celldino_weights_path: str | None = None,
) -> _CacheContext:
    """Open and validate the *side*-specific artifact cache for the run.

    Parameters
    ----------
    config
        Full Hydra config.
    side
        ``"gt"`` opens the GT-side cache at ``io.gt_cache_dir``; ``"pred"``
        opens the prediction-side cache at ``io.pred_cache_dir``. The GT
        cache is mandatory when ``io.require_complete_cache=true``; the pred
        cache stays opt-in (a missing ``io.pred_cache_dir`` returns a
        disabled context).
    dinov3_model_name
        DINOv3 pretrained name; ``None`` when feature metrics are disabled.
    dynaclr_ckpt_path
        DynaCLR checkpoint path; ``None`` when feature metrics are disabled.
    dynaclr_encoder_cfg
        DynaCLR encoder config (resolved dict); ``None`` when disabled.
    celldino_weights_path
        CELL-DINO ``.pth`` state_dict path; ``None`` when the CELL-DINO
        backbone is not configured.
    """
    io = config.io
    force = _resolve_force(config.force_recompute)
    require_complete_requested = bool(io.require_complete_cache)
    spacing = list(config.pixel_metrics.spacing)
    patch_size = int(config.feature_metrics.patch_size)

    dynaclr_ckpt_sha12 = ckpt_sha256_12(dynaclr_ckpt_path) if dynaclr_ckpt_path is not None else None
    dynaclr_encoder_sha12 = encoder_config_sha256_12(dynaclr_encoder_cfg) if dynaclr_encoder_cfg is not None else None
    celldino_weights_sha12 = ckpt_sha256_12(celldino_weights_path) if celldino_weights_path is not None else None

    cache_dir_key, plate_key, channel_key = _SIDE_IO_KEYS[side]
    cache_dir = OmegaConf.select(config, f"io.{cache_dir_key}", default=None)

    base_kwargs: dict[str, Any] = dict(
        force=force,
        target_name=config.target_name,
        spacing=spacing,
        patch_size=patch_size,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_sha12=dynaclr_ckpt_sha12,
        dynaclr_encoder_sha12=dynaclr_encoder_sha12,
        celldino_weights_sha12=celldino_weights_sha12,
    )

    if cache_dir is None:
        if side == "gt" and require_complete_requested:
            raise ValueError("io.require_complete_cache=true requires io.gt_cache_dir to be set")
        return _CacheContext(paths=None, manifest={}, require_complete=False, side=side, **base_kwargs)

    if side == "pred":
        gt_cache_dir = OmegaConf.select(config, "io.gt_cache_dir", default=None)
        if gt_cache_dir is not None and Path(cache_dir).expanduser().resolve(strict=False) == Path(
            gt_cache_dir
        ).expanduser().resolve(strict=False):
            raise ValueError("io.pred_cache_dir must be distinct from io.gt_cache_dir")

    paths = cache_paths(Path(cache_dir))
    manifest = load_manifest(paths)

    other_side = _OTHER_SIDE[side]
    if manifest.get(other_side) is not None:
        raise StaleCacheError(
            f"io.{cache_dir_key}={cache_dir!r} contains a {other_side.upper()} cache manifest; "
            f"use a separate {side} cache dir"
        )

    cell_seg_path = str(io.cell_segmentation_path) if io.cell_segmentation_path is not None else None
    plate_path = str(getattr(io, plate_key))
    channel_name = str(getattr(io, channel_key))
    check_cache_identity(
        manifest,
        source=side,
        plate_path=plate_path,
        channel_name=channel_name,
        cell_segmentation_path=cell_seg_path,
    )
    seed_cache_identity(
        manifest,
        source=side,
        plate_path=plate_path,
        channel_name=channel_name,
        cell_segmentation_path=cell_seg_path,
    )

    ctx = _CacheContext(
        paths=paths,
        manifest=manifest,
        require_complete=require_complete_requested,
        side=side,
        **base_kwargs,
    )
    _validate_artifact_params(ctx)
    return ctx


def _validate_artifact_params(ctx: _CacheContext) -> None:
    """Raise if any existing per-artifact manifest entry disagrees with ctx params."""
    artifacts = ctx.manifest.get("artifacts", {})

    masks_section = artifacts.get("organelle_masks", {})
    check_artifact_params(
        masks_section.get(ctx.target_name),
        {"target_name": ctx.target_name, **ctx.source_tag},
        artifact_label=f"organelle_masks[{ctx.target_name}]",
    )
    check_artifact_params(
        artifacts.get("cp_features"),
        {"spacing": ctx.spacing, **ctx.source_tag},
        artifact_label=f"{ctx.label_prefix}cp_features",
        numeric_keys=("spacing",),
    )
    if ctx.dinov3_model_name is not None:
        dinov3_section = artifacts.get("dinov3_features", {})
        check_artifact_params(
            dinov3_section.get(feature_slug(ctx.dinov3_model_name)),
            {"model_name": ctx.dinov3_model_name, "patch_size": ctx.patch_size, **ctx.source_tag},
            artifact_label=f"dinov3_features[{ctx.dinov3_model_name}]",
        )
    if ctx.dynaclr_ckpt_sha12 is not None:
        dynaclr_section = artifacts.get("dynaclr_features", {})
        check_artifact_params(
            dynaclr_section.get(ctx.dynaclr_ckpt_sha12),
            {
                "checkpoint_sha256_12": ctx.dynaclr_ckpt_sha12,
                "encoder_config_sha256_12": ctx.dynaclr_encoder_sha12,
                "patch_size": ctx.patch_size,
                **ctx.source_tag,
            },
            artifact_label=f"dynaclr_features[{ctx.dynaclr_ckpt_sha12}]",
        )
    if ctx.celldino_weights_sha12 is not None:
        celldino_section = artifacts.get("celldino_features", {})
        check_artifact_params(
            celldino_section.get(ctx.celldino_weights_sha12),
            {
                "weights_sha256_12": ctx.celldino_weights_sha12,
                "patch_size": ctx.patch_size,
                **ctx.source_tag,
            },
            artifact_label=f"celldino_features[{ctx.celldino_weights_sha12}]",
        )


def _raise_if_require_complete(ctx: _CacheContext, artifact: str, pos_name: str, t: int | None = None) -> None:
    """Raise when ``require_complete_cache=true`` forces a miss to be fatal."""
    if ctx.require_complete:
        where = f"{pos_name}" if t is None else f"{pos_name}/t{t}"
        raise StaleCacheError(f"{artifact} cache miss at {where} and io.require_complete_cache=true")


def _update_manifest_entry(manifest: dict, keys: list[str], entry: dict) -> None:
    """Walk-and-create nested dict path, then shallow-merge *entry* into leaf."""
    current = manifest.setdefault("artifacts", {})
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    leaf = current.setdefault(keys[-1], {})
    leaf.update(entry)


def _add_position(manifest: dict, keys: list[str], pos_name: str) -> None:
    """Append *pos_name* to an artifact entry's ``positions`` list if absent."""
    current = manifest.get("artifacts", {})
    for key in keys:
        current = current.get(key, {})
        if not isinstance(current, dict):
            return
    positions = current.setdefault("positions", [])
    if pos_name not in positions:
        positions.append(pos_name)


@contextlib.contextmanager
def _pos_write_lock(ctx: _CacheContext, kind: str, pos_name: str):
    """Exclusive advisory file lock for one (kind, pos_name) cache slot.

    Concurrent eval jobs on the same (organelle, source plate) all write to
    the same cache zarrs. Without locking, two writers race: writer A
    starts creating the position group, writer B reads the OME plate
    metadata before A finishes updating it (read_mask → None), then B
    calls create_position → zarr.errors.ContainsGroupError because the
    underlying group already exists from A. The lock serializes writers
    of the same slot; cache hits stay lock-free in the callers.

    ``kind`` is a short tag distinguishing artifact families (e.g.
    ``"masks_sec61b"``, ``"features_dinov3_<slug>"``) so different
    families don't contend on each other's locks.
    """
    if ctx.paths is None:
        raise RuntimeError("_pos_write_lock requires a cache-enabled context")
    locks_dir = ctx.paths.root / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)

    def _flatten(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_")

    lock_path = locks_dir / f"{_flatten(kind)}_{_flatten(pos_name)}.lock"
    # Hold the file open for the lifetime of the lock; fcntl.flock is
    # released either explicitly or on file close. Use an "r+" open so the
    # file is not truncated on each acquisition (we don't write to it).
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def fov_masks(
    ctx: _CacheContext,
    pos_name: str,
    image_arr: np.ndarray,
    seg_model,
) -> np.ndarray:
    """Return a ``(T, D, H, W)`` bool mask stack, loading from cache or computing+writing.

    The cache side (GT vs prediction) is taken from ``ctx.side``; the
    artifact label, force-recompute flag, and manifest entry are derived
    from it. When caching is disabled (``ctx.enabled == False``), the masks
    are computed fresh from *image_arr* without any cache interaction.
    """
    manifest_entry: dict[str, Any] = {
        "path": f"organelle_masks/{ctx.target_name}.zarr",
        "target_name": ctx.target_name,
        "built_at": built_at_now(),
        **ctx.source_tag,
    }
    return _fov_masks(
        ctx,
        pos_name=pos_name,
        image_arr=image_arr,
        seg_model=seg_model,
        force_key=f"{ctx.side}_masks",
        artifact_label=f"{ctx.label_prefix}organelle_masks[{ctx.target_name}]",
        manifest_entry=manifest_entry,
    )


def _fov_masks(
    ctx: _CacheContext,
    *,
    pos_name: str,
    image_arr: np.ndarray,
    seg_model,
    force_key: str,
    artifact_label: str,
    manifest_entry: dict[str, Any],
) -> np.ndarray:
    """Shared load-or-compute implementation for one side's organelle masks."""
    from dynacell.evaluation.segmentation import segment

    t_count = image_arr.shape[0]
    channel_name = _MASK_CHANNEL_BY_SIDE[ctx.side]

    def _compute_masks() -> np.ndarray:
        return np.stack(
            [
                np.asarray(segment(image_arr[t], ctx.target_name, seg_model=seg_model)).astype(bool)
                for t in range(t_count)
            ]
        )

    def _validate_cached_shape(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] != t_count:
            raise StaleCacheError(
                f"Cached mask timepoint count mismatch for {pos_name}: cached={arr.shape[0]} vs current={t_count}"
            )
        return arr

    def _record_write() -> None:
        _update_manifest_entry(
            ctx.manifest,
            ["organelle_masks", ctx.target_name],
            manifest_entry,
        )
        _add_position(ctx.manifest, ["organelle_masks", ctx.target_name], pos_name)
        ctx.mark_manifest_dirty()

    def _write_masks(masks: np.ndarray) -> None:
        write_mask(ctx.paths, ctx.target_name, pos_name, masks, channel_name=channel_name)

    # Cache disabled — compute fresh, no locking, no caching.
    if not ctx.enabled:
        return _compute_masks()

    # Force-recompute: take the lock so we don't race against a concurrent
    # cache-miss writer for the same slot, then overwrite.
    if ctx.force[force_key]:
        with _pos_write_lock(ctx, f"{ctx.side}_masks_{ctx.target_name}", pos_name):
            masks = _compute_masks()
            _write_masks(masks)
            _record_write()
        return masks

    # Fast-path: cache hit without taking a lock.
    cached = read_mask(ctx.paths, ctx.target_name, pos_name)
    if cached is not None:
        return _validate_cached_shape(cached)

    # Cache miss — take the lock and re-check inside it. A concurrent writer
    # may have populated the slot between our fast-path read and the lock
    # acquisition; if so, the re-read returns a hit and we skip recomputation.
    with _pos_write_lock(ctx, f"{ctx.side}_masks_{ctx.target_name}", pos_name):
        cached = read_mask(ctx.paths, ctx.target_name, pos_name)
        if cached is not None:
            return _validate_cached_shape(cached)
        _raise_if_require_complete(ctx, artifact_label, pos_name)
        masks = _compute_masks()
        _write_masks(masks)
        _record_write()
    return masks


def _load_or_compute_feature_timepoints(
    ctx: _CacheContext,
    *,
    kind: FeatureKind,
    pos_name: str,
    t_count: int,
    force_key: str,
    artifact_label: str,
    cache_kwargs: dict[str, Any],
    compute_fn,
) -> tuple[list[np.ndarray], bool]:
    """Per-timepoint load-or-compute loop for one feature family.

    Reads from the backing zarr group lockless (concurrent readers are
    safe on append-only feature zarrs); only acquires the per-FOV write
    lock when at least one timepoint is missing or force-recompute is
    set. Returns ``(per_t_features, manifest_updated)``.
    """
    if not ctx.enabled:
        return [np.asarray(compute_fn(t)) for t in range(t_count)], False

    force_recompute = ctx.force[force_key]
    per_t: list[np.ndarray | None] = [None] * t_count

    # Lockless prefetch pass. Skipped under force_recompute because we'll
    # rewrite every timepoint anyway.
    if not force_recompute:
        with open_features_group(ctx.paths, kind, mode="r", **cache_kwargs) as group:
            if group is not None:
                for t in range(t_count):
                    per_t[t] = read_features_from_group(group, pos_name, t)

    pending = [t for t in range(t_count) if per_t[t] is None] if not force_recompute else list(range(t_count))
    if not pending:
        return per_t, False  # type: ignore[return-value]

    # Lock domain: per (feature family, position). Different families have
    # separate backing zarrs, so they don't contend on each other's slots;
    # different positions within one family can write concurrently.
    kwargs_tag = "_".join(str(v) for _, v in sorted(cache_kwargs.items()))
    lock_tag = f"features_{kind}_{kwargs_tag}" if kwargs_tag else f"features_{kind}"

    manifest_updated = False
    with (
        _pos_write_lock(ctx, lock_tag, pos_name),
        open_features_group(ctx.paths, kind, mode="a", **cache_kwargs) as group,
    ):
        for t in pending:
            if not force_recompute:
                # A concurrent writer may have populated this slot between
                # the lockless prefetch and the lock acquisition.
                feats = read_features_from_group(group, pos_name, t)
                if feats is not None:
                    per_t[t] = feats
                    continue
                _raise_if_require_complete(ctx, artifact_label, pos_name, t)
            feats = np.asarray(compute_fn(t))
            write_features_to_group(group, pos_name, t, feats)
            per_t[t] = feats
            manifest_updated = True
    return per_t, manifest_updated  # type: ignore[return-value]


def fov_cp_features(
    ctx: _CacheContext,
    pos_name: str,
    image_arr: np.ndarray,
    cell_segmentation_arr: np.ndarray,
) -> list[np.ndarray]:
    """Return CP regionprops per timepoint for one FOV, loading from cache or computing+writing.

    The cache side is taken from ``ctx.side``. Result is a list of ``T``
    arrays, each shape ``(n_cells_t, n_props_raw)``.
    """
    per_t, manifest_updated = _load_or_compute_feature_timepoints(
        ctx,
        kind="cp",
        pos_name=pos_name,
        t_count=image_arr.shape[0],
        force_key=f"{ctx.side}_cp",
        artifact_label=f"{ctx.label_prefix}cp_features",
        cache_kwargs={},
        compute_fn=lambda t: cp_regionprops(image_arr[t], cell_segmentation_arr[t], ctx.spacing),
    )

    if ctx.enabled and manifest_updated:
        entry: dict[str, Any] = {
            "path": "features/cp.zarr",
            "spacing": ctx.spacing,
            "built_at": built_at_now(),
            **ctx.source_tag,
        }
        _update_manifest_entry(ctx.manifest, ["cp_features"], entry)
        _add_position(ctx.manifest, ["cp_features"], pos_name)
        ctx.mark_manifest_dirty()

    return per_t


def fov_deep_features(
    ctx: _CacheContext,
    pos_name: str,
    image_arr: np.ndarray,
    cell_segmentation_arr: np.ndarray,
    feature_extractor,
    kind: FeatureKind,
) -> list[np.ndarray]:
    """Return per-cell deep embeddings per timepoint for one FOV.

    The cache side is taken from ``ctx.side``. ``kind`` is ``"dinov3"``,
    ``"dynaclr"``, or ``"celldino"``; the cache key (model name or
    checkpoint/weights hash) is pulled from *ctx*.
    """
    return _fov_deep_features(
        ctx,
        pos_name=pos_name,
        image_arr=image_arr,
        kind=kind,
        compute_fn=lambda t: deep_features(image_arr[t], cell_segmentation_arr[t], feature_extractor, ctx.patch_size),
    )


def _deep_feature_cache_metadata(
    ctx: _CacheContext,
    kind: FeatureKind,
) -> tuple[str, str, dict[str, Any], list[str], dict[str, Any]]:
    """Return force key, artifact label, cache kwargs, manifest path, and manifest entry."""
    if kind == "dinov3":
        if ctx.dinov3_model_name is None:
            raise ValueError("dinov3_model_name is required for DINOv3 feature caching")
        force_key = f"{ctx.side}_dinov3"
        artifact_label = f"{ctx.label_prefix}dinov3_features[{ctx.dinov3_model_name}]"
        cache_kwargs = {"model_name": ctx.dinov3_model_name}
        slug = feature_slug(ctx.dinov3_model_name)
        manifest_keys = ["dinov3_features", slug]
        entry = {
            "path": f"features/dinov3/{slug}.zarr",
            "model_name": ctx.dinov3_model_name,
            "patch_size": ctx.patch_size,
            **ctx.source_tag,
            "built_at": built_at_now(),
        }
    elif kind == "dynaclr":
        if ctx.dynaclr_ckpt_sha12 is None:
            raise ValueError("dynaclr_ckpt_sha12 is required for DynaCLR feature caching")
        force_key = f"{ctx.side}_dynaclr"
        artifact_label = f"{ctx.label_prefix}dynaclr_features[{ctx.dynaclr_ckpt_sha12}]"
        cache_kwargs = {"ckpt_sha12": ctx.dynaclr_ckpt_sha12}
        manifest_keys = ["dynaclr_features", ctx.dynaclr_ckpt_sha12]
        entry = {
            "path": f"features/dynaclr/{ctx.dynaclr_ckpt_sha12}.zarr",
            "checkpoint_sha256_12": ctx.dynaclr_ckpt_sha12,
            "encoder_config_sha256_12": ctx.dynaclr_encoder_sha12,
            "patch_size": ctx.patch_size,
            **ctx.source_tag,
            "built_at": built_at_now(),
        }
    elif kind == "celldino":
        if ctx.celldino_weights_sha12 is None:
            raise ValueError("celldino_weights_sha12 is required for CELL-DINO feature caching")
        force_key = f"{ctx.side}_celldino"
        artifact_label = f"{ctx.label_prefix}celldino_features[{ctx.celldino_weights_sha12}]"
        cache_kwargs = {"weights_sha12": ctx.celldino_weights_sha12}
        manifest_keys = ["celldino_features", ctx.celldino_weights_sha12]
        entry = {
            "path": f"features/celldino/{ctx.celldino_weights_sha12}.zarr",
            "weights_sha256_12": ctx.celldino_weights_sha12,
            "patch_size": ctx.patch_size,
            **ctx.source_tag,
            "built_at": built_at_now(),
        }
    else:
        raise ValueError(f"Unknown deep-feature kind: {kind!r}")
    return force_key, artifact_label, cache_kwargs, manifest_keys, entry


def _fov_deep_features(
    ctx: _CacheContext,
    *,
    pos_name: str,
    image_arr: np.ndarray,
    kind: FeatureKind,
    compute_fn,
) -> list[np.ndarray]:
    """Shared load-or-compute implementation for one side's deep embeddings."""
    force_key, artifact_label, cache_kwargs, manifest_keys, entry = _deep_feature_cache_metadata(ctx, kind)

    per_t, manifest_updated = _load_or_compute_feature_timepoints(
        ctx,
        kind=kind,
        pos_name=pos_name,
        t_count=image_arr.shape[0],
        force_key=force_key,
        artifact_label=artifact_label,
        cache_kwargs=cache_kwargs,
        compute_fn=compute_fn,
    )

    if ctx.enabled and manifest_updated:
        _update_manifest_entry(ctx.manifest, manifest_keys, entry)
        _add_position(ctx.manifest, manifest_keys, pos_name)
        ctx.mark_manifest_dirty()

    return per_t


def flush_manifest(ctx: _CacheContext) -> None:
    """Persist the manifest to disk if it has been mutated since last flush.

    Concurrent eval jobs sharing the same cache dir would otherwise race on
    the manifest YAML — last writer wins, all earlier additions are lost.
    Under the lock we reload the on-disk manifest, merge our pending edits
    on top, then save. This preserves additions from concurrent writers
    even though zarr-level cache correctness doesn't depend on the manifest.
    """
    if not (ctx.enabled and ctx.consume_manifest_dirty()):
        return
    with _pos_write_lock(ctx, "manifest", "global"):
        on_disk = load_manifest(ctx.paths)
        merged = _merge_manifests(on_disk, ctx.manifest)
        save_manifest(ctx.paths, merged)
        ctx.manifest = merged


def _merge_manifests(on_disk: dict[str, Any], in_memory: dict[str, Any]) -> dict[str, Any]:
    """Merge an in-memory manifest on top of the on-disk one, unioning position lists.

    Identity fields and per-artifact metadata (``built_at``, ``path``, model
    identifiers) are taken from the in-memory copy. ``positions`` lists are
    unioned so additions from concurrent writers are preserved.
    """
    merged = {**on_disk, **{k: v for k, v in in_memory.items() if k != "artifacts"}}
    merged["artifacts"] = dict(on_disk.get("artifacts", {}))
    in_artifacts = in_memory.get("artifacts", {})
    for top_key, top_val in in_artifacts.items():
        if not isinstance(top_val, dict):
            merged["artifacts"][top_key] = top_val
            continue
        existing = merged["artifacts"].setdefault(top_key, {})
        if not isinstance(existing, dict):
            merged["artifacts"][top_key] = top_val
            continue
        for sub_key, sub_val in top_val.items():
            if not isinstance(sub_val, dict):
                existing[sub_key] = sub_val
                continue
            target = existing.setdefault(sub_key, {})
            if not isinstance(target, dict):
                existing[sub_key] = sub_val
                continue
            on_disk_positions = target.get("positions", [])
            in_mem_positions = sub_val.get("positions", [])
            unioned = list(dict.fromkeys([*on_disk_positions, *in_mem_positions]))
            target.update(sub_val)
            if unioned:
                target["positions"] = unioned
    return merged


def resolve_dynaclr_encoder_cfg(config: DictConfig) -> dict[str, Any] | None:
    """Extract and resolve the DynaCLR encoder config as a plain dict (for hashing)."""
    encoder = OmegaConf.select(config, "feature_extractor.dynaclr.encoder", default=None)
    if encoder is None:
        return None
    return OmegaConf.to_container(encoder, resolve=True)
