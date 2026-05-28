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
import warnings
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
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
    build_crops,
    cp_regionprops,
    deep_features,
    features_from_crops,
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
    _: KW_ONLY
    use_gpu: bool = True
    dinov3_model_name: str | None = None
    dynaclr_ckpt_sha12: str | None = None
    dynaclr_encoder_sha12: str | None = None
    celldino_weights_sha12: str | None = None
    dinov3_preprocess_version: str | None = None
    dynaclr_preprocess_version: str | None = None
    celldino_preprocess_version: str | None = None
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
    dinov3_preprocess_version: str | None = None,
    dynaclr_preprocess_version: str | None = None,
    celldino_preprocess_version: str | None = None,
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
    dinov3_preprocess_version, dynaclr_preprocess_version,
    celldino_preprocess_version
        Per-extractor preprocess-recipe version tags (e.g.
        ``"self_normalize_v1"``). On a known mismatch against the cached
        manifest entry, the corresponding ``force_recompute.<side>_<kind>``
        flag is set so the per-FOV cache is bypassed and features get
        recomputed with the current preprocessing. Missing values (e.g.
        the kind isn't loaded, or the cached manifest pre-dates version
        tracking) are treated as "no constraint" — no auto-invalidation.
    """
    io = config.io
    force = _resolve_force(config.force_recompute)
    require_complete_requested = bool(io.require_complete_cache)
    spacing = list(config.pixel_metrics.spacing)
    patch_size = int(config.feature_metrics.patch_size)
    use_gpu = bool(getattr(config, "use_gpu", True))

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
        use_gpu=use_gpu,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_sha12=dynaclr_ckpt_sha12,
        dynaclr_encoder_sha12=dynaclr_encoder_sha12,
        celldino_weights_sha12=celldino_weights_sha12,
        dinov3_preprocess_version=dinov3_preprocess_version,
        dynaclr_preprocess_version=dynaclr_preprocess_version,
        celldino_preprocess_version=celldino_preprocess_version,
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
    _auto_invalidate_on_preprocess_version_mismatch(ctx)
    return ctx


def _auto_invalidate_on_preprocess_version_mismatch(ctx: _CacheContext) -> None:
    """Soft-invalidate per-extractor cache when ``preprocess_version`` changes.

    The cache manifest records a per-extractor ``preprocess_version`` tag
    starting with this commit. When the version on disk differs from the
    extractor's current ``PREPROCESS_VERSION``, the corresponding
    ``ctx.force[<side>_<kind>]`` flag is set so the per-FOV cache read is
    bypassed and features are recomputed with the current preprocessing.

    Missing tags (cached manifest pre-dates version tracking, or the kind
    isn't loaded for this run) are treated as "no constraint" — they do
    NOT trigger invalidation. Operators handle that bootstrap transition
    explicitly via ``force_recompute.<side>_<kind>``.
    """
    if not ctx.enabled:
        return
    artifacts = ctx.manifest.get("artifacts", {})
    checks: list[tuple[str, str, str | None, str | None]] = [
        (
            "dinov3",
            "dinov3_features",
            ctx.dinov3_preprocess_version,
            feature_slug(ctx.dinov3_model_name) if ctx.dinov3_model_name is not None else None,
        ),
        ("dynaclr", "dynaclr_features", ctx.dynaclr_preprocess_version, ctx.dynaclr_ckpt_sha12),
        ("celldino", "celldino_features", ctx.celldino_preprocess_version, ctx.celldino_weights_sha12),
    ]
    for kind, section_key, current_version, sub_key in checks:
        if current_version is None or sub_key is None:
            continue
        entry = artifacts.get(section_key, {}).get(sub_key)
        if entry is None:
            continue
        cached_version = entry.get("preprocess_version")
        if cached_version is None or cached_version == current_version:
            continue
        force_key = f"{ctx.side}_{kind}"
        ctx.force[force_key] = True
        warnings.warn(
            f"{section_key}[{sub_key}]: preprocess_version mismatch "
            f"(cached={cached_version!r}, current={current_version!r}); "
            f"auto-invalidating {force_key} cache for this run.",
            stacklevel=2,
        )


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


def _kind_cache_kwargs(ctx: _CacheContext, kind: FeatureKind) -> dict[str, Any]:
    """Per-kind kwargs for ``open_features_group``."""
    return _deep_feature_cache_metadata(ctx, kind)[2]


def _kind_lock_tag(kind: FeatureKind, cache_kwargs: dict[str, Any]) -> str:
    """Per-(family, model identity) lock tag for ``_pos_write_lock``.

    Shared by :func:`_load_or_compute_feature_timepoints` and
    :func:`_flush_kind` so the per-(pos, kind) write lock is identical
    regardless of which path produced the write.
    """
    kwargs_tag = "_".join(str(v) for _, v in sorted(cache_kwargs.items()))
    return f"features_{kind}_{kwargs_tag}" if kwargs_tag else f"features_{kind}"


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
    lock_tag = _kind_lock_tag(kind, cache_kwargs)

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
        compute_fn=lambda t: cp_regionprops(image_arr[t], cell_segmentation_arr[t], ctx.spacing, use_gpu=ctx.use_gpu),
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
        if ctx.dinov3_preprocess_version is not None:
            entry["preprocess_version"] = ctx.dinov3_preprocess_version
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
        if ctx.dynaclr_preprocess_version is not None:
            entry["preprocess_version"] = ctx.dynaclr_preprocess_version
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
        if ctx.celldino_preprocess_version is not None:
            entry["preprocess_version"] = ctx.celldino_preprocess_version
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


class DeepFeatureBatcher:
    """Accumulate per-cell crops across ``(pos_name, t)`` and flush in batches.

    Owns a single side's :class:`_CacheContext`. The caller's outer position
    scan decides which ``(pos_name, t)`` slots need work (via
    :meth:`pending_kinds_per_t`) and feeds crops in (via :meth:`push`). When
    the pending crop count for a backbone reaches *flush_threshold*, the
    extractor runs over the accumulated crops, results are split by row
    count, and per-``(pos, t)`` entries are written to the cache.

    Parameters
    ----------
    ctx
        The side's cache context. ``ctx.enabled`` must be ``True``.
    extractors
        Map from feature kind (``"dinov3" | "dynaclr" | "celldino"``) to a
        configured extractor. Calls are routed through
        :func:`dynacell.evaluation.metrics.features_from_crops`, which
        accepts either a batch-capable or per-cell extractor.
    flush_threshold
        Pending crop count per backbone before a batched forward fires.
    """

    def __init__(
        self,
        ctx: _CacheContext,
        extractors: dict[FeatureKind, Any],
        flush_threshold: int = 256,
    ) -> None:
        if not ctx.enabled:
            raise ValueError("DeepFeatureBatcher requires a cache-enabled context")
        self.ctx = ctx
        self.extractors = extractors
        self.flush_threshold = flush_threshold
        self._pending: dict[FeatureKind, list[tuple[str, int, list[np.ndarray]]]] = {k: [] for k in extractors}
        self._pending_counts: dict[FeatureKind, int] = {k: 0 for k in extractors}
        # Snapshot force flags at construction. The scan reads from the
        # snapshot, NEVER from ``ctx.force``. Mid-scan flushes that would
        # otherwise clear ``ctx.force[...]`` cannot change the scan
        # semantics for later positions.
        self._force_snapshot: dict[FeatureKind, bool] = {k: ctx.force[f"{ctx.side}_{k}"] for k in extractors}
        self._flushed_kinds: set[FeatureKind] = set()

    def pending_kinds_per_t(self, pos_name: str, t_count: int) -> dict[FeatureKind, list[int]]:
        """Lockless prefetch: return per-kind list of timepoints needing work."""
        out: dict[FeatureKind, list[int]] = {}
        for kind in self.extractors:
            if self._force_snapshot[kind]:
                out[kind] = list(range(t_count))
                continue
            cache_kwargs = _kind_cache_kwargs(self.ctx, kind)
            with open_features_group(self.ctx.paths, kind, mode="r", **cache_kwargs) as group:
                if group is None:
                    out[kind] = list(range(t_count))
                else:
                    out[kind] = [t for t in range(t_count) if read_features_from_group(group, pos_name, t) is None]
        return out

    def push(
        self,
        pos_name: str,
        t: int,
        crops: list[np.ndarray],
        kinds: Iterable[FeatureKind],
    ) -> None:
        """Queue *crops* for the listed *kinds* at ``(pos_name, t)``.

        Parameters
        ----------
        pos_name
            FOV name (matches the ome-zarr position path).
        t
            Timepoint index.
        crops
            Per-cell 2-D crops from :func:`build_crops`. Empty list is a
            valid zero-cell slot and will write the same ``(0, 0)``
            sentinel as ``features_from_crops([])``.
        kinds
            Backbones to queue for; typically the filtered subset of
            :meth:`pending_kinds_per_t` at this ``t``.
        """
        n = len(crops)
        for kind in kinds:
            self._pending[kind].append((pos_name, t, crops))
            self._pending_counts[kind] += n
            if self._pending_counts[kind] >= self.flush_threshold:
                _flush_kind(self.ctx, kind, self.extractors[kind], self._pending[kind])
                self._flushed_kinds.add(kind)
                self._pending[kind].clear()
                self._pending_counts[kind] = 0

    def drain(self) -> None:
        """Flush all pending; clear force flags for kinds we populated.

        The force clear is deferred to drain time so mid-scan flushes can't
        flip scan semantics for later positions. Only kinds the batcher
        actually flushed are cleared — kinds that had nothing to do (every
        slot already cached) keep their force flag unchanged.
        """
        for kind, items in self._pending.items():
            if items:
                _flush_kind(self.ctx, kind, self.extractors[kind], items)
                self._flushed_kinds.add(kind)
                items.clear()
                self._pending_counts[kind] = 0
        for kind in self._flushed_kinds:
            # Cache is now up-to-date for this kind; defang force_recompute
            # so the downstream per-FOV path treats slots we just wrote as
            # hits instead of re-extracting them.
            self.ctx.force[f"{self.ctx.side}_{kind}"] = False


def _flush_kind(
    ctx: _CacheContext,
    kind: FeatureKind,
    extractor,
    items: list[tuple[str, int, list[np.ndarray]]],
) -> None:
    """Run *extractor* on accumulated crops; split by row counts; write cache.

    Calls through :func:`features_from_crops` so extractors without
    ``extract_features_batch`` fall back to per-cell ``extract_features``.
    Empty ``(pos, t)`` slots (zero cells) write ``np.empty((0, 0))`` to match
    the per-FOV path's ``features_from_crops([])`` output.
    """
    flat: list[np.ndarray] = [c for _, _, crops in items for c in crops]
    counts = [len(crops) for _, _, crops in items]

    feats = features_from_crops(flat, extractor)

    cache_kwargs = _kind_cache_kwargs(ctx, kind)
    lock_tag = _kind_lock_tag(kind, cache_kwargs)
    _, _, _, manifest_keys, entry = _deep_feature_cache_metadata(ctx, kind)

    by_pos: dict[str, list[tuple[int, np.ndarray]]] = {}
    cursor = 0
    for (pos_name, t, _), n in zip(items, counts):
        if n:
            chunk = feats[cursor : cursor + n]
        else:
            # Empty-cell slot writes (0, 0) — same as features_from_crops([]).
            # NEVER write (0, feature_dim); that would diverge from the per-FOV
            # path and break cache equivalence.
            chunk = np.empty((0, 0), dtype=np.float32)
        by_pos.setdefault(pos_name, []).append((t, chunk))
        cursor += n

    for pos_name, ts_chunks in by_pos.items():
        with (
            _pos_write_lock(ctx, lock_tag, pos_name),
            open_features_group(ctx.paths, kind, mode="a", **cache_kwargs) as group,
        ):
            for t, chunk in ts_chunks:
                write_features_to_group(group, pos_name, t, chunk)
        _update_manifest_entry(ctx.manifest, manifest_keys, entry)
        _add_position(ctx.manifest, manifest_keys, pos_name)
        ctx.mark_manifest_dirty()
    # Manifest persistence is deferred to the caller (after batcher.drain()).
    # Zarr slot writes above are durable per-slot; on resume the lockless
    # prefetch finds them whether or not the manifest is up to date.


def precompute_deep_features(
    sides: dict[Literal["gt", "pred"], _CacheContext],
    side_positions: dict[Literal["gt", "pred"], list[tuple[str, Any]]],
    side_channel_names: dict[Literal["gt", "pred"], str],
    seg_positions: list[tuple[str, Any]],
    extractors: dict[FeatureKind, Any],
    *,
    flush_threshold: int = 256,
) -> None:
    """Batched precompute of deep features across all FOVs/timepoints.

    Iterates positions in lockstep across the configured sides, loads
    ``cell_seg`` once per position (shared across sides), loads each side's
    image once, builds 2-D crops, and pushes to per-side
    :class:`DeepFeatureBatcher` instances.

    This function never raises on cold-cache misses — it IS the cache
    builder. Whether to RUN precompute under ``ctx.require_complete`` is a
    policy decision made by the caller.

    Parameters
    ----------
    sides
        Per-side ``_CacheContext`` map. Sides with ``not ctx.enabled`` are
        skipped silently.
    side_positions
        Per-side ome-zarr position lists. Length must equal
        ``len(seg_positions)`` and pos names must match across sides + seg.
    side_channel_names
        Per-side channel name used to index the image arrays.
    seg_positions
        Cell-segmentation ome-zarr position list.
    extractors
        Map from feature kind to a configured extractor.
    flush_threshold
        Crops accumulated per (side, backbone) before a batched forward fires.

    Raises
    ------
    ValueError
        If any side's positions are misaligned with ``seg_positions`` by
        name or length, or if any ``pos_seg`` is ``None``.
    """
    active = {s: ctx for s, ctx in sides.items() if ctx.enabled}
    if not active:
        return

    # Validate paired-position alignment upfront so we don't write crops to
    # the wrong cache slot if names disagree.
    for side, positions in side_positions.items():
        if side not in active:
            continue
        if len(positions) != len(seg_positions):
            raise ValueError(f"Position count mismatch: {side}={len(positions)} vs seg={len(seg_positions)}")
        for (pos_name, _), (seg_name, _) in zip(positions, seg_positions):
            if pos_name != seg_name:
                raise ValueError(f"Position name mismatch: {side}={pos_name!r} vs seg={seg_name!r}")
    if len(active) > 1:
        names_by_side = {s: [n for n, _ in side_positions[s]] for s in active}
        ref_side, ref_names = next(iter(names_by_side.items()))
        for s, names in names_by_side.items():
            if names != ref_names:
                raise ValueError(f"Position name mismatch between sides: {ref_side}={ref_names!r} vs {s}={names!r}")

    batchers = {s: DeepFeatureBatcher(ctx, extractors, flush_threshold=flush_threshold) for s, ctx in active.items()}

    for pos_idx, (seg_name, pos_seg) in enumerate(seg_positions):
        del seg_name
        if pos_seg is None:
            raise ValueError("cell_segmentation_path is required for deep feature precompute")

        # Metadata-only T from the zarr array shape; no I/O.
        t_count = int(pos_seg.data.shape[0])

        # Walk every active side WITHOUT loading any arrays first. Only
        # materialize cell_seg + image arrays once at least one side has
        # missing slots — warm-cache runs skip the I/O entirely.
        work_per_side: dict[str, dict] = {}
        for side, ctx in active.items():
            pos_name, pos = side_positions[side][pos_idx]
            side_t = int(pos.data.shape[0])
            if side_t != t_count:
                raise ValueError(
                    f"Timepoint count mismatch at {pos_name!r}: {side}.image T={side_t} vs cell_seg T={t_count}"
                )
            needs = batchers[side].pending_kinds_per_t(pos_name, t_count)
            ts_needed = sorted({t for ts in needs.values() for t in ts})
            if ts_needed:
                work_per_side[side] = {
                    "pos_name": pos_name,
                    "pos": pos,
                    "needs": needs,
                    "ts_needed": ts_needed,
                }

        if not work_per_side:
            continue

        cell_seg = np.asarray(pos_seg.data[:, 0])
        for side, work in work_per_side.items():
            ctx = active[side]
            pos = work["pos"]
            pos_name = work["pos_name"]
            needs = work["needs"]
            ts_needed = work["ts_needed"]
            channel_index = pos.get_channel_index(side_channel_names[side])
            image = np.asarray(pos.data[:, channel_index])
            for t in ts_needed:
                crops = build_crops(image[t], cell_seg[t], ctx.patch_size)
                kinds_for_t = [k for k in extractors if t in needs[k]]
                batchers[side].push(pos_name, t, crops, kinds_for_t)

    for batcher in batchers.values():
        batcher.drain()


def resolve_dynaclr_encoder_cfg(config: DictConfig) -> dict[str, Any] | None:
    """Extract and resolve the DynaCLR encoder config as a plain dict (for hashing)."""
    encoder = OmegaConf.select(config, "feature_extractor.dynaclr.encoder", default=None)
    if encoder is None:
        return None
    return OmegaConf.to_container(encoder, resolve=True)
