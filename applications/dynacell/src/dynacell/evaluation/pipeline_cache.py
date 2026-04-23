"""Pipeline-level helpers for the GT artifact cache.

Sits between :mod:`dynacell.evaluation.cache` (filesystem layout + raw
read/write) and :mod:`dynacell.evaluation.pipeline` (per-FOV orchestration).
Each per-FOV helper loads target-side artifacts from cache when present,
otherwise computes and writes them — while honoring the per-artifact
``force_recompute`` flags and the ``io.require_complete_cache`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    cp_target_regionprops,
    deep_target_features,
)


@dataclass
class _CacheContext:
    """Per-eval-run cache state passed into FOV-level helpers."""

    paths: CachePaths | None
    manifest: dict[str, Any]
    force: dict[str, bool]
    require_complete: bool
    target_name: str
    spacing: list[float]
    patch_size: int
    dinov3_model_name: str | None = None
    dynaclr_ckpt_sha12: str | None = None
    dynaclr_encoder_sha12: str | None = None
    _manifest_dirty: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        """Whether cache read/write is active for this run."""
        return self.paths is not None

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
        "final_metrics": all_flag or bool(force.final_metrics),
    }


def init_cache_context(
    config: DictConfig,
    *,
    dinov3_model_name: str | None = None,
    dynaclr_ckpt_path: str | None = None,
    dynaclr_encoder_cfg: dict[str, Any] | None = None,
) -> _CacheContext:
    """Open and validate the GT cache for the current run.

    Parameters
    ----------
    config
        Full Hydra config.
    dinov3_model_name
        DINOv3 pretrained name; ``None`` when feature metrics are disabled.
    dynaclr_ckpt_path
        DynaCLR checkpoint path; ``None`` when feature metrics are disabled.
    dynaclr_encoder_cfg
        DynaCLR encoder config (resolved dict); ``None`` when disabled.
    """
    io = config.io
    force = _resolve_force(config.force_recompute)
    require_complete = bool(io.require_complete_cache)

    spacing = list(config.pixel_metrics.spacing)
    patch_size = int(config.feature_metrics.patch_size)

    if io.gt_cache_dir is None:
        if require_complete:
            raise ValueError("io.require_complete_cache=true requires io.gt_cache_dir to be set")
        dynaclr_ckpt_sha12 = ckpt_sha256_12(dynaclr_ckpt_path) if dynaclr_ckpt_path is not None else None
        dynaclr_encoder_sha12 = (
            encoder_config_sha256_12(dynaclr_encoder_cfg) if dynaclr_encoder_cfg is not None else None
        )
        return _CacheContext(
            paths=None,
            manifest={},
            force=force,
            require_complete=False,
            target_name=config.target_name,
            spacing=spacing,
            patch_size=patch_size,
            dinov3_model_name=dinov3_model_name,
            dynaclr_ckpt_sha12=dynaclr_ckpt_sha12,
            dynaclr_encoder_sha12=dynaclr_encoder_sha12,
        )

    paths = cache_paths(Path(io.gt_cache_dir))
    manifest = load_manifest(paths)

    cell_seg_path = str(io.cell_segmentation_path) if io.cell_segmentation_path is not None else None
    check_cache_identity(
        manifest,
        gt_plate_path=str(io.gt_path),
        gt_channel_name=str(io.gt_channel_name),
        cell_segmentation_path=cell_seg_path,
    )
    seed_cache_identity(
        manifest,
        gt_plate_path=str(io.gt_path),
        gt_channel_name=str(io.gt_channel_name),
        cell_segmentation_path=cell_seg_path,
    )

    dynaclr_ckpt_sha12 = ckpt_sha256_12(dynaclr_ckpt_path) if dynaclr_ckpt_path is not None else None
    dynaclr_encoder_sha12 = encoder_config_sha256_12(dynaclr_encoder_cfg) if dynaclr_encoder_cfg is not None else None

    ctx = _CacheContext(
        paths=paths,
        manifest=manifest,
        force=force,
        require_complete=require_complete,
        target_name=config.target_name,
        spacing=spacing,
        patch_size=patch_size,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_sha12=dynaclr_ckpt_sha12,
        dynaclr_encoder_sha12=dynaclr_encoder_sha12,
    )
    _validate_artifact_params(ctx)
    return ctx


def _validate_artifact_params(ctx: _CacheContext) -> None:
    """Raise if any existing per-artifact manifest entry disagrees with ctx params."""
    artifacts = ctx.manifest.get("artifacts", {})

    masks_section = artifacts.get("organelle_masks", {})
    check_artifact_params(
        masks_section.get(ctx.target_name),
        {"target_name": ctx.target_name},
        artifact_label=f"organelle_masks[{ctx.target_name}]",
    )
    check_artifact_params(
        artifacts.get("cp_features"),
        {"spacing": ctx.spacing},
        artifact_label="cp_features",
        numeric_keys=("spacing",),
    )
    if ctx.dinov3_model_name is not None:
        dinov3_section = artifacts.get("dinov3_features", {})
        check_artifact_params(
            dinov3_section.get(feature_slug(ctx.dinov3_model_name)),
            {"model_name": ctx.dinov3_model_name, "patch_size": ctx.patch_size},
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
            },
            artifact_label=f"dynaclr_features[{ctx.dynaclr_ckpt_sha12}]",
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


def fov_gt_masks(
    ctx: _CacheContext,
    pos_name: str,
    target_arr: np.ndarray,
    seg_model,
) -> np.ndarray:
    """Return a ``(T, D, H, W)`` bool mask stack, loading from cache or computing+writing.

    When caching is disabled (``ctx.enabled == False``), the masks are
    computed fresh from *target_arr* without any cache interaction.
    """
    from dynacell.evaluation.segmentation import segment

    t_count = target_arr.shape[0]

    if ctx.enabled and not ctx.force["gt_masks"]:
        cached = read_mask(ctx.paths, ctx.target_name, pos_name)
        if cached is not None:
            if cached.shape[0] != t_count:
                raise StaleCacheError(
                    f"Cached mask timepoint count mismatch for {pos_name}: "
                    f"cached={cached.shape[0]} vs current={t_count}"
                )
            return cached
        _raise_if_require_complete(ctx, f"organelle_masks[{ctx.target_name}]", pos_name)

    masks = np.stack(
        [np.asarray(segment(target_arr[t], ctx.target_name, seg_model=seg_model)).astype(bool) for t in range(t_count)]
    )

    if ctx.enabled:
        write_mask(ctx.paths, ctx.target_name, pos_name, masks)
        _update_manifest_entry(
            ctx.manifest,
            ["organelle_masks", ctx.target_name],
            {
                "path": f"organelle_masks/{ctx.target_name}.zarr",
                "target_name": ctx.target_name,
                "built_at": built_at_now(),
            },
        )
        _add_position(ctx.manifest, ["organelle_masks", ctx.target_name], pos_name)
        ctx.mark_manifest_dirty()

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

    Opens the backing zarr group once per FOV (not per timepoint) and funnels
    every read/write through it. Returns ``(per_t_features, manifest_updated)``.
    ``compute_fn`` is called as ``compute_fn(t)`` on misses and must return a
    2-D ``(n_cells_t, feature_dim)`` array.
    """
    per_t: list[np.ndarray] = []
    if not ctx.enabled:
        for t in range(t_count):
            per_t.append(np.asarray(compute_fn(t)))
        return per_t, False

    manifest_updated = False
    with open_features_group(ctx.paths, kind, mode="a", **cache_kwargs) as group:
        for t in range(t_count):
            feats = None
            if not ctx.force[force_key]:
                feats = read_features_from_group(group, pos_name, t)
                if feats is None:
                    _raise_if_require_complete(ctx, artifact_label, pos_name, t)
            if feats is None:
                feats = np.asarray(compute_fn(t))
                write_features_to_group(group, pos_name, t, feats)
                manifest_updated = True
            per_t.append(feats)
    return per_t, manifest_updated


def fov_gt_cp_features(
    ctx: _CacheContext,
    pos_name: str,
    target_arr: np.ndarray,
    cell_segmentation_arr: np.ndarray,
) -> list[np.ndarray]:
    """Return target-side CP regionprops per timepoint, loading from cache or computing+writing.

    Result is a list of ``T`` arrays, each shape ``(n_cells_t, n_props_raw)``.
    """
    per_t, manifest_updated = _load_or_compute_feature_timepoints(
        ctx,
        kind="cp",
        pos_name=pos_name,
        t_count=target_arr.shape[0],
        force_key="gt_cp",
        artifact_label="cp_features",
        cache_kwargs={},
        compute_fn=lambda t: cp_target_regionprops(target_arr[t], cell_segmentation_arr[t], ctx.spacing),
    )

    if ctx.enabled and manifest_updated:
        _update_manifest_entry(
            ctx.manifest,
            ["cp_features"],
            {"path": "features/cp.zarr", "spacing": ctx.spacing, "built_at": built_at_now()},
        )
        _add_position(ctx.manifest, ["cp_features"], pos_name)
        ctx.mark_manifest_dirty()

    return per_t


def fov_gt_deep_features(
    ctx: _CacheContext,
    pos_name: str,
    target_arr: np.ndarray,
    cell_segmentation_arr: np.ndarray,
    feature_extractor,
    kind: FeatureKind,
) -> list[np.ndarray]:
    """Return target-side deep embeddings per timepoint for one feature family.

    ``kind`` is ``"dinov3"`` or ``"dynaclr"``. The cache key (model name or
    checkpoint hash) is pulled from *ctx*.
    """
    if kind == "dinov3":
        force_key = "gt_dinov3"
        artifact_label = f"dinov3_features[{ctx.dinov3_model_name}]"
        cache_kwargs = {"model_name": ctx.dinov3_model_name}
        slug = feature_slug(ctx.dinov3_model_name)
        manifest_keys = ["dinov3_features", slug]
        entry = {
            "path": f"features/dinov3/{slug}.zarr",
            "model_name": ctx.dinov3_model_name,
            "patch_size": ctx.patch_size,
            "built_at": built_at_now(),
        }
    elif kind == "dynaclr":
        force_key = "gt_dynaclr"
        artifact_label = f"dynaclr_features[{ctx.dynaclr_ckpt_sha12}]"
        cache_kwargs = {"ckpt_sha12": ctx.dynaclr_ckpt_sha12}
        manifest_keys = ["dynaclr_features", ctx.dynaclr_ckpt_sha12]
        entry = {
            "path": f"features/dynaclr/{ctx.dynaclr_ckpt_sha12}.zarr",
            "checkpoint_sha256_12": ctx.dynaclr_ckpt_sha12,
            "encoder_config_sha256_12": ctx.dynaclr_encoder_sha12,
            "patch_size": ctx.patch_size,
            "built_at": built_at_now(),
        }
    else:
        raise ValueError(f"Unknown deep-feature kind: {kind!r}")

    per_t, manifest_updated = _load_or_compute_feature_timepoints(
        ctx,
        kind=kind,
        pos_name=pos_name,
        t_count=target_arr.shape[0],
        force_key=force_key,
        artifact_label=artifact_label,
        cache_kwargs=cache_kwargs,
        compute_fn=lambda t: deep_target_features(
            target_arr[t], cell_segmentation_arr[t], feature_extractor, ctx.patch_size
        ),
    )

    if ctx.enabled and manifest_updated:
        _update_manifest_entry(ctx.manifest, manifest_keys, entry)
        _add_position(ctx.manifest, manifest_keys, pos_name)
        ctx.mark_manifest_dirty()

    return per_t


def flush_manifest(ctx: _CacheContext) -> None:
    """Persist the manifest to disk if it has been mutated since last flush."""
    if ctx.enabled and ctx.consume_manifest_dirty():
        save_manifest(ctx.paths, ctx.manifest)


def resolve_dynaclr_encoder_cfg(config: DictConfig) -> dict[str, Any] | None:
    """Extract and resolve the DynaCLR encoder config as a plain dict (for hashing)."""
    encoder = OmegaConf.select(config, "feature_extractor.dynaclr.encoder", default=None)
    if encoder is None:
        return None
    return OmegaConf.to_container(encoder, resolve=True)
