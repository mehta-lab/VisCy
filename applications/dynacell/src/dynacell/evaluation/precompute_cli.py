"""CLI entry point for pre-filling the GT artifact cache.

Runs the same load-or-compute helpers that ``evaluate_predictions`` uses,
but without the evaluation loop — iterates GT positions and writes any
missing artifacts to ``io.gt_cache_dir`` so that subsequent
``dynacell evaluate`` runs hit the cache.

Invoked as ``dynacell precompute-gt ...`` via the CLI router in
:mod:`dynacell.__main__`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.focus import build_focus_slabs, read_focus_slab_config, write_focus_slice_metadata
from dynacell.evaluation.metrics import build_crops
from dynacell.evaluation.model_loader import LoadFlags, init_gt_cache_context, load_eval_models
from dynacell.evaluation.pipeline_cache import (
    DeepFeatureBatcher,
    flush_manifest,
    fov_cp_features,
    fov_masks,
)


def _focus_slabs(config: DictConfig, pos_gt, t_count: int) -> list[slice | None]:
    """Per-timepoint in-focus slabs for a GT position, or ``[None]*t`` when disabled."""
    slab_cfg = read_focus_slab_config(config)
    if slab_cfg is None:
        return [None] * t_count
    return build_focus_slabs(pos_gt, channel_name=slab_cfg.channel_name, halfwidth=slab_cfg.halfwidth, t_count=t_count)


def precompute_gt_artifacts(config: DictConfig) -> None:
    """Build every GT-side artifact toggled on in ``config.build``."""
    from dynacell.evaluation.runtime import apply_thread_budget, resolve_runtime

    if config.io.gt_cache_dir is None:
        raise ValueError("io.gt_cache_dir is required for dynacell precompute-gt")

    # Precompute is single-process by design (DeepFeatureBatcher accumulates
    # state across FOVs). Apply the thread cap but raise if the user requested
    # FOV-level parallelism here — that belongs to evaluate_predictions only.
    runtime = resolve_runtime(config)
    if runtime.executor != "serial" or runtime.fov_workers != 1:
        raise ValueError(
            "dynacell precompute-gt does not support FOV-level parallelism. "
            f"Got runtime.executor={runtime.executor!r}, "
            f"runtime.fov_workers={runtime.fov_workers}. "
            "Set runtime.executor='serial' and runtime.fov_workers=1 (or omit)."
        )
    apply_thread_budget(runtime.threads_per_worker)

    build = config.build
    build_any_features = bool(build.cp or build.dinov3 or build.dynaclr or build.celldino)

    if build_any_features and config.io.cell_segmentation_path is None:
        raise ValueError(
            "io.cell_segmentation_path is required when any of "
            "build.cp / build.dinov3 / build.dynaclr / build.celldino is true"
        )

    # Stricter celldino gate than evaluate_predictions: precompute-gt
    # requires an explicit weights path when build.celldino=true (the
    # eval path soft-skips a null weights_path; here it would silently
    # produce no cache, so we surface it as an error).
    if build.celldino and config.feature_extractor.celldino.weights_path is None:
        raise ValueError("feature_extractor.celldino.weights_path is required when build.celldino=true")

    # Focus metadata is written directly to the GT store (zattrs), not the
    # artifact cache, and needs none of the models below — so do it first. The
    # phase channel must exist in io.gt_path; packed .ozx stores are read-only
    # (run against the unpacked OME-Zarr and repackage).
    if OmegaConf.select(config, "build.focus", default=False):
        focus_channel = str(OmegaConf.select(config, "focus.channel_name", default="Phase3D"))
        pixel_size = OmegaConf.select(config, "focus.pixel_size", default=None)
        if pixel_size is None:
            pixel_size = float(config.pixel_metrics.spacing[-1])
        print(f"Writing focus_slice to {config.io.gt_path} (channel={focus_channel})")
        stats = write_focus_slice_metadata(
            str(config.io.gt_path),
            channel_name=focus_channel,
            na_det=float(OmegaConf.select(config, "focus.na_det", default=1.35)),
            lambda_ill=float(OmegaConf.select(config, "focus.lambda_ill", default=0.450)),
            pixel_size=float(pixel_size),
            device=str(OmegaConf.select(config, "focus.device", default="cpu")),
        )
        print(f"  focus_slice[{focus_channel}].dataset_statistics = {stats}")

    models = load_eval_models(
        config,
        flags=LoadFlags(
            masks=bool(build.masks),
            dinov3=bool(build.dinov3),
            dynaclr=bool(build.dynaclr),
            celldino=bool(build.celldino),
        ),
    )
    seg_model = models.seg_model
    dinov3_feature_extractor = models.dinov3
    dynaclr_feature_extractor = models.dynaclr
    celldino_feature_extractor = models.celldino

    cache_ctx = init_gt_cache_context(config, models)

    gt_path = Path(config.io.gt_path)
    seg_path = Path(config.io.cell_segmentation_path) if config.io.cell_segmentation_path is not None else None

    with open_ome_zarr(gt_path, mode="r") as gt_plate:
        gt_positions = list(gt_plate.positions())
        seg_plate = open_ome_zarr(seg_path, mode="r") if seg_path is not None else None
        try:
            if seg_plate is not None:
                seg_positions = list(seg_plate.positions())
                if len(seg_positions) != len(gt_positions):
                    raise ValueError(f"Position count mismatch: gt={len(gt_positions)}, seg={len(seg_positions)}")
            else:
                seg_positions = [(name, None) for name, _ in gt_positions]

            limit = getattr(config, "limit_positions", None)
            if limit is not None:
                gt_positions = gt_positions[:limit]
                seg_positions = seg_positions[:limit]

            deep_extractors = {}
            if build.dinov3:
                deep_extractors["dinov3"] = dinov3_feature_extractor
            if build.dynaclr:
                deep_extractors["dynaclr"] = dynaclr_feature_extractor
            if build.celldino:
                deep_extractors["celldino"] = celldino_feature_extractor

            flush_threshold = int(OmegaConf.select(config, "feature_metrics.deep_feature_batch_threshold", default=256))
            batcher = (
                DeepFeatureBatcher(cache_ctx, deep_extractors, flush_threshold=flush_threshold)
                if deep_extractors and cache_ctx.enabled
                else None
            )

            for (pos_name_gt, pos_gt), (pos_name_seg, pos_seg) in tqdm(
                zip(gt_positions, seg_positions),
                total=len(gt_positions),
                desc="Precomputing GT artifacts",
            ):
                if seg_plate is not None and pos_name_gt != pos_name_seg:
                    raise ValueError(f"Position name mismatch: gt={pos_name_gt!r}, seg={pos_name_seg!r}")

                gt_channel_index = pos_gt.get_channel_index(config.io.gt_channel_name)
                target = np.asarray(pos_gt.data[:, gt_channel_index])
                cell_segmentation = np.asarray(pos_seg.data[:, 0]) if pos_seg is not None else None
                z_slabs = _focus_slabs(config, pos_gt, target.shape[0])

                if build.masks:
                    fov_masks(cache_ctx, pos_name_gt, target, seg_model)
                if build.cp:
                    fov_cp_features(cache_ctx, pos_name_gt, target, cell_segmentation)

                # Deep features stream in-loop via the batcher — no second
                # plate read. The batcher's pending_kinds_per_t reflects
                # already-cached slots so warm-cache positions skip work.
                if batcher is not None and cell_segmentation is not None:
                    t_count = target.shape[0]
                    needs = batcher.pending_kinds_per_t(pos_name_gt, t_count)
                    for t in range(t_count):
                        kinds_for_t = [k for k in deep_extractors if t in needs[k]]
                        if not kinds_for_t:
                            continue
                        crops = build_crops(target[t], cell_segmentation[t], cache_ctx.patch_size, z_slab=z_slabs[t])
                        batcher.push(pos_name_gt, t, crops, kinds_for_t)

                flush_manifest(cache_ctx)

            if batcher is not None:
                batcher.drain()
                flush_manifest(cache_ctx)
        finally:
            if seg_plate is not None:
                seg_plate.close()


@hydra.main(version_base="1.2", config_path="_configs", config_name="precompute")
def precompute_gt(config: DictConfig) -> None:
    """Hydra entry point for ``dynacell precompute-gt``."""
    apply_dataset_ref(config)
    precompute_gt_artifacts(config)


if __name__ == "__main__":
    precompute_gt()
