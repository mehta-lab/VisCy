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
from dynacell.evaluation.metrics import build_crops
from dynacell.evaluation.pipeline_cache import (
    DeepFeatureBatcher,
    flush_manifest,
    fov_cp_features,
    fov_masks,
    init_cache_context,
    resolve_dynaclr_encoder_cfg,
)


def precompute_gt_artifacts(config: DictConfig) -> None:
    """Build every GT-side artifact toggled on in ``config.build``."""
    from dynacell.evaluation.runtime import apply_thread_budget, resolve_runtime
    from dynacell.evaluation.segmentation import prepare_segmentation_model
    from dynacell.evaluation.utils import CellDinoFeatureExtractor, DinoV3FeatureExtractor, DynaCLRFeatureExtractor

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

    seg_model = prepare_segmentation_model(config) if build.masks else None

    dinov3_model_name = None
    dynaclr_ckpt_path = None
    dynaclr_encoder_cfg = None
    celldino_weights_path = None
    dinov3_feature_extractor = None
    dynaclr_feature_extractor = None
    celldino_feature_extractor = None

    if build.dinov3:
        dinov3_model_name = config.feature_extractor.dinov3.pretrained_model_name
        dinov3_feature_extractor = DinoV3FeatureExtractor(dinov3_model_name)
    if build.dynaclr:
        dynaclr_config = config.feature_extractor.dynaclr
        dynaclr_ckpt_path = str(dynaclr_config.checkpoint)
        dynaclr_encoder_cfg = resolve_dynaclr_encoder_cfg(config)
        dynaclr_feature_extractor = DynaCLRFeatureExtractor(
            checkpoint=dynaclr_config.checkpoint,
            encoder_config=dynaclr_encoder_cfg,
        )
    if build.celldino:
        celldino_cfg = config.feature_extractor.celldino
        if celldino_cfg.weights_path is None:
            raise ValueError("feature_extractor.celldino.weights_path is required when build.celldino=true")
        celldino_weights_path = str(celldino_cfg.weights_path)
        celldino_feature_extractor = CellDinoFeatureExtractor(
            weights_path=celldino_weights_path,
            img_size=int(celldino_cfg.img_size),
            patch_size=int(celldino_cfg.patch_size),
        )

    cache_ctx = init_cache_context(
        config,
        side="gt",
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_path=dynaclr_ckpt_path,
        dynaclr_encoder_cfg=dynaclr_encoder_cfg,
        celldino_weights_path=celldino_weights_path,
    )

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
                        crops = build_crops(target[t], cell_segmentation[t], cache_ctx.patch_size)
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
