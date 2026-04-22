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
from omegaconf import DictConfig
from tqdm import tqdm

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.pipeline_cache import (
    flush_manifest,
    fov_gt_cp_features,
    fov_gt_deep_features,
    fov_gt_masks,
    init_cache_context,
    resolve_dynaclr_encoder_cfg,
)


def precompute_gt_artifacts(config: DictConfig) -> None:
    """Build every GT-side artifact toggled on in ``config.build``."""
    from dynacell.evaluation.segmentation import prepare_segmentation_model
    from dynacell.evaluation.utils import DinoV3FeatureExtractor, DynaCLRFeatureExtractor

    if config.io.gt_cache_dir is None:
        raise ValueError("io.gt_cache_dir is required for dynacell precompute-gt")

    build = config.build
    build_any_features = bool(build.cp or build.dinov3 or build.dynaclr)

    if build_any_features and config.io.cell_segmentation_path is None:
        raise ValueError(
            "io.cell_segmentation_path is required when any of build.cp / build.dinov3 / build.dynaclr is true"
        )

    seg_model = prepare_segmentation_model(config) if build.masks else None

    dinov3_model_name = None
    dynaclr_ckpt_path = None
    dynaclr_encoder_cfg = None
    dinov3_feature_extractor = None
    dynaclr_feature_extractor = None

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

    cache_ctx = init_cache_context(
        config,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_path=dynaclr_ckpt_path,
        dynaclr_encoder_cfg=dynaclr_encoder_cfg,
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
                    fov_gt_masks(cache_ctx, pos_name_gt, target, seg_model)
                if build.cp:
                    fov_gt_cp_features(cache_ctx, pos_name_gt, target, cell_segmentation)
                if build.dinov3:
                    fov_gt_deep_features(
                        cache_ctx, pos_name_gt, target, cell_segmentation, dinov3_feature_extractor, "dinov3"
                    )
                if build.dynaclr:
                    fov_gt_deep_features(
                        cache_ctx, pos_name_gt, target, cell_segmentation, dynaclr_feature_extractor, "dynaclr"
                    )

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
