"""Shared model-loading entry point for the dynacell eval pipeline.

A single source of truth for instantiating the segmenter + deep-feature
extractors used by ``evaluate_predictions``. Lifting the load out of the
pipeline lets the grouped multi-condition driver build models once and
reuse them across conditions that differ only in I/O paths.

The dataclass carries the **identity tags** (model name, checkpoint sha
sources) that ``init_cache_context`` needs, so callers don't have to
recompute them per condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class EvalModels:
    """Bundle of pre-loaded segmenter + feature extractors with their identity tags.

    Attributes
    ----------
    seg_model : Any | None
        SuperModel returned by ``prepare_segmentation_model``. May be None
        when ``io.require_complete_cache=true`` short-circuits the load.
    dinov3, dynaclr, celldino : Any | None
        Deep feature extractor instances. All None when
        ``compute_feature_metrics=false``. ``celldino`` is None when its
        ``weights_path`` is unset.
    dinov3_model_name, dynaclr_ckpt_path, dynaclr_encoder_cfg,
    celldino_weights_path : str | dict | None
        Identity tags consumed by ``init_cache_context``. Stay None when
        the corresponding extractor was not loaded.
    """

    seg_model: Any | None
    dinov3: Any | None
    dynaclr: Any | None
    celldino: Any | None
    dinov3_model_name: str | None
    dynaclr_ckpt_path: str | None
    dynaclr_encoder_cfg: dict[str, Any] | None
    celldino_weights_path: str | None


def load_eval_models(config: DictConfig) -> EvalModels:
    """Instantiate the segmenter + deep feature extractors from ``config``.

    No GPU serialization wrapping — callers (worker setup) wrap with
    ``gpu_serialization_lock`` as needed. The parent of
    ``evaluate_predictions`` calls this unwrapped today, so this entry
    preserves that exact pattern.

    Parameters
    ----------
    config : DictConfig
        Resolved eval config. Reads ``target_name``,
        ``compute_feature_metrics``, ``feature_extractor.{dinov3,dynaclr,celldino}``,
        and (transitively) ``io.require_complete_cache``.

    Returns
    -------
    EvalModels
        All model handles + identity tags. ``seg_model`` is None under
        ``require_complete_cache=true``; extractors are None when
        ``compute_feature_metrics=false`` (and celldino additionally
        when its ``weights_path`` is null).
    """
    from dynacell.evaluation.pipeline_cache import resolve_dynaclr_encoder_cfg
    from dynacell.evaluation.segmentation import prepare_segmentation_model
    from dynacell.evaluation.utils import (
        CellDinoFeatureExtractor,
        DinoV3FeatureExtractor,
        DynaCLRFeatureExtractor,
    )

    seg_model = prepare_segmentation_model(config)

    dinov3_model_name: str | None = None
    dynaclr_ckpt_path: str | None = None
    dynaclr_encoder_cfg: dict[str, Any] | None = None
    celldino_weights_path: str | None = None
    dinov3 = None
    dynaclr = None
    celldino = None

    if config.compute_feature_metrics:
        dinov3_model_name = config.feature_extractor.dinov3.pretrained_model_name
        dinov3 = DinoV3FeatureExtractor(dinov3_model_name)
        dynaclr_config = config.feature_extractor.dynaclr
        dynaclr_ckpt_path = str(dynaclr_config.checkpoint)
        dynaclr_encoder_cfg = resolve_dynaclr_encoder_cfg(config)
        dynaclr = DynaCLRFeatureExtractor(
            checkpoint=dynaclr_config.checkpoint,
            encoder_config=dynaclr_encoder_cfg,
        )
        celldino_cfg = config.feature_extractor.celldino
        if celldino_cfg.weights_path is not None:
            celldino_weights_path = str(celldino_cfg.weights_path)
            celldino = CellDinoFeatureExtractor(
                weights_path=celldino_weights_path,
                img_size=int(celldino_cfg.img_size),
                patch_size=int(celldino_cfg.patch_size),
            )

    return EvalModels(
        seg_model=seg_model,
        dinov3=dinov3,
        dynaclr=dynaclr,
        celldino=celldino,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_path=dynaclr_ckpt_path,
        dynaclr_encoder_cfg=dynaclr_encoder_cfg,
        celldino_weights_path=celldino_weights_path,
    )
