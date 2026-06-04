"""Shared model-loading entry point for the dynacell eval pipeline.

A single source of truth for instantiating the segmenter + deep-feature
extractors used by ``evaluate_predictions`` and the precompute-gt CLI.
Lifting the load out of the pipeline lets the grouped multi-condition
driver build models once and reuse them across conditions that differ
only in I/O paths; the precompute-gt CLI shares the same code path with
its own per-extractor gates (``build.masks`` / ``build.dinov3`` / ...).

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
        when ``io.require_complete_cache=true`` short-circuits the load
        or when ``LoadFlags.masks=False``.
    dinov3, dynaclr, celldino : Any | None
        Deep feature extractor instances. Each is None when its
        ``LoadFlags`` gate is False, or (for celldino) when its
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
    dinov3_preprocess_version: str | None = None
    dynaclr_preprocess_version: str | None = None
    celldino_preprocess_version: str | None = None


@dataclass(frozen=True)
class LoadFlags:
    """Per-model gate for :func:`load_eval_models`.

    Lets ``precompute-gt`` (per-extractor flags ``build.masks`` /
    ``build.dinov3`` / ``build.dynaclr`` / ``build.celldino``) and
    ``evaluate-predictions`` (all extractors gated together by
    ``compute_feature_metrics``) share one loader.
    """

    masks: bool = True
    dinov3: bool = False
    dynaclr: bool = False
    celldino: bool = False

    @classmethod
    def for_evaluate(cls, config: DictConfig) -> LoadFlags:
        """Flags matching the ``evaluate_predictions`` call path.

        Masks always on (``prepare_segmentation_model`` handles its own
        ``require_complete_cache`` short-circuit). Extractors gated as a
        group by ``config.compute_feature_metrics``; celldino additionally
        soft-skips inside the loader when its ``weights_path`` is null.
        """
        ext_on = bool(config.compute_feature_metrics)
        return cls(masks=True, dinov3=ext_on, dynaclr=ext_on, celldino=ext_on)


def load_eval_models(config: DictConfig, *, flags: LoadFlags | None = None) -> EvalModels:
    """Instantiate the segmenter + deep feature extractors from ``config``.

    No GPU serialization wrapping — callers (worker setup) wrap with
    ``gpu_serialization_lock`` as needed. The parent of
    ``evaluate_predictions`` calls this unwrapped today, so this entry
    preserves that exact pattern.

    Parameters
    ----------
    config : DictConfig
        Resolved eval config. Reads ``target_name``,
        ``feature_extractor.{dinov3,dynaclr,celldino}``,
        and (transitively) ``io.require_complete_cache``.
    flags : LoadFlags, optional
        Per-model gates. Defaults to :meth:`LoadFlags.for_evaluate`,
        which preserves the historical ``evaluate_predictions``
        behavior. Pass a custom :class:`LoadFlags` to load a subset
        (e.g. for precompute-gt where ``build.dinov3`` / etc. are
        toggled independently).

    Returns
    -------
    EvalModels
        All model handles + identity tags. Each slot is None when its
        flag is off; celldino additionally soft-skips when its
        ``weights_path`` is null even with the flag on.
    """
    from dynacell.evaluation.pipeline_cache import resolve_dynaclr_encoder_cfg
    from dynacell.evaluation.segmentation import prepare_segmentation_model
    from dynacell.evaluation.utils import (
        CellDinoFeatureExtractor,
        DinoV3FeatureExtractor,
        DynaCLRFeatureExtractor,
    )

    if flags is None:
        flags = LoadFlags.for_evaluate(config)

    seg_model = prepare_segmentation_model(config) if flags.masks else None

    dinov3_model_name: str | None = None
    dynaclr_ckpt_path: str | None = None
    dynaclr_encoder_cfg: dict[str, Any] | None = None
    celldino_weights_path: str | None = None
    dinov3 = None
    dynaclr = None
    celldino = None

    if flags.dinov3:
        dinov3_model_name = config.feature_extractor.dinov3.pretrained_model_name
        dinov3 = DinoV3FeatureExtractor(dinov3_model_name)
    if flags.dynaclr:
        dynaclr_config = config.feature_extractor.dynaclr
        dynaclr_ckpt_path = str(dynaclr_config.checkpoint)
        dynaclr_encoder_cfg = resolve_dynaclr_encoder_cfg(config)
        dynaclr = DynaCLRFeatureExtractor(
            checkpoint=dynaclr_config.checkpoint,
            encoder_config=dynaclr_encoder_cfg,
        )
    if flags.celldino:
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
        dinov3_preprocess_version=DinoV3FeatureExtractor.PREPROCESS_VERSION if dinov3 is not None else None,
        dynaclr_preprocess_version=DynaCLRFeatureExtractor.PREPROCESS_VERSION if dynaclr is not None else None,
        celldino_preprocess_version=CellDinoFeatureExtractor.PREPROCESS_VERSION if celldino is not None else None,
    )


def _identity_kwargs(models: EvalModels) -> dict[str, Any]:
    """Identity tags forwarded into ``init_cache_context``."""
    return {
        "dinov3_model_name": models.dinov3_model_name,
        "dynaclr_ckpt_path": models.dynaclr_ckpt_path,
        "dynaclr_encoder_cfg": models.dynaclr_encoder_cfg,
        "celldino_weights_path": models.celldino_weights_path,
        "dinov3_preprocess_version": models.dinov3_preprocess_version,
        "dynaclr_preprocess_version": models.dynaclr_preprocess_version,
        "celldino_preprocess_version": models.celldino_preprocess_version,
    }


def init_cache_contexts(config: DictConfig, models: EvalModels) -> tuple[Any, Any]:
    """Build ``(gt_ctx, pred_ctx)`` cache contexts using ``models``'s identity tags.

    Replaces the four-call-site duplication in ``evaluate_predictions``
    and ``_worker_setup`` where the same six kwargs flowed into
    ``init_cache_context(..., side="gt")`` then again into
    ``init_cache_context(..., side="pred")``.
    """
    from dynacell.evaluation.pipeline_cache import init_cache_context

    kwargs = _identity_kwargs(models)
    gt_ctx = init_cache_context(config, side="gt", **kwargs)
    pred_ctx = init_cache_context(config, side="pred", **kwargs)
    return gt_ctx, pred_ctx


def init_gt_cache_context(config: DictConfig, models: EvalModels) -> Any:
    """Build only the GT cache context — used by ``precompute-gt`` which is GT-only."""
    from dynacell.evaluation.pipeline_cache import init_cache_context

    return init_cache_context(config, side="gt", **_identity_kwargs(models))
