"""CLI entry point for pre-filling the GT artifact cache.

Runs the same load-or-compute helpers that ``evaluate_predictions`` uses,
but without the evaluation loop — iterates GT positions and writes any
missing artifacts to ``io.gt_cache_dir`` so that subsequent
``dynacell evaluate`` runs hit the cache.

Invoked as ``dynacell precompute-gt ...`` via the CLI router in
:mod:`dynacell.__main__`.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import get_args

import hydra
import numpy as np
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.cache import FeatureKind
from dynacell.evaluation.focus import (
    build_focus_slabs,
    read_focus_compute_config,
    read_focus_slab_config,
    resolve_focus_instance_planes,
    slab_mip,
    write_focus_slice_metadata,
)
from dynacell.evaluation.metrics import build_crops
from dynacell.evaluation.model_loader import LoadFlags, init_gt_cache_context, load_eval_models
from dynacell.evaluation.pipeline_cache import (
    DeepFeatureBatcher,
    cpdino_infer_kwargs,
    flush_manifest,
    fov_cp_features,
    fov_cpdino_nucleus_instances,
    fov_cpdino_whole_cell_instances,
    fov_masks,
)
from dynacell.evaluation.segmentation_cpdino import segment_cpdino_instances
from dynacell.evaluation.segmentation_whole_cell import slice_index

# Deep-feature backbones in FeatureKind order; "cp" is regionprops, not an extractor.
_DEEP_KINDS: tuple[FeatureKind, ...] = tuple(k for k in get_args(FeatureKind) if k != "cp")


def _focus_slabs(config: DictConfig, pos_gt, pos_name: str, t_count: int) -> list[slice | None]:
    """Per-timepoint in-focus slabs for a GT position, or ``[None]*t`` when disabled."""
    slab_cfg = read_focus_slab_config(config)
    if slab_cfg is None:
        return [None] * t_count
    fc = read_focus_compute_config(config, channel_name=slab_cfg.channel_name)
    return build_focus_slabs(
        pos_gt,
        halfwidth=slab_cfg.halfwidth,
        t_count=t_count,
        compute=fc,
        cache_dir=config.io.gt_cache_dir,
        pos_name=pos_name,
    )


def _gt_instance_slices(config: DictConfig, pos_gt, target: np.ndarray, pos_name: str, *, nucleus_vol=None):
    """Return ``(target_cells, z_idx, is3d, slab_hw)`` for the GT instance build.

    Mirrors the eval instance dispatch (``focus.resolve_focus_instance_planes`` +
    ``slab_mip``): 3-D uses the full ``(T, Z, Y, X)`` volume; 2-D picks one plane per
    timepoint (in-focus anchor or ``frac``) and returns the ``(T, Y, X)`` +/-slab MIP.
    ``nucleus_vol`` (``(T, Z, Y, X)``) is the nucleus channel for the ``nucleus_area``
    anchor (the target for nucleus, the GT-nuclei source for whole-cell).
    """
    t_count = target.shape[0]
    is3d = OmegaConf.select(config, "segmentation.dimension", default="2d") == "3d"
    if is3d:
        return target, None, True, 0
    sel = OmegaConf.select(config, "segmentation.slice_selection", default="frac")
    slab_hw = int(OmegaConf.select(config, "segmentation.focus_slab_halfwidth", default=0)) if sel == "focus" else 0
    if sel == "focus":
        z_idx = resolve_focus_instance_planes(
            config, t_count=t_count, pos_gt=pos_gt, pos_name=pos_name, nucleus_vol=nucleus_vol
        )
    else:
        frac = float(OmegaConf.select(config, "segmentation.slice_fraction", default=0.30))
        z_idx = [slice_index(target[t], selection=sel, fraction=frac) for t in range(t_count)]
    target_cells = slab_mip(target, z_idx, slab_hw)
    return target_cells, z_idx, False, slab_hw


def _build_gt_instances(config, cache_ctx, seg_model, pos_gt, pos_name, target, nuclei_plate) -> None:
    """Prewarm the GT instance-label cache for the configured cpdino backend/target.

    Nucleus target → cpdino on the GT nucleus channel. Membrane target → cpdino on the GT
    membrane channel + carve the GT-nucleus footprint (cpdino nucleus instances; nuclei
    read from *nuclei_plate* when the GT nuclei live in a separate store, else *pos_gt*).
    Writes the same ``instance_masks/<target>__cpdino.zarr`` identity the eval reads.
    """
    backend = OmegaConf.select(config, "segmentation.backend", default="supermodel")
    if backend != "cpdino":
        raise ValueError(f"build.instances currently supports segmentation.backend='cpdino' only; got {backend!r}")
    target_name = config.target_name
    # Read GT nuclei up front: whole-cell always needs them (carve seeds), and the
    # nucleus_area focus anchor uses the nucleus channel as its focus signal (the target
    # itself for nucleus). Passing nucleus_vol keeps the plane identical to the eval.
    nuclei = None
    if target_name == "membrane":
        nuclei_channel = OmegaConf.select(config, "segmentation.nuclei_channel_name", default=None)
        if nuclei_channel is None:
            raise ValueError(
                "build.instances for target_name='membrane' requires "
                "segmentation.nuclei_channel_name (the GT-nucleus channel carved into "
                "whole-cell seeds). It is unset — set it, or run precompute-gt without "
                "build.instances if you only need GT features."
            )
        nuclei_src = nuclei_plate[pos_name] if nuclei_plate is not None else pos_gt
        nuclei = np.asarray(nuclei_src.data[:, nuclei_src.get_channel_index(nuclei_channel)])
    nucleus_vol = target if target_name == "nucleus" else nuclei
    target_cells, z_idx, is3d, slab_hw = _gt_instance_slices(config, pos_gt, target, pos_name, nucleus_vol=nucleus_vol)
    if target_name == "nucleus":
        fov_cpdino_nucleus_instances(cache_ctx, pos_name, target_cells, seg_model)
        return
    if target_name == "membrane":
        nuclei_cells = nuclei if is3d else slab_mip(nuclei, z_idx, slab_hw)
        spacing = tuple(config.pixel_metrics.spacing) if is3d else tuple(config.pixel_metrics.spacing[-2:])
        infer = cpdino_infer_kwargs(cache_ctx)
        seed_stack = np.stack(
            [
                segment_cpdino_instances(nuclei_cells[t], spacing, seg_model, do_3d=is3d, **infer)
                for t in range(len(nuclei_cells))
            ]
        )
        fov_cpdino_whole_cell_instances(cache_ctx, pos_name, target_cells, seed_stack, seg_model)
        return
    raise ValueError(f"build.instances (cpdino) requires target_name in {{nucleus, membrane}}; got {target_name!r}")


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
    build_any_features = bool(build.cp or build.dinov3 or build.dynaclr or build.celldino or build.morphem)

    if build_any_features and config.io.cell_segmentation_path is None:
        raise ValueError(
            "io.cell_segmentation_path is required when any of "
            "build.cp / build.dinov3 / build.dynaclr / build.celldino / build.morphem is true"
        )

    # Stricter celldino gate than evaluate_predictions: precompute-gt
    # requires an explicit weights path when build.celldino=true (the
    # eval path soft-skips a null weights_path; here it would silently
    # produce no cache, so we surface it as an error).
    if build.celldino and config.feature_extractor.celldino.weights_path is None:
        raise ValueError("feature_extractor.celldino.weights_path is required when build.celldino=true")

    # Same strict gate for morphem: a null hub id with build.morphem=true
    # would silently fill no cache, so surface it as an error here.
    if build.morphem and config.feature_extractor.morphem.pretrained_model_name is None:
        raise ValueError("feature_extractor.morphem.pretrained_model_name is required when build.morphem=true")

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

    build_instances = bool(OmegaConf.select(config, "build.instances", default=False))
    models = load_eval_models(
        config,
        flags=LoadFlags(
            # build.instances needs the segmentation model too (prepare_segmentation_model
            # loads cpdino for segmentation.backend=cpdino), so trigger the masks load.
            masks=bool(build.masks or build_instances),
            dinov3=bool(build.dinov3),
            dynaclr=bool(build.dynaclr),
            celldino=bool(build.celldino),
            morphem=bool(build.morphem),
        ),
    )
    seg_model = models.seg_model

    cache_ctx = init_gt_cache_context(config, models)

    gt_path = Path(config.io.gt_path)
    seg_path = Path(config.io.cell_segmentation_path) if config.io.cell_segmentation_path is not None else None

    # ExitStack so the two optional auxiliary stores close even if opening the
    # second one raises (mirrors pipeline.evaluate_predictions).
    with open_ome_zarr(gt_path, mode="r") as gt_plate, ExitStack() as aux_stack:
        gt_positions = list(gt_plate.positions())
        seg_plate = aux_stack.enter_context(open_ome_zarr(seg_path, mode="r")) if seg_path is not None else None
        # Whole-cell instance prewarm needs the GT nucleus channel for the carve seeds.
        # A separate store (A549 H2B_*.ozx) is opened here; when io.nuclei_gt_path is null
        # or equals io.gt_path (iPSC cell.zarr), _build_gt_instances reads it from pos_gt.
        nuclei_plate = None
        if build_instances and config.target_name == "membrane":
            nuclei_gt_path = OmegaConf.select(config, "io.nuclei_gt_path", default=None)
            if nuclei_gt_path is not None and str(nuclei_gt_path) != str(gt_path):
                nuclei_plate = aux_stack.enter_context(open_ome_zarr(nuclei_gt_path, mode="r"))
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

        # build.<kind> and EvalModels.<kind> share the FeatureKind name, so the
        # requested extractors are a filter over the deep kinds rather than four
        # hand-written branches.
        deep_extractors = {k: getattr(models, k) for k in _DEEP_KINDS if build[k]}

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
            z_slabs = _focus_slabs(config, pos_gt, pos_name_gt, target.shape[0])

            if build.masks:
                fov_masks(cache_ctx, pos_name_gt, target, seg_model)
            if build_instances:
                _build_gt_instances(config, cache_ctx, seg_model, pos_gt, pos_name_gt, target, nuclei_plate)
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


@hydra.main(version_base="1.2", config_path="_configs", config_name="precompute")
def precompute_gt(config: DictConfig) -> None:
    """Hydra entry point for ``dynacell precompute-gt``."""
    apply_dataset_ref(config)
    precompute_gt_artifacts(config)


if __name__ == "__main__":
    precompute_gt()
