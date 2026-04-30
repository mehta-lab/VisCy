"""Evaluation config generator for DynaCLR trained models.

Generates per-step YAML configs from a single eval YAML and prints a JSON manifest
mapping step names to config paths. Called internally by the Nextflow PREPARE_CONFIGS step.

Usage
-----
dynaclr prepare-eval-configs -c eval_config.yaml
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import click
import yaml

from dynaclr.evaluation.evaluate_config import EvaluationConfig
from viscy_utils.cli_utils import load_config

_Z_REDUCTION_CLASS = "viscy_transforms.BatchedChannelWiseZReductiond"

# Placeholders used in template YAMLs that operate per-experiment zarr.
# Nextflow processes substitute these at runtime when handling per-experiment channels.
_ZARR_PLACEHOLDER = "__ZARR_PATH__"
_PLOT_DIR_PLACEHOLDER = "__PLOT_DIR__"


def _load_training_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_predict_data_config(training_cfg: dict, eval_cfg: EvaluationConfig) -> dict:
    """Extract data init_args for the predict YAML from the training config.

    Strips augmentations (except BatchedChannelWiseZReductiond which is
    architecturally required), overrides batch_size and split_ratio.
    """
    data_init = dict(training_cfg["data"]["init_args"])

    # Override cell_index_path if user supplied one
    if eval_cfg.cell_index_path is not None:
        data_init["cell_index_path"] = eval_cfg.cell_index_path

    # Move z-reduction transform from augmentations to end of normalizations
    augmentations = data_init.pop("augmentations", []) or []
    z_reduction = [t for t in augmentations if _is_z_reduction(t)]
    normalizations = list(data_init.get("normalizations") or [])
    data_init["normalizations"] = normalizations + z_reduction
    data_init["augmentations"] = []

    # Predict-specific overrides
    data_init["batch_size"] = eval_cfg.predict.batch_size
    data_init["num_workers"] = eval_cfg.predict.num_workers
    data_init["split_ratio"] = 1.0

    # Remove training-only keys that are irrelevant for predict
    for key in ["stratify_by", "batch_group_by", "temporal_enrichment", "leaky", "group_weights"]:
        data_init.pop(key, None)

    return data_init


def _is_z_reduction(transform: Any) -> bool:
    """Check if a transform config is BatchedChannelWiseZReductiond."""
    if isinstance(transform, dict):
        return transform.get("class_path", "") == _Z_REDUCTION_CLASS
    return False


def _extract_model_config(training_cfg: dict) -> dict:
    """Extract model config, setting drop_path_rate=0 for inference.

    Only sets drop_path_rate if the encoder already declares it (e.g. ContrastiveEncoder).
    Encoders like DINOv3Model do not accept this parameter and must not receive it.
    """
    model = dict(training_cfg["model"])
    init_args = dict(model.get("init_args", {}))
    encoder = dict(init_args.get("encoder", {}))
    encoder_init = dict(encoder.get("init_args", {}))
    if "drop_path_rate" in encoder_init:
        encoder_init["drop_path_rate"] = 0.0
    encoder["init_args"] = encoder_init
    init_args["encoder"] = encoder
    model["init_args"] = init_args
    return model


# ---------------------------------------------------------------------------
# YAML config generators
# ---------------------------------------------------------------------------


def _generate_predict_yaml(eval_cfg: EvaluationConfig, training_cfg: dict, output_dir: Path) -> Path:
    """Generate the Lightning predict YAML config."""
    embeddings_path = str(output_dir / "embeddings" / "embeddings.zarr")
    data_init = _extract_predict_data_config(training_cfg, eval_cfg)
    model_cfg = _extract_model_config(training_cfg)

    embedding_writer: dict = {
        "class_path": "viscy_utils.callbacks.embedding_writer.EmbeddingWriter",
        "init_args": {
            "output_path": embeddings_path,
            "overwrite": True,
        },
    }

    predict_cfg: dict = {
        "seed_everything": 42,
        "trainer": {
            "accelerator": "gpu",
            "devices": eval_cfg.predict.devices,
            "num_nodes": 1,
            "precision": eval_cfg.predict.precision,
            "inference_mode": True,
            "logger": False,
            "callbacks": [embedding_writer],
        },
        "model": model_cfg,
        "data": {
            "class_path": training_cfg["data"]["class_path"],
            "init_args": data_init,
        },
    }
    # Only emit ckpt_path when set. Foundation-model baselines (e.g. DINOv3-frozen)
    # load weights from HuggingFace inside the model __init__ and have no
    # Lightning checkpoint; passing ``ckpt_path: null`` to ``viscy predict``
    # would still try to restore from a None path.
    if eval_cfg.ckpt_path is not None:
        predict_cfg["ckpt_path"] = eval_cfg.ckpt_path

    out_path = output_dir / "configs" / "predict.yml"
    with open(out_path, "w") as f:
        yaml.dump(predict_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def _generate_reduce_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate dim reduction template config YAML.

    Uses a placeholder for ``input_path`` because the actual per-experiment
    zarr paths are only known after the split step runs.
    """
    cfg_dict: dict = {
        "input_path": _ZARR_PLACEHOLDER,
        "overwrite_keys": eval_cfg.reduce_dimensionality.overwrite_keys,
    }
    if eval_cfg.reduce_dimensionality.pca:
        cfg_dict["pca"] = eval_cfg.reduce_dimensionality.pca.model_dump()
    if eval_cfg.reduce_dimensionality.umap:
        cfg_dict["umap"] = eval_cfg.reduce_dimensionality.umap.model_dump()
    if eval_cfg.reduce_dimensionality.phate:
        cfg_dict["phate"] = eval_cfg.reduce_dimensionality.phate.model_dump()

    out_path = output_dir / "configs" / "reduce.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_reduce_combined_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate joint dimensionality reduction config YAML.

    ``input_paths`` is populated at runtime by Nextflow (collecting per-experiment zarrs).
    """
    rc = eval_cfg.reduce_combined
    cfg_dict: dict = {
        "input_paths": [_ZARR_PLACEHOLDER],
        "overwrite_keys": rc.overwrite_keys,
    }
    if rc.pca:
        cfg_dict["pca"] = rc.pca.model_dump()
    if rc.umap:
        cfg_dict["umap"] = rc.umap.model_dump()
    if rc.phate:
        cfg_dict["phate"] = rc.phate.model_dump()

    out_path = output_dir / "configs" / "reduce_combined.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_smoothness_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate smoothness evaluation config YAML."""
    model_name = Path(eval_cfg.training_config).stem

    cfg_dict = {
        "models": [{"path": _ZARR_PLACEHOLDER, "label": model_name}],
        "evaluation": {
            "distance_metric": eval_cfg.smoothness.distance_metric,
            "output_dir": str(output_dir / "smoothness"),
            "save_plots": eval_cfg.smoothness.save_plots,
            "save_distributions": eval_cfg.smoothness.save_distributions,
            "verbose": eval_cfg.smoothness.verbose,
        },
    }

    out_path = output_dir / "configs" / "smoothness.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_plot_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate per-experiment plot config YAML (template with placeholders)."""
    cfg_dict = {
        "input_path": _ZARR_PLACEHOLDER,
        "output_dir": _PLOT_DIR_PLACEHOLDER,
        "embedding_keys": eval_cfg.plot.embedding_keys,
        "color_by": eval_cfg.plot.color_by,
        "point_size": eval_cfg.plot.point_size,
        "components": list(eval_cfg.plot.components),
        "pairplot_components": eval_cfg.plot.pairplot_components,
        "format": eval_cfg.plot.format,
    }

    out_path = output_dir / "configs" / "plot.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_plot_combined_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate combined plot config YAML.

    The input_paths list is patched at runtime by Nextflow.
    """
    cfg_dict = {
        "input_paths": [_ZARR_PLACEHOLDER],
        "output_dir": str(output_dir / "plots" / "combined"),
        "embedding_keys": eval_cfg.plot.combined_embedding_keys,
        "color_by": eval_cfg.plot.combined_color_by,
        "point_size": eval_cfg.plot.point_size,
        "components": list(eval_cfg.plot.components),
        "pairplot_components": eval_cfg.plot.pairplot_components,
        "format": eval_cfg.plot.format,
    }

    out_path = output_dir / "configs" / "plot_combined.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_append_annotations_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate append-annotations config YAML.

    Sources annotations from ``eval_cfg.append_annotations`` when present
    (Wave-2 datasets that have ground truth but do not train LCs), else
    falls back to ``eval_cfg.linear_classifiers.annotations`` (Wave-1
    legacy path where annotations live alongside LC training config).
    """
    if eval_cfg.append_annotations is not None and eval_cfg.append_annotations.annotations:
        annotations = eval_cfg.append_annotations.annotations
        # Tasks list is informational for the writer; when running standalone
        # we emit an empty list (annotation columns are inferred from CSV).
        tasks: list[dict] = []
    else:
        lc = eval_cfg.linear_classifiers
        annotations = lc.annotations
        tasks = [{"task": t.task, "marker_filters": t.marker_filters} for t in lc.tasks]

    cfg_dict = {
        "embeddings_path": str(output_dir / "embeddings"),
        "annotations": [{"experiment": a.experiment, "path": a.path} for a in annotations],
        "tasks": tasks,
    }
    out_path = output_dir / "configs" / "append_annotations.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def _generate_append_predictions_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate append-predictions config YAML.

    Honors ``eval_cfg.append_predictions.pipelines_dir`` when set (Wave-2
    evaluations fetching from the central LC registry, typically
    ``{registry_root}/{model_name}/latest``). Otherwise falls back to the
    legacy in-run location ``output_dir/linear_classifiers/pipelines/``.
    """
    ap = eval_cfg.append_predictions
    if ap is not None and ap.pipelines_dir:
        pipelines_dir = ap.pipelines_dir
    else:
        pipelines_dir = str(output_dir / "linear_classifiers" / "pipelines")

    cfg_dict = {
        "embeddings_path": str(output_dir / "embeddings"),
        "pipelines_dir": pipelines_dir,
    }
    out_path = output_dir / "configs" / "append_predictions.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_linear_classifiers_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate linear classifiers config YAML for dynaclr run-linear-classifiers.

    Propagates ``publish_dir`` (central LC registry root) when set — the writer
    atomically promotes the trained bundle to ``{publish_dir}/vN/`` and updates
    the ``latest`` symlink.
    """
    lc = eval_cfg.linear_classifiers
    embeddings_dir = str(output_dir / "embeddings")
    lc_output_dir = str(output_dir / "linear_classifiers")

    cfg_dict: dict = {
        "embeddings_path": embeddings_dir,
        "output_dir": lc_output_dir,
        "annotations": [{"experiment": a.experiment, "path": a.path} for a in lc.annotations],
        "tasks": [{"task": t.task, "marker_filters": t.marker_filters} for t in lc.tasks],
        "use_scaling": lc.use_scaling,
        "use_pca": lc.use_pca,
        "n_pca_components": lc.n_pca_components,
        "max_iter": lc.max_iter,
        "class_weight": lc.class_weight,
        "solver": lc.solver,
        "split_train_data": lc.split_train_data,
        "random_seed": lc.random_seed,
    }
    if lc.publish_dir:
        cfg_dict["publish_dir"] = lc.publish_dir

    out_path = output_dir / "configs" / "linear_classifiers.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def _mmd_block_name(mmd: "MMDStepConfig", idx: int) -> str:  # noqa: F821
    """Derive a filesystem-safe name for an MMD block."""
    if mmd.name:
        return mmd.name
    return f"mmd_{idx}"


def _generate_mmd_yaml(mmd: "MMDStepConfig", output_dir: Path, block_name: str) -> Path:  # noqa: F821
    """Generate per-experiment MMD config YAML template (uses __ZARR_PATH__ placeholder)."""
    cfg_dict = {
        "input_path": _ZARR_PLACEHOLDER,
        "output_dir": str(output_dir / "mmd" / block_name),
        "comparisons": [{"cond_a": c.cond_a, "cond_b": c.cond_b, "label": c.label} for c in mmd.comparisons],
        "group_by": mmd.group_by,
        "obs_filter": mmd.obs_filter,
        "embedding_key": mmd.embedding_key,
        "mmd": mmd.mmd.model_dump(),
        "map_settings": mmd.map_settings.model_dump(),
        "temporal_bin_size": mmd.temporal_bin_size,
        "save_plots": mmd.save_plots,
    }
    out_path = output_dir / "configs" / f"{block_name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_mmd_combined_yaml(mmd: "MMDStepConfig", output_dir: Path, block_name: str) -> Path:  # noqa: F821
    """Generate cross-experiment MMD config YAML template (input_paths patched at runtime)."""
    combined_name = f"{block_name}_cross_exp"
    combined_bin_size = (
        mmd.combined_temporal_bin_size if mmd.combined_temporal_bin_size is not None else mmd.temporal_bin_size
    )
    cfg_dict = {
        "input_paths": [_ZARR_PLACEHOLDER],
        "output_dir": str(output_dir / "mmd" / combined_name),
        "group_by": mmd.group_by,
        "obs_filter": mmd.obs_filter,
        "embedding_key": mmd.embedding_key,
        "mmd": mmd.mmd.model_dump(),
        "map_settings": mmd.map_settings.model_dump(),
        "temporal_bin_size": combined_bin_size,
        "save_plots": mmd.save_plots,
    }
    out_path = output_dir / "configs" / f"{combined_name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _resolve_cell_index_path(eval_cfg: EvaluationConfig, training_cfg: dict) -> str:
    """Resolve the cell index parquet path from eval config or training config fallback."""
    if eval_cfg.cell_index_path is not None:
        return eval_cfg.cell_index_path
    return training_cfg["data"]["init_args"]["cell_index_path"]


# ---------------------------------------------------------------------------
# Main prepare_configs function
# ---------------------------------------------------------------------------


def prepare_configs(config: Path) -> None:
    """Generate all per-step YAML configs and print a JSON manifest to stdout.

    The manifest maps step names to generated config paths and includes paths
    needed by Nextflow to wire the pipeline (embeddings_dir, output_dir,
    cell_index_path, mmd_blocks).
    """
    raw = load_config(config)
    eval_cfg = EvaluationConfig(**raw)

    training_cfg = _load_training_config(eval_cfg.training_config)
    output_dir = Path(eval_cfg.output_dir)

    # Create output directories for active steps
    subdirs = ["configs", "embeddings"]
    step_subdirs = {
        "smoothness": "smoothness",
        "mmd": "mmd",
        "plot": "plots",
        "linear_classifiers": "linear_classifiers",
    }
    for step in eval_cfg.steps:
        if step in step_subdirs:
            subdirs.append(step_subdirs[step])
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Save a copy of the input eval config for reproducibility and re-runs
    shutil.copy(config, output_dir / "configs" / "eval.yaml")

    manifest: dict = {
        "output_dir": str(output_dir),
        "embeddings_dir": str(output_dir / "embeddings"),
        "cell_index_path": _resolve_cell_index_path(eval_cfg, training_cfg),
        "mmd_blocks": [],
        "mmd_combined_blocks": [],
    }

    for step in eval_cfg.steps:
        if step == "predict":
            predict_yml = _generate_predict_yaml(eval_cfg, training_cfg, output_dir)
            manifest["predict"] = str(predict_yml)
            click.echo(f"[predict]  {predict_yml}", err=True)

        elif step == "split":
            click.echo(
                f"[split]    viewer.yaml will be written to {output_dir / 'configs' / 'viewer.yaml'} after split runs",
                err=True,
            )

        elif step == "reduce_dimensionality":
            reduce_yaml = _generate_reduce_yaml(eval_cfg, output_dir)
            manifest["reduce"] = str(reduce_yaml)
            click.echo(f"[reduce]   {reduce_yaml}", err=True)

        elif step == "reduce_combined":
            reduce_combined_yaml = _generate_reduce_combined_yaml(eval_cfg, output_dir)
            manifest["reduce_combined"] = str(reduce_combined_yaml)
            click.echo(f"[combined] {reduce_combined_yaml}", err=True)

        elif step == "smoothness":
            smoothness_yaml = _generate_smoothness_yaml(eval_cfg, output_dir)
            manifest["smoothness"] = str(smoothness_yaml)
            click.echo(f"[smooth]   {smoothness_yaml}", err=True)

        elif step == "plot":
            plot_yaml = _generate_plot_yaml(eval_cfg, output_dir)
            manifest["plot"] = str(plot_yaml)
            click.echo(f"[plot]     {plot_yaml}", err=True)

        elif step == "plot_combined":
            plot_combined_yaml = _generate_plot_combined_yaml(eval_cfg, output_dir)
            manifest["plot_combined"] = str(plot_combined_yaml)
            click.echo(f"[plot]     {plot_combined_yaml}", err=True)

        elif step == "mmd":
            if not eval_cfg.mmd:
                click.echo("[mmd] skipped: no blocks configured", err=True)
                continue
            for i, mmd_block in enumerate(eval_cfg.mmd):
                block_name = _mmd_block_name(mmd_block, i)
                mmd_yaml = _generate_mmd_yaml(mmd_block, output_dir, block_name)
                manifest[f"mmd_{block_name}"] = str(mmd_yaml)
                manifest[f"mmd_{block_name}_dir"] = str(output_dir / "mmd" / block_name)
                manifest["mmd_blocks"].append(block_name)
                click.echo(f"[mmd]      {mmd_yaml}", err=True)
                if mmd_block.combined_mode:
                    mmd_combined_yaml = _generate_mmd_combined_yaml(mmd_block, output_dir, block_name)
                    combined_name = f"{block_name}_cross_exp"
                    manifest[f"mmd_{combined_name}"] = str(mmd_combined_yaml)
                    manifest["mmd_combined_blocks"].append(block_name)
                    click.echo(f"[mmd]      {mmd_combined_yaml}", err=True)

        elif step == "linear_classifiers":
            if eval_cfg.linear_classifiers is None:
                click.echo("[linear_classifiers] skipped: no config provided", err=True)
                continue
            if not eval_cfg.linear_classifiers.annotations:
                click.echo(
                    "[linear_classifiers] Warning: annotations is empty. "
                    "Add experiment + annotation CSV paths before running.",
                    err=True,
                )
            if not eval_cfg.linear_classifiers.tasks:
                click.echo(
                    "[linear_classifiers] Warning: tasks is empty. "
                    "Add task specs (task + optional marker_filters) before running.",
                    err=True,
                )
            lc_yaml = _generate_linear_classifiers_yaml(eval_cfg, output_dir)
            manifest["linear_classifiers"] = str(lc_yaml)
            click.echo(f"[lc]       {lc_yaml}", err=True)

        elif step == "append_annotations":
            # Annotations may live in either:
            #  (a) eval_cfg.append_annotations.annotations  — Wave-2 datasets
            #      that have ground truth but do not train LCs (alfi).
            #  (b) eval_cfg.linear_classifiers.annotations  — Wave-1 legacy
            #      path where annotations are colocated with LC training.
            has_aa = eval_cfg.append_annotations is not None and eval_cfg.append_annotations.annotations
            has_lc = eval_cfg.linear_classifiers is not None and eval_cfg.linear_classifiers.annotations
            if not (has_aa or has_lc):
                click.echo(
                    "[append_annotations] skipped: no annotations configured "
                    "(set append_annotations.annotations or linear_classifiers.annotations)",
                    err=True,
                )
                continue
            aa_yaml = _generate_append_annotations_yaml(eval_cfg, output_dir)
            manifest["append_annotations"] = str(aa_yaml)
            click.echo(f"[append_ann] {aa_yaml}", err=True)

        elif step == "append_predictions":
            # Two ways to satisfy append_predictions:
            #  (a) in-run: the same eval also trains LCs (linear_classifiers in
            #      steps + LinearClassifiersStepConfig present), and we fetch
            #      from output_dir/linear_classifiers/pipelines/.
            #  (b) external: eval_cfg.append_predictions.pipelines_dir points
            #      at a central registry directory (typically the `latest`
            #      symlink under a model's registry root). Wave-2 runs.
            has_external = eval_cfg.append_predictions is not None and eval_cfg.append_predictions.pipelines_dir
            has_in_run = eval_cfg.linear_classifiers is not None and "linear_classifiers" in eval_cfg.steps
            if not (has_external or has_in_run):
                raise ValueError(
                    "'append_predictions' requires either:\n"
                    "  (a) 'linear_classifiers' in steps (train LCs in this run), or\n"
                    "  (b) append_predictions.pipelines_dir set to an existing LC bundle\n"
                    "      (fetch pipelines from a separate run / central registry)."
                )
            ap_yaml = _generate_append_predictions_yaml(eval_cfg, output_dir)
            manifest["append_predictions"] = str(ap_yaml)
            click.echo(f"[append_pred] {ap_yaml}", err=True)

        else:
            click.echo(f"Unknown step '{step}', skipping", err=True)

    # Print JSON manifest to stdout for Nextflow to consume
    click.echo(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation YAML configuration file",
)
def main(config: Path) -> None:
    """Generate evaluation configs for a trained DynaCLR model.

    Writes per-step YAML configs to output_dir/configs/ and prints a JSON manifest
    to stdout mapping step names to config paths. Used as the entry point for the
    Nextflow evaluation pipeline.
    """
    prepare_configs(config)


if __name__ == "__main__":
    main()
