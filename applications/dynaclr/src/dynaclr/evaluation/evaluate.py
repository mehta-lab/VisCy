"""Evaluation orchestrator for DynaCLR trained models.

Generates per-step YAML configs and SLURM scripts from a single eval YAML.
Each generated script is independently submittable; the orchestrator also
prints a chained submission one-liner.

Usage
-----
dynaclr evaluate -c eval_config.yaml
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import click
import yaml

from dynaclr.evaluation.evaluate_config import EvaluationConfig
from viscy_utils.cli_utils import load_config

_Z_REDUCTION_CLASS = "viscy_transforms.BatchedChannelWiseZReductiond"

# Placeholders used in template YAMLs that operate per-experiment zarr.
# Shell scripts replace these at runtime when looping over globbed zarr paths.
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
        "ckpt_path": eval_cfg.ckpt_path,
    }

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

    ``input_paths`` is populated at runtime by the shell script (globbing
    per-experiment zarrs), so we write a placeholder list here.
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
    """Generate smoothness evaluation config YAML.

    Uses a placeholder path because the actual per-experiment zarr paths
    are only known after the split step.
    """
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
    """Generate per-experiment plot config YAML (template with placeholders).

    Plots per-experiment embedding keys (e.g. X_pca) into plots/{experiment}/ subdirs.
    Both input_path and output_dir use placeholders substituted at runtime.
    """
    cfg_dict = {
        "input_path": _ZARR_PLACEHOLDER,
        "output_dir": _PLOT_DIR_PLACEHOLDER,
        "embedding_keys": eval_cfg.plot.embedding_keys,
        "color_by": eval_cfg.plot.color_by,
        "point_size": eval_cfg.plot.point_size,
        "components": list(eval_cfg.plot.components),
        "format": eval_cfg.plot.format,
    }

    out_path = output_dir / "configs" / "plot.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_plot_combined_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate combined plot config YAML (template with input_paths placeholder list).

    Plots combined embedding keys (X_pca_combined, X_phate_combined) from all
    experiments concatenated into a single figure in plots/combined/.
    The input_paths list is patched at runtime by the shell script or local runner.
    """
    cfg_dict = {
        "input_paths": [_ZARR_PLACEHOLDER],
        "output_dir": str(output_dir / "plots" / "combined"),
        "embedding_keys": eval_cfg.plot.combined_embedding_keys,
        "color_by": eval_cfg.plot.color_by,
        "point_size": eval_cfg.plot.point_size,
        "components": list(eval_cfg.plot.components),
        "format": eval_cfg.plot.format,
    }

    out_path = output_dir / "configs" / "plot_combined.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def _generate_linear_classifiers_yaml(eval_cfg: EvaluationConfig, output_dir: Path) -> Path:
    """Generate linear classifiers config YAML for dynaclr run-linear-classifiers."""
    lc = eval_cfg.linear_classifiers
    embeddings_dir = str(output_dir / "embeddings")
    lc_output_dir = str(output_dir / "linear_classifiers")

    cfg_dict = {
        "embeddings_path": embeddings_dir,
        "output_dir": lc_output_dir,
        "annotations": [{"experiment": a.experiment, "path": a.path} for a in lc.annotations],
        "tasks": [{"task": t.task, "marker_filter": t.marker_filter} for t in lc.tasks],
        "use_scaling": lc.use_scaling,
        "use_pca": lc.use_pca,
        "n_pca_components": lc.n_pca_components,
        "max_iter": lc.max_iter,
        "class_weight": lc.class_weight,
        "solver": lc.solver,
        "split_train_data": lc.split_train_data,
        "random_seed": lc.random_seed,
    }

    out_path = output_dir / "configs" / "linear_classifiers.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------


def _slurm_header(partition: str, mem: str, time: str, cpus: int, job_name: str, log_path: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={partition}
        #SBATCH --mem={mem}
        #SBATCH --time={time}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --output={log_path}

        export PYTHONNOUSERSITE=1
        """)


def _slurm_gpu_header(partition: str, mem: str, time: str, job_name: str, log_path: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={partition}
        #SBATCH --mem={mem}
        #SBATCH --time={time}
        #SBATCH --gres=gpu:1
        #SBATCH --output={log_path}

        export PYTHONNOUSERSITE=1
        """)


def _workspace_cd(workspace_dir: str) -> str:
    return f"cd {workspace_dir}\n"


def _uv_run_prefix(workspace_dir: str) -> str:
    return f"uv run --project {workspace_dir}"


def _per_zarr_loop(embeddings_dir: str, body: str) -> str:
    """Generate a bash for-loop over per-experiment zarrs.

    Parameters
    ----------
    embeddings_dir : str
        Directory containing per-experiment zarrs.
    body : str
        Loop body. Use ``$zarr`` to reference the current zarr path and
        ``$name`` for the experiment name (stem without .zarr).
    """
    return textwrap.dedent(f"""\
        EMBEDDINGS_DIR="{embeddings_dir}"
        for zarr in "$EMBEDDINGS_DIR"/*.zarr; do
          name=$(basename "$zarr" .zarr)
          echo "=== Processing $name ==="
          {body}
        done
        """)


def _sed_replace_placeholder(yaml_path: str, placeholder: str, replacement: str) -> str:
    """Generate a sed command to replace a placeholder in a YAML template."""
    return f'sed "s|{placeholder}|{replacement}|g" {yaml_path}'


# ---------------------------------------------------------------------------
# SLURM script generators
# ---------------------------------------------------------------------------


def _generate_predict_sh(eval_cfg: EvaluationConfig, output_dir: Path, predict_yml: Path) -> Path:
    slurm = eval_cfg.slurm
    log = str(output_dir / "logs" / "predict_%j.out")
    content = _slurm_gpu_header(slurm.gpu_partition, slurm.gpu_mem, slurm.gpu_time, "dynaclr-predict", log)
    content += _workspace_cd(slurm.workspace_dir)
    content += f"srun {_uv_run_prefix(slurm.workspace_dir)} --package viscy-utils viscy predict -c {predict_yml}\n"

    out_path = output_dir / "configs" / "predict.sh"
    out_path.write_text(content)
    return out_path


def _resolve_cell_index_path(eval_cfg: EvaluationConfig, training_cfg: dict) -> str:
    """Resolve the cell index parquet path from eval config or training config fallback."""
    if eval_cfg.cell_index_path is not None:
        return eval_cfg.cell_index_path
    return training_cfg["data"]["init_args"]["cell_index_path"]


def _generate_viewer_yaml(split_zarr_paths: list[Path], output_dir: Path, cell_index_path: str) -> Path:
    """Generate a viewer YAML with the datasets structure for nd-embedding viewer.

    Reads experiment -> store_path from the cell index parquet to get hcs_plate paths.
    Written to configs/viewer.yaml after the split step.

    Parameters
    ----------
    split_zarr_paths : list[Path]
        Per-experiment zarr paths produced by split-embeddings.
    output_dir : Path
        Evaluation output root directory.
    cell_index_path : str
        Path to the cell index parquet for experiment -> hcs_plate lookup.

    Returns
    -------
    Path
        Path to the written viewer.yaml.
    """
    import pandas as pd

    df = pd.read_parquet(cell_index_path, columns=["experiment", "store_path"])
    exp_to_plate = df.drop_duplicates("experiment").set_index("experiment")["store_path"].to_dict()

    datasets: dict = {}
    for zarr_path in sorted(split_zarr_paths):
        exp_name = zarr_path.stem
        datasets[exp_name] = {
            "hcs_plate": exp_to_plate[exp_name],
            "anndata": str(zarr_path),
        }

    cfg_dict = {"datasets": datasets}
    out_path = output_dir / "configs" / "viewer.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def _generate_split_sh(eval_cfg: EvaluationConfig, output_dir: Path, cell_index_path: str) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    combined_zarr = str(output_dir / "embeddings" / "embeddings.zarr")
    viewer_yaml = str(output_dir / "configs" / "viewer.yaml")
    log = str(output_dir / "logs" / "split_%j.out")
    content = _slurm_header(slurm.cpu_partition, "32G", "0-00:30:00", 4, "dynaclr-split", log)
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    content += (
        f"{uv} --package dynaclr dynaclr split-embeddings --input {combined_zarr} --output-dir {embeddings_dir}\n"
    )
    # Generate viewer YAML after split: look up hcs_plate from the cell index parquet
    content += textwrap.dedent(f"""\
        {uv} --package dynaclr python3 -c "
        import pandas as pd, yaml, pathlib
        embeddings_dir = pathlib.Path('{embeddings_dir}')
        df = pd.read_parquet('{cell_index_path}', columns=['experiment', 'store_path'])
        exp_to_plate = df.drop_duplicates('experiment').set_index('experiment')['store_path'].to_dict()
        datasets = {{}}
        for zarr_path in sorted(embeddings_dir.glob('*.zarr')):
            exp_name = zarr_path.stem
            datasets[exp_name] = {{
                'hcs_plate': exp_to_plate[exp_name],
                'anndata': str(zarr_path),
            }}
        with open('{viewer_yaml}', 'w') as f:
            yaml.dump({{'datasets': datasets}}, f, default_flow_style=False, sort_keys=False)
        print('Viewer YAML written to {viewer_yaml}')
        "
        """)

    out_path = output_dir / "configs" / "split.sh"
    out_path.write_text(content)
    return out_path


def _generate_reduce_sh(eval_cfg: EvaluationConfig, output_dir: Path, reduce_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    log = str(output_dir / "logs" / "reduce_%j.out")
    content = _slurm_header(
        slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-reduce", log
    )
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    sed_cmd = _sed_replace_placeholder(str(reduce_yaml), _ZARR_PLACEHOLDER, "$zarr")
    body = f"{sed_cmd} > /tmp/reduce_$name.yaml && {uv} --package dynaclr dynaclr reduce-dimensionality -c /tmp/reduce_$name.yaml"
    content += _per_zarr_loop(embeddings_dir, body)

    out_path = output_dir / "configs" / "reduce.sh"
    out_path.write_text(content)
    return out_path


def _generate_reduce_combined_sh(eval_cfg: EvaluationConfig, output_dir: Path, reduce_combined_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    log = str(output_dir / "logs" / "reduce_combined_%j.out")
    content = _slurm_header(
        slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-reduce-combined", log
    )
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    # Build input_paths list from per-experiment zarrs at runtime
    content += textwrap.dedent(f"""\
        EMBEDDINGS_DIR="{embeddings_dir}"
        # Build a YAML list of input_paths from the per-experiment zarrs
        INPUT_PATHS=""
        for zarr in "$EMBEDDINGS_DIR"/*.zarr; do
          INPUT_PATHS="$INPUT_PATHS\\n- $zarr"
        done

        # Patch the template YAML: replace the placeholder list with actual paths
        python3 -c "
        import yaml, sys
        with open('{reduce_combined_yaml}') as f:
            cfg = yaml.safe_load(f)
        import glob
        cfg['input_paths'] = sorted(glob.glob('{embeddings_dir}/*.zarr'))
        with open('/tmp/reduce_combined_patched.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        "

        {uv} --package dynaclr dynaclr combined-dim-reduction -c /tmp/reduce_combined_patched.yaml
        """)

    out_path = output_dir / "configs" / "reduce_combined.sh"
    out_path.write_text(content)
    return out_path


def _generate_smoothness_sh(eval_cfg: EvaluationConfig, output_dir: Path, smoothness_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    log = str(output_dir / "logs" / "smoothness_%j.out")
    content = _slurm_header(
        slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-smoothness", log
    )
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    sed_cmd = _sed_replace_placeholder(str(smoothness_yaml), _ZARR_PLACEHOLDER, "$zarr")
    body = f"{sed_cmd} > /tmp/smoothness_$name.yaml && {uv} --package dynaclr dynaclr evaluate-smoothness -c /tmp/smoothness_$name.yaml"
    content += _per_zarr_loop(embeddings_dir, body)

    out_path = output_dir / "configs" / "smoothness.sh"
    out_path.write_text(content)
    return out_path


def _generate_plot_sh(eval_cfg: EvaluationConfig, output_dir: Path, plot_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    plots_dir = str(output_dir / "plots")
    log = str(output_dir / "logs" / "plot_%j.out")
    content = _slurm_header(
        slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-plot", log
    )
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    # Substitute both placeholders: zarr path and per-experiment plot subdir
    sed_cmd = f'sed "s|{_ZARR_PLACEHOLDER}|$zarr|g; s|{_PLOT_DIR_PLACEHOLDER}|{plots_dir}/$name|g" {plot_yaml}'
    body = f"{sed_cmd} > /tmp/plot_$name.yaml && {uv} --package dynaclr dynaclr plot-embeddings -c /tmp/plot_$name.yaml"
    content += _per_zarr_loop(embeddings_dir, body)

    out_path = output_dir / "configs" / "plot.sh"
    out_path.write_text(content)
    return out_path


def _generate_plot_combined_sh(eval_cfg: EvaluationConfig, output_dir: Path, plot_combined_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    embeddings_dir = str(output_dir / "embeddings")
    log = str(output_dir / "logs" / "plot_combined_%j.out")
    content = _slurm_header(
        slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-plot-combined", log
    )
    content += _workspace_cd(slurm.workspace_dir)
    uv = _uv_run_prefix(slurm.workspace_dir)
    content += textwrap.dedent(f"""\
        {uv} --package dynaclr python3 -c "
        import yaml, glob
        with open('{plot_combined_yaml}') as f:
            cfg = yaml.safe_load(f)
        cfg['input_paths'] = sorted(glob.glob('{embeddings_dir}/*.zarr'))
        with open('/tmp/plot_combined_patched.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        "
        {uv} --package dynaclr dynaclr plot-embeddings -c /tmp/plot_combined_patched.yaml
        """)

    out_path = output_dir / "configs" / "plot_combined.sh"
    out_path.write_text(content)
    return out_path


def _generate_linear_classifiers_sh(eval_cfg: EvaluationConfig, output_dir: Path, lc_yaml: Path) -> Path:
    slurm = eval_cfg.slurm
    log = str(output_dir / "logs" / "linear_classifiers_%j.out")
    content = _slurm_header(slurm.cpu_partition, slurm.cpu_mem, slurm.cpu_time, slurm.cpus_per_task, "dynaclr-lc", log)
    content += _workspace_cd(slurm.workspace_dir)
    content += f"{_uv_run_prefix(slurm.workspace_dir)} --package dynaclr dynaclr run-linear-classifiers -c {lc_yaml}\n"

    out_path = output_dir / "configs" / "linear_classifiers.sh"
    out_path.write_text(content)
    return out_path


# ---------------------------------------------------------------------------
# Submission summary
# ---------------------------------------------------------------------------


def _print_submission_summary(
    output_dir: Path,
    steps: list[str],
    generated_scripts: dict[str, Path],
) -> None:
    """Print submission instructions with correct dependency ordering.

    Dependency chain:
      predict → split → reduce_dimensionality → reduce_combined → plot
                      → smoothness
                      → linear_classifiers
    reduce_dimensionality must complete before reduce_combined and plot.
    smoothness and linear_classifiers read raw embeddings (.X) so only need split.
    """
    click.echo("\n" + "=" * 70)
    click.echo("EVALUATION PIPELINE READY")
    click.echo("=" * 70)
    click.echo(f"\nConfigs written to: {output_dir / 'configs'}\n")

    predict_sh = generated_scripts.get("predict")
    split_sh = generated_scripts.get("split")
    reduce_sh = generated_scripts.get("reduce_dimensionality")
    reduce_combined_sh = generated_scripts.get("reduce_combined")
    plot_sh = generated_scripts.get("plot")
    # Steps that depend on split only (read raw embeddings)
    split_dependents = ["smoothness", "linear_classifiers"]

    click.echo("## Submit individually:")
    for step_name, sh in generated_scripts.items():
        click.echo(f"  sbatch {sh}   # {step_name}")

    click.echo("\n## Chain all jobs automatically:")
    lines = []

    # predict
    if predict_sh:
        lines.append(f"  JOB_PREDICT=$(sbatch --parsable {predict_sh})")

    # split depends on predict
    if split_sh:
        dep = " --dependency=afterok:$JOB_PREDICT" if predict_sh else ""
        lines.append(f"  JOB_SPLIT=$(sbatch --parsable{dep} {split_sh})")

    # reduce_dimensionality depends on split
    if reduce_sh:
        dep = " --dependency=afterok:$JOB_SPLIT" if split_sh else ""
        lines.append(f"  JOB_REDUCE=$(sbatch --parsable{dep} {reduce_sh})")

    # reduce_combined depends on reduce_dimensionality
    if reduce_combined_sh:
        dep = " --dependency=afterok:$JOB_REDUCE" if reduce_sh else ""
        lines.append(f"  JOB_REDUCE_COMBINED=$(sbatch --parsable{dep} {reduce_combined_sh})")

    # plot depends on reduce_combined (needs X_pca_combined / X_phate_combined)
    if plot_sh:
        if reduce_combined_sh:
            lines.append(f"  sbatch --dependency=afterok:$JOB_REDUCE_COMBINED {plot_sh}")
        elif reduce_sh:
            lines.append(f"  sbatch --dependency=afterok:$JOB_REDUCE {plot_sh}")
        elif split_sh:
            lines.append(f"  sbatch --dependency=afterok:$JOB_SPLIT {plot_sh}")
        else:
            lines.append(f"  sbatch {plot_sh}")

    # smoothness and linear_classifiers depend on split
    for step in split_dependents:
        sh = generated_scripts.get(step)
        if sh:
            if split_sh:
                lines.append(f"  sbatch --dependency=afterok:$JOB_SPLIT {sh}")
            elif predict_sh:
                lines.append(f"  sbatch --dependency=afterok:$JOB_PREDICT {sh}")
            else:
                lines.append(f"  sbatch {sh}")

    click.echo("\n".join(lines))
    click.echo("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def _run_local_cpu_step(step: str, yaml_path: Path, workspace_dir: str) -> None:
    """Run a single CPU step in a subprocess."""
    cmd_map = {
        "reduce_dimensionality": ["dynaclr", "reduce-dimensionality", "-c", str(yaml_path)],
        "reduce_combined": ["dynaclr", "combined-dim-reduction", "-c", str(yaml_path)],
        "smoothness": ["dynaclr", "evaluate-smoothness", "-c", str(yaml_path)],
        "plot": ["dynaclr", "plot-embeddings", "-c", str(yaml_path)],
        "plot_combined": ["dynaclr", "plot-embeddings", "-c", str(yaml_path)],
        "linear_classifiers": ["dynaclr", "run-linear-classifiers", "-c", str(yaml_path)],
    }
    cmd = ["uv", "run", f"--project={workspace_dir}", "--package=dynaclr"] + cmd_map[step]
    click.echo(f"  Running: {' '.join(cmd_map[step])}")
    result = subprocess.run(cmd, cwd=workspace_dir)
    if result.returncode != 0:
        raise click.ClickException(f"Step '{step}' failed with exit code {result.returncode}")


def _run_local_split(output_dir: Path, workspace_dir: str) -> None:
    """Run split-embeddings locally."""
    combined_zarr = output_dir / "embeddings" / "embeddings.zarr"
    embeddings_dir = output_dir / "embeddings"
    cmd = [
        "uv",
        "run",
        f"--project={workspace_dir}",
        "--package=dynaclr",
        "dynaclr",
        "split-embeddings",
        "--input",
        str(combined_zarr),
        "--output-dir",
        str(embeddings_dir),
    ]
    click.echo("  Running: dynaclr split-embeddings")
    result = subprocess.run(cmd, cwd=workspace_dir)
    if result.returncode != 0:
        raise click.ClickException(f"split failed with exit code {result.returncode}")


def _patch_yaml_for_zarr(template_yaml: Path, zarr_path: Path, plots_dir: Path | None = None) -> Path:
    """Create a patched copy of a template YAML with the actual zarr path.

    If plots_dir is provided, also substitutes _PLOT_DIR_PLACEHOLDER with
    plots_dir / zarr_path.stem (per-experiment plot subdirectory).
    """
    import tempfile

    with open(template_yaml) as f:
        content = f.read()
    content = content.replace(_ZARR_PLACEHOLDER, str(zarr_path))
    if plots_dir is not None:
        exp_plot_dir = plots_dir / zarr_path.stem
        content = content.replace(_PLOT_DIR_PLACEHOLDER, str(exp_plot_dir))
    patched = Path(tempfile.mktemp(suffix=".yaml"))
    with open(patched, "w") as f:
        f.write(content)
    return patched


def _patch_reduce_combined_yaml(template_yaml: Path, embeddings_dir: Path) -> Path:
    """Create a patched reduce_combined YAML with actual per-experiment zarr paths."""
    import tempfile

    with open(template_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["input_paths"] = sorted(str(p) for p in embeddings_dir.glob("*.zarr"))
    patched = Path(tempfile.mktemp(suffix=".yaml"))
    with open(patched, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return patched


def _run_local(
    eval_cfg: EvaluationConfig,
    training_cfg: dict,
    output_dir: Path,
    yaml_configs: dict[str, Path],
) -> None:
    """Execute all steps locally: predict (blocking), split, then CPU steps."""
    import concurrent.futures

    steps = eval_cfg.steps
    workspace_dir = eval_cfg.slurm.workspace_dir
    embeddings_dir = output_dir / "embeddings"

    # --- predict (GPU, must finish before everything else) ---
    if "predict" in steps:
        predict_yml = yaml_configs["predict"]
        click.echo("\n[predict] Running viscy predict (blocking)...")
        cmd = [
            "uv",
            f"--project={workspace_dir}",
            "run",
            "--package=viscy-utils",
            "viscy",
            "predict",
            "-c",
            str(predict_yml),
        ]
        result = subprocess.run(cmd, cwd=workspace_dir)
        if result.returncode != 0:
            raise click.ClickException(f"predict failed with exit code {result.returncode}")
        click.echo("[predict] Done.")

    # --- split (must finish before per-experiment steps) ---
    if "split" in steps:
        click.echo("\n[split] Running split-embeddings...")
        _run_local_split(output_dir, workspace_dir)
        click.echo("[split] Done.")
        click.echo("\n[split] Generating viewer YAML...")
        cell_index_path = _resolve_cell_index_path(eval_cfg, training_cfg)
        viewer_yaml = _generate_viewer_yaml(sorted(embeddings_dir.glob("*.zarr")), output_dir, cell_index_path)
        click.echo(f"[split] Viewer YAML written to {viewer_yaml}")

    # --- reduce_dimensionality (per-experiment, must finish before reduce_combined and plot) ---
    if "reduce_dimensionality" in steps:
        click.echo("\n[reduce_dimensionality] Running per-experiment...")
        for zarr_path in sorted(embeddings_dir.glob("*.zarr")):
            patched = _patch_yaml_for_zarr(yaml_configs["reduce_dimensionality"], zarr_path)
            _run_local_cpu_step("reduce_dimensionality", patched, workspace_dir)
        click.echo("[reduce_dimensionality] Done.")

    # --- reduce_combined (must finish before plot) ---
    if "reduce_combined" in steps:
        click.echo("\n[reduce_combined] Running joint reduction...")
        patched = _patch_reduce_combined_yaml(yaml_configs["reduce_combined"], embeddings_dir)
        _run_local_cpu_step("reduce_combined", patched, workspace_dir)
        click.echo("[reduce_combined] Done.")

    # --- Remaining CPU steps run in parallel (per-experiment where needed) ---
    serial_steps = {"predict", "split", "reduce_dimensionality", "reduce_combined"}
    parallel_steps = [s for s in steps if s not in serial_steps]
    # plot_combined is generated alongside plot but not listed in steps; add if plot is a step
    if "plot" in steps and "plot_combined" in yaml_configs:
        parallel_steps = [s if s != "plot" else s for s in parallel_steps] + ["plot_combined"]
    if not parallel_steps:
        return

    per_zarr_steps = {"smoothness", "plot"}
    # Steps that need input_paths patched from all zarrs (like reduce_combined)
    all_zarr_steps = {"plot_combined"}

    click.echo(f"\nRunning in parallel: {parallel_steps}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_steps)) as executor:
        futures: dict[concurrent.futures.Future, str] = {}
        for step in parallel_steps:
            if step not in yaml_configs:
                continue
            if step in per_zarr_steps:
                for zarr_path in sorted(embeddings_dir.glob("*.zarr")):
                    plots_dir = output_dir / "plots" if step == "plot" else None
                    patched = _patch_yaml_for_zarr(yaml_configs[step], zarr_path, plots_dir=plots_dir)
                    f = executor.submit(_run_local_cpu_step, step, patched, workspace_dir)
                    futures[f] = f"{step}/{zarr_path.stem}"
            elif step in all_zarr_steps:
                patched = _patch_reduce_combined_yaml(yaml_configs[step], embeddings_dir)
                f = executor.submit(_run_local_cpu_step, "plot_combined", patched, workspace_dir)
                futures[f] = step
            else:
                f = executor.submit(_run_local_cpu_step, step, yaml_configs[step], workspace_dir)
                futures[f] = step

        for future in concurrent.futures.as_completed(futures):
            step_label = futures[future]
            try:
                future.result()
                click.echo(f"[{step_label}] Done.")
            except Exception as exc:
                click.echo(f"[{step_label}] Failed: {exc}", err=True)
                raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation YAML configuration file",
)
@click.option(
    "--mode",
    type=click.Choice(["slurm", "local"], case_sensitive=False),
    default="slurm",
    show_default=True,
    help="slurm: generate SLURM scripts and print sbatch commands. local: run all steps in the current process.",
)
def main(config: Path, mode: str) -> None:
    """Generate evaluation configs and SLURM scripts for a trained DynaCLR model."""
    raw = load_config(config)
    eval_cfg = EvaluationConfig(**raw)

    training_cfg = _load_training_config(eval_cfg.training_config)
    output_dir = Path(eval_cfg.output_dir)

    # Create output directories
    for subdir in ["configs", "embeddings", "smoothness", "plots", "linear_classifiers", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Save a copy of the input eval config for reproducibility and re-runs
    shutil.copy(config, output_dir / "configs" / "eval.yaml")

    generated_scripts: dict[str, Path] = {}
    yaml_configs: dict[str, Path] = {}

    for step in eval_cfg.steps:
        if step == "predict":
            predict_yml = _generate_predict_yaml(eval_cfg, training_cfg, output_dir)
            yaml_configs["predict"] = predict_yml
            click.echo(f"[predict]  {predict_yml}")
            if mode == "slurm":
                predict_sh = _generate_predict_sh(eval_cfg, output_dir, predict_yml)
                generated_scripts["predict"] = predict_sh
                click.echo(f"           {predict_sh}")

        elif step == "split":
            viewer_yaml_path = output_dir / "configs" / "viewer.yaml"
            click.echo(f"[split]    viewer.yaml will be written to {viewer_yaml_path} after split runs")
            if mode == "slurm":
                cell_index_path = _resolve_cell_index_path(eval_cfg, training_cfg)
                split_sh = _generate_split_sh(eval_cfg, output_dir, cell_index_path)
                generated_scripts["split"] = split_sh
                click.echo(f"           {split_sh}")

        elif step == "reduce_dimensionality":
            reduce_yaml = _generate_reduce_yaml(eval_cfg, output_dir)
            yaml_configs["reduce_dimensionality"] = reduce_yaml
            click.echo(f"[reduce]   {reduce_yaml}")
            if mode == "slurm":
                reduce_sh = _generate_reduce_sh(eval_cfg, output_dir, reduce_yaml)
                generated_scripts["reduce_dimensionality"] = reduce_sh
                click.echo(f"           {reduce_sh}")

        elif step == "reduce_combined":
            reduce_combined_yaml = _generate_reduce_combined_yaml(eval_cfg, output_dir)
            yaml_configs["reduce_combined"] = reduce_combined_yaml
            click.echo(f"[combined] {reduce_combined_yaml}")
            if mode == "slurm":
                rc_sh = _generate_reduce_combined_sh(eval_cfg, output_dir, reduce_combined_yaml)
                generated_scripts["reduce_combined"] = rc_sh
                click.echo(f"           {rc_sh}")

        elif step == "smoothness":
            smoothness_yaml = _generate_smoothness_yaml(eval_cfg, output_dir)
            yaml_configs["smoothness"] = smoothness_yaml
            click.echo(f"[smooth]   {smoothness_yaml}")
            if mode == "slurm":
                smoothness_sh = _generate_smoothness_sh(eval_cfg, output_dir, smoothness_yaml)
                generated_scripts["smoothness"] = smoothness_sh
                click.echo(f"           {smoothness_sh}")

        elif step == "plot":
            plot_yaml = _generate_plot_yaml(eval_cfg, output_dir)
            yaml_configs["plot"] = plot_yaml
            click.echo(f"[plot]     {plot_yaml}")
            plot_combined_yaml = _generate_plot_combined_yaml(eval_cfg, output_dir)
            yaml_configs["plot_combined"] = plot_combined_yaml
            click.echo(f"[plot]     {plot_combined_yaml}")
            if mode == "slurm":
                plot_sh = _generate_plot_sh(eval_cfg, output_dir, plot_yaml)
                generated_scripts["plot"] = plot_sh
                click.echo(f"           {plot_sh}")
                plot_combined_sh = _generate_plot_combined_sh(eval_cfg, output_dir, plot_combined_yaml)
                generated_scripts["plot_combined"] = plot_combined_sh
                click.echo(f"           {plot_combined_sh}")

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
                    "Add task specs (task + optional marker_filter) before running.",
                    err=True,
                )
            lc_yaml = _generate_linear_classifiers_yaml(eval_cfg, output_dir)
            yaml_configs["linear_classifiers"] = lc_yaml
            click.echo(f"[lc]       {lc_yaml}")
            if mode == "slurm":
                lc_sh = _generate_linear_classifiers_sh(eval_cfg, output_dir, lc_yaml)
                generated_scripts["linear_classifiers"] = lc_sh
                click.echo(f"           {lc_sh}")

        else:
            click.echo(f"Unknown step '{step}', skipping", err=True)

    if mode == "slurm":
        _print_submission_summary(output_dir, eval_cfg.steps, generated_scripts)
    else:
        _run_local(eval_cfg, training_cfg, output_dir, yaml_configs)


if __name__ == "__main__":
    main()
