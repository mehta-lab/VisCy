"""CLI for config-driven dataset preparation (NFS -> VAST)."""

from __future__ import annotations

import logging
import re
import subprocess

import click

from airtable_utils.prepare import (
    PrepareConfig,
    check_dataset_status,
    check_preprocessed,
    check_zarr_version,
    discover_channels,
    discover_wells,
    filter_raw_channels,
    format_status_table,
    generate_concatenate_script,
    generate_crop_concat_config,
    generate_preprocess_slurm,
    generate_qc_config,
    generate_qc_slurm,
    generate_sbatch_override_file,
    resolve_nfs_paths,
    resolve_vast_paths,
    write_yaml,
)

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _load_prepare_config(config_path: str) -> PrepareConfig:
    """Load and validate a prepare config YAML."""
    from viscy_utils.cli_utils import load_config

    raw = load_config(config_path)
    return PrepareConfig(**raw)


def _parse_slurm_job_id(sbatch_output: str) -> str:
    """Extract job ID from sbatch stdout like 'Submitted batch job 12345'."""
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {sbatch_output}")
    return match.group(1)


@click.group(context_settings=CONTEXT_SETTINGS)
def prepare():
    """Prepare datasets for training on VAST storage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@prepare.command()
@click.argument("dataset_name")
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to prepare config YAML.",
)
@click.option("--dry-run", is_flag=True, help="Generate configs without submitting SLURM jobs.")
@click.option("--force", is_flag=True, help="Overwrite existing VAST zarr even if it is zarr v2.")
def run(dataset_name: str, config_path: str, dry_run: bool, force: bool) -> None:
    """Run the full preparation pipeline for DATASET_NAME.

    Steps: Airtable validation -> discover positions/channels -> generate
    crop_concat.yml + qc_config.yml + SLURM scripts -> submit jobs.
    """
    cfg = _load_prepare_config(config_path)

    # 1. Validate dataset is registered in Airtable
    click.echo(f"Validating {dataset_name} in Airtable...")
    from airtable_utils.database import AirtableDatasets

    db = AirtableDatasets()
    records = db.get_dataset_records(dataset_name)
    if not records:
        raise click.ClickException(
            f"Dataset '{dataset_name}' not found in Airtable. Register it first with the airtable-register workflow."
        )
    click.echo(f"  Found {len(records)} FOV records in Airtable.")

    # 2. Resolve NFS paths
    nfs = resolve_nfs_paths(dataset_name, cfg.nfs_root)
    click.echo(f"  NFS zarr: {nfs['zarr']}")

    # 3. Resolve VAST paths
    vast = resolve_vast_paths(dataset_name, cfg.vast_root)
    click.echo(f"  VAST output: {vast['output_dir']}")

    # 4. Check existing VAST zarr
    if vast["zarr"].exists():
        ver = check_zarr_version(vast["zarr"])
        is_v3 = ver["zarr_format"] == 3
        is_ome05 = ver["ome_version"] == "0.5"
        is_preprocessed = check_preprocessed(vast["zarr"])

        if is_v3 and is_ome05 and is_preprocessed:
            click.echo(
                f"  VAST zarr already exists: zarr v{ver['zarr_format']}, "
                f"OME {ver['ome_version']}, preprocessed. Skipping."
            )
            return

        if not force:
            msg = (
                f"VAST zarr already exists at {vast['zarr']} "
                f"(zarr v{ver['zarr_format']}, OME {ver['ome_version']}, "
                f"preprocessed={is_preprocessed}). "
                "Use --force to overwrite."
            )
            raise click.ClickException(msg)

        click.echo(f"  WARNING: Overwriting existing VAST zarr (zarr v{ver['zarr_format']}, OME {ver['ome_version']}).")

    # 5. Discover wells and resolve channels from NFS zarr
    click.echo("Discovering wells and channels from NFS zarr...")
    wells = discover_wells(nfs["zarr"])
    zarr_channels = discover_channels(nfs["zarr"])

    if cfg.concatenate.channel_names is not None:
        concat_channels = cfg.concatenate.channel_names
        missing = [ch for ch in concat_channels if ch not in zarr_channels]
        if missing:
            raise click.ClickException(f"Channels {missing} from config not found in zarr. Available: {zarr_channels}")
    else:
        concat_channels = filter_raw_channels(zarr_channels)
        if not concat_channels:
            raise click.ClickException(f"No raw channels found in zarr. Available: {zarr_channels}")

    click.echo(f"  Wells: {wells}")
    click.echo(f"  Zarr channels: {zarr_channels}")
    click.echo(f"  Extracting: {concat_channels}")

    # 6. Create output directory
    vast["output_dir"].mkdir(parents=True, exist_ok=True)

    # 7. Generate crop_concat.yml
    crop_concat_cfg = generate_crop_concat_config(nfs["zarr"], wells, concat_channels, cfg.concatenate)
    crop_concat_path = vast["output_dir"] / "crop_concat.yml"
    write_yaml(crop_concat_cfg, crop_concat_path)
    click.echo(f"  Wrote: {crop_concat_path}")

    # 8. Generate qc_config.yml
    qc_cfg = generate_qc_config(vast["zarr"], cfg.qc)
    qc_config_path = vast["output_dir"] / "qc_config.yml"
    write_yaml(qc_cfg, qc_config_path)
    click.echo(f"  Wrote: {qc_config_path}")

    # 9. Generate scripts
    sbatch_override_path = None
    if cfg.concatenate.sbatch_overrides:
        sbatch_content = generate_sbatch_override_file(cfg.concatenate.sbatch_overrides)
        sbatch_override_path = vast["output_dir"] / "sbatch_overrides.sh"
        sbatch_override_path.write_text(sbatch_content)
        click.echo(f"  Wrote: {sbatch_override_path}")

    concat_script = generate_concatenate_script(
        crop_concat_path=crop_concat_path,
        vast_zarr_path=vast["zarr"],
        nfs_tracking_path=nfs["tracking"],
        vast_tracking_path=vast["tracking"],
        conda_env=cfg.concatenate.conda_env,
        sbatch_override_path=sbatch_override_path,
    )
    concat_script_path = vast["output_dir"] / "01_concatenate.sh"
    concat_script_path.write_text(concat_script)
    click.echo(f"  Wrote: {concat_script_path}")

    qc_script = generate_qc_slurm(
        dataset_name=dataset_name,
        vast_output_dir=vast["output_dir"],
        qc_config_path=qc_config_path,
        workspace_dir=cfg.workspace_dir,
        slurm_cfg=cfg.slurm.qc,
    )
    qc_script_path = vast["output_dir"] / "02_qc.sh"
    qc_script_path.write_text(qc_script)
    click.echo(f"  Wrote: {qc_script_path}")

    preprocess_script = generate_preprocess_slurm(
        dataset_name=dataset_name,
        vast_output_dir=vast["output_dir"],
        vast_zarr_path=vast["zarr"],
        workspace_dir=cfg.workspace_dir,
        preprocess_params=cfg.preprocess,
        slurm_cfg=cfg.slurm.preprocess,
    )
    preprocess_script_path = vast["output_dir"] / "03_preprocess.sh"
    preprocess_script_path.write_text(preprocess_script)
    click.echo(f"  Wrote: {preprocess_script_path}")

    if dry_run:
        click.echo("\n--dry-run: configs and scripts generated, nothing executed.")
        return

    # 10. Run concatenation (biahub submits its own SLURM jobs via submitit)
    click.echo("\nRunning biahub concatenate + tracking copy...")
    click.echo("  (biahub will submit SLURM jobs internally and -m will monitor them)")
    subprocess.run(["bash", str(concat_script_path)], check=True)
    click.echo("Concatenation and tracking copy complete.")

    # 11. Submit QC and preprocess as separate SLURM jobs (no dependency, no race condition)
    click.echo("\nSubmitting QC and preprocess SLURM jobs...")
    result_qc = subprocess.run(
        ["sbatch", str(qc_script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    qc_job_id = _parse_slurm_job_id(result_qc.stdout)
    click.echo(f"  QC job: {qc_job_id} (GPU, ~5-20 min)")

    result_pp = subprocess.run(
        ["sbatch", str(preprocess_script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    pp_job_id = _parse_slurm_job_id(result_pp.stdout)
    click.echo(f"  Preprocess job: {pp_job_id} (CPU, ~3 hrs)")

    click.echo(f"\nPipeline running for {dataset_name}.")
    click.echo(f"  Output: {vast['output_dir']}")
    click.echo(f"  Monitor: squeue -j {qc_job_id},{pp_job_id}")


@prepare.command()
@click.argument("dataset_names", nargs=-1, required=True)
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to prepare config YAML.",
)
def status(dataset_names: tuple[str, ...], config_path: str) -> None:
    """Check NFS/VAST existence and version status for one or more datasets."""
    cfg = _load_prepare_config(config_path)

    rows = [check_dataset_status(name, cfg.nfs_root, cfg.vast_root) for name in dataset_names]
    click.echo(format_status_table(rows))


def main() -> None:
    """Entry point for the prepare CLI."""
    prepare()


if __name__ == "__main__":
    main()
