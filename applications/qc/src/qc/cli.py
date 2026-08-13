"""Click CLI for QC metrics."""

import click

from qc.annotation import write_annotation_metadata
from qc.config import QCConfig
from qc.focus import FocusSliceMetric, audit_focus_slice
from qc.qc_metrics import generate_qc_metadata
from viscy_utils.cli_utils import load_config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def qc():
    """Quality control metrics for OME-Zarr datasets."""
    pass


@qc.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
def run(config_path: str):
    """Run QC metrics on an OME-Zarr dataset."""
    raw = load_config(config_path)
    cfg = QCConfig(**raw)

    # Write annotation metadata if configured
    if cfg.annotation is not None:
        write_annotation_metadata(zarr_dir=cfg.data_path, annotation=cfg.annotation)
        click.echo("Annotation metadata written.")

    # Build and run QC metrics
    metrics = []
    if cfg.focus_slice is not None:
        metrics.append(
            FocusSliceMetric(
                NA_det=cfg.focus_slice.NA_det,
                lambda_ill=cfg.focus_slice.lambda_ill,
                pixel_size=cfg.focus_slice.pixel_size,
                channel_names=cfg.focus_slice.channel_names,
                midband_fractions=cfg.focus_slice.midband_fractions,
                device=cfg.focus_slice.device,
            )
        )

    if not metrics and cfg.annotation is None:
        click.echo("No QC metrics configured. Nothing to do.")
        return

    if metrics:
        generate_qc_metadata(
            zarr_dir=cfg.data_path,
            metrics=metrics,
            num_workers=cfg.num_workers,
        )
        click.echo("QC metrics complete.")


@qc.command("audit-focus")
@click.option(
    "-d",
    "--data-path",
    required=True,
    type=click.Path(exists=True),
    help="OME-Zarr plate whose written focus_slice metadata to audit.",
)
@click.option("--channel", required=True, help="Channel whose focus indices to audit (e.g. Phase3D).")
def audit_focus(data_path: str, channel: str):
    """Report suspect focus_slice detections (edge z-index) across a plate.

    Z-depth-aware: a 2D acquisition (Z == 1) trivially focuses at slice 0, so it is
    reported as 2D with no flags; for 3D stacks, focus indices at a stack edge
    (0 or Z-1) are flagged as failed detections.
    """
    summary = audit_focus_slice(data_path, channel)
    click.echo(f"## Focus-slice audit — `{channel}`")
    click.echo(f"- Z depth: {summary['z_depth']}" + (" (2D — focus check N/A)" if summary["is_2d"] else ""))
    click.echo(f"- FOVs audited: {summary['n_fovs']}")
    click.echo(f"- Timepoints audited: {summary['n_timepoints_total']}")
    click.echo(f"- Suspect (edge) indices: {summary['n_suspect']}")
    if summary["fovs_affected"]:
        click.echo(f"- FOVs affected: {len(summary['fovs_affected'])}")
        for name, count in sorted(summary["fovs_affected"].items(), key=lambda kv: -kv[1]):
            click.echo(f"    - {name}: {count} suspect timepoint(s)")
    if summary["valid_focus"] is not None:
        vf = summary["valid_focus"]
        click.echo(f"- Valid focus z: min={vf['min']} max={vf['max']} mean={vf['mean']:.1f} median={vf['median']:.0f}")


def main():
    """Run the QC CLI."""
    qc()


if __name__ == "__main__":
    main()
