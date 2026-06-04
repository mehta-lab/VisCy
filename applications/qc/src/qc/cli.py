"""Click CLI for QC metrics."""

import click

from qc.annotation import write_annotation_metadata
from qc.config import QCConfig
from qc.focus import FocusSliceMetric
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


def main():
    """Run the QC CLI."""
    qc()


if __name__ == "__main__":
    main()
