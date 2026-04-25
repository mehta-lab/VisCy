"""CLI for viscy-phenotyping image-based feature extraction."""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr

from viscy_data._typing import ULTRACK_INDEX_COLUMNS
from viscy_phenotyping.io import crop_2d
from viscy_phenotyping.profiler import compute_cell_features

_logger = logging.getLogger(__name__)

_INDEX_COLS = list(ULTRACK_INDEX_COLUMNS)


@click.group()
def main() -> None:
    """viscy-phenotyping: image-based phenotyping tools."""


def _get_fov_names(data_path: Path) -> list[str]:
    """Return all FOV names from an OME-Zarr plate."""
    with open_ome_zarr(data_path, mode="r") as plate:
        return [
            position.zgroup.name.strip("/")
            for _, well in plate.wells()
            for _, position in well.positions()
        ]


def _process_fov(
    fov_name: str,
    data_path: Path,
    tracks_path: Path,
    source_channels: tuple[str, ...],
    patch_size: tuple[int, int],
    nuclear_label_channel: str,
) -> pd.DataFrame:
    """Process a single FOV and return a DataFrame of features + obs columns."""
    csv_files = list((tracks_path / fov_name).glob("*.csv"))
    if not csv_files:
        _logger.warning("No tracking CSV found for %s — skipping.", fov_name)
        return pd.DataFrame()
    tracks_df = pd.read_csv(csv_files[0])

    rows: list[dict] = []

    with open_ome_zarr(data_path, mode="r") as data_plate, open_ome_zarr(
        tracks_path, mode="r"
    ) as tracks_plate:
        position = data_plate[fov_name]
        channel_names = list(position.channel_names)
        try:
            channel_indices = [channel_names.index(c) for c in source_channels]
        except ValueError as exc:
            raise click.ClickException(f"Channel not found in {fov_name}: {exc}") from exc

        tracks_position = tracks_plate[fov_name]
        try:
            label_channel_idx = list(tracks_position.channel_names).index(nuclear_label_channel)
        except ValueError as exc:
            raise click.ClickException(
                f"Nuclear label channel '{nuclear_label_channel}' not found in {fov_name}: {exc}"
            ) from exc

        img_array = position["0"]
        label_array = tracks_position["0"]

        click.echo(f"Processing {fov_name} — {len(tracks_df)} cells")

        for t, t_group in tracks_df.groupby("t"):
            t = int(t)
            label_img = np.squeeze(np.asarray(label_array[t, label_channel_idx]))
            img_frame = np.asarray(img_array[t, channel_indices])  # (C, [Z,] Y, X)
            if img_frame.ndim == 4:
                img_frame = img_frame.max(axis=1)  # (C, Y, X)

            for _, row in t_group.iterrows():
                y, x = int(row["y"]), int(row["x"])
                cell_id = int(row["track_id"])

                label_patch = crop_2d(label_img, y, x, patch_size)
                img_patch = crop_2d(img_frame, y, x, patch_size)

                feat = compute_cell_features(img_patch, label_patch, cell_id, list(source_channels))
                if not feat:
                    _logger.warning(
                        "No features for cell id=%d at %s t=%d — skipping.",
                        cell_id,
                        fov_name,
                        t,
                    )
                    continue

                obs = {"fov_name": fov_name}
                obs.update({col: row.get(col) for col in _INDEX_COLS if col != "fov_name"})
                rows.append({**obs, **feat})

    return pd.DataFrame(rows)


@main.command("write-header")
@click.option(
    "--source-channel",
    "source_channels",
    multiple=True,
    required=True,
    help="Channel name(s) — must match what will be passed to compute-features.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="CSV path to create with the header row.",
)
def write_header(source_channels: tuple[str, ...], output: Path) -> None:
    """Write an empty CSV with the correct header for a given set of channels.

    Run this before submitting the SLURM array so the shared output CSV is
    initialised with column names. Each array job then appends rows without
    re-writing the header.

    Example
    -------
    .. code-block:: bash

        viscy-phenotyping write-header \\
            --source-channel "raw mCherry EX561 EM600-37" \\
            --output features.csv
    """
    # Run compute_cell_features on a minimal dummy patch to discover all column names
    dummy_img = np.ones((len(source_channels), 64, 64), dtype=np.float32)
    dummy_label = np.zeros((64, 64), dtype=np.int32)
    dummy_label[20:44, 20:44] = 1
    feat = compute_cell_features(dummy_img, dummy_label, cell_id=1, channel_names=list(source_channels))
    columns = _INDEX_COLS + list(feat.keys())
    pd.DataFrame(columns=columns).to_csv(output, index=False)
    click.echo(f"Wrote header with {len(columns)} columns to {output}")


@main.command("list-fovs")
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="OME-Zarr fluorescence image store.",
)
def list_fovs(data_path: Path) -> None:
    """Print all FOV names in an OME-Zarr store, one per line."""
    for fov in _get_fov_names(data_path):
        click.echo(fov)


@main.command("compute-features")
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="OME-Zarr fluorescence image store.",
)
@click.option(
    "--tracks-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="OME-Zarr tracking store containing 2-D nuclear label images and per-FOV tracking CSVs.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output CSV file path.",
)
@click.option(
    "--source-channel",
    "source_channels",
    multiple=True,
    required=True,
    help="Channel name(s) to load from the fluorescence store. Repeat for multiple channels.",
)
@click.option(
    "--patch-size",
    nargs=2,
    type=int,
    default=(160, 160),
    show_default=True,
    help="YX patch size — must match the dynaCLR inference config.",
)
@click.option(
    "--nuclear-label-channel",
    required=True,
    help="Channel name for 2-D nuclear label images inside each tracking-zarr position.",
)
@click.option(
    "--fov-name",
    default=None,
    help="Process a single FOV only (for SLURM array jobs). If omitted, all FOVs are processed.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output CSV.",
)
def compute_features(
    data_path: Path,
    tracks_path: Path,
    output: Path,
    source_channels: tuple[str, ...],
    patch_size: tuple[int, int],
    nuclear_label_channel: str,
    fov_name: str | None,
    overwrite: bool,
) -> None:
    """Compute image-based phenotyping features and write to a CSV file.

    For parallel execution across FOVs, use ``--fov-name`` with a SLURM array job
    so each task writes its own CSV. Combine afterwards with ``merge-features``.

    Example (single run, all FOVs)
    --------------------------------
    .. code-block:: bash

        viscy-phenotyping compute-features \\
            --data-path /data/registered.zarr \\
            --tracks-path /data/tracks.zarr \\
            --output /results/features.csv \\
            --source-channel "raw mCherry EX561 EM600-37"

    Example (per-FOV, for SLURM array)
    ------------------------------------
    .. code-block:: bash

        viscy-phenotyping compute-features ... \\
            --fov-name A/1/000000 --output /results/fovs/A_1_000000.csv
    """
    if output.exists() and not overwrite:
        raise click.ClickException(f"{output} already exists. Use --overwrite to replace it.")

    fov_names = [fov_name] if fov_name is not None else _get_fov_names(data_path)

    dfs = [
        _process_fov(fov, data_path, tracks_path, source_channels, patch_size, nuclear_label_channel)
        for fov in fov_names
    ]
    result = pd.concat([df for df in dfs if not df.empty], ignore_index=True)

    if result.empty:
        raise click.ClickException("No features were extracted. Check that label IDs match the tracking zarr.")

    click.echo(f"Writing {len(result)} cells to {output}")
    result.to_csv(output, index=False)
    click.echo("Done.")


@main.command("merge-features")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing per-FOV CSV files.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output merged CSV path.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output CSV.",
)
def merge_features(input_dir: Path, output: Path, overwrite: bool) -> None:
    """Concatenate per-FOV CSV files into a single merged CSV.

    Example
    -------
    .. code-block:: bash

        viscy-phenotyping merge-features \\
            --input-dir /results/fovs/ \\
            --output /results/features.csv
    """
    if output.exists() and not overwrite:
        raise click.ClickException(f"{output} already exists. Use --overwrite to replace it.")

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise click.ClickException(f"No CSV files found in {input_dir}")

    click.echo(f"Merging {len(csv_paths)} CSVs from {input_dir}")
    merged = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    click.echo(f"Writing {len(merged)} total cells to {output}")
    merged.to_csv(output, index=False)
    click.echo("Done.")
