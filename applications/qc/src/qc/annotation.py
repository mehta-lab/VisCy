"""Write channel annotation and experiment metadata to OME-Zarr zattrs."""

from iohub import open_ome_zarr

from airtable_utils.schemas import parse_position_name
from qc.config import AnnotationConfig


def write_annotation_metadata(zarr_dir: str, annotation: AnnotationConfig) -> None:
    """Write channel_annotation and experiment_metadata to .zattrs.

    channel_annotation is written to plate-level and every FOV position.
    experiment_metadata is written per-position based on well-path matching.

    Parameters
    ----------
    zarr_dir : str
        Path to the HCS OME-Zarr dataset.
    annotation : AnnotationConfig
        Annotation configuration with channel and experiment metadata.

    Raises
    ------
    ValueError
        If a channel name in config is not found in the plate, or if a well
        path in config does not exist in the plate.
    """
    plate = open_ome_zarr(zarr_dir, mode="r+")

    # Validate channel names
    plate_channels = set(plate.channel_names)
    for ch_name in annotation.channel_annotation:
        if ch_name not in plate_channels:
            plate.close()
            raise ValueError(
                f"Channel '{ch_name}' in annotation config not found in plate. "
                f"Available channels: {sorted(plate_channels)}"
            )

    # Collect well paths present in the plate
    plate_well_paths: set[str] = set()
    position_list = list(plate.positions())
    for name, _ in position_list:
        plate_well_paths.add(parse_position_name(name)[0])

    # Validate well paths
    for well_path in annotation.experiment_metadata:
        if well_path not in plate_well_paths:
            plate.close()
            raise ValueError(
                f"Well path '{well_path}' in annotation config not found in plate. "
                f"Available wells: {sorted(plate_well_paths)}"
            )

    # Serialize channel_annotation once
    channel_annotation_dict = {k: v.model_dump() for k, v in annotation.channel_annotation.items()}

    # Write channel_annotation to plate-level zattrs
    plate.zattrs["channel_annotation"] = channel_annotation_dict

    # Write per-position metadata
    for name, pos in position_list:
        # channel_annotation at every FOV
        pos.zattrs["channel_annotation"] = channel_annotation_dict

        # experiment_metadata per well
        well_path = parse_position_name(name)[0]
        if well_path in annotation.experiment_metadata:
            pos.zattrs["experiment_metadata"] = annotation.experiment_metadata[well_path].model_dump()

    plate.close()
