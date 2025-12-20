# %%
import logging
import os
from datetime import datetime
from pathlib import Path

import click
import napari
import numpy as np
import pandas as pd
from _reader import fov_to_layers
from iohub import open_ome_zarr
from napari.types import LayerDataTuple

from viscy.data.triplet import INDEX_COLUMNS

_logger = logging.getLogger("viscy")
_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
_logger.addHandler(console_handler)


def _ultrack_inv_tracks_df_forest(df: pd.DataFrame, no_parent=-1) -> dict[int, int]:
    """
    Vendored from ultrack.tracks.graph.inv_tracks_df_forest.
    """
    for col in ["track_id", "parent_track_id"]:
        if col not in df.columns:
            raise ValueError(
                f"The input dataframe does not contain the column '{col}'."
            )

    df = df.drop_duplicates("track_id")
    df = df[df["parent_track_id"] != no_parent]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[id] = parent_id
    return graph


def _ultrack_read_csv(path: Path | str) -> LayerDataTuple:
    """
    Vendored from ultrack.reader.napari_reader.read_csv.
    """
    if isinstance(path, str):
        path = Path(path)

    df = pd.read_csv(path)

    _logger.info(f"Read {len(df)} tracks from {path}")
    _logger.info(df.head())

    # For napari tracks layer, only use position columns: [track_id, t, y, x,z]
    tracks_cols = [
        "track_id",
        "t",
        "z",
        "y",
        "x",
    ]
    if "z" not in df.columns:
        tracks_cols.remove("z")

    if "parent_track_id" in df.columns:
        graph = _ultrack_inv_tracks_df_forest(df)
        _logger.info(f"Track lineage graph with length {len(graph)}")
    else:
        graph = None

    kwargs = {
        "features": df,  # Full dataframe with all columns is stored in features
        "name": path.name.removesuffix(".csv"),
        "graph": graph,
    }

    return (df[tracks_cols], kwargs, "tracks")


# %%
def open_image_and_tracks(
    images_dataset: Path,
    tracks_dataset: Path,
    fov_name: str,
    expand_z_for_tracking_labels: bool = True,
    load_tracks_layer: bool = True,
    tracks_z_index: int = -1,
) -> list[napari.types.LayerDataTuple]:
    """
    Load images and tracking labels.
    Also load predicted features (if supplied)
    and associate them with the tracking labels.
    To be used with napari-clusters-plotter plugin.

    Parameters
    ----------
    images_dataset : pathlib.Path
        Path to the images dataset (HCS OME-Zarr).
    tracks_dataset : pathlib.Path
        Path to the tracking labels dataset (HCS OME-Zarr).
        Potentially with a singleton Z dimension.
    fov_name : str
        Name of the FOV to load, e.g. `"A/12/2"`.
    expand_z_for_tracking_labels : bool
        Whether to expand the tracking labels to the Z dimension of the images.
    load_tracks_layer : bool
        Whether to load the tracks layer.
    tracks_z_index : int
        Index of the Z slice to place the 2D tracks, by default -1 (middle slice).

    Returns
    -------
    List[napari.types.LayerDataTuple]
        List of layers to add to the viewer.
        (image layers and one labels layer)
    """
    _logger.info(f"Loading images from {images_dataset}")
    image_plate = open_ome_zarr(images_dataset)
    image_fov = image_plate[fov_name]
    image_layers = fov_to_layers(image_fov)
    _logger.info(f"Loading tracking labels from {tracks_dataset}")
    tracks_plate = open_ome_zarr(tracks_dataset)
    tracks_fov = tracks_plate[fov_name]
    labels_layer = fov_to_layers(tracks_fov, layer_type="labels")[0]
    # TODO: remove this after https://github.com/napari/napari/issues/7327 is fixed
    labels_layer[0][0] = labels_layer[0][0].astype("uint32")
    image_z = image_fov["0"].slices
    if expand_z_for_tracking_labels:
        _logger.info(f"Expanding tracks to Z={image_z}")
        labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)
    image_layers.append(labels_layer)
    tracks_csv = next((tracks_dataset / fov_name.strip("/")).glob("*.csv"))
    if load_tracks_layer:
        _logger.info(f"Loading tracks from {str(tracks_csv)} with ultrack")
        tracks_layer = _ultrack_read_csv(tracks_csv)
        if tracks_z_index is not None:
            tracks_z_index = image_z // 2
        _logger.info(f"Placing tracks at Z={tracks_z_index}")
        tracks_layer[0].insert(loc=2, column="z", value=tracks_z_index)
        image_layers.append(tracks_layer)
    _logger.info(f"Finished loading {len(image_layers)} layers")
    _logger.debug(f"Layers: {image_layers}")
    return image_layers


def setup_annotation_layers(viewer: napari.Viewer) -> None:
    """
    Create four annotation points layers (one per event type).
    All layers default to 'add' mode for easy annotation.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to add layers to.
    """
    # Cell Division layer - mark mitosis events
    layer = viewer.add_points(
        ndim=4,
        size=20,
        face_color="blue",
        name="_mitosis_events",
    )
    layer.mode = "add"

    # Infection layer - mark infected events
    layer = viewer.add_points(
        ndim=4,
        size=20,
        face_color="orange",
        name="_infected_events",
    )
    layer.mode = "add"

    # Organelle remodeling layer - mark remodel events
    layer = viewer.add_points(
        ndim=4,
        size=20,
        face_color="purple",
        name="_remodel_events",
    )
    layer.mode = "add"

    # Cell death layer - mark death events
    layer = viewer.add_points(
        ndim=4,
        size=20,
        face_color="red",
        name="_death_events",
    )
    layer.mode = "add"


def save_annotations(
    viewer: napari.Viewer,
    output_path: Path,
    fov_name: str,
    tracks_zarr,
    tracks_csv_path: Path,
    diameter: int = 10,
) -> None:
    """
    Save napari point annotations to ultrack-style CSV.
    Expands annotations to all timepoints based on binary logic.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    output_path : Path
        Path to save the annotations CSV.
    fov_name : str
        FOV name for the fov_name column.
    tracks_zarr : Position
        Opened OME-Zarr position with segmentation labels.
    tracks_csv_path : Path
        Path to tracks CSV file.
    diameter : int
        Window diameter for robust label lookup.
    """
    # Load tracks CSV to get all track-timepoint combinations
    tracks_df = pd.read_csv(tracks_csv_path)

    # Capture annotation metadata
    annotator = os.getlogin()
    annotation_date = datetime.now().isoformat()
    annotation_version = "1.0"

    # Collect marked events from each layer
    marked_events = {
        "cell_division_state": [],  # mitosis events
        "infection_state": [],  # infected events
        "organelle_state": [],  # remodel events
        "cell_death_state": [],  # death events
    }

    # Process the four annotation layers
    layer_mapping = [
        ("_mitosis_events", "cell_division_state", "mitosis"),
        ("_infected_events", "infection_state", "infected"),
        ("_remodel_events", "organelle_state", "remodel"),
        ("_death_events", "cell_death_state", "dead"),
    ]

    for layer_name, event_type, event_state in layer_mapping:
        if layer_name in viewer.layers:
            points_layer = viewer.layers[layer_name]
            points_data = points_layer.data  # Shape: (n_points, 4) for [t, z, y, x]

            for point in points_data:
                t, z, y, x = [int(coord) for coord in point]

                # Load segmentation for this timepoint
                labels_image = tracks_zarr["0"][t, 0, 0]  # (C, Z, Y, X) → take C=0, Z=0

                # Get label value in window around point
                y_slice = slice(
                    max(0, y - diameter), min(labels_image.shape[0], y + diameter)
                )
                x_slice = slice(
                    max(0, x - diameter), min(labels_image.shape[1], x + diameter)
                )
                label_value = int(labels_image[y_slice, x_slice].mean())

                if label_value > 0:
                    marked_events[event_type].append({"track_id": label_value, "t": t})
                else:
                    _logger.warning(
                        f"Point at t={t}, y={y}, x={x} maps to background (label=0)"
                    )

    # Expand annotations to all timepoints based on binary logic
    all_annotations = []

    # Get all track-timepoint combinations
    all_track_timepoints = tracks_df[["track_id", "t"]].drop_duplicates()

    # Process each event type
    for track_id in all_track_timepoints["track_id"].unique():
        track_timepoints = all_track_timepoints[
            all_track_timepoints["track_id"] == track_id
        ]["t"].sort_values()

        # Cell Division: marked timepoints = mitosis, all others = interphase
        division_events = [
            e for e in marked_events["cell_division_state"] if e["track_id"] == track_id
        ]
        mitosis_timepoints = [e["t"] for e in division_events]

        # Infection: first marked timepoint onwards = infected, before = uninfected
        infection_events = [
            e for e in marked_events["infection_state"] if e["track_id"] == track_id
        ]
        first_infected_t = (
            min([e["t"] for e in infection_events]) if infection_events else None
        )

        # Organelle: first marked timepoint onwards = remodel, before = noremodel
        organelle_events = [
            e for e in marked_events["organelle_state"] if e["track_id"] == track_id
        ]
        first_remodel_t = (
            min([e["t"] for e in organelle_events]) if organelle_events else None
        )

        # Cell death: first marked timepoint onwards = dead, before = alive
        _death_events = [
            e for e in marked_events["cell_death_state"] if e["track_id"] == track_id
        ]
        first_death_t = min([e["t"] for e in _death_events]) if _death_events else None

        # Create one row per timepoint with all event states
        for t in track_timepoints:
            # Check if cell is dead at this timepoint
            is_dead = first_death_t is not None and t >= first_death_t

            if is_dead:
                # If dead, all other states are None
                cell_division_state = None
                infection_state = None
                organelle_state = None
                cell_death_state = "dead"
            else:
                # If alive, compute states normally
                cell_division_state = (
                    "mitosis" if t in mitosis_timepoints else "interphase"
                )
                infection_state = (
                    "infected"
                    if (first_infected_t is not None and t >= first_infected_t)
                    else "uninfected"
                    if first_infected_t is not None
                    else None
                )
                # Organelle: always has a value - remodel if marked, otherwise noremodel
                organelle_state = (
                    "remodel"
                    if (first_remodel_t is not None and t >= first_remodel_t)
                    else "noremodel"
                )
                cell_death_state = "alive" if first_death_t is not None else None

            all_annotations.append(
                {
                    "track_id": track_id,
                    "t": t,
                    "cell_division_state": cell_division_state,
                    "infection_state": infection_state,
                    "organelle_state": organelle_state,
                    "cell_death_state": cell_death_state,
                    "annotator": annotator,
                    "annotation_date": annotation_date,
                    "annotation_version": annotation_version,
                }
            )

    # Save to CSV
    if all_annotations:
        annotations_df = pd.DataFrame(all_annotations)

        # Merge with original tracks dataframe to preserve all INDEX_COLUMNS
        # Add fov_name column first
        tracks_df["fov_name"] = fov_name

        # Merge on track_id and t
        merged_df = tracks_df.merge(annotations_df, on=["track_id", "t"], how="left")

        # Reorder columns to have fov_name first, followed by INDEX_COLUMNS, then annotation columns, then metadata
        index_cols = [col for col in INDEX_COLUMNS if col in merged_df.columns]
        annotation_cols = [
            "cell_division_state",
            "infection_state",
            "organelle_state",
            "cell_death_state",
        ]
        metadata_cols = ["annotator", "annotation_date", "annotation_version"]
        column_order = index_cols + annotation_cols + metadata_cols
        merged_df = merged_df[column_order]

        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename for history
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"annotations_{fov_name.replace('/', '_')}"
        timestamped_csv = output_path / f"{base_name}_{annotator}_{timestamp}.csv"
        canonical_csv = output_path / f"{base_name}.csv"

        # Save with timestamp (for history/consensus)
        merged_df.to_csv(timestamped_csv, index=False)
        _logger.info(f"Saved {len(merged_df)} annotations to {timestamped_csv}")

        # Update canonical version (most recent)
        merged_df.to_csv(canonical_csv, index=False)
        _logger.info(f"Updated canonical annotations at {canonical_csv}")
    else:
        _logger.warning("No annotations to save")


@click.command()
@click.option(
    "--images-dataset",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to OME-Zarr dataset with images",
)
@click.option(
    "--tracks-dataset",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to OME-Zarr dataset with tracking labels",
)
@click.option(
    "--fov-name",
    "-f",
    type=str,
    required=True,
    help="FOV name to annotate (e.g., 'A/1/000000')",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Path folder to save annotations CSV (default: tracks_dataset/fov_name). It will use the fov_name to create the file name.",
)
def main(images_dataset, tracks_dataset, fov_name, output_path):
    """
    Interactive napari tool for annotating cell division, infection, remodeling, and death events.

    Keyboard shortcuts:
    - a/d: Step backward/forward in time
    - q/e: Cycle through annotation layers (mitosis → infected → remodel → death)
    - r: Enable interpolation mode (click start point → press 'r' → click end point to auto-interpolate)
         (For cell divisiona and organelle remodeling only)
    - s: Save annotations

    Annotation logic:
    - Mitosis: marked timepoints = mitosis, others = interphase
    - Infected: first marked timepoint onwards = infected, before = uninfected
    - Remodel: first marked timepoint onwards = remodel, before = noremodel
    - Death: first marked timepoint onwards = dead (all other states become None), before = alive
    """
    # Load image and track layers
    _logger.info("Loading images and tracks...")
    layers = open_image_and_tracks(images_dataset, tracks_dataset, fov_name)

    # Create napari viewer
    viewer = napari.Viewer()

    # Add all layers to viewer
    for layer_data, layer_kwargs, layer_type in layers:
        if layer_type == "image":
            viewer.add_image(layer_data, **layer_kwargs)
        elif layer_type == "labels":
            viewer.add_labels(layer_data, **layer_kwargs)
        elif layer_type == "tracks":
            viewer.add_tracks(layer_data, **layer_kwargs)

    # Open tracks zarr for label lookup
    tracks_plate = open_ome_zarr(tracks_dataset)
    tracks_fov = tracks_plate[fov_name]

    # Get tracks CSV path
    tracks_csv_path = next((Path(tracks_dataset) / fov_name.strip("/")).glob("*.csv"))

    # Setup annotation layers
    _logger.info("Setting up annotation layers...")
    setup_annotation_layers(viewer)

    # Set default output path if not provided
    if output_path is None:
        output_path = Path(tracks_dataset) / fov_name.strip("/")

    # State for interpolation mode
    interpolation_mode = {"enabled": False, "start_point": None}

    # List of annotation layers for cycling
    annotation_layers = [
        "_mitosis_events",
        "_infected_events",
        "_remodel_events",
        "_death_events",
    ]
    current_layer_index = {"index": 0}

    # Add mouse callback for interpolation and tracking last point
    def interpolate_points(layer, event):
        if (
            interpolation_mode["enabled"]
            and interpolation_mode["start_point"] is not None
        ):
            # Get click position for end point
            end_coords = np.array(layer.world_to_data(event.position))
            start_coords = interpolation_mode["start_point"]

            t1, t2 = int(start_coords[0]), int(end_coords[0])
            if t1 > t2:
                t1, t2 = t2, t1
                start_coords, end_coords = end_coords, start_coords

            # Add all intermediate timepoints (skip endpoints as they're already added)
            for t in range(t1 + 1, t2):
                alpha = (t - t1) / (t2 - t1)
                interpolated = start_coords + alpha * (end_coords - start_coords)
                interpolated[0] = t  # Set exact timepoint
                layer.add(interpolated)

            _logger.info(f"Interpolated {t2 - t1 - 1} points between t={t1} and t={t2}")

            # Reset interpolation mode
            interpolation_mode["enabled"] = False
            interpolation_mode["start_point"] = None
        else:
            # Track last added point for potential interpolation
            coords = np.array(layer.world_to_data(event.position))
            interpolation_mode["start_point"] = coords

    # Connect the callback to each annotation layer and add custom keybindings
    for layer_name in [
        "_mitosis_events",
        "_infected_events",
        "_remodel_events",
        "_death_events",
    ]:
        layer = viewer.layers[layer_name]
        layer.mouse_drag_callbacks.append(interpolate_points)

        # Bind shortcuts directly to each layer so they work when the layer is active
        @layer.bind_key("a")
        def layer_step_backward(layer):
            current_step = viewer.dims.current_step
            if current_step[0] > 0:
                viewer.dims.current_step = (current_step[0] - 1, *current_step[1:])
                _logger.info(f"Time: {viewer.dims.current_step[0]}")

        @layer.bind_key("d")
        def layer_step_forward(layer):
            current_step = viewer.dims.current_step
            max_step = viewer.dims.range[0][1] - 1
            if current_step[0] < max_step:
                viewer.dims.current_step = (current_step[0] + 1, *current_step[1:])
                _logger.info(f"Time: {viewer.dims.current_step[0]}")

        @layer.bind_key("s")
        def layer_save(layer):
            _logger.info("Saving annotations...")
            save_annotations(viewer, output_path, fov_name, tracks_fov, tracks_csv_path)

        @layer.bind_key("q")
        def layer_cycle_backward(layer):
            current_layer_index["index"] = (current_layer_index["index"] - 1) % len(
                annotation_layers
            )
            new_layer_name = annotation_layers[current_layer_index["index"]]
            viewer.layers.selection.active = viewer.layers[new_layer_name]
            _logger.info(f"Switched to {new_layer_name}")

        @layer.bind_key("e")
        def layer_cycle_forward(layer):
            current_layer_index["index"] = (current_layer_index["index"] + 1) % len(
                annotation_layers
            )
            new_layer_name = annotation_layers[current_layer_index["index"]]
            viewer.layers.selection.active = viewer.layers[new_layer_name]
            _logger.info(f"Switched to {new_layer_name}")

        @layer.bind_key("r")
        def layer_toggle_interpolation(layer):
            if interpolation_mode["start_point"] is not None:
                interpolation_mode["enabled"] = True
                start_t = int(interpolation_mode["start_point"][0])
                _logger.info(
                    f"Interpolation mode ENABLED - click end point to interpolate from t={start_t}"
                )
            else:
                _logger.info(
                    "No start point - add a point first, then press 'r' to enable interpolation"
                )

    # Set initial active layer
    viewer.layers.selection.active = viewer.layers["_mitosis_events"]

    _logger.info("Viewer ready! Annotation layers in 'add' mode by default")
    _logger.info("  Navigation: a/d = step backward/forward in time")
    _logger.info("  Layers: q/e = cycle through annotation layers")
    _logger.info("  Interpolation: click start point → press 'r' → click end point")
    _logger.info("  Save: s = save annotations")
    _logger.info("  Annotation layers: mitosis → infected → remodel → death")

    # Run napari
    napari.run()


if __name__ == "__main__":
    main()
