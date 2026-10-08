"""Per-FOV tracking table reader.

Tracks for a FOV ``<row>/<col>/<fov>`` live under ``<tracks_root>/<row>/<col>/<fov>/``
in one of two layouts:

* ``tracks.geff`` — a GEFF graph (https://liveimagetrackingtools.org/geff/), as written
  by ``biahub track`` inside the image plate FOV. Preferred when present.
* exactly one ``*.csv`` — the ultrack ``to_tracks_layer`` table, either inside the plate
  FOV (``tracks_<row>_<col>_<fov>.csv``) or in a separate tracking zarr.

Both are returned as the same DataFrame: ``track_id, t, [z], y, x, id,
parent_track_id, parent_id``, with ``-1`` marking roots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["GEFF_TRACKS_NAME", "read_fov_tracks", "read_geff_tracks"]

GEFF_TRACKS_NAME = "tracks.geff"

_NO_PARENT = -1


def read_fov_tracks(tracks_root: str | Path, fov_path: str) -> pd.DataFrame:
    """Read the tracking table of one FOV.

    Parameters
    ----------
    tracks_root : str | Path
        Root holding per-FOV track directories: the HCS plate itself for in-plate
        tracks, or a separate tracking zarr.
    fov_path : str
        FOV path relative to ``tracks_root``, e.g. ``"A/1/000000"``.

    Returns
    -------
    pd.DataFrame
        Columns ``track_id, t, [z], y, x, id, parent_track_id, parent_id``.

    Raises
    ------
    FileNotFoundError
        If the FOV has neither a ``tracks.geff`` nor a tracking CSV.
    ValueError
        If the FOV has no ``tracks.geff`` and more than one CSV.
    """
    tracks_dir = Path(tracks_root) / fov_path
    geff_path = tracks_dir / GEFF_TRACKS_NAME
    if geff_path.is_dir():
        return read_geff_tracks(geff_path)
    csv_files = list(tracks_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tracking CSV in {tracks_dir}")
    if len(csv_files) > 1:
        raise ValueError(f"Expected exactly one tracking CSV in {tracks_dir}, found: {csv_files}")
    return pd.read_csv(csv_files[0])


def _import_geff():
    try:
        from geff.core_io import read_to_memory
        from geff_spec import GeffMetadata
    except ImportError as e:
        raise ImportError(
            "Reading tracks.geff requires the 'geff' package. Install with: pip install 'viscy-data[tracks]'"
        ) from e
    return GeffMetadata, read_to_memory


def read_geff_tracks(geff_path: str | Path) -> pd.DataFrame:
    """Read a GEFF track graph into the ultrack tracks CSV layout.

    Node ids become ``id``. ``parent_id`` is the source node of each node's
    incoming edge (``-1`` for roots). ``parent_track_id`` is read from the node
    property of that name when stored; otherwise it is the track of the parent
    of the track's first node, constant over the track (``-1`` for root
    tracks), matching ultrack's ``to_tracks_layer``.

    Parameters
    ----------
    geff_path : str | Path
        Path to the GEFF zarr group (e.g. ``<plate>/A/1/000000/tracks.geff``).

    Returns
    -------
    pd.DataFrame
        Columns ``track_id, t, [z], y, x, id, parent_track_id, parent_id``,
        sorted by ``(track_id, t)``. ``z`` is present only if the GEFF has a
        ``z`` node property.
    """
    GeffMetadata, read_to_memory = _import_geff()
    geff_path = Path(geff_path)

    # Only load the node properties we need; skip masks, bboxes, etc.
    metadata = GeffMetadata.read(str(geff_path))
    available = set(metadata.node_props_metadata or {})
    track_prop = (metadata.track_node_props or {}).get("tracklet", "track_id")
    missing = {track_prop, "t", "y", "x"} - available
    if missing:
        raise ValueError(f"GEFF {geff_path} is missing node properties {sorted(missing)}; found {sorted(available)}")
    wanted = [track_prop, "t", "y", "x"] + [p for p in ("z", "parent_track_id") if p in available]
    graph = read_to_memory(str(geff_path), node_props=wanted, edge_props=[])

    node_ids = np.asarray(graph["node_ids"]).astype(np.int64)
    props = {name: np.asarray(prop["values"]) for name, prop in graph["node_props"].items()}

    df = pd.DataFrame({"track_id": props[track_prop].astype(np.int64), "t": props["t"].astype(np.int64)})
    if "z" in props:
        df["z"] = props["z"]
    df["y"] = props["y"]
    df["x"] = props["x"]
    df["id"] = node_ids

    # parent_id: source of the (single) incoming edge, -1 for roots.
    edges = np.asarray(graph["edge_ids"]).astype(np.int64).reshape(-1, 2)
    parent_of = pd.Series(edges[:, 0], index=edges[:, 1])
    if parent_of.index.has_duplicates:
        raise ValueError(f"GEFF {geff_path} has nodes with more than one parent; expected a forest.")
    df["parent_id"] = parent_of.reindex(node_ids).fillna(_NO_PARENT).astype(np.int64).to_numpy()

    if "parent_track_id" in props:
        df["parent_track_id"] = props["parent_track_id"].astype(np.int64)
    else:
        track_of = pd.Series(df["track_id"].to_numpy(), index=node_ids)
        first = df.sort_values("t").drop_duplicates("track_id")
        first_parent = first["parent_id"].to_numpy()
        parent_track = np.where(
            first_parent == _NO_PARENT,
            _NO_PARENT,
            track_of.reindex(first_parent).fillna(_NO_PARENT).astype(np.int64).to_numpy(),
        )
        df["parent_track_id"] = (
            df["track_id"].map(pd.Series(parent_track, index=first["track_id"].to_numpy())).astype(np.int64)
        )

    columns = ["track_id", "t", *(["z"] if "z" in df.columns else []), "y", "x", "id", "parent_track_id", "parent_id"]
    return df[columns].sort_values(["track_id", "t"], kind="stable").reset_index(drop=True)
