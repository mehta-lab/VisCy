"""
Interactive PHATE Track Viewer with Infection Annotations

This Dash application visualizes PHATE embeddings colored by infection status,
allows multi-track selection, and displays track timelines with images.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anndata import read_zarr
from dash import Dash, Input, Output, State, dcc, html
from iohub import open_ome_zarr
from PIL import Image

from viscy.representation.evaluation.annotation import load_annotation_anndata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Edit these paths and settings before running
# =============================================================================

# Data paths
ADATA_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/4-phenotyping/1-predictions/organellebox_mip_v3/timeaware/organelle_160patch_104ckpt_ver3max.zarr"
)
ANNOTATION_CSV_PATH = None
DATA_PATH = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/4-phenotyping/0-train-test/2025_08_26_A549_SEC61_TOMM20_ZIKV.zarr"
)

# Annotation settings (optional - set to None if no annotations)
ANNOTATION_CSV = None  # Set to None to skip annotation loading
ANNOTATION_COLUMN = "infection_status"
# Optional: Map annotation values to standardized categories
# Example for numeric labels: CATEGORIES = {0: "uninfected", 1: "infected", 2: "unknown"}
# Example for string labels: CATEGORIES = {"healthy": "uninfected", "sick": "infected"}
# Set to None if no remapping needed
CATEGORIES = {0: "uninfected", 1: "infected", 2: "unknown"}

# Color mode: "annotation", "time", or "track_id"
# This will be selectable in the UI, but you can set a default here
DEFAULT_COLOR_MODE = "annotation"  # or "time" or "track_id"

# FOV filtering (optional - set to None to include all FOVs)
# Can be a list of exact FOV names: ["A/1/0", "A/2/0"]
# Or a list of patterns to match: ["A_1", "A_2"] will match any FOV containing these strings
FOV_FILTER = ["A/1", "A/2"]  # Set to None to include all FOVs

# Image settings
CHANNELS = ["GFP EX488 EM525-45"]
Z_RANGE = (0, 1)  # Z slice range
YX_PATCH_SIZE = (160, 160)

# Server settings
PORT = 8050
DEBUG = False

# Colorblind-friendly palette
INFECTION_COLORS = {
    "uninfected": "#3498db",  # Blue
    "infected": "#e67e22",  # Orange
    "unknown": "#95a5a6",  # Gray
}

# =============================================================================
# DATA LOADING
# =============================================================================


def load_and_prepare_data(
    adata_path: Path,
    annotation_csv: Path | None = None,
    annotation_column: str | None = None,
    categories: dict | None = None,
    fov_filter: list[str] | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, list, bool]:
    """
    Load AnnData with PHATE embeddings and optional annotations.

    Parameters
    ----------
    adata_path : Path
        Path to AnnData zarr store with PHATE embeddings.
    annotation_csv : Path, optional
        Path to CSV file with annotations. If None, no annotations are loaded.
    annotation_column : str, optional
        Column name in CSV for annotation values.
    categories : dict, optional
        Dictionary to remap annotation categories (e.g., {0: "uninfected", 1: "infected"}).
    fov_filter : list[str], optional
        List of FOV names or patterns to filter. If provided, only FOVs containing
        any of these strings will be included.

    Returns
    -------
    adata : ad.AnnData
        AnnData object with optional annotations.
    plot_df : pd.DataFrame
        DataFrame with PHATE coordinates and metadata for plotting.
    track_options : list
        List of unique track identifiers for dropdown.
    has_annotations : bool
        Whether annotations were loaded.
    """
    logger.info(f"Loading AnnData from {adata_path}")
    adata = read_zarr(adata_path)
    logger.info(f"Loaded {adata.shape[0]} observations with {adata.shape[1]} features")

    # Filter by FOV if specified
    if fov_filter is not None and len(fov_filter) > 0:
        logger.info(f"Filtering FOVs by patterns: {fov_filter}")

        # Show sample of available FOV names for debugging
        unique_fovs = adata.obs["fov_name"].unique()
        logger.info(f"Available FOVs (showing first 10): {list(unique_fovs[:10])}")
        logger.info(f"Total unique FOVs: {len(unique_fovs)}")

        initial_count = adata.shape[0]

        # Convert fov_name to string for pattern matching
        fov_names = adata.obs["fov_name"].astype(str)

        # Create mask: keep FOVs where ANY pattern is IN the FOV name
        # e.g., "A_1" matches "A_1_0000000", "A_1_0000001", etc.
        mask = pd.Series([False] * len(fov_names), index=fov_names.index)
        for pattern in fov_filter:
            mask |= fov_names.str.contains(pattern, regex=False)

        adata = adata[mask]
        logger.info(
            f"Filtered to {adata.shape[0]} observations from {len(adata.obs['fov_name'].unique())} FOVs "
            f"(removed {initial_count - adata.shape[0]} observations)"
        )

    has_annotations = False

    # Load annotations if provided
    if annotation_csv is not None and annotation_csv.exists() and annotation_column:
        logger.info(f"Loading annotations from {annotation_csv}")
        adata = load_annotation_anndata(
            adata, str(annotation_csv), annotation_column, categories=categories
        )
        has_annotations = True

        # Filter out unknown infection status and NaN values
        initial_count = adata.shape[0]
        valid_mask = (adata.obs[annotation_column] != "unknown") & (
            adata.obs[annotation_column].notna()
        )
        adata = adata[valid_mask]
        logger.info(
            f"Filtered {initial_count - adata.shape[0]} invalid observations (unknown/NaN), {adata.shape[0]} remaining"
        )
    else:
        logger.info("No annotations provided, skipping annotation loading")

    # Check for PHATE embeddings
    if "X_phate" not in adata.obsm:
        raise ValueError("PHATE embeddings not found in AnnData.obsm['X_phate']")

    # Extract PHATE coordinates
    phate_coords = adata.obsm["X_phate"]
    logger.info(f"PHATE embedding shape: {phate_coords.shape}")

    # Create plotting DataFrame
    plot_df_dict = {
        "PHATE1": phate_coords[:, 0],
        "PHATE2": phate_coords[:, 1],
        "track_id": adata.obs["track_id"].values,
        "fov_name": adata.obs["fov_name"].values,
        "t": adata.obs["t"].values,
        "y": adata.obs["y"].values,
        "x": adata.obs["x"].values,
        "id": adata.obs["id"].values,
    }

    # Add annotation column if available
    if has_annotations and annotation_column in adata.obs:
        plot_df_dict["annotation"] = adata.obs[annotation_column].values

    plot_df = pd.DataFrame(plot_df_dict, index=adata.obs.index)

    # Create track options for dropdown (format: "fov_name/track_id")
    plot_df["track_key"] = (
        plot_df["fov_name"].astype(str) + "/" + plot_df["track_id"].astype(str)
    )
    track_options = sorted(plot_df["track_key"].unique())
    logger.info(f"Found {len(track_options)} unique tracks")

    return adata, plot_df, track_options, has_annotations


# =============================================================================
# PHATE VISUALIZATION
# =============================================================================


def create_phate_figure(
    df: pd.DataFrame,
    color_by: str = "annotation",
    selected_values: Optional[list] = None,
    selected_tracks: Optional[list[str]] = None,
    show_trajectories: bool = False,
    highlight_timepoint: Optional[int] = None,
) -> go.Figure:
    """
    Create interactive PHATE scatter plot with flexible coloring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PHATE coordinates and metadata.
    color_by : str
        Coloring mode: "annotation", "time", or "track_id".
    selected_values : list, optional
        Values to display (for categorical coloring).
    selected_tracks : list[str], optional
        Track keys to highlight.
    show_trajectories : bool, optional
        If True, draw lines connecting points in temporal order for selected tracks.
    highlight_timepoint : int, optional
        Specific timepoint to highlight for selected tracks.

    Returns
    -------
    fig : go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    if color_by == "annotation" and "annotation" in df.columns:
        # Color by annotation (categorical)
        if selected_values is None:
            selected_values = df["annotation"].unique().tolist()

        filtered_df = df[df["annotation"].isin(selected_values)]

        for status in selected_values:
            status_df = filtered_df[filtered_df["annotation"] == status]

            if len(status_df) == 0:
                continue

            # Check if any of these points are in selected tracks
            if selected_tracks:
                highlighted = status_df[status_df["track_key"].isin(selected_tracks)]
                background = status_df[~status_df["track_key"].isin(selected_tracks)]

                # Add background points (smaller, more transparent)
                if len(background) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=background["PHATE1"],
                            y=background["PHATE2"],
                            mode="markers",
                            name=f"{status} (background)",
                            marker=dict(
                                color=INFECTION_COLORS.get(status, "#95a5a6"),
                                size=4,
                                opacity=0.3,
                            ),
                            customdata=background[
                                ["track_key", "t", "fov_name", "track_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<b>Status:</b> " + status + "<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                # Add highlighted points (larger, more opaque)
                if len(highlighted) > 0:
                    fig.add_trace(
                        go.Scattergl(
                            x=highlighted["PHATE1"],
                            y=highlighted["PHATE2"],
                            mode="markers",
                            name=f"{status} (selected)",
                            marker=dict(
                                color=INFECTION_COLORS.get(status, "#95a5a6"),
                                size=8,
                                opacity=0.9,
                                line=dict(width=1, color="white"),
                            ),
                            customdata=highlighted[
                                ["track_key", "t", "fov_name", "track_id"]
                            ].values,
                            hovertemplate=(
                                "<b>Track:</b> %{customdata[0]}<br>"
                                "<b>Time:</b> %{customdata[1]}<br>"
                                "<b>FOV:</b> %{customdata[2]}<br>"
                                "<b>Status:</b> " + status + "<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
            else:
                # No tracks selected, show all points normally
                fig.add_trace(
                    go.Scattergl(
                        x=status_df["PHATE1"],
                        y=status_df["PHATE2"],
                        mode="markers",
                        name=status.replace("_", " ").title(),
                        marker=dict(
                            color=INFECTION_COLORS.get(status, "#95a5a6"),
                            size=5,
                            opacity=0.6,
                        ),
                        customdata=status_df[
                            ["track_key", "t", "fov_name", "track_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<b>Status:</b> " + status + "<br>"
                            "<extra></extra>"
                        ),
                    )
                )

    elif color_by == "time":
        # Color by timepoint (continuous colorscale)
        filtered_df = df.copy()

        # Separate selected tracks and background
        if selected_tracks:
            highlighted = filtered_df[filtered_df["track_key"].isin(selected_tracks)]
            background = filtered_df[~filtered_df["track_key"].isin(selected_tracks)]

            # Add background
            if len(background) > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=background["PHATE1"],
                        y=background["PHATE2"],
                        mode="markers",
                        name="background",
                        marker=dict(
                            color=background["t"],
                            colorscale="Viridis",
                            size=4,
                            opacity=0.3,
                            showscale=False,
                        ),
                        customdata=background[
                            ["track_key", "t", "fov_name", "track_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            # Add highlighted
            if len(highlighted) > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=highlighted["PHATE1"],
                        y=highlighted["PHATE2"],
                        mode="markers",
                        name="time (selected tracks)",
                        marker=dict(
                            color=highlighted["t"],
                            colorscale="Viridis",
                            size=8,
                            opacity=0.9,
                            showscale=True,
                            colorbar=dict(title="Time"),
                            line=dict(width=1, color="white"),
                        ),
                        customdata=highlighted[
                            ["track_key", "t", "fov_name", "track_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
        else:
            # No tracks selected
            fig.add_trace(
                go.Scattergl(
                    x=filtered_df["PHATE1"],
                    y=filtered_df["PHATE2"],
                    mode="markers",
                    name="time",
                    marker=dict(
                        color=filtered_df["t"],
                        colorscale="Viridis",
                        size=5,
                        opacity=0.6,
                        showscale=True,
                        colorbar=dict(title="Time"),
                    ),
                    customdata=filtered_df[
                        ["track_key", "t", "fov_name", "track_id"]
                    ].values,
                    hovertemplate=(
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    elif color_by == "track_id":
        # Color by track_id
        filtered_df = df.copy()

        if selected_tracks:
            # Only show selected tracks
            filtered_df = filtered_df[filtered_df["track_key"].isin(selected_tracks)]

            # Generate colors for selected tracks
            colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
            track_colors = {
                track: colors[i % len(colors)]
                for i, track in enumerate(selected_tracks)
            }

            for track_key in selected_tracks:
                track_df = filtered_df[filtered_df["track_key"] == track_key]

                if len(track_df) == 0:
                    continue

                fig.add_trace(
                    go.Scattergl(
                        x=track_df["PHATE1"],
                        y=track_df["PHATE2"],
                        mode="markers",
                        name=track_key,
                        marker=dict(
                            color=track_colors[track_key],
                            size=8,
                            opacity=0.7,
                            line=dict(width=1, color="white"),
                        ),
                        customdata=track_df[
                            ["track_key", "t", "fov_name", "track_id"]
                        ].values,
                        hovertemplate=(
                            "<b>Track:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>FOV:</b> %{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
        else:
            # Show all points in gray if no tracks selected
            fig.add_trace(
                go.Scattergl(
                    x=filtered_df["PHATE1"],
                    y=filtered_df["PHATE2"],
                    mode="markers",
                    name="all tracks",
                    marker=dict(
                        color="#95a5a6",
                        size=5,
                        opacity=0.4,
                    ),
                    customdata=filtered_df[
                        ["track_key", "t", "fov_name", "track_id"]
                    ].values,
                    hovertemplate=(
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    # Add trajectory lines for selected tracks
    if show_trajectories and selected_tracks:
        for track_key in selected_tracks:
            track_df = df[df["track_key"] == track_key].sort_values("t")

            if len(track_df) < 2:
                continue

            # Get color based on mode
            if color_by == "annotation" and "annotation" in track_df.columns:
                annotation_value = track_df["annotation"].iloc[0]
                color = INFECTION_COLORS.get(annotation_value, "#95a5a6")
            else:
                color = "#666666"  # Default gray for other modes

            # Add line trace connecting points in temporal order
            fig.add_trace(
                go.Scattergl(
                    x=track_df["PHATE1"],
                    y=track_df["PHATE2"],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2,
                        dash="solid",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add arrow markers at intermediate points to show direction
            # Use every 3rd point to avoid cluttering
            for i in range(0, len(track_df) - 1, 3):
                row_current = track_df.iloc[i]
                row_next = track_df.iloc[i + 1]

                # Calculate arrow direction
                row_next["PHATE1"] - row_current["PHATE1"]
                row_next["PHATE2"] - row_current["PHATE2"]

                # Add arrow annotation
                fig.add_annotation(
                    x=row_next["PHATE1"],
                    y=row_next["PHATE2"],
                    ax=row_current["PHATE1"],
                    ay=row_current["PHATE2"],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    opacity=0.6,
                )

    # Highlight specific timepoint for selected tracks
    if highlight_timepoint is not None and selected_tracks:
        highlight_points = []
        for track_key in selected_tracks:
            track_df = df[df["track_key"] == track_key]
            timepoint_data = track_df[track_df["t"] == highlight_timepoint]

            if len(timepoint_data) > 0:
                highlight_points.append(timepoint_data)

        if highlight_points:
            highlight_df = pd.concat(highlight_points, ignore_index=True)

            # Add highlighted points as a separate trace with larger markers
            fig.add_trace(
                go.Scattergl(
                    x=highlight_df["PHATE1"],
                    y=highlight_df["PHATE2"],
                    mode="markers",
                    name=f"t={highlight_timepoint}",
                    marker=dict(
                        size=15,
                        color="yellow",
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    customdata=highlight_df[
                        ["track_key", "t", "fov_name", "track_id"]
                    ].values,
                    hovertemplate=(
                        "<b>⭐ HIGHLIGHTED ⭐</b><br>"
                        "<b>Track:</b> %{customdata[0]}<br>"
                        "<b>Time:</b> %{customdata[1]}<br>"
                        "<b>FOV:</b> %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    # Update layout
    fig.update_layout(
        title="PHATE Embedding - Click points to select tracks",
        xaxis_title="PHATE1",
        yaxis_title="PHATE2",
        hovermode="closest",
        template="plotly_white",
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig


# =============================================================================
# IMAGE LOADING AND CACHING
# =============================================================================


class ImageCache:
    """Cache for loading and storing microscopy images."""

    def __init__(
        self,
        data_path: Path,
        channels: list[str],
        z_range: tuple[int, int],
        yx_patch_size: tuple[int, int],
    ):
        """
        Initialize image cache.

        Parameters
        ----------
        data_path : Path
            Path to microscopy data zarr store.
        channels : list[str]
            Channel names to load.
        z_range : tuple[int, int]
            Z slice range (start, end).
        yx_patch_size : tuple[int, int]
            Patch size in (Y, X) dimensions.
        """
        self.data_path = data_path
        self.channels = channels
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size
        self.cache = {}  # (fov_name, track_id, t, channel) -> base64 string

        logger.info(f"Initializing ImageCache with data from {data_path}")
        try:
            self.data_store = open_ome_zarr(str(data_path), mode="r")
            logger.info("Successfully opened data store")
        except Exception as e:
            logger.error(f"Failed to open data store: {e}")
            self.data_store = None

    @staticmethod
    def _normalize_image(img_array: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 255] range.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            Normalized image as uint8.
        """
        min_val = img_array.min()
        max_val = img_array.max()
        if min_val == max_val:
            return np.zeros_like(img_array, dtype=np.uint8)
        return ((img_array - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

    @staticmethod
    def _numpy_to_base64(img_array: np.ndarray) -> str:
        """
        Convert numpy array to base64-encoded JPEG string.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array (uint8).

        Returns
        -------
        str
            Base64-encoded JPEG string with data URI prefix.
        """
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def load_image(
        self, fov_name: str, track_id: int, t: int, channel: str, y: float, x: float
    ) -> Optional[str]:
        """
        Load image for specific observation.

        Parameters
        ----------
        fov_name : str
            Field of view name.
        track_id : int
            Track identifier.
        t : int
            Timepoint.
        channel : str
            Channel name.
        y : float
            Y coordinate (centroid).
        x : float
            X coordinate (centroid).

        Returns
        -------
        str or None
            Base64-encoded image string, or None if loading fails.
        """
        cache_key = (fov_name, track_id, t, channel)

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.data_store is None:
            logger.error("Data store not initialized")
            return None

        try:
            # Get position from data store
            position = self.data_store[fov_name]

            # Get channel index
            channel_idx = position.get_channel_index(channel)

            # Calculate patch bounds
            y_int, x_int = int(round(y)), int(round(x))
            y_half, x_half = self.yx_patch_size[0] // 2, self.yx_patch_size[1] // 2

            # Access image using tensorstore
            image = position["0"].tensorstore()
            patch = (
                image.oindex[
                    t,
                    [channel_idx],
                    slice(self.z_range[0], self.z_range[1]),
                    slice(y_int - y_half, y_int + y_half),
                    slice(x_int - x_half, x_int + x_half),
                ]
                .read()
                .result()
            )

            # Take max projection over Z and remove channel dimension
            patch_2d = patch[0].max(axis=0)

            # Normalize and convert to base64
            patch_normalized = self._normalize_image(patch_2d)
            img_base64 = self._numpy_to_base64(patch_normalized)

            # Cache the result
            self.cache[cache_key] = img_base64

            return img_base64

        except Exception as e:
            logger.error(f"Failed to load image for {fov_name}/{track_id}/t={t}: {e}")
            return None


# =============================================================================
# TRACK TIMELINE DISPLAY
# =============================================================================


def create_track_timeline(
    selected_tracks: list[str],
    adata: ad.AnnData,
    plot_df: pd.DataFrame,
    image_cache: ImageCache,
    channels: list[str],
) -> list:
    """
    Create timeline displays for selected tracks.

    Parameters
    ----------
    selected_tracks : list[str]
        List of track keys (format: "fov_name/track_id").
    adata : ad.AnnData
        AnnData object with full data.
    plot_df : pd.DataFrame
        DataFrame with track metadata.
    image_cache : ImageCache
        Image cache instance.
    channels : list[str]
        Channel names to display.

    Returns
    -------
    list
        List of Dash HTML components for track timelines.
    """
    if not selected_tracks:
        return []

    timelines = []

    for track_key in selected_tracks[:10]:  # Limit to 10 tracks
        # Parse track key
        fov_name, track_id_str = track_key.rsplit("/", 1)
        track_id = int(track_id_str)

        # Get all observations for this track
        track_data = plot_df[plot_df["track_key"] == track_key].sort_values("t")

        if len(track_data) == 0:
            continue

        # Get annotation status if available
        if "annotation" in track_data.columns:
            annotation_value = track_data["annotation"].iloc[0]
            header_text = f"Track: {track_key} | Status: {annotation_value}"
            header_color = INFECTION_COLORS.get(annotation_value, "#95a5a6")
        else:
            header_text = f"Track: {track_key}"
            header_color = "#95a5a6"

        # Create header
        header = html.Div(
            [
                html.H4(
                    header_text,
                    style={
                        "margin": "10px 0",
                        "padding": "10px",
                        "backgroundColor": header_color,
                        "color": "white",
                        "borderRadius": "5px",
                    },
                )
            ]
        )

        # Create timepoint labels
        timepoint_labels = []
        for idx, row in track_data.iterrows():
            timepoint_labels.append(
                html.Div(
                    f"t={int(row['t'])}",
                    style={
                        "width": "150px",
                        "minWidth": "150px",
                        "textAlign": "center",
                        "fontWeight": "bold",
                        "padding": "5px",
                    },
                )
            )

        # Create image rows for each channel
        channel_rows = []
        for channel in channels:
            images = []
            for idx, row in track_data.iterrows():
                img_base64 = image_cache.load_image(
                    fov_name=fov_name,
                    track_id=track_id,
                    t=int(row["t"]),
                    channel=channel,
                    y=row.get("y", 0),
                    x=row.get("x", 0),
                )

                if img_base64:
                    images.append(
                        html.Div(
                            html.Img(
                                src=img_base64,
                                style={
                                    "width": "150px",
                                    "height": "150px",
                                    "objectFit": "contain",
                                },
                            ),
                            style={"padding": "5px", "minWidth": "150px"},
                        )
                    )
                else:
                    images.append(
                        html.Div(
                            "Failed to load",
                            style={
                                "width": "150px",
                                "minWidth": "150px",
                                "height": "150px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "backgroundColor": "#f0f0f0",
                                "color": "#999",
                            },
                        )
                    )

            channel_row = html.Div(
                images,
                style={
                    "display": "flex",
                    "flexDirection": "row",
                },
            )

            channel_rows.append((channel, channel_row))

        # Create scrollable content area with labels and images
        scrollable_content = html.Div(
            [
                # Labels row
                html.Div(
                    timepoint_labels,
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "borderBottom": "2px solid #ddd",
                        "marginBottom": "10px",
                    },
                ),
                # Channel rows
                *[
                    html.Div(
                        channel_row,
                        style={"marginBottom": "10px"},
                    )
                    for _, channel_row in channel_rows
                ],
            ],
            style={
                "overflowX": "auto",
                "marginLeft": "100px",
            },
        )

        # Create channel labels column (fixed on left)
        channel_labels = html.Div(
            [
                html.Div(
                    "",  # Empty space above labels for alignment
                    style={"height": "43px"},  # Match height of labels row + border
                ),
                *[
                    html.Div(
                        channel_name,
                        style={
                            "fontWeight": "bold",
                            "padding": "5px",
                            "height": "160px",  # Match image height + margin
                            "display": "flex",
                            "alignItems": "center",
                        },
                    )
                    for channel_name, _ in channel_rows
                ],
            ],
            style={
                "position": "absolute",
                "left": "15px",
                "width": "100px",
                "backgroundColor": "#fafafa",
            },
        )

        # Combine into track section with relative positioning
        track_section = html.Div(
            [header, channel_labels, scrollable_content],
            style={
                "position": "relative",
                "marginBottom": "30px",
                "padding": "15px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "backgroundColor": "#fafafa",
            },
        )

        timelines.append(track_section)

    if len(selected_tracks) > 10:
        timelines.append(
            html.Div(
                f"Showing first 10 of {len(selected_tracks)} selected tracks",
                style={
                    "padding": "10px",
                    "color": "#e67e22",
                    "fontStyle": "italic",
                },
            )
        )

    return timelines


# =============================================================================
# DASH APPLICATION
# =============================================================================

# Load data
logger.info("Loading data...")
adata, plot_df, track_options, has_annotations = load_and_prepare_data(
    ADATA_PATH,
    ANNOTATION_CSV_PATH if ANNOTATION_CSV is not None else None,
    ANNOTATION_COLUMN,
    CATEGORIES,
    FOV_FILTER,
)

# Initialize image cache
logger.info("Initializing image cache...")
image_cache = ImageCache(DATA_PATH, CHANNELS, Z_RANGE, YX_PATCH_SIZE)

# Get unique annotation values if available
if has_annotations and "annotation" in plot_df.columns:
    annotation_values = sorted(
        [s for s in plot_df["annotation"].unique() if pd.notna(s)]
    )
else:
    annotation_values = []

# Create Dash app
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "PHATE Track Viewer with Infection Annotations",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        # Controls
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Color points by:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="color-mode",
                            options=[
                                opt
                                for opt in [
                                    (
                                        {"label": "Annotation", "value": "annotation"}
                                        if has_annotations
                                        else None
                                    ),
                                    {"label": "Time", "value": "time"},
                                    {"label": "Track ID", "value": "track_id"},
                                ]
                                if opt is not None
                            ],
                            value=DEFAULT_COLOR_MODE if has_annotations else "time",
                            clearable=False,
                            style={"width": "200px", "marginLeft": "10px"},
                        ),
                    ],
                    style={
                        "marginBottom": "15px",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Filter by annotation:", style={"fontWeight": "bold"}
                        ),
                        dcc.Checklist(
                            id="annotation-filter",
                            options=[
                                {
                                    "label": status.replace("_", " ").title(),
                                    "value": status,
                                }
                                for status in annotation_values
                            ],
                            value=annotation_values,
                            inline=True,
                            style={"marginLeft": "10px"},
                        ),
                    ],
                    style={
                        "marginBottom": "15px",
                        "display": "none" if not has_annotations else "block",
                    },
                    id="annotation-filter-container",
                ),
                html.Div(
                    [
                        html.Label("Selected Tracks:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="track-selector",
                            options=[
                                {"label": track, "value": track}
                                for track in track_options
                            ],
                            multi=True,
                            placeholder="Click points on plot to select tracks, or choose from dropdown...",
                            value=[],
                            style={"width": "100%"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id="show-trajectories",
                            options=[
                                {
                                    "label": " Show track trajectories (lines connecting points in temporal order)",
                                    "value": "show",
                                }
                            ],
                            value=[],
                            inline=True,
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Highlight timepoint (for selected tracks):",
                            style={"fontWeight": "bold", "marginRight": "10px"},
                        ),
                        dcc.Input(
                            id="highlight-timepoint",
                            type="number",
                            placeholder="Enter timepoint (e.g., 0, 1, 2...)",
                            min=0,
                            step=1,
                            style={"width": "200px"},
                        ),
                        html.Span(
                            " ⭐ Highlights as yellow star on plot",
                            style={
                                "marginLeft": "10px",
                                "color": "#666",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "10px",
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
            ],
            style={
                "padding": "20px",
                "backgroundColor": "#f9f9f9",
                "borderRadius": "5px",
                "marginBottom": "20px",
            },
        ),
        # PHATE plot
        dcc.Loading(
            dcc.Graph(
                id="phate-scatter",
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                },
            ),
            type="default",
        ),
        # Track timelines
        html.Div(
            id="track-timelines",
            style={
                "marginTop": "30px",
                "padding": "20px",
            },
        ),
    ],
    style={
        "maxWidth": "1800px",
        "margin": "0 auto",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif",
    },
)


# Callback 1: Update PHATE plot based on color mode and filters
@app.callback(
    Output("phate-scatter", "figure"),
    [
        Input("color-mode", "value"),
        Input("annotation-filter", "value"),
        Input("track-selector", "value"),
        Input("show-trajectories", "value"),
        Input("highlight-timepoint", "value"),
    ],
)
def update_phate_plot(
    color_mode,
    selected_annotations,
    selected_tracks,
    show_trajectories,
    highlight_timepoint,
):
    """Update PHATE scatter plot based on filters and selections."""
    show_traj = "show" in show_trajectories if show_trajectories else False
    return create_phate_figure(
        plot_df,
        color_mode,
        selected_annotations,
        selected_tracks,
        show_traj,
        highlight_timepoint,
    )


# Callback 1.5: Show/hide annotation filter based on color mode
@app.callback(
    Output("annotation-filter-container", "style"), Input("color-mode", "value")
)
def toggle_annotation_filter(color_mode):
    """Show annotation filter only when coloring by annotation."""
    if color_mode == "annotation" and has_annotations:
        return {"marginBottom": "15px", "display": "block"}
    return {"marginBottom": "15px", "display": "none"}


# Callback 2: Add track on click
@app.callback(
    Output("track-selector", "value"),
    Input("phate-scatter", "clickData"),
    State("track-selector", "value"),
)
def add_track_on_click(click_data, current_tracks):
    """Add clicked track to selection."""
    if click_data is None:
        return current_tracks

    # Parse customdata to get track_key
    point = click_data["points"][0]
    track_key = point["customdata"][0]

    # Add to selection if not already present
    if current_tracks is None:
        current_tracks = []

    if track_key not in current_tracks:
        return current_tracks + [track_key]

    return current_tracks


# Callback 3: Update track timelines based on selected tracks
@app.callback(Output("track-timelines", "children"), Input("track-selector", "value"))
def update_timelines(selected_tracks):
    """Update track timeline displays."""
    if not selected_tracks:
        return html.Div(
            "Select tracks by clicking points on the PHATE plot or using the dropdown above.",
            style={
                "textAlign": "center",
                "color": "#999",
                "padding": "50px",
                "fontSize": "18px",
            },
        )

    return create_track_timeline(selected_tracks, adata, plot_df, image_cache, CHANNELS)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting PHATE Track Viewer on http://localhost:{PORT}")
    logger.info(f"Loaded {len(track_options)} tracks from {ADATA_PATH}")
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
