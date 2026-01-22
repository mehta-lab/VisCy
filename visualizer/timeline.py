"""
Track timeline visualization component.

This module provides functions for creating HTML-based track timelines that
display images from multiple channels across timepoints.

Functions
---------
create_track_timeline : Generate timeline HTML components for selected tracks
"""

import logging

import anndata as ad
import pandas as pd
from dash import html

from .config import INFECTION_COLORS
from .image_cache import ImageCache, MultiDatasetImageCache

logger = logging.getLogger(__name__)


def create_track_timeline(
    selected_tracks: list[str],
    adata: ad.AnnData,
    plot_df: pd.DataFrame,
    image_cache: ImageCache | MultiDatasetImageCache,
    default_channels: list[str],
) -> list:
    """
    Create timeline displays for selected tracks.

    Parameters
    ----------
    selected_tracks : list[str]
        List of track keys (format: "dataset_id/fov_name/track_id").
    adata : ad.AnnData
        AnnData object with full data.
    plot_df : pd.DataFrame
        DataFrame with track metadata.
    image_cache : ImageCache or MultiDatasetImageCache
        Image cache instance.
    default_channels : list[str]
        Default channel names (used for single dataset or if dataset doesn't specify).

    Returns
    -------
    list
        List of Dash HTML components for track timelines.
    """
    if not selected_tracks:
        return []

    timelines = []
    is_multi_dataset = isinstance(image_cache, MultiDatasetImageCache)

    for track_key in selected_tracks[:10]:
        prefix, track_id_str = track_key.rsplit("/", 1)
        track_id = int(track_id_str)
        dataset_id, fov_name = prefix.split("/", 1)

        if is_multi_dataset:
            channels = image_cache.get_channels(dataset_id)
        else:
            channels = default_channels

        track_data = plot_df[plot_df["track_key"] == track_key].sort_values("t")

        if len(track_data) == 0:
            continue

        if "annotation" in track_data.columns:
            annotation_value = track_data["annotation"].iloc[0]
            header_text = f"Dataset: {dataset_id} | Track: {fov_name}/{track_id} | Status: {annotation_value}"
            header_color = INFECTION_COLORS.get(annotation_value, "#95a5a6")
        else:
            header_text = f"Dataset: {dataset_id} | Track: {fov_name}/{track_id}"
            header_color = "#95a5a6"

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

        channel_rows = []
        first_image_logged = False
        for channel in channels:
            images = []
            for idx, row in track_data.iterrows():
                if is_multi_dataset:
                    if not first_image_logged:
                        logger.info(
                            f"Loading image: dataset_id={dataset_id}, fov={fov_name}, "
                            f"track={track_id}, t={int(row['t'])}, channel={channel}"
                        )
                        first_image_logged = True
                    img_base64 = image_cache.load_image(
                        dataset_id=dataset_id,
                        fov_name=fov_name,
                        track_id=track_id,
                        t=int(row["t"]),
                        channel=channel,
                        y=row.get("y", 0),
                        x=row.get("x", 0),
                    )
                else:
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

        scrollable_content = html.Div(
            [
                html.Div(
                    timepoint_labels,
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "borderBottom": "2px solid #ddd",
                        "marginBottom": "10px",
                    },
                ),
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

        channel_labels = html.Div(
            [
                html.Div(
                    "",
                    style={"height": "43px"},
                ),
                *[
                    html.Div(
                        channel_name,
                        style={
                            "fontWeight": "bold",
                            "padding": "5px",
                            "height": "160px",
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
