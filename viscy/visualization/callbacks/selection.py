"""Selection-related callbacks for the visualization app."""

import json
from typing import Dict, Any, List, Optional

import dash
from dash import html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from viscy.visualization.components.scrollable_container import ScrollableContainer


class SelectionCallbacks:
    """Class to handle selection-related callbacks."""

    def __init__(self, app):
        """Initialize the selection callbacks.

        Args:
            app: The main visualization app instance.
        """
        self.app = app

    def register(self):
        """Register all selection-related callbacks."""
        self._register_selection_callback()
        self._register_clear_selection_callback()
        self._register_image_click_callback()

    def _register_selection_callback(self):
        """Register callback for point selection."""

        @self.app.app.callback(
            [
                Output("track-timeline", "children", allow_duplicate=True),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [Input("scatter-plot", "clickData")],
            [
                State("scatter-plot", "selectedData"),
                State("view-tabs", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_selection(
            click_data: Optional[Dict[str, Any]],
            selected_data: Optional[Dict[str, Any]],
            active_tab: str,
        ):
            """Update selection based on click."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # Handle single point click for timeline view
            if click_data and click_data.get("points"):
                point = click_data["points"][0]  # Only take the first point
                text = point["text"]
                lines = text.split("<br>")

                # Extract track information
                if "Cluster:" in lines[0]:
                    # Clicked on a cluster point
                    track_id = int(lines[1].split(": ")[1])
                    t = int(lines[2].split(": ")[1])
                    fov = lines[3].split(": ")[1]
                else:
                    # Clicked on a regular point
                    track_id = int(lines[0].split(": ")[1])
                    t = int(lines[1].split(": ")[1])
                    fov = lines[2].split(": ")[1]

                return self._create_timeline_view(track_id, t, fov, point)

            return dash.no_update, dash.no_update

    def _register_image_click_callback(self):
        """Register callback for image clicks in the timeline."""

        @self.app.app.callback(
            [
                Output("track-timeline", "children", allow_duplicate=True),
                Output("scatter-plot", "selectedData", allow_duplicate=True),
            ],
            [
                Input(
                    {"type": "image", "track_id": ALL, "t": ALL, "fov": ALL}, "n_clicks"
                )
            ],
            [State("scatter-plot", "figure")],
            prevent_initial_call=True,
        )
        def handle_image_click(n_clicks, figure):
            """Handle clicks on timeline images."""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks):
                raise PreventUpdate

            # Get the clicked image's properties
            clicked_props = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            track_id = clicked_props["track_id"]
            t = clicked_props["t"]
            fov = clicked_props["fov"]

            # Find the corresponding point in the scatter plot
            point_data = self.app.filtered_features_df[
                (self.app.filtered_features_df["fov_name"] == fov)
                & (self.app.filtered_features_df["track_id"] == track_id)
                & (self.app.filtered_features_df["t"] == t)
            ]

            if point_data.empty:
                raise PreventUpdate

            # Create point data structure
            x_axis = figure["layout"]["xaxis"]["title"]["text"]
            y_axis = figure["layout"]["yaxis"]["title"]["text"]
            point = {
                "x": float(point_data[x_axis].iloc[0]),
                "y": float(point_data[y_axis].iloc[0]),
                "text": f"Track: {track_id}<br>Time: {t}<br>FOV: {fov}",
            }

            return self._create_timeline_view(track_id, t, fov, point)

    def _create_timeline_view(
        self, track_id: int, clicked_t: int, fov: str, point: Dict
    ) -> tuple:
        """Create the timeline view for a track.

        Parameters
        ----------
        track_id : int
            The track ID
        clicked_t : int
            The clicked timepoint
        fov : str
            The FOV name
        point : Dict
            The point data for selection

        Returns
        -------
        tuple
            (timeline view, selected data)
        """
        # Create selectedData for single point
        new_selected_data = {
            "points": [
                {
                    "x": point["x"],
                    "y": point["y"],
                    "text": point["text"],
                    "curveNumber": point.get("curveNumber", 0),
                    "pointNumber": point.get("pointNumber", 0),
                    "pointIndex": point.get("pointIndex", 0),
                }
            ]
        }

        # Get all timepoints for this track from the features dataframe
        track_data = self.app.filtered_features_df[
            (self.app.filtered_features_df["fov_name"] == fov)
            & (self.app.filtered_features_df["track_id"] == track_id)
        ].sort_values("t")

        if track_data.empty:
            return (
                html.Div(f"No timeline data found for Track {track_id}"),
                new_selected_data,
            )

        # Create image grids for each channel
        channel_grids = []
        for channel in self.app.channels_to_display:
            images = []
            for _, row in track_data.iterrows():
                cache_key = (row["fov_name"], row["track_id"], row["t"])
                if (
                    cache_key in self.app.image_cache
                    and channel in self.app.image_cache[cache_key]
                ):
                    images.append(
                        html.Div(
                            [
                                html.Img(
                                    src=self.app.image_cache[cache_key][channel],
                                    style={
                                        "width": "150px",
                                        "height": "150px",
                                        "border": (
                                            "2px solid #007bff"  # Blue border for clicked timepoint
                                            if row["t"] == clicked_t
                                            else "1px solid #ddd"
                                        ),
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                    id={
                                        "type": "image",
                                        "track_id": row["track_id"],
                                        "t": row["t"],
                                        "fov": row["fov_name"],
                                    },
                                ),
                                html.Div(
                                    f"Time: {row['t']}",
                                    style={
                                        "fontSize": "12px",
                                        "textAlign": "center",
                                        "marginTop": "5px",
                                        "fontWeight": (
                                            "bold"
                                            if row["t"] == clicked_t
                                            else "normal"
                                        ),
                                        "color": (
                                            "#007bff"  # Blue text for clicked timepoint
                                            if row["t"] == clicked_t
                                            else "#2c3e50"
                                        ),
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin": "5px",
                                "verticalAlign": "top",
                            },
                        )
                    )

            if images:
                channel_grids.append(
                    html.Div(
                        [
                            html.H4(
                                channel,
                                style={
                                    "marginTop": "10px",
                                    "marginBottom": "10px",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "color": "#2c3e50",
                                    "paddingLeft": "10px",
                                },
                            ),
                            html.Div(
                                images,
                                style={
                                    "whiteSpace": "nowrap",
                                    "marginBottom": "20px",
                                },
                            ),
                        ]
                    )
                )

        if not channel_grids:
            return (
                html.Div("No images found in cache for this track."),
                new_selected_data,
            )

        return (
            ScrollableContainer(
                title=f"Track {track_id} Timeline (FOV: {fov})",
                content=html.Div(
                    channel_grids,
                    style={"padding": "20px"},
                ),
                max_height="80vh",
                direction="horizontal",
            ).create_layout(),
            new_selected_data,
        )

    def _register_clear_selection_callback(self):
        """Register callback for clearing selection."""

        @self.app.app.callback(
            [
                Output("scatter-plot", "selectedData"),
                Output("scatter-plot", "figure", allow_duplicate=True),
            ],
            [Input("clear-selection", "n_clicks")],
            [
                State("show-arrows", "value"),
                State("x-axis", "value"),
                State("y-axis", "value"),
                State("color-mode", "value"),
            ],
            prevent_initial_call=True,
        )
        def clear_selection(n_clicks, show_arrows, x_axis, y_axis, color_mode):
            """Clear the current selection and reset opacities."""
            if n_clicks:
                # Create new figure based on color mode
                if color_mode == "time":
                    fig = self.app._create_time_colored_figure(
                        show_arrows=len(show_arrows or []) > 0,
                        x_axis=x_axis,
                        y_axis=y_axis,
                    )
                else:  # track color mode
                    fig = self.app._create_track_colored_figure(
                        show_arrows=len(show_arrows or []) > 0,
                        x_axis=x_axis,
                        y_axis=y_axis,
                    )
                return None, fig
            raise PreventUpdate
