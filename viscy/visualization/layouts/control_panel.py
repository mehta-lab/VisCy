"""Control panel layout component."""

from typing import Dict, List

from dash import dcc, html

from viscy.visualization.base import DashComponent
from viscy.visualization.components.button import Button
from viscy.visualization.styles.common import CommonStyles


class ControlPanel(DashComponent):
    """Control panel component containing all input controls."""

    def __init__(self, pc_options: List[Dict[str, str]]):
        """Initialize the control panel.

        Parameters
        ----------
        pc_options : List[Dict[str, str]]
            List of PC options with labels and values.
        """
        self.pc_options = pc_options

    def create_layout(self) -> html.Div:
        """Create the control panel layout.

        Returns
        -------
        html.Div
            The control panel component.
        """
        return html.Div(
            [
                # Single row containing all controls
                html.Div(
                    [
                        # Left side: Axis and Display controls
                        html.Div(
                            [
                                # Axis controls
                                html.Div(
                                    [
                                        html.Label(
                                            "X-axis:", style={"marginRight": "10px"}
                                        ),
                                        dcc.Dropdown(
                                            id="x-axis",
                                            options=self.pc_options,
                                            value=self.pc_options[0]["value"],
                                            style={"width": "200px"},
                                        ),
                                    ],
                                    style={
                                        "marginRight": "20px",
                                        "display": "inline-block",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Y-axis:", style={"marginRight": "10px"}
                                        ),
                                        dcc.Dropdown(
                                            id="y-axis",
                                            options=self.pc_options,
                                            value=self.pc_options[1]["value"],
                                            style={"width": "200px"},
                                        ),
                                    ],
                                    style={
                                        "marginRight": "20px",
                                        "display": "inline-block",
                                    },
                                ),
                                # Display controls
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="show-arrows",
                                            options=[
                                                {
                                                    "label": "Show Arrows",
                                                    "value": "show",
                                                }
                                            ],
                                            value=[],
                                            style={"marginRight": "20px"},
                                        ),
                                        dcc.RadioItems(
                                            id="color-mode",
                                            options=[
                                                {
                                                    "label": "Track Color",
                                                    "value": "track",
                                                },
                                                {
                                                    "label": "Time Color",
                                                    "value": "time",
                                                },
                                            ],
                                            value="track",
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "marginRight": "20px",
                                    },
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "20px"},
                        ),
                        # Right side: Cluster buttons
                        html.Div(
                            [
                                html.Button(
                                    "Add New Cluster",
                                    id="cluster-button",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#28a745",  # Green
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "marginRight": "10px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Clear Selection",
                                    id="clear-selection",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#6c757d",  # Gray
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "marginRight": "10px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Clear All Clusters",
                                    id="clear-clusters",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545",  # Red
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "marginRight": "10px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Export Clusters",
                                    id="export-clusters",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#17a2b8",  # Blue
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "marginRight": "10px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Import Clusters",
                                    id="import-clusters",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#17a2b8",  # Blue
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 20px",
                                        "marginRight": "10px",
                                        "borderRadius": "4px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                            style={"display": "inline-block"},
                        ),
                        dcc.Download(id="download-clusters"),
                        dcc.Upload(
                            id="upload-clusters",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select a File")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px 0",
                                "display": "none",
                            },
                            multiple=False,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "marginBottom": "20px",
                        "padding": "10px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
            ],
            style=CommonStyles.get_style("container"),
        )
