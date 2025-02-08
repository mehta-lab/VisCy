"""Control panel layout component."""

from typing import Any, Dict, List

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
                self._create_color_controls(),
                self._create_axis_controls(),
                self._create_selection_controls(),
                self._create_trajectory_controls(),
                self._create_clustering_controls(),
            ],
            style=CommonStyles.get_style("flex_container"),
        )

    def _create_color_controls(self) -> html.Div:
        """Create color mode and arrow controls."""
        return html.Div(
            [
                html.Label("Color by:", style={"marginRight": "10px"}),
                dcc.Dropdown(
                    id="color-mode",
                    options=[
                        {"label": "Track ID", "value": "track"},
                        {"label": "Time", "value": "time"},
                    ],
                    value="track",
                    style={"width": "200px"},
                ),
                dcc.Checklist(
                    id="show-arrows",
                    options=[{"label": "Show arrows", "value": "show"}],
                    value=[],
                    style={"marginLeft": "20px"},
                ),
            ]
        )

    def _create_axis_controls(self) -> html.Div:
        """Create axis selection controls."""
        return html.Div(
            [
                html.Div(
                    [
                        html.Label("X-axis:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id="x-axis",
                            options=self.pc_options,
                            value="PCA1",
                            style={"width": "200px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Y-axis:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id="y-axis",
                            options=self.pc_options,
                            value="PCA2",
                            style={"width": "200px"},
                        ),
                    ]
                ),
            ]
        )

    def _create_selection_controls(self) -> html.Div:
        """Create selection mode and cluster controls."""
        return html.Div(
            [
                html.Label("Selection mode:", style={"marginRight": "10px"}),
                dcc.RadioItems(
                    id="selection-mode",
                    options=[
                        {"label": "Shaded region", "value": "region"},
                        {"label": "Lasso", "value": "lasso"},
                    ],
                    value="region",
                    inline=True,
                ),
                self._create_cluster_controls(),
            ]
        )

    def _create_cluster_controls(self) -> html.Div:
        """Create cluster assignment buttons."""
        return html.Div(
            [
                Button(
                    "Assign to New Cluster", "cluster-button", "success"
                ).create_layout(),
                Button(
                    "Clear All Clusters", "clear-clusters", "danger"
                ).create_layout(),
                Button(
                    "Clear Selection", "clear-selection", "secondary"
                ).create_layout(),
            ],
            style={"marginLeft": "10px", "display": "inline-block"},
        )

    def _create_trajectory_controls(self) -> html.Div:
        """Create trajectory mode controls."""
        return html.Div(
            [
                html.Label("Trajectory:", style={"marginRight": "10px"}),
                dcc.RadioItems(
                    id="trajectory-mode",
                    options=[
                        {"label": "X-axis", "value": "x"},
                        {"label": "Y-axis", "value": "y"},
                    ],
                    value="x",
                    inline=True,
                ),
            ]
        )

    def _create_clustering_controls(self) -> html.Div:
        """Create clustering parameter controls."""
        return html.Div(
            [
                html.Label("Clustering Parameters:", style={"marginBottom": "10px"}),
                html.Div(
                    [
                        html.Label("Epsilon:", style={"marginRight": "10px"}),
                        dcc.Slider(
                            id="eps-slider",
                            min=0.1,
                            max=2.0,
                            step=0.1,
                            value=0.5,
                            marks={i / 2: str(i / 2) for i in range(1, 9)},
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Min Samples:", style={"marginRight": "10px"}),
                        dcc.Slider(
                            id="min-samples-slider",
                            min=2,
                            max=10,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(2, 11)},
                        ),
                    ],
                ),
            ],
            style=CommonStyles.get_style(
                "container",
                padding="15px",
                margin_top="20px",
                width="100%",
            ),
        )
