"""Scatter plot component."""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from viscy.visualization.base import DashComponent
from viscy.visualization.styles.common import CommonStyles


class ScatterPlot(DashComponent):
    """Scatter plot component."""

    def __init__(self, figure):
        """Initialize the scatter plot.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            The initial figure to display.
        """
        self.figure = figure
        # Set initial layout settings
        self.figure.update_layout(
            dragmode="lasso",
            clickmode="event+select",
            uirevision="same",  # Keep selection state
            selectdirection="any",
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20),
            modebar=dict(
                remove=["select2d", "lasso2d", "autoScale2d"],
                orientation="v",
            ),
        )

    def create_layout(self) -> html.Div:
        """Create the scatter plot layout.

        Returns
        -------
        html.Div
            The scatter plot component.
        """
        return html.Div(
            [
                dcc.Loading(
                    id="scatter-plot-loading",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="scatter-plot",
                            figure=self.figure,
                            style={"height": "70vh"},
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": [
                                    "select2d",
                                    "lasso2d",
                                    "autoScale2d",
                                ],
                                "displaylogo": False,
                                "scrollZoom": True,
                            },
                            clear_on_unhover=True,
                            responsive=True,
                            selectedData=None,  # Initialize with no selection
                        )
                    ],
                )
            ],
            style=CommonStyles.get_style("container"),
        )
