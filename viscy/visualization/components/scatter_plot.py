"""Reusable scatter plot component."""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from viscy.visualization.base import DashComponent
from viscy.visualization.styles.common import CommonStyles


class ScatterPlot(DashComponent):
    """A reusable scatter plot component."""

    def __init__(
        self,
        figure: go.Figure,
        plot_id: str = "scatter-plot",
        height: str = "50vh",
        loading: bool = True,
    ):
        """Initialize the scatter plot.

        Parameters
        ----------
        figure : go.Figure
            The plotly figure to display.
        plot_id : str, optional
            The plot ID, by default "scatter-plot"
        height : str, optional
            The plot height, by default "50vh"
        loading : bool, optional
            Whether to show loading indicator, by default True
        """
        self.figure = figure
        self.plot_id = plot_id
        self.height = height
        self.loading = loading

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default plot configuration.

        Returns
        -------
        Dict[str, Any]
            The default configuration.
        """
        return {
            "displayModeBar": True,
            "editable": False,
            "showEditInChartStudio": False,
            "modeBarButtonsToRemove": ["select2d", "resetScale2d"],
            "edits": {
                "annotationPosition": False,
                "annotationTail": False,
                "annotationText": False,
                "shapePosition": True,
            },
            "scrollZoom": True,
        }

    def create_layout(self) -> html.Div:
        """Create the scatter plot layout.

        Returns
        -------
        html.Div
            The scatter plot component.
        """
        plot = dcc.Graph(
            id=self.plot_id,
            figure=self.figure,
            config=self.get_default_config(),
            style={"height": self.height},
        )

        if self.loading:
            return html.Div(
                dcc.Loading(
                    id=f"{self.plot_id}-loading",
                    children=[plot],
                    type="default",
                )
            )
        return html.Div(plot)
