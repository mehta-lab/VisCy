"""Reusable tabs component."""

from typing import Any, Dict, List

from dash import dcc, html

from viscy.visualization.base import DashComponent
from viscy.visualization.components.cluster_manager import ClusterManager
from viscy.visualization.styles.common import CommonStyles


class ViewTabs(DashComponent):
    """A component for view tabs (timeline, clusters)."""

    def __init__(self):
        """Initialize the view tabs."""
        pass

    def create_layout(self) -> html.Div:
        """Create the tabs layout.

        Returns
        -------
        html.Div
            The tabs component.
        """
        return dcc.Tabs(
            id="view-tabs",
            value="clusters-tab",
            children=[
                self._create_clusters_tab(),
                self._create_timeline_tab(),
            ],
            style={"marginTop": "20px"},
        )

    def _create_timeline_tab(self) -> dcc.Tab:
        """Create the track timeline tab."""
        return dcc.Tab(
            label="Track Timeline",
            value="timeline-tab",
            children=[
                html.Div(
                    id="track-timeline",
                    style=CommonStyles.get_style(
                        "container",
                        height="auto",
                        overflow_y="auto",
                        max_height="80vh",
                    ),
                ),
            ],
        )

    def _create_clusters_tab(self) -> dcc.Tab:
        """Create the clusters tab."""
        return dcc.Tab(
            label="Clusters",
            value="clusters-tab",
            children=[
                html.Div(
                    id="cluster-list",
                    style=CommonStyles.get_style(
                        "container",
                        max_height="80vh",
                        overflow_y="auto",
                    ),
                ),
            ],
        )
