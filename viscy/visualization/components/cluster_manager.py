"""Component for managing clusters."""

from dash import dcc, html
from typing import List, Dict, Any

from viscy.visualization.base import DashComponent
from viscy.visualization.components.button import Button
from viscy.visualization.styles.common import CommonStyles


class ClusterManager(DashComponent):
    """Component for managing and annotating clusters."""

    def __init__(self):
        """Initialize the cluster manager."""
        pass

    def create_layout(self) -> html.Div:
        """Create the cluster manager layout.

        Returns
        -------
        html.Div
            The cluster manager component.
        """
        return html.Div(
            children=[
                # Export/Import controls
                html.Div(
                    className="cluster-controls",
                    style=CommonStyles.get_style("container", margin_bottom="20px"),
                    children=[
                        Button(
                            "Export Clusters",
                            "export-clusters",
                            style={"marginRight": "10px"},
                        ).create_layout(),
                        dcc.Download(id="download-clusters"),
                        Button(
                            "Import Clusters",
                            "import-clusters",
                            style={"marginRight": "10px"},
                        ).create_layout(),
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
                ),
                # Cluster list container
                html.Div(
                    id="cluster-list",
                    style=CommonStyles.get_style(
                        "container",
                        max_height="60vh",
                        overflow_y="auto",
                    ),
                ),
            ],
        )

    @staticmethod
    def create_cluster_card(
        cluster_idx: int,
        cluster_data: List[Dict[str, Any]],
        annotation: Dict[str, str],
        image_grid: html.Div,
    ) -> html.Div:
        """Create a card for a single cluster.

        Parameters
        ----------
        cluster_idx : int
            Index of the cluster
        cluster_data : List[Dict[str, Any]]
            List of points in the cluster
        annotation : Dict[str, str]
            Cluster annotation data
        image_grid : html.Div
            Grid of images for this cluster

        Returns
        -------
        html.Div
            The cluster card component
        """
        return html.Div(
            className=f"cluster-card-{cluster_idx}",
            style={
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "15px",
                "marginBottom": "20px",
            },
            children=[
                # Cluster header with delete button
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                    },
                    children=[
                        html.H3(f"Cluster {cluster_idx + 1}"),
                        html.Button(
                            "×",
                            id={"type": "delete-cluster", "index": cluster_idx},
                            style={
                                "border": "none",
                                "background": "none",
                                "fontSize": "20px",
                                "cursor": "pointer",
                                "color": "#ff0000",
                            },
                        ),
                    ],
                ),
                # Cluster annotation inputs
                dcc.Input(
                    id={"type": "cluster-label", "index": cluster_idx},
                    type="text",
                    placeholder="Enter cluster label",
                    value=annotation.get("label", ""),
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Textarea(
                    id={"type": "cluster-description", "index": cluster_idx},
                    placeholder="Enter cluster description",
                    value=annotation.get("description", ""),
                    style={"width": "100%", "height": "100px", "marginBottom": "10px"},
                ),
                # Point count
                html.P(f"Points: {len(cluster_data)}"),
                # Image grid
                image_grid,
            ],
        )
