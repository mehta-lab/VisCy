"""Scrollable container component."""

from typing import List, Dict, Any

from dash import html

from viscy.visualization.base import DashComponent
from viscy.visualization.styles.common import CommonStyles


class ScrollableContainer(DashComponent):
    """A reusable scrollable container component."""

    def __init__(
        self,
        title: str,
        content: List[Dict[str, Any]],
        max_height: str = "80vh",
        direction: str = "horizontal",
    ):
        """Initialize the scrollable container.

        Parameters
        ----------
        title : str
            The title of the container.
        content : List[Dict[str, Any]]
            List of content items to display.
        max_height : str, optional
            Maximum height of the container, by default "80vh"
        direction : str, optional
            Scroll direction ("horizontal" or "vertical"), by default "horizontal"
        """
        self.title = title
        self.content = content
        self.max_height = max_height
        self.direction = direction

    def create_layout(self) -> html.Div:
        """Create the scrollable container layout.

        Returns
        -------
        html.Div
            The scrollable container component.
        """
        return html.Div(
            [
                html.H2(
                    self.title,
                    style=CommonStyles.get_style(
                        "header",
                        font_size="24px",
                        margin_bottom="20px",
                    ),
                ),
                html.Div(
                    self.content,
                    style={
                        "overflowX": (
                            "auto" if self.direction == "horizontal" else "hidden"
                        ),
                        "overflowY": (
                            "auto" if self.direction == "vertical" else "hidden"
                        ),
                        "whiteSpace": (
                            "nowrap" if self.direction == "horizontal" else "normal"
                        ),
                        "maxHeight": self.max_height,
                        "backgroundColor": "#ffffff",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
            ],
            style=CommonStyles.get_style("container"),
        )
