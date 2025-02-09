"""Image grid component."""

from typing import Dict, List, Optional, Tuple

from dash import html

from viscy.visualization.base import DashComponent
from viscy.visualization.styles.common import CommonStyles


class ImageGrid(DashComponent):
    """A reusable image grid component."""

    def __init__(
        self,
        images: List[Dict],
        channel_name: str,
        image_size: Tuple[int, int] = (150, 150),
        highlight_key: Optional[Tuple] = None,
    ):
        """Initialize the image grid.

        Parameters
        ----------
        images : List[Dict]
            List of image data dictionaries containing src, track_id, t, and fov.
        channel_name : str
            Name of the channel being displayed.
        image_size : Tuple[int, int], optional
            Size of each image (width, height), by default (150, 150)
        highlight_key : Optional[Tuple], optional
            Key of the image to highlight (fov, track_id, t), by default None
        """
        self.images = images
        self.channel_name = channel_name
        self.image_size = image_size
        self.highlight_key = highlight_key

    def create_layout(self) -> html.Div:
        """Create the image grid layout.

        Returns
        -------
        html.Div
            The image grid component.
        """
        if not self.images:
            return html.Div(f"No images available for {self.channel_name}")

        return html.Div(
            [
                html.H4(
                    self.channel_name,
                    style=CommonStyles.get_style(
                        "header",
                        font_size="16px",
                        margin_bottom="10px",
                    ),
                ),
                html.Div(
                    [self._create_image_cell(img) for img in self.images],
                    style={
                        "display": "flex",
                        "flexWrap": "nowrap",
                        "gap": "10px",
                        "marginBottom": "20px",
                    },
                ),
            ]
        )

    def _create_image_cell(self, image_data: Dict) -> html.Div:
        """Create a single image cell.

        Parameters
        ----------
        image_data : Dict
            Image data containing src, track_id, t, and fov.

        Returns
        -------
        html.Div
            The image cell component.
        """
        is_highlighted = (
            self.highlight_key
            and (
                image_data["fov"],
                image_data["track_id"],
                image_data["t"],
            )
            == self.highlight_key
        )

        # Create base props for the div
        div_props = {
            "children": [
                html.Img(
                    src=image_data["src"],
                    style={
                        "width": f"{self.image_size[0]}px",
                        "height": f"{self.image_size[1]}px",
                        "border": (
                            "3px solid #007bff" if is_highlighted else "1px solid #ddd"
                        ),
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "boxShadow": (
                            "0 2px 4px rgba(0,0,0,0.1)" if is_highlighted else "none"
                        ),
                    },
                    id={
                        "type": "image",
                        "track_id": image_data["track_id"],
                        "t": image_data["t"],
                        "fov": image_data["fov"],
                    },
                ),
                html.Div(
                    f"t={image_data['t']}",
                    style={
                        "textAlign": "center",
                        "fontSize": "12px",
                        "marginTop": "5px",
                        "color": "#007bff" if is_highlighted else "#666",
                        "fontWeight": "bold" if is_highlighted else "normal",
                    },
                ),
            ],
            "style": {
                "display": "inline-block",
                "margin": "5px",
                "verticalAlign": "top",
                "padding": "4px",
                "backgroundColor": "#f8f9fa" if is_highlighted else "transparent",
                "borderRadius": "8px",
            },
        }

        # Only add id if the cell is highlighted
        if is_highlighted:
            div_props["id"] = f"timepoint-{image_data['t']}"

        return html.Div(**div_props)
