"""Image-related callbacks for the visualization app."""

from typing import Dict, Any, Optional
import dash
from dash import Input, Output, State, callback_context, html, no_update


class ImageCallbacks:
    """Callbacks for handling image interactions."""

    def __init__(self, app_instance):
        """Initialize the callbacks.

        Parameters
        ----------
        app_instance : EmbeddingVisualizationApp
            The main application instance.
        """
        self.app_instance = app_instance

    def register(self):
        """Register all image-related callbacks."""
        self._register_image_click_callback()

    def _register_image_click_callback(self):
        """Register callback for handling image clicks."""

        @self.app_instance.app.callback(
            Output("image-click-store", "data"),
            [
                Input(
                    {
                        "type": "image",
                        "track_id": dash.ALL,
                        "t": dash.ALL,
                        "fov": dash.ALL,
                    },
                    "n_clicks",
                )
            ],
            prevent_initial_call=True,
        )
        def handle_image_click(n_clicks):
            """Handle image click events and store the clicked point data."""
            ctx = callback_context
            if not ctx.triggered:
                return no_update

            # Get the ID of the clicked image
            prop_id = ctx.triggered[0]["prop_id"]
            if not prop_id:
                return no_update

            # Extract the clicked image's data from its ID
            clicked_id = eval(prop_id.split(".")[0])
            return {
                "track_id": clicked_id["track_id"],
                "t": clicked_id["t"],
                "fov_name": clicked_id["fov"],
            }
