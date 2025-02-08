"""Base classes for visualization components."""

from typing import Any, Dict, List, Optional
from dash import Dash, html, dcc


class DashComponent:
    """Base class for reusable Dash components."""

    def create_layout(self) -> html.Div:
        """Create and return the component layout.

        Returns
        -------
        html.Div
            The component layout.
        """
        raise NotImplementedError


class CallbackManager:
    """Base class for callback management."""

    def register_callbacks(self, app: Dash) -> None:
        """Register callbacks with the Dash app.

        Parameters
        ----------
        app : Dash
            The Dash application instance.
        """
        raise NotImplementedError


class StyleManager:
    """Base class for style management."""

    @staticmethod
    def get_style(style_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Get style dictionary for a component.

        Parameters
        ----------
        style_type : str
            The type of style to get.
        **kwargs : Any
            Additional style parameters.

        Returns
        -------
        Dict[str, Any]
            The style dictionary.
        """
        raise NotImplementedError
