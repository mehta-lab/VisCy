"""Reusable button component."""

from typing import Dict, Any, Optional
from dash import html

from viscy.visualization.base import DashComponent
from viscy.visualization.styles.common import CommonStyles


class Button(DashComponent):
    """A reusable button component."""

    def __init__(
        self,
        text: str,
        button_id: str,
        button_type: str = "secondary",
        style: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the button.

        Parameters
        ----------
        text : str
            The button text.
        button_id : str
            The button ID.
        button_type : str, optional
            The button type (success, danger, secondary), by default "secondary"
        style : Optional[Dict[str, Any]], optional
            Additional style properties, by default None
        """
        self.text = text
        self.button_id = button_id
        self.button_type = button_type
        self.additional_style = style or {}

    def create_layout(self) -> html.Button:
        """Create the button layout.

        Returns
        -------
        html.Button
            The button component.
        """
        style = CommonStyles.get_style("button", color=self.button_type)
        style.update(self.additional_style)

        return html.Button(
            self.text,
            id=self.button_id,
            style=style,
        )
