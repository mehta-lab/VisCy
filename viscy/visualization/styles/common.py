"""Common styles used across visualization components."""

from typing import Any, Dict

from viscy.visualization.base import StyleManager


class CommonStyles(StyleManager):
    """Common styles used across components."""

    COLORS = {
        "primary": "#007bff",
        "success": "#28a745",
        "danger": "#dc3545",
        "secondary": "#6c757d",
        "light": "#f8f9fa",
        "dark": "#2c3e50",
        "border": "#ddd",
    }

    @staticmethod
    def get_style(style_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Get common style dictionary.

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
        if style_type == "button":
            return {
                "backgroundColor": CommonStyles.COLORS.get(
                    kwargs.get("color", "secondary")
                ),
                "color": "white",
                "border": "none",
                "padding": "5px 10px",
                "borderRadius": "4px",
                "cursor": "pointer",
                "marginRight": "10px",
            }
        elif style_type == "container":
            return {
                "padding": kwargs.get("padding", "10px"),
                "marginTop": kwargs.get("margin_top", "10px"),
                "marginBottom": kwargs.get("margin_bottom", "10px"),
                "backgroundColor": kwargs.get("background_color", "white"),
                "borderRadius": kwargs.get("border_radius", "8px"),
                "boxShadow": kwargs.get("box_shadow", "0 2px 4px rgba(0,0,0,0.1)"),
            }
        elif style_type == "header":
            return {
                "fontSize": kwargs.get("font_size", "20px"),
                "fontWeight": "bold",
                "color": CommonStyles.COLORS["dark"],
                "marginBottom": kwargs.get("margin_bottom", "20px"),
            }
        elif style_type == "flex_container":
            return {
                "display": "flex",
                "flexDirection": kwargs.get("direction", "row"),
                "alignItems": kwargs.get("align", "center"),
                "gap": kwargs.get("gap", "20px"),
                "flexWrap": kwargs.get("wrap", "wrap"),
                "marginBottom": kwargs.get("margin_bottom", "20px"),
            }
        else:
            return {}
