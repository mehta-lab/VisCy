"""CLI utility functions for formatting and configuration loading."""

from pathlib import Path

import yaml


def format_markdown_table(
    data: dict | list[dict], title: str = None, headers: list[str] = None
) -> str:
    """Format data as a markdown table.

    Parameters
    ----------
    data : dict | list[dict]
        Data to format. If dict, will create two columns (key, value).
        If list of dicts, each dict becomes a row with columns from headers
        or dict keys.
    title : str, optional
        Optional title to add above the table.
    headers : list[str], optional
        Column headers. If None and data is dict, uses ["Metric", "Value"].
        If None and data is list[dict], uses keys from first dict.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = []

    if title:
        lines.append(f"## {title}")
        lines.append("")

    if isinstance(data, dict):
        if headers is None:
            headers = ["Metric", "Value"]

        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"|{'|'.join(['---' + '-' * len(h) for h in headers])}|")

        for key, value in data.items():
            formatted_key = str(key).replace("_", " ").title()
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            lines.append(f"| {formatted_key} | {formatted_value} |")

    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if headers is None:
            headers = list(data[0].keys())

        header_titles = [str(h).replace("_", " ").title() for h in headers]
        lines.append(f"| {' | '.join(header_titles)} |")
        lines.append(f"|{'|'.join(['---' + '-' * len(h) for h in header_titles])}|")

        for row in data:
            values = []
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append(f"| {' | '.join(values)} |")

    lines.append("")
    return "\n".join(lines)


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
