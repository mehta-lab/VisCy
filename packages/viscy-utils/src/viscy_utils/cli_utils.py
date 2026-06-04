"""CLI utility functions for formatting and configuration loading."""

from pathlib import Path

from viscy_utils.compose import load_composed_config


def format_markdown_table(data: dict | list[dict], title: str = None, headers: list[str] = None) -> str:
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
    """Load a YAML configuration file with optional recipe composition.

    A top-level ``base:`` key is interpreted as a list of relative paths
    to other YAML files that are merged before this file's own keys
    (later entries override earlier ones; this file overrides the bases).
    YAML files without a ``base:`` key behave identically to
    ``yaml.safe_load`` — there is no special handling beyond that one
    key. See ``viscy_utils.compose.load_composed_config`` for the merge
    rules.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Composed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file (or any referenced base) does not exist.
    """
    return load_composed_config(Path(config_path))


def load_config_section(config_path: str | Path, section: str | None, default_section: str | None = None) -> dict:
    """Load a YAML config file, optionally selecting a subsection.

    This enables reusing a single YAML file for multiple CLI steps by storing
    per-command configuration under a top-level key (``section``), while keeping
    shared keys (e.g., ``datasets``) at the root.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file.
    section : str | None
        If provided, selects ``config[section]`` and merges in any shared root
        keys that are not already present in the section.
    default_section : str | None
        If ``section`` is None and ``default_section`` exists in the YAML, that section is used.

    Returns
    -------
    dict
        Configuration dictionary (either full or merged subsection).
    """
    cfg = load_config(config_path)
    if section is None:
        if default_section is None or default_section not in cfg:
            return cfg
        section = default_section

    if section not in cfg:
        raise KeyError(f"Config section not found: {section}")

    section_cfg = cfg[section] or {}
    if not isinstance(section_cfg, dict):
        raise TypeError(f"Config section must be a mapping: {section}")

    merged = dict(section_cfg)
    for k, v in cfg.items():
        if k == section:
            continue
        merged.setdefault(k, v)
    return merged
