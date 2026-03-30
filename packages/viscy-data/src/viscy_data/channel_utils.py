"""Utilities for parsing and classifying microscopy channel names."""

from __future__ import annotations

import re

__all__ = ["parse_channel_name"]


def parse_channel_name(name: str) -> dict:
    """Extract channel metadata from a zarr channel label.

    Parameters
    ----------
    name : str
        Channel label from ``omero.channels[].label``,
        e.g. ``"Phase3D"``, ``"raw GFP EX488 EM525-45"``,
        ``"nuclei_prediction"``.

    Returns
    -------
    dict
        Parsed metadata with keys:
        - ``channel_type``: ``"labelfree"`` | ``"fluorescence"`` | ``"virtual_stain"``
        - ``filter_cube``: microscope filter name (e.g. ``"GFP"``) if fluorescence
        - ``excitation_nm``: excitation wavelength if parseable
        - ``emission_nm``: emission center wavelength if parseable
    """
    result: dict = {}
    name_lower = name.lower()

    # Fluorescence pattern: "raw <FILTER> EX<num> EM<num>[-<num>]"
    fl_match = re.match(
        r"raw\s+(\w+)\s+EX(\d+)\s+EM(\d+)(?:-(\d+))?",
        name,
        re.IGNORECASE,
    )
    if fl_match:
        result["channel_type"] = "fluorescence"
        result["filter_cube"] = fl_match.group(1)
        result["excitation_nm"] = int(fl_match.group(2))
        result["emission_nm"] = int(fl_match.group(3))
        return result

    # Virtual stain patterns (check before labelfree to avoid substring collisions)
    vs_keywords = ("prediction", "virtual", "vs_")
    if any(kw in name_lower for kw in vs_keywords):
        result["channel_type"] = "virtual_stain"
        return result

    # Label-free patterns (use word boundaries for short keywords)
    labelfree_substrings = ("phase", "brightfield", "retardance")
    labelfree_word_patterns = (r"\bbf[\b_]", r"\bdic\b", r"\bpol\b")
    if any(kw in name_lower for kw in labelfree_substrings) or any(
        re.search(p, name_lower) for p in labelfree_word_patterns
    ):
        result["channel_type"] = "labelfree"
        return result

    # Fallback: if contains EX/EM pattern without "raw" prefix
    ex_em_match = re.search(r"EX(\d+)\s*EM(\d+)", name, re.IGNORECASE)
    if ex_em_match:
        result["channel_type"] = "fluorescence"
        result["excitation_nm"] = int(ex_em_match.group(1))
        result["emission_nm"] = int(ex_em_match.group(2))
        return result

    result["channel_type"] = "unknown"
    return result
