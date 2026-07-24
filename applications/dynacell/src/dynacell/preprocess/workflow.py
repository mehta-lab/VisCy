"""AICS-HiPSC workflow ID parsing utilities."""

from __future__ import annotations


def extract_numeric_part(workflow_id_str: str) -> str:
    """Extract the numeric pipeline ID from a WorkflowId string.

    Parameters
    ----------
    workflow_id_str : str
        Raw workflow ID string, e.g. ``"[Pipeline 4.1]"``.

    Returns
    -------
    str
        Cleaned numeric part, e.g. ``"4.1"``.
    """
    return workflow_id_str.strip("[]'").replace("Pipeline ", "").strip()


def is_target_workflow(workflow_id_str: str, targets: list[str]) -> bool:
    """Check whether a workflow ID matches any target.

    Parameters
    ----------
    workflow_id_str : str
        Raw workflow ID string.
    targets : list[str]
        List of target numeric IDs.

    Returns
    -------
    bool
        True if the extracted ID is in ``targets``.
    """
    return extract_numeric_part(workflow_id_str) in targets
