"""DEPRECATED shim — canonical path logic moved to ``paths.py``.

This module is a **transitional backward-compat shim** for consumers not yet
migrated to :mod:`dynacell.evaluation.paths` (the submitters and the grouped-eval
generator test, refactored in the Phase-5 / PR-3 codemod). It re-exports the
retained public API from ``paths.py`` and keeps the LEGACY ``eval_save_dir``
(paper-key + ``*_with_embeddings`` scheme) alive until those consumers move to the
new grammar (``eval_leaf`` / ``prediction_store`` in ``paths.py``).

Do NOT add new call sites here — import from ``paths.py`` instead. This file is
slated for deletion once the Phase-5 consumers are refactored.
"""

from __future__ import annotations

from pathlib import Path

from dynacell.evaluation.paths import (
    DATA_ROOT as _DEFAULT_DATA_ROOT,
)
from dynacell.evaluation.paths import (
    DEFAULT_EVAL_RUN_ROOT,
    ORGANELLE_EVAL_TARGET,
    ORGANELLE_PAPER,
    PAPER_KEY,
    eval_predict_set_group,
    extract_predict_output_store,
    paper_key,
)

__all__ = [
    "DEFAULT_EVAL_RUN_ROOT",
    "ORGANELLE_EVAL_TARGET",
    "ORGANELLE_PAPER",
    "PAPER_KEY",
    "eval_predict_set_group",
    "eval_save_dir",
    "extract_predict_output_store",
    "paper_key",
]


def _a549trained_key(code_model: str) -> str:
    """A549-trained legacy naming uses the bare paper key (no celldiff variant suffix)."""
    if code_model.startswith("celldiff"):
        return "celldiff_r2"
    return paper_key(code_model)


def _joint_key(code_model: str) -> str:
    """Joint-trained legacy naming collapses celldiff variants and otherwise = paper key."""
    if code_model.startswith("celldiff"):
        return "celldiff_r2"
    return paper_key(code_model)


def eval_save_dir(
    organelle: str,
    code_model: str,
    train_set: str,
    test_plate: str,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> Path:
    """LEGACY eval save_dir (paper-key + ``*_with_embeddings`` scheme).

    Retained only for the un-refactored Phase-5 submitters. New code MUST use
    :func:`dynacell.evaluation.paths.eval_leaf`. Produces the pre-canonical layout
    ``<test_set>/evaluations[_a549trained|_jointtrained]_with_embeddings/eval_<paperkey>...``.
    """
    if organelle not in ORGANELLE_PAPER:
        raise ValueError(f"unknown organelle {organelle!r}; expected one of {sorted(ORGANELLE_PAPER)}")
    if test_plate not in {"ipsc", "mock", "denv", "zikv"}:
        raise ValueError(f"unknown test_plate {test_plate!r}; expected one of 'ipsc' | 'mock' | 'denv' | 'zikv'")
    if train_set not in {"ipsc_confocal", "a549_mantis", "joint_ipsc_confocal_a549_mantis"}:
        raise ValueError(
            f"unknown train_set {train_set!r}; expected one of "
            f"'ipsc_confocal' | 'a549_mantis' | 'joint_ipsc_confocal_a549_mantis'"
        )
    organelle_paper = ORGANELLE_PAPER[organelle]
    root = Path(data_root)
    if test_plate == "ipsc":
        if train_set == "ipsc_confocal":
            return root / "ipsc" / "evaluations_with_embeddings" / f"eval_{paper_key(code_model)}_{organelle_paper}"
        if train_set == "a549_mantis":
            return (
                root
                / "ipsc"
                / "evaluations_a549trained_with_embeddings"
                / f"eval_{_a549trained_key(code_model)}_a549trained_{organelle_paper}"
            )
        return (
            root
            / "ipsc"
            / "evaluations_jointtrained_with_embeddings"
            / f"eval_{_joint_key(code_model)}_jointtrained_{organelle_paper}"
        )
    if train_set == "ipsc_confocal":
        return (
            root
            / "a549"
            / "evaluations_with_embeddings"
            / f"eval_{paper_key(code_model)}_{organelle_paper}_{test_plate}"
        )
    if train_set == "a549_mantis":
        return (
            root
            / "a549"
            / "evaluations_a549trained_with_embeddings"
            / f"eval_{_a549trained_key(code_model)}_a549trained_{organelle_paper}_{test_plate}"
        )
    return (
        root
        / "a549"
        / "evaluations_jointtrained_with_embeddings"
        / f"eval_{_joint_key(code_model)}_jointtrained_{organelle_paper}_{test_plate}"
    )
