"""Canonical eval-output directory naming for dynacell virtual-staining benchmarks.

# IMPORTANT: This file is DELIBERATELY duplicated with
#   /hpc/mydata/alex.kalinin/dynacell/paper/scripts/compute_all_organelle_precision_recall.py
#     (the `eval_dir_for` function, lines 55-110)
# Both must stay in sync — VisCy and the paper repo are independent git repos.

Returns the same on-disk save_dir that downstream paper aggregation scripts
expect. Used by ``applications/dynacell/tools/submit_evaluation_job.py`` to
emit ``save.save_dir=<path>`` Hydra overrides.
"""

from __future__ import annotations

from pathlib import Path

# Mapping from code-side model key (config dir name) to paper key.
# Source of truth: applications/dynacell/CLAUDE.md
PAPER_KEY: dict[str, str] = {
    "fcmae_vscyto3d_scratch": "unext2",
    "fcmae_vscyto3d_pretrained": "vscyto3d",
    "fnet3d_paper": "fnet3d",
    "unetvit3d": "unetvit3d",
    # CELL-Diff variants all collapse to a single iterative paper key for
    # iPSC-trained evaluations (matches paper script line 51).
    "celldiff": "celldiff_iterative",
    "celldiff_iterative": "celldiff_iterative",
    "celldiff_sliding_window": "celldiff_iterative",
    "celldiff_denoise": "celldiff_iterative",
    # VSCyto3D ablations: random-init, external-ckpt (no-FT), and dynacell-FT
    # from cytoland / infection-FT sources. Launched outside the standard
    # submitter; eval YAMLs hand-authored. Paper-side aggregation script
    # must learn these keys lockstep.
    "fcmae_vscyto3d_pretrained_randinit": "vscyto3d_randinit",
    "fcmae_vscyto3d_pretrained_cytoland": "vscyto3d_cytoland",
    "fcmae_vscyto3d_pretrained_infectionft": "vscyto3d_infectionft",
    "vscyto3d_cytolandft": "vscyto3d_cytolandft",
    "vscyto3d_infectionft_dynacellft": "vscyto3d_infectionft_dynacellft",
}

# Organelle-name translation: code config dir → paper-script organelle key.
# Mito uses the long form `mitochondria` in paper outputs (paper script
# lines 41-44).
ORGANELLE_PAPER: dict[str, str] = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er",
    "mito": "mitochondria",
}

# Organelle → eval-side Hydra `target` group name. Membrane and nucleus are
# 1:1; ER and Mito disambiguate by gene to match the target YAMLs under
# `configs/benchmarks/virtual_staining/_internal/shared/eval/target/`.
ORGANELLE_EVAL_TARGET: dict[str, str] = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er_sec61b",
    "mito": "mito_tomm20",
}

_DEFAULT_DATA_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell")
DEFAULT_EVAL_RUN_ROOT = _DEFAULT_DATA_ROOT / "eval_runs"


def eval_predict_set_group(dataset_name: str) -> str:
    """Return the eval-side Hydra ``predict_set`` group name for one leaf.

    iPSC composes back to itself; A549 leaves carry the per-condition dataset
    slug ``a549-mantis-<marker>-<cond>`` and the group name uses underscores.
    """
    if dataset_name == "aics-hipsc":
        return "ipsc_confocal"
    if dataset_name.startswith("a549-mantis-"):
        return "a549_mantis_" + dataset_name.removeprefix("a549-mantis-").replace("-", "_")
    raise ValueError(
        f"cannot map dataset {dataset_name!r} to a predict_set group; "
        f"expected 'aics-hipsc' or 'a549-mantis-<marker>-<cond>'"
    )


def extract_predict_output_store(composed: dict, leaf_path: Path) -> Path:
    """Pull ``HCSPredictionWriter.init_args.output_store`` from a composed predict config."""
    callbacks = composed.get("trainer", {}).get("callbacks", [])
    if not isinstance(callbacks, list):
        raise ValueError(f"{leaf_path}: trainer.callbacks must be a list (got {type(callbacks).__name__})")
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        if str(cb.get("class_path", "")).endswith("HCSPredictionWriter"):
            init_args = cb.get("init_args", {}) or {}
            store = init_args.get("output_store")
            if not store:
                raise ValueError(f"{leaf_path}: HCSPredictionWriter has no init_args.output_store")
            return Path(store)
    raise ValueError(f"{leaf_path}: no HCSPredictionWriter callback found under trainer.callbacks")


def paper_key(code_model: str) -> str:
    """Translate the code-side model key (e.g. config dir name) to the paper key."""
    if code_model not in PAPER_KEY:
        raise ValueError(f"unknown model key {code_model!r}; expected one of {sorted(PAPER_KEY)}")
    return PAPER_KEY[code_model]


def _a549trained_key(code_model: str) -> str:
    """A549-trained naming uses the bare paper key (no celldiff variant suffix)."""
    if code_model.startswith("celldiff"):
        return "celldiff"
    return paper_key(code_model)


def _joint_key(code_model: str) -> str:
    """Joint-trained naming collapses celldiff variants and otherwise = paper key."""
    if code_model.startswith("celldiff"):
        return "celldiff"
    return paper_key(code_model)


def eval_save_dir(
    organelle: str,
    code_model: str,
    train_set: str,
    test_plate: str,
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> Path:
    """Return canonical eval save_dir following the paper-script convention.

    Must produce identical paths to
    paper/scripts/compute_all_organelle_precision_recall.py:eval_dir_for.

    Parameters
    ----------
    organelle : str
        Config-side organelle key: ``nucleus`` | ``membrane`` | ``er`` | ``mito``.
    code_model : str
        Config-side model key (e.g. ``fnet3d_paper``, ``fcmae_vscyto3d_pretrained``,
        ``celldiff_iterative``).
    train_set : str
        Train-set group name: ``ipsc_confocal`` | ``a549_mantis`` |
        ``joint_ipsc_confocal_a549_mantis``.
    test_plate : str
        Test plate identifier: ``ipsc`` for iPSC test; ``mock`` | ``denv`` | ``zikv``
        for A549 plates.
    data_root : str or Path, optional
        Base directory under which the canonical layout is rooted. Defaults to
        ``/hpc/projects/virtual_staining/training/dynacell``.

    Returns
    -------
    Path
        Absolute path to the eval save_dir. Does not create the directory.

    Raises
    ------
    ValueError
        If any of ``organelle``, ``code_model``, ``train_set``, or ``test_plate``
        is not one of the supported values.
    """
    if organelle not in ORGANELLE_PAPER:
        raise ValueError(f"unknown organelle {organelle!r}; expected one of {sorted(ORGANELLE_PAPER)}")
    if test_plate not in {"ipsc", "mock", "denv", "zikv"}:
        raise ValueError(f"unknown test_plate {test_plate!r}; expected one of 'ipsc' | 'mock' | 'denv' | 'zikv'")
    if train_set not in {
        "ipsc_confocal",
        "a549_mantis",
        "joint_ipsc_confocal_a549_mantis",
    }:
        raise ValueError(
            f"unknown train_set {train_set!r}; expected one of "
            f"'ipsc_confocal' | 'a549_mantis' | 'joint_ipsc_confocal_a549_mantis'"
        )
    organelle_paper = ORGANELLE_PAPER[organelle]
    root = Path(data_root)
    if test_plate == "ipsc":
        if train_set == "ipsc_confocal":
            return root / "ipsc" / "evaluations" / f"eval_{paper_key(code_model)}_{organelle_paper}"
        if train_set == "a549_mantis":
            return (
                root
                / "ipsc"
                / "evaluations_a549trained_with_embeddings"
                / f"eval_{_a549trained_key(code_model)}_a549trained_{organelle_paper}"
            )
        # joint
        return root / "ipsc" / "joint_evaluations" / f"eval_{_joint_key(code_model)}_joint_{organelle_paper}"
    # A549 plate (mock | denv | zikv)
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
    # joint
    return root / "a549" / "joint_evaluations" / f"eval_{_joint_key(code_model)}_joint_{organelle_paper}_{test_plate}"
