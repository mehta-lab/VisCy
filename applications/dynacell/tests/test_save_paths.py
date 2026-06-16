"""Pin canonical eval save_dir paths emitted by save_paths.eval_save_dir.

These paths are deliberately duplicated with the paper repo's
``compute_all_organelle_precision_recall.py::eval_dir_for``. Any drift here
must be matched in that file or downstream paper aggregation scripts will
fail to find the metrics. These pins are the cross-repo contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynacell.evaluation.save_paths import eval_save_dir

_DATA_ROOT = "/hpc/projects/virtual_staining/training/dynacell"

# (organelle, code_model, train_set, test_plate) -> expected relative path
# under _DATA_ROOT. Pins one representative case from each branch of
# `eval_save_dir`. Match the paper script convention exactly.
_PINS: list[tuple[tuple[str, str, str, str], str]] = [
    # iPSC test, iPSC train: focus-2D dir is evaluations_with_embeddings/.
    (
        ("er", "fnet3d_paper", "ipsc_confocal", "ipsc"),
        "ipsc/evaluations_with_embeddings/eval_fnet3d_er",
    ),
    (
        ("mito", "fcmae_vscyto3d_scratch", "ipsc_confocal", "ipsc"),
        "ipsc/evaluations_with_embeddings/eval_unext2_mitochondria",
    ),
    # iPSC test, A549 train: a549trained subdir + a549trained suffix; celldiff
    # variants collapse to the R2 bare key "celldiff_r2".
    (
        ("nucleus", "celldiff_iterative", "a549_mantis", "ipsc"),
        "ipsc/evaluations_a549trained_with_embeddings/eval_celldiff_r2_a549trained_nucleus",
    ),
    # iPSC test, joint train: evaluations_jointtrained_with_embeddings/ + jointtrained suffix.
    (
        ("membrane", "fcmae_vscyto3d_pretrained", "joint_ipsc_confocal_a549_mantis", "ipsc"),
        "ipsc/evaluations_jointtrained_with_embeddings/eval_vscyto3d_jointtrained_membrane",
    ),
    # A549 test, iPSC train.
    (
        ("er", "fnet3d_paper", "ipsc_confocal", "mock"),
        "a549/evaluations_with_embeddings/eval_fnet3d_er_mock",
    ),
    # A549 test, A549 train: celldiff variant collapses to "celldiff_r2".
    (
        ("mito", "celldiff_sliding_window", "a549_mantis", "denv"),
        "a549/evaluations_a549trained_with_embeddings/eval_celldiff_r2_a549trained_mitochondria_denv",
    ),
    # A549 test, joint train: evaluations_jointtrained_with_embeddings/ + jointtrained suffix.
    (
        ("nucleus", "unetvit3d", "joint_ipsc_confocal_a549_mantis", "zikv"),
        "a549/evaluations_jointtrained_with_embeddings/eval_unetvit3d_jointtrained_nucleus_zikv",
    ),
]


@pytest.mark.parametrize(("inputs", "expected_rel"), _PINS)
def test_eval_save_dir_matches_paper_convention(inputs: tuple[str, str, str, str], expected_rel: str) -> None:
    organelle, code_model, train_set, test_plate = inputs
    got = eval_save_dir(
        organelle=organelle,
        code_model=code_model,
        train_set=train_set,
        test_plate=test_plate,
        data_root=_DATA_ROOT,
    )
    assert got == Path(_DATA_ROOT) / expected_rel
