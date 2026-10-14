"""Unit tests for the predict-triplet planning logic.

Covers ``plan_predict_runs`` — the pure per-reporter run planner that turns a
collection's channels into model/run/checkpoint-scoped predict jobs. No GPU or
real zarr required; the predict execution itself is exercised by
``test_inference_reproducibility.py`` (hpc_integration).
"""

from pathlib import Path

import pandas as pd
import pytest

from dynaclr.evaluation.predict_triplet import (
    COLLECTION_OBS_COLUMNS,
    build_obs_metadata,
    plan_predict_runs,
)
from viscy_data.collection import ChannelEntry, Collection, ExperimentEntry

MODEL_FAMILY = "DynaCLR-2D-MIP-BagOfChannels"
RUN = "2d-mip-fix-shuffler"
CKPT = "epoch105-step84800"


def _box_collection() -> Collection:
    """Multi-organelle box plate: GFP reporter varies by column, mCherry uniform, plus phase."""
    exp = ExperimentEntry(
        name="2026_07_01_ZIKV",
        data_path="/data/2026_07_01_ZIKV/2026_07_01_ZIKV.zarr",
        tracks_path="/data/2026_07_01_ZIKV/tracking.zarr",
        channels=[
            ChannelEntry(name="raw GFP EX488 EM525-45", marker="SEC61B", wells=["A/2", "B/2"]),
            ChannelEntry(name="raw GFP EX488 EM525-45", marker="TOMM20", wells=["A/3", "B/3"]),
            ChannelEntry(name="raw GFP EX488 EM525-45", marker="G3BP1", wells=["A/4", "B/4"]),
            ChannelEntry(name="raw mCherry EX561 EM600-37", marker="pAL17"),  # empty wells = all
            ChannelEntry(name="Phase3D", marker="Phase3D"),
        ],
        perturbation_wells={"uninfected": ["A/2"], "ZIKV": ["B/2"]},
        pixel_size_xy_um=0.1133,
    )
    return Collection(name="box", experiments=[exp])


def _plan(**kwargs):
    return plan_predict_runs(
        _box_collection(),
        model_family=MODEL_FAMILY,
        run=RUN,
        ckpt_name=CKPT,
        datasets_root="/data",
        **kwargs,
    )


def test_one_run_per_channel_entry():
    """Each channel entry (incl. repeated GFP) becomes its own run."""
    runs = _plan()
    assert [r.marker for r in runs] == ["SEC61B", "TOMM20", "G3BP1", "pAL17", "Phase3D"]


def test_per_reporter_well_subset():
    """Wells carry through as fit_include_wells; empty -> None (all wells)."""
    runs = {r.marker: r for r in _plan()}
    assert runs["SEC61B"].wells == ["A/2", "B/2"]
    assert runs["TOMM20"].wells == ["A/3", "B/3"]
    assert runs["pAL17"].wells is None  # empty list -> all wells


def test_output_path_is_dataset_centric():
    """Output tree is <dataset>/2-phenotyping/predictions/{model}/{run}/{ckpt}/embeddings/{marker}.zarr."""
    run = next(r for r in _plan() if r.marker == "SEC61B")
    assert run.output_path == Path(
        "/data/2026_07_01_ZIKV/2-phenotyping/predictions/"
        "DynaCLR-2D-MIP-BagOfChannels/2d-mip-fix-shuffler/epoch105-step84800/embeddings/SEC61B.zarr"
    )


def test_markers_of_one_dataset_colocate():
    """Multiple markers of one physical dataset share the {model}/{run}/{ckpt} dir."""
    runs = {r.marker: r for r in _plan()}
    assert runs["SEC61B"].output_path.parent == runs["TOMM20"].output_path.parent
    assert runs["SEC61B"].output_path.name == "SEC61B.zarr"
    assert runs["TOMM20"].output_path.name == "TOMM20.zarr"


def test_no_labelfree_drops_phase():
    """--no-labelfree removes phase/brightfield channels, keeps fluorescence."""
    markers = [r.marker for r in _plan(include_labelfree=False)]
    assert "Phase3D" not in markers
    assert markers == ["SEC61B", "TOMM20", "G3BP1", "pAL17"]


def test_labelfree_flag_marks_phase():
    """Phase3D is flagged label-free; fluorescence reporters are not."""
    runs = {r.marker: r for r in _plan()}
    assert runs["Phase3D"].is_labelfree is True
    assert runs["SEC61B"].is_labelfree is False


def test_markers_subset():
    """--markers restricts to the requested marker labels."""
    markers = [r.marker for r in _plan(markers=["SEC61B", "pAL17"])]
    assert markers == ["SEC61B", "pAL17"]


def test_empty_plan_raises():
    """A filter that matches nothing is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="No reporter runs planned"):
        _plan(markers=["does-not-exist"])


# ---------------------------------------------------------------------------
# obs enrichment (build_obs_metadata) — the fix for triplet obs lacking
# perturbation / hpi / experiment / marker that the parquet path carries.
# ---------------------------------------------------------------------------


def _ultrack_obs() -> pd.DataFrame:
    """Ultrack-only obs like the triplet EmbeddingWriter writes (A row + B row)."""
    return pd.DataFrame(
        {
            "fov_name": ["A/2/000000", "A/2/000001", "B/2/000000", "B/2/001003"],
            "track_id": [3, 4, 3, 9],
            "t": [0, 10, 5, 66],
        }
    )


def _exp_for_enrich() -> ExperimentEntry:
    return ExperimentEntry(
        name="2026_07_01_ZIKV",
        data_path="/data/2026_07_01_ZIKV/2026_07_01_ZIKV.zarr",
        tracks_path="/data/2026_07_01_ZIKV/tracking.zarr",
        channels=[ChannelEntry(name="raw GFP EX488 EM525-45", marker="SEC61B", wells=["A/2", "B/2"])],
        perturbation_wells={"uninfected": ["A/2", "A/3", "A/4"], "ZIKV": ["B/2", "B/3", "B/4"]},
        organelle="endoplasmic_reticulum",
        microscope="mantis",
        interval_minutes=30.0,
        start_hpi=3.0,
    )


def test_build_obs_metadata_columns_and_index():
    """Returns exactly the parquet-schema columns, aligned to obs index."""
    obs = _ultrack_obs()
    meta = build_obs_metadata(obs, _exp_for_enrich(), marker="SEC61B")
    assert list(meta.columns) == list(COLLECTION_OBS_COLUMNS)
    assert meta.index.equals(obs.index)
    assert (meta["experiment"] == "2026_07_01_ZIKV").all()
    assert (meta["marker"] == "SEC61B").all()
    assert (meta["organelle"] == "endoplasmic_reticulum").all()


def test_build_obs_metadata_perturbation_per_well():
    """perturbation resolves from fov_name well: A row uninfected, B row ZIKV."""
    meta = build_obs_metadata(_ultrack_obs(), _exp_for_enrich(), marker="SEC61B")
    assert meta["perturbation"].tolist() == ["uninfected", "uninfected", "ZIKV", "ZIKV"]


def test_build_obs_metadata_hpi_formula():
    """hours_post_perturbation = start_hpi + t * interval_minutes / 60."""
    meta = build_obs_metadata(_ultrack_obs(), _exp_for_enrich(), marker="SEC61B")
    # start_hpi=3, interval=30min=0.5h: t=0->3.0, t=10->8.0, t=5->5.5, t=66->36.0
    assert meta["hours_post_perturbation"].tolist() == [3.0, 8.0, 5.5, 36.0]


def test_build_obs_metadata_unknown_well():
    """A well absent from perturbation_wells resolves to 'unknown' (not an error)."""
    obs = pd.DataFrame({"fov_name": ["C/9/000000"], "track_id": [1], "t": [0]})
    meta = build_obs_metadata(obs, _exp_for_enrich(), marker="SEC61B")
    assert meta["perturbation"].tolist() == ["unknown"]
