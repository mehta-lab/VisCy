"""Tests for annotation metadata writing."""

import pytest
from iohub import open_ome_zarr
from pydantic import ValidationError

from qc.annotation import _well_from_position_name, write_annotation_metadata
from qc.config import (
    AnnotationConfig,
    BiologicalAnnotation,
    ChannelAnnotationEntry,
    Perturbation,
    WellExperimentMetadata,
)


# -- Pydantic validation tests --


def test_labelfree_entry():
    entry = ChannelAnnotationEntry(channel_type="labelfree")
    assert entry.channel_type == "labelfree"
    assert entry.biological_annotation is None


def test_fluorescence_entry():
    entry = ChannelAnnotationEntry(
        channel_type="fluorescence",
        biological_annotation=BiologicalAnnotation(
            organelle="endoplasmic_reticulum",
            marker="SEC61B",
            marker_type="protein_tag",
            fluorophore="eGFP",
        ),
    )
    assert entry.biological_annotation.organelle == "endoplasmic_reticulum"
    assert entry.biological_annotation.fluorophore == "eGFP"


def test_perturbation_extra_fields():
    p = Perturbation(name="ZIKV", type="virus", hours_post=24.0, moi=0.5)
    assert p.moi == 0.5


def test_invalid_channel_type_rejected():
    with pytest.raises(ValidationError):
        ChannelAnnotationEntry(channel_type="brightfield")


def test_invalid_marker_type_rejected():
    with pytest.raises(ValidationError):
        BiologicalAnnotation(
            organelle="nucleus",
            marker="H2B",
            marker_type="invalid_type",
        )


# -- Helper tests --


def test_well_from_position_name():
    assert _well_from_position_name("A/1/0") == "A/1"
    assert _well_from_position_name("B/3/2") == "B/3"


# -- Integration tests --


def _make_annotation_config(
    channel_names: list[str],
    well_paths: list[str],
) -> AnnotationConfig:
    """Build an AnnotationConfig matching the given channels and wells."""
    channel_annotation = {}
    for ch in channel_names:
        channel_annotation[ch] = ChannelAnnotationEntry(channel_type="labelfree")

    experiment_metadata = {}
    for i, wp in enumerate(well_paths):
        experiment_metadata[wp] = WellExperimentMetadata(
            perturbations=(
                [Perturbation(name="ZIKV", type="virus", hours_post=24.0)]
                if i == 0
                else []
            ),
            time_sampling_minutes=30.0,
        )

    return AnnotationConfig(
        channel_annotation=channel_annotation,
        experiment_metadata=experiment_metadata,
    )


def test_write_channel_annotation_to_all_fovs(multi_well_hcs_dataset):
    annotation = _make_annotation_config(
        channel_names=["Phase", "Fluorescence_405"],
        well_paths=["A/1", "A/2"],
    )
    write_annotation_metadata(str(multi_well_hcs_dataset), annotation)

    with open_ome_zarr(multi_well_hcs_dataset, mode="r") as plate:
        # Plate-level
        assert "channel_annotation" in plate.zattrs
        assert "Phase" in plate.zattrs["channel_annotation"]
        assert "Fluorescence_405" in plate.zattrs["channel_annotation"]

        # Every FOV
        for _, pos in plate.positions():
            assert "channel_annotation" in pos.zattrs
            assert "Phase" in pos.zattrs["channel_annotation"]
            assert "Fluorescence_405" in pos.zattrs["channel_annotation"]


def test_write_experiment_metadata_per_well(multi_well_hcs_dataset):
    annotation = _make_annotation_config(
        channel_names=["Phase", "Fluorescence_405"],
        well_paths=["A/1", "A/2"],
    )
    write_annotation_metadata(str(multi_well_hcs_dataset), annotation)

    with open_ome_zarr(multi_well_hcs_dataset, mode="r") as plate:
        for name, pos in plate.positions():
            meta = pos.zattrs["experiment_metadata"]
            well_path = _well_from_position_name(name)
            if well_path == "A/1":
                assert len(meta["perturbations"]) == 1
                assert meta["perturbations"][0]["name"] == "ZIKV"
            elif well_path == "A/2":
                assert len(meta["perturbations"]) == 0
            assert meta["time_sampling_minutes"] == 30.0


def test_unknown_channel_raises(multi_well_hcs_dataset):
    annotation = AnnotationConfig(
        channel_annotation={
            "NonexistentChannel": ChannelAnnotationEntry(channel_type="labelfree"),
        },
        experiment_metadata={
            "A/1": WellExperimentMetadata(time_sampling_minutes=30.0),
        },
    )
    with pytest.raises(ValueError, match="NonexistentChannel"):
        write_annotation_metadata(str(multi_well_hcs_dataset), annotation)


def test_unknown_well_raises(multi_well_hcs_dataset):
    annotation = AnnotationConfig(
        channel_annotation={
            "Phase": ChannelAnnotationEntry(channel_type="labelfree"),
        },
        experiment_metadata={
            "Z/99": WellExperimentMetadata(time_sampling_minutes=30.0),
        },
    )
    with pytest.raises(ValueError, match="Z/99"):
        write_annotation_metadata(str(multi_well_hcs_dataset), annotation)
