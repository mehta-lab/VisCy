"""Tests for airtable_utils.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from airtable_utils.schemas import (
    BiologicalAnnotation,
    ChannelAnnotationEntry,
    DatasetRecord,
    Perturbation,
    WellExperimentMetadata,
    parse_channel_name,
    parse_position_name,
)

# ============================================================================
# parse_channel_name
# ============================================================================


class TestParseChannelName:
    """Test parse_channel_name for various channel label formats."""

    # -- fluorescence --------------------------------------------------------

    def test_fluorescence_full_pattern(self):
        result = parse_channel_name("raw GFP EX488 EM525-45")
        assert result["channel_type"] == "fluorescence"
        assert result["filter_cube"] == "GFP"
        assert result["excitation_nm"] == 488
        assert result["emission_nm"] == 525

    def test_fluorescence_no_bandwidth(self):
        result = parse_channel_name("raw DAPI EX405 EM450")
        assert result["channel_type"] == "fluorescence"
        assert result["filter_cube"] == "DAPI"
        assert result["excitation_nm"] == 405
        assert result["emission_nm"] == 450

    def test_fluorescence_case_insensitive(self):
        result = parse_channel_name("RAW mCherry ex561 em600-50")
        assert result["channel_type"] == "fluorescence"
        assert result["filter_cube"] == "mCherry"

    def test_fluorescence_fallback_ex_em_without_raw(self):
        """EX/EM pattern without 'raw' prefix still detected as fluorescence."""
        result = parse_channel_name("GFP EX488 EM525")
        assert result["channel_type"] == "fluorescence"
        assert result["excitation_nm"] == 488
        assert result["emission_nm"] == 525
        # filter_cube not extracted in fallback path
        assert "filter_cube" not in result

    # -- labelfree -----------------------------------------------------------

    def test_labelfree_phase(self):
        result = parse_channel_name("Phase3D")
        assert result["channel_type"] == "labelfree"

    def test_labelfree_brightfield(self):
        result = parse_channel_name("Brightfield_LED")
        assert result["channel_type"] == "labelfree"

    def test_labelfree_retardance(self):
        result = parse_channel_name("Retardance_PolScope")
        assert result["channel_type"] == "labelfree"

    def test_labelfree_bf_prefix(self):
        result = parse_channel_name("BF_LED_Matrix_Full")
        assert result["channel_type"] == "labelfree"

    def test_labelfree_dic(self):
        result = parse_channel_name("DIC")
        assert result["channel_type"] == "labelfree"

    # -- virtual_stain -------------------------------------------------------

    def test_virtual_stain_prediction(self):
        result = parse_channel_name("nuclei_prediction")
        assert result["channel_type"] == "virtual_stain"

    def test_virtual_stain_virtual(self):
        result = parse_channel_name("virtual_fluorescence")
        assert result["channel_type"] == "virtual_stain"

    def test_virtual_stain_vs_prefix(self):
        result = parse_channel_name("vs_nucleus")
        assert result["channel_type"] == "virtual_stain"

    # -- unknown / edge cases ------------------------------------------------

    def test_unknown_channel(self):
        result = parse_channel_name("some_random_channel")
        assert result["channel_type"] == "unknown"

    def test_empty_string(self):
        result = parse_channel_name("")
        assert result["channel_type"] == "unknown"


# ============================================================================
# parse_position_name
# ============================================================================


class TestParsePositionName:
    """Test parse_position_name for OME-Zarr position paths."""

    def test_standard_three_part_path(self):
        well, fov = parse_position_name("B/1/000000")
        assert well == "B/1"
        assert fov == "000000"

    def test_deep_path(self):
        well, fov = parse_position_name("A/3/000005")
        assert well == "A/3"
        assert fov == "000005"

    def test_two_part_path_no_fov(self):
        well, fov = parse_position_name("C/2")
        assert well == "C/2"
        assert fov == ""

    def test_single_part_path(self):
        well, fov = parse_position_name("A")
        assert well == "A"
        assert fov == ""

    def test_four_part_path(self):
        """Extra parts beyond 3 are ignored; only first 2 form the well."""
        well, fov = parse_position_name("D/4/000010/extra")
        assert well == "D/4"
        assert fov == "000010"


# ============================================================================
# DatasetRecord.from_airtable_record
# ============================================================================


class TestDatasetRecordFromAirtable:
    """Test DatasetRecord.from_airtable_record with various response shapes."""

    def test_full_record_with_select_dicts(self, sample_airtable_records):
        """Record where select fields are dicts with 'name' key."""
        rec = DatasetRecord.from_airtable_record(sample_airtable_records[0])
        assert rec.dataset == "dataset_alpha"
        assert rec.well_id == "A/1"
        assert rec.fov == "000000"
        assert rec.cell_type == "HEK293T"
        assert rec.cell_state == "healthy"
        assert rec.cell_line == ["HEK293T-H2B-mCherry"]
        assert rec.organelle == "nucleus"
        assert rec.perturbation == "DMSO"
        assert rec.hours_post_perturbation == 24.0
        assert rec.time_interval_min == 5.0
        assert rec.seeding_density == 50000
        assert rec.treatment_concentration_nm == 100.0
        assert rec.channel_0_name == "Phase3D"
        assert rec.channel_0_marker == "Membrane"
        assert rec.channel_1_name == "raw GFP EX488 EM525-45"
        assert rec.channel_1_marker == "Endoplasmic Reticulum"
        assert rec.data_path == "/hpc/datasets/alpha.zarr"
        assert rec.fluorescence_modality == "widefield"
        assert rec.t_shape == 50
        assert rec.c_shape == 2
        assert rec.z_shape == 30
        assert rec.y_shape == 2048
        assert rec.x_shape == 2048
        assert rec.record_id == "rec001"

    def test_record_with_plain_string_fields(self, sample_airtable_records):
        """Record where select fields are plain strings (no dict wrapper)."""
        rec = DatasetRecord.from_airtable_record(sample_airtable_records[1])
        assert rec.dataset == "dataset_beta"
        assert rec.cell_type == "A549"
        assert rec.cell_state == "infected"
        assert rec.organelle == "mitochondria"
        assert rec.perturbation == "ZIKV"
        assert rec.moi == 0.5
        assert rec.cell_line is None

    def test_minimal_record(self):
        """Record with only required fields."""
        minimal = {
            "id": "recMIN",
            "fields": {
                "dataset": "minimal_ds",
                "well_id": "A/1",
            },
        }
        rec = DatasetRecord.from_airtable_record(minimal)
        assert rec.dataset == "minimal_ds"
        assert rec.well_id == "A/1"
        assert rec.fov is None
        assert rec.cell_type is None
        assert rec.channel_0_name is None
        assert rec.record_id == "recMIN"

    def test_empty_fields_record(self):
        """Record with empty 'fields' dict."""
        empty = {"id": "recEMPTY", "fields": {}}
        rec = DatasetRecord.from_airtable_record(empty)
        assert rec.dataset == ""
        assert rec.well_id == ""
        assert rec.record_id == "recEMPTY"

    def test_record_without_id(self):
        """Record without an 'id' key."""
        no_id = {"fields": {"dataset": "no_id_ds", "well_id": "X/1"}}
        rec = DatasetRecord.from_airtable_record(no_id)
        assert rec.record_id is None
        assert rec.dataset == "no_id_ds"

    def test_multiselect_cell_line(self):
        """cell_line with list-of-dicts multipleSelects format."""
        record = {
            "id": "recMS",
            "fields": {
                "dataset": "multi",
                "well_id": "A/1",
                "cell_line": [
                    {"name": "Line-A"},
                    {"name": "Line-B"},
                ],
            },
        }
        rec = DatasetRecord.from_airtable_record(record)
        assert rec.cell_line == ["Line-A", "Line-B"]

    def test_multiselect_cell_line_plain_strings(self):
        """cell_line with list-of-strings format."""
        record = {
            "id": "recMS2",
            "fields": {
                "dataset": "multi2",
                "well_id": "B/2",
                "cell_line": ["Line-C", "Line-D"],
            },
        }
        rec = DatasetRecord.from_airtable_record(record)
        assert rec.cell_line == ["Line-C", "Line-D"]


# ============================================================================
# BiologicalAnnotation
# ============================================================================


class TestBiologicalAnnotation:
    """Test BiologicalAnnotation pydantic model validation."""

    def test_valid_protein_tag(self):
        ba = BiologicalAnnotation(
            organelle="nucleus",
            marker="H2B",
            marker_type="protein_tag",
            fluorophore="mCherry",
        )
        assert ba.organelle == "nucleus"
        assert ba.marker == "H2B"
        assert ba.marker_type == "protein_tag"
        assert ba.fluorophore == "mCherry"

    def test_valid_without_fluorophore(self):
        ba = BiologicalAnnotation(
            organelle="mitochondria",
            marker="COX8A",
            marker_type="direct_label",
        )
        assert ba.fluorophore is None

    def test_valid_nuclear_dye(self):
        ba = BiologicalAnnotation(
            organelle="nucleus",
            marker="Hoechst",
            marker_type="nuclear_dye",
        )
        assert ba.marker_type == "nuclear_dye"

    def test_valid_virtual_stain(self):
        ba = BiologicalAnnotation(
            organelle="endoplasmic_reticulum",
            marker="predicted",
            marker_type="virtual_stain",
        )
        assert ba.marker_type == "virtual_stain"

    def test_invalid_marker_type_rejected(self):
        with pytest.raises(ValidationError):
            BiologicalAnnotation(
                organelle="nucleus",
                marker="H2B",
                marker_type="invalid_type",
            )

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            BiologicalAnnotation(organelle="nucleus")


# ============================================================================
# Perturbation
# ============================================================================


class TestPerturbation:
    """Test Perturbation pydantic model validation."""

    def test_valid_perturbation(self):
        p = Perturbation(name="ZIKV", type="virus", hours_post=48.0)
        assert p.name == "ZIKV"
        assert p.type == "virus"
        assert p.hours_post == 48.0

    def test_default_type(self):
        p = Perturbation(name="DMSO", hours_post=24.0)
        assert p.type == "unknown"

    def test_extra_fields_allowed(self):
        p = Perturbation(
            name="ZIKV",
            type="virus",
            hours_post=48.0,
            moi=0.5,
            concentration_nm=100.0,
        )
        assert p.moi == 0.5
        assert p.concentration_nm == 100.0

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            Perturbation(hours_post=24.0)

    def test_missing_hours_post_rejected(self):
        with pytest.raises(ValidationError):
            Perturbation(name="DMSO")


# ============================================================================
# WellExperimentMetadata (aliased as ExperimentMetadata in the request)
# ============================================================================


class TestWellExperimentMetadata:
    """Test WellExperimentMetadata pydantic model validation."""

    def test_valid_metadata(self):
        m = WellExperimentMetadata(
            perturbations=[
                Perturbation(name="ZIKV", type="virus", hours_post=48.0),
            ],
            time_sampling_minutes=5.0,
        )
        assert len(m.perturbations) == 1
        assert m.time_sampling_minutes == 5.0

    def test_empty_perturbations(self):
        m = WellExperimentMetadata(time_sampling_minutes=10.0)
        assert m.perturbations == []

    def test_missing_time_sampling_rejected(self):
        with pytest.raises(ValidationError):
            WellExperimentMetadata(
                perturbations=[],
            )

    def test_multiple_perturbations(self):
        m = WellExperimentMetadata(
            perturbations=[
                Perturbation(name="ZIKV", type="virus", hours_post=48.0),
                Perturbation(name="Drug_A", type="drug", hours_post=24.0),
            ],
            time_sampling_minutes=5.0,
        )
        assert len(m.perturbations) == 2
        assert m.perturbations[0].name == "ZIKV"
        assert m.perturbations[1].name == "Drug_A"


# ============================================================================
# ChannelAnnotationEntry
# ============================================================================


class TestChannelAnnotationEntry:
    """Test ChannelAnnotationEntry pydantic model."""

    def test_fluorescence_with_annotation(self):
        entry = ChannelAnnotationEntry(
            channel_type="fluorescence",
            biological_annotation=BiologicalAnnotation(
                organelle="nucleus",
                marker="H2B",
                marker_type="protein_tag",
                fluorophore="mCherry",
            ),
        )
        assert entry.channel_type == "fluorescence"
        assert entry.biological_annotation.organelle == "nucleus"

    def test_labelfree_without_annotation(self):
        entry = ChannelAnnotationEntry(channel_type="labelfree")
        assert entry.channel_type == "labelfree"
        assert entry.biological_annotation is None

    def test_invalid_channel_type_rejected(self):
        with pytest.raises(ValidationError):
            ChannelAnnotationEntry(channel_type="invalid")
