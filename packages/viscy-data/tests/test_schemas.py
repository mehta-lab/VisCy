"""Tests for viscy_data.schemas.FOVRecord."""

from viscy_data.schemas import FOVRecord


class TestFOVRecordCreation:
    """Test FOVRecord instantiation with various field combinations."""

    def test_all_fields(self):
        """Verify FOVRecord accepts every field."""
        record = FOVRecord(
            dataset="exp001",
            well_id="A/1",
            fov="0",
            data_path="/data/exp001.zarr",
            tracks_path="/tracks/exp001",
            channel_names=["Phase", "GFP", "RFP"],
            time_interval_min=15.0,
            hours_post_perturbation=2.0,
            moi=0.5,
            marker="TOMM20",
            organelle="mitochondria",
            cell_state="infected",
            cell_type="A549",
            cell_line=["A549-GFP"],
            perturbation="drug_x",
            seeding_density=50000,
            treatment_concentration_nm=100.0,
            fluorescence_modality="widefield",
            t_shape=100,
            c_shape=3,
            z_shape=10,
            y_shape=2048,
            x_shape=2048,
        )
        assert record.dataset == "exp001"
        assert record.well_id == "A/1"
        assert record.fov == "0"
        assert record.data_path == "/data/exp001.zarr"
        assert record.tracks_path == "/tracks/exp001"
        assert record.channel_names == ["Phase", "GFP", "RFP"]
        assert record.time_interval_min == 15.0
        assert record.hours_post_perturbation == 2.0
        assert record.moi == 0.5
        assert record.marker == "TOMM20"
        assert record.organelle == "mitochondria"
        assert record.cell_state == "infected"
        assert record.cell_type == "A549"
        assert record.cell_line == ["A549-GFP"]
        assert record.perturbation == "drug_x"
        assert record.seeding_density == 50000
        assert record.treatment_concentration_nm == 100.0
        assert record.fluorescence_modality == "widefield"
        assert record.t_shape == 100
        assert record.c_shape == 3
        assert record.z_shape == 10
        assert record.y_shape == 2048
        assert record.x_shape == 2048

    def test_minimal_fields(self):
        """Verify FOVRecord requires only dataset and well_id."""
        record = FOVRecord(dataset="exp002", well_id="B/3")
        assert record.dataset == "exp002"
        assert record.well_id == "B/3"

    def test_minimal_defaults(self):
        """Verify default values for optional fields."""
        record = FOVRecord(dataset="exp002", well_id="B/3")
        assert record.fov is None
        assert record.data_path is None
        assert record.tracks_path is None
        assert record.channel_names == []
        assert record.time_interval_min is None
        assert record.hours_post_perturbation is None
        assert record.moi is None
        assert record.marker is None
        assert record.organelle is None
        assert record.cell_state is None
        assert record.cell_type is None
        assert record.cell_line is None
        assert record.perturbation is None
        assert record.seeding_density is None
        assert record.treatment_concentration_nm is None
        assert record.fluorescence_modality is None
        assert record.t_shape is None
        assert record.c_shape is None
        assert record.z_shape is None
        assert record.y_shape is None
        assert record.x_shape is None

    def test_channel_names_list(self):
        """Verify channel_names accepts a list of strings."""
        record = FOVRecord(
            dataset="exp003",
            well_id="C/2",
            channel_names=["DAPI", "Brightfield", "mCherry"],
        )
        assert record.channel_names == ["DAPI", "Brightfield", "mCherry"]
        assert len(record.channel_names) == 3
