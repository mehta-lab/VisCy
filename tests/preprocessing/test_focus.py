import pytest
from iohub import open_ome_zarr

from viscy.preprocessing.focus import FocusSliceMetric
from viscy.preprocessing.qc_metrics import generate_qc_metadata


@pytest.fixture
def focus_metric():
    return FocusSliceMetric(
        NA_det=0.55,
        lambda_ill=0.532,
        pixel_size=0.325,
        channel_names=["Phase"],
    )


@pytest.fixture
def focus_metric_all_channels():
    return FocusSliceMetric(
        NA_det=0.55,
        lambda_ill=0.532,
        pixel_size=0.325,
        channel_names=-1,
    )


def test_focus_slice_metric_call(temporal_hcs_dataset, focus_metric):
    with open_ome_zarr(temporal_hcs_dataset, mode="r") as plate:
        channel_index = plate.channel_names.index("Phase")
        _, pos = next(iter(plate.positions()))
        result = focus_metric(pos, "Phase", channel_index, num_workers=1)

    assert "fov_statistics" in result
    assert "per_timepoint" in result
    assert "z_focus_mean" in result["fov_statistics"]
    assert "z_focus_std" in result["fov_statistics"]
    for t in range(5):
        assert str(t) in result["per_timepoint"]
        idx = result["per_timepoint"][str(t)]
        assert isinstance(idx, int)
        assert 0 <= idx < 10


def test_generate_qc_metadata_focus(temporal_hcs_dataset, focus_metric):
    generate_qc_metadata(
        zarr_dir=temporal_hcs_dataset,
        metrics=[focus_metric],
        num_workers=1,
    )

    with open_ome_zarr(temporal_hcs_dataset, mode="r") as plate:
        assert "focus_slice" in plate.zattrs
        assert "Phase" in plate.zattrs["focus_slice"]
        ds_stats = plate.zattrs["focus_slice"]["Phase"]["dataset_statistics"]
        assert "z_focus_mean" in ds_stats
        assert "z_focus_std" in ds_stats
        assert "z_focus_min" in ds_stats
        assert "z_focus_max" in ds_stats

        for _, pos in plate.positions():
            assert "focus_slice" in pos.zattrs
            pos_meta = pos.zattrs["focus_slice"]["Phase"]
            assert "dataset_statistics" in pos_meta
            assert "fov_statistics" in pos_meta
            assert "per_timepoint" in pos_meta


def test_generate_qc_metadata_skips_unconfigured_channel(
    temporal_hcs_dataset, focus_metric
):
    generate_qc_metadata(
        zarr_dir=temporal_hcs_dataset,
        metrics=[focus_metric],
        num_workers=1,
    )

    with open_ome_zarr(temporal_hcs_dataset, mode="r") as plate:
        assert "Retardance" not in plate.zattrs.get("focus_slice", {})
        for _, pos in plate.positions():
            assert "Retardance" not in pos.zattrs.get("focus_slice", {})


def test_generate_qc_metadata_per_timepoint_count(temporal_hcs_dataset, focus_metric):
    generate_qc_metadata(
        zarr_dir=temporal_hcs_dataset,
        metrics=[focus_metric],
        num_workers=1,
    )

    with open_ome_zarr(temporal_hcs_dataset, mode="r") as plate:
        for _, pos in plate.positions():
            per_tp = pos.zattrs["focus_slice"]["Phase"]["per_timepoint"]
            assert len(per_tp) == 5
            for t in range(5):
                assert str(t) in per_tp


def test_generate_qc_metadata_all_channels(
    temporal_hcs_dataset, focus_metric_all_channels
):
    generate_qc_metadata(
        zarr_dir=temporal_hcs_dataset,
        metrics=[focus_metric_all_channels],
        num_workers=1,
    )

    with open_ome_zarr(temporal_hcs_dataset, mode="r") as plate:
        for ch in plate.channel_names:
            assert ch in plate.zattrs["focus_slice"]
            for _, pos in plate.positions():
                assert ch in pos.zattrs["focus_slice"]
