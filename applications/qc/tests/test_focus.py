"""Tests for focus slice QC metric."""

import numpy as np
import pytest
from iohub import open_ome_zarr
from iohub.core.config import TensorStoreConfig

from qc.focus import FocusSliceMetric, audit_focus_slice
from qc.qc_metrics import generate_qc_metadata


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
        channel_names=["Phase", "Retardance"],
    )


def test_focus_slice_metric_call(temporal_hcs_dataset, focus_metric):
    with open_ome_zarr(
        temporal_hcs_dataset,
        mode="r",
        implementation="tensorstore",
        implementation_config=TensorStoreConfig(data_copy_concurrency=1),
    ) as plate:
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


def test_generate_qc_metadata_skips_unconfigured_channel(temporal_hcs_dataset, focus_metric):
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


def test_generate_qc_metadata_all_channels(temporal_hcs_dataset, focus_metric_all_channels):
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


# ---------------------------------------------------------------------------
# audit_focus_slice — validate ALREADY-written focus metadata (2D vs 3D aware)
# ---------------------------------------------------------------------------


def _write_focus_per_timepoint(zarr_path, channel_name, per_timepoint_by_fov):
    """Inject a ``focus_slice.{channel}.per_timepoint`` block into each FOV's zattrs."""
    with open_ome_zarr(zarr_path, mode="r+") as plate:
        for name, pos in plate.positions():
            indices = per_timepoint_by_fov[name]
            pos.zattrs["focus_slice"] = {
                channel_name: {"per_timepoint": {str(t): int(v) for t, v in enumerate(indices)}}
            }


def test_audit_focus_slice_flags_edge_indices(temporal_hcs_dataset):
    """On a 3D stack (Z=10), focus indices at 0 or Z-1 are flagged as suspect."""
    # A/1/0: two edge frames (z=0, z=9); A/1/1: all mid-stack (clean).
    _write_focus_per_timepoint(
        temporal_hcs_dataset,
        "Phase",
        {"A/1/0": [0, 5, 5, 9, 4], "A/1/1": [5, 5, 6, 5, 4]},
    )
    summary = audit_focus_slice(temporal_hcs_dataset, "Phase")
    assert summary["z_depth"] == 10
    assert summary["is_2d"] is False
    assert summary["n_fovs"] == 2
    assert summary["n_timepoints_total"] == 10
    assert summary["n_suspect"] == 2  # only the two edge frames in A/1/0
    assert summary["fovs_affected"] == {"A/1/0": 2}
    assert summary["valid_focus"]["min"] == 4 and summary["valid_focus"]["max"] == 6


def test_audit_focus_slice_2d_never_flags(tmp_path):
    """A 2D acquisition (Z=1) trivially focuses at slice 0 — nothing is flagged."""
    zarr_path = tmp_path / "focus2d.zarr"
    dataset = open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=["Phase"])
    rng = np.random.default_rng(0)
    for fov in ("0", "1"):
        pos = dataset.create_position("A", "1", fov)
        pos.create_image("0", rng.random((3, 1, 1, 32, 32)).astype(np.float32), chunks=(1, 1, 1, 32, 32))
    dataset.close()
    _write_focus_per_timepoint(zarr_path, "Phase", {"A/1/0": [0, 0, 0], "A/1/1": [0, 0, 0]})

    summary = audit_focus_slice(zarr_path, "Phase")
    assert summary["z_depth"] == 1
    assert summary["is_2d"] is True
    assert summary["n_suspect"] == 0  # z=0 is expected in 2D, not a failure
    assert summary["fovs_affected"] == {}
    assert summary["valid_focus"] is None


def test_audit_focus_slice_missing_channel_raises(temporal_hcs_dataset):
    _write_focus_per_timepoint(temporal_hcs_dataset, "Phase", {"A/1/0": [5, 5, 5, 5, 5], "A/1/1": [5, 5, 5, 5, 5]})
    with pytest.raises(KeyError, match="No focus_slice metadata"):
        audit_focus_slice(temporal_hcs_dataset, "Retardance")
