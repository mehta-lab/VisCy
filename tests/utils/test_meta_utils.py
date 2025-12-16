from pathlib import Path

import numpy as np
from iohub import open_ome_zarr

from viscy.utils.meta_utils import (
    _grid_sample_timepoint,
    generate_normalization_metadata,
)


def test_grid_sample_timepoint(temporal_hcs_dataset: Path):
    """Test that _grid_sample_timepoint samples the correct timepoint."""
    plate = open_ome_zarr(temporal_hcs_dataset, mode="r")

    # Get first position
    position_keys = list(plate.positions())
    _, position = position_keys[0]

    grid_spacing = 10
    channel_index = 0
    num_workers = 2
    num_timepoints = position["0"].shape[0]

    # Sample different timepoints
    samples_per_timepoint = []
    for timepoint_index in range(num_timepoints):
        samples = _grid_sample_timepoint(
            position=position,
            grid_spacing=grid_spacing,
            channel_index=channel_index,
            timepoint_index=timepoint_index,
            num_workers=num_workers,
        )
        samples_per_timepoint.append(samples)

        # Verify shape is correct (should be 3D: Z, Y_sampled, X_sampled)
        assert len(samples.shape) == 3, (
            f"Expected 3D samples, got shape {samples.shape}"
        )

    # Verify that samples from different timepoints differ
    # (due to random data generation)
    means = [np.mean(s) for s in samples_per_timepoint]
    unique_means = len(set(means))
    assert unique_means > 1, "All timepoint samples are identical, expected variation"

    plate.close()


def test_generate_normalization_metadata_structure(temporal_hcs_dataset: Path):
    """Test that generate_normalization_metadata creates correct metadata structure."""
    num_timepoints = 5  # As specified in the temporal_hcs_dataset fixture

    # Generate normalization metadata
    generate_normalization_metadata(
        str(temporal_hcs_dataset), num_workers=2, channel_ids=-1, grid_spacing=10
    )

    # Reopen and check metadata
    plate = open_ome_zarr(temporal_hcs_dataset, mode="r")

    # Check plate-level metadata
    assert "normalization" in plate.zattrs, "Normalization field not found in metadata"

    for channel_name in plate.channel_names:
        assert channel_name in plate.zattrs["normalization"], (
            f"Channel {channel_name} not found in normalization metadata"
        )

        channel_norm = plate.zattrs["normalization"][channel_name]

        # Check that dataset statistics exist
        assert "dataset_statistics" in channel_norm, (
            "dataset_statistics not found in metadata"
        )

        # Check that timepoint statistics exist
        assert "timepoint_statistics" in channel_norm, (
            "timepoint_statistics not found in metadata"
        )

        timepoint_stats = channel_norm["timepoint_statistics"]

        # Check that all timepoints are present
        for t in range(num_timepoints):
            assert str(t) in timepoint_stats, (
                f"Timepoint {t} not found in timepoint_statistics"
            )

            # Verify statistics structure
            t_stats = timepoint_stats[str(t)]
            required_keys = ["mean", "std", "median", "iqr", "p5", "p95", "p95_p5"]
            for key in required_keys:
                assert key in t_stats, f"{key} not found in timepoint {t} statistics"

    # Check position-level metadata
    position_keys = list(plate.positions())
    for _, position in position_keys:
        assert "normalization" in position.zattrs, (
            "Normalization field not found in position metadata"
        )

        for channel_name in plate.channel_names:
            assert channel_name in position.zattrs["normalization"], (
                f"Channel {channel_name} not found in position normalization metadata"
            )

            pos_channel_norm = position.zattrs["normalization"][channel_name]

            # Check that all three types of statistics exist
            assert "dataset_statistics" in pos_channel_norm, (
                "dataset_statistics not found in position metadata"
            )
            assert "fov_statistics" in pos_channel_norm, (
                "fov_statistics not found in position metadata"
            )
            assert "timepoint_statistics" in pos_channel_norm, (
                "timepoint_statistics not found in position metadata"
            )

    plate.close()


def test_generate_normalization_timepoint_values_differ(temporal_hcs_dataset: Path):
    """Test that per-timepoint statistics have different values across timepoints."""
    num_timepoints = 5

    # Generate normalization metadata
    generate_normalization_metadata(
        str(temporal_hcs_dataset), num_workers=2, channel_ids=0, grid_spacing=10
    )

    # Reopen and check metadata
    plate = open_ome_zarr(temporal_hcs_dataset, mode="r")

    channel_name = plate.channel_names[0]
    timepoint_stats = plate.zattrs["normalization"][channel_name][
        "timepoint_statistics"
    ]

    # Extract median values for each timepoint
    medians = [timepoint_stats[str(t)]["median"] for t in range(num_timepoints)]

    # Since data is randomly generated per timepoint,
    # medians should vary across timepoints
    unique_medians = len(set(medians))
    assert unique_medians > 1, "All timepoint medians are identical, expected variation"

    # All medians should be positive floats
    for t, median in enumerate(medians):
        assert isinstance(median, (int, float)), f"Timepoint {t} median is not numeric"
        assert median >= 0, f"Timepoint {t} median is negative"

    plate.close()


def test_generate_normalization_single_channel(temporal_hcs_dataset: Path):
    """Test normalization metadata generation for a single channel."""

    # Generate normalization for only channel 0
    generate_normalization_metadata(
        str(temporal_hcs_dataset), num_workers=2, channel_ids=0, grid_spacing=15
    )

    # Reopen and check metadata
    plate = open_ome_zarr(temporal_hcs_dataset, mode="r")

    # Only channel 0 should have normalization metadata
    assert "normalization" in plate.zattrs, (
        "Normalization metadata not created at plate level"
    )
    assert len(plate.zattrs["normalization"]) == 1, (
        "Expected only one channel in normalization metadata"
    )

    channel_name = plate.channel_names[0]
    assert channel_name in plate.zattrs["normalization"], (
        f"Channel {channel_name} not found in metadata"
    )

    plate.close()


def test_grid_sample_timepoint_shape(temporal_hcs_dataset: Path):
    """Test that _grid_sample_timepoint returns correctly shaped array."""
    plate = open_ome_zarr(temporal_hcs_dataset, mode="r")

    # Get first position
    position_keys = list(plate.positions())
    _, position = position_keys[0]

    grid_spacing = 10
    channel_index = 0
    timepoint_index = 0
    num_workers = 2

    samples = _grid_sample_timepoint(
        position=position,
        grid_spacing=grid_spacing,
        channel_index=channel_index,
        timepoint_index=timepoint_index,
        num_workers=num_workers,
    )

    # Expected shape: (Z, Y//grid_spacing, X//grid_spacing)
    expected_z = position["0"].shape[2]
    expected_y = (position["0"].shape[3] + grid_spacing - 1) // grid_spacing
    expected_x = (position["0"].shape[4] + grid_spacing - 1) // grid_spacing

    assert samples.shape[0] == expected_z, (
        f"Expected Z={expected_z}, got {samples.shape[0]}"
    )
    # Y and X dimensions might be slightly different due to grid sampling
    assert samples.shape[1] <= expected_y, "Y dimension larger than expected"
    assert samples.shape[2] <= expected_x, "X dimension larger than expected"

    plate.close()
