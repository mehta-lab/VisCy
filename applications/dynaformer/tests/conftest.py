"""Synthetic timelapse plate and tracking tables for dynaformer tests."""

import numpy as np
import pytest
from iohub import open_ome_zarr

from viscy_data.track_window import TrackWindowDataModule
from viscy_transforms import NormalizeSampled

SOURCE_CHANNEL = "Phase3D"
TARGET_CHANNELS = ["mKO", "mVenus"]
MASK_CHANNEL = "nuclei_mask"
CHANNEL_NAMES = [SOURCE_CHANNEL, *TARGET_CHANNELS, MASK_CHANNEL]

TIMEPOINTS = 12
HEIGHT = WIDTH = 160
# The final patch must be divisible by the encoder's total stride (32 for a
# stride-4 stem and four stages), so 64 is the smallest usable size.
PATCH = 96
FINAL_PATCH = 64

# Track lengths chosen so some tracks are longer than the default window and
# some are shorter, which is what exercises the variable-length clip path.
# Enough tracks that both sides of a split are populated. Track 3 sits on the
# border to exercise patch clamping.
SHORT_LENGTH = 5
TRACK_LENGTHS = {1: TIMEPOINTS, 2: SHORT_LENGTH, 3: TIMEPOINTS, 4: TIMEPOINTS, 5: SHORT_LENGTH, 6: TIMEPOINTS}


@pytest.fixture
def timelapse_plate(tmp_path):
    """Write a two-FOV timelapse HCS OME-Zarr with normalization metadata."""
    zarr_path = tmp_path / "timelapse.zarr"
    rng = np.random.default_rng(0)
    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=CHANNEL_NAMES) as dataset:
        for fov in ("0", "1"):
            position = dataset.create_position("A", "1", fov)
            array = rng.random((TIMEPOINTS, len(CHANNEL_NAMES), 1, HEIGHT, WIDTH)).astype(np.float32)
            # Binarize the mask channel so masked pooling has something to pool.
            array[:, -1] = (array[:, -1] > 0.5).astype(np.float32)
            position.create_image("0", array, chunks=(1, 1, 1, HEIGHT, WIDTH))
    norm_meta = {name: {"fov_statistics": {"mean": 0.5, "std": 0.29}} for name in CHANNEL_NAMES}
    with open_ome_zarr(zarr_path, mode="r+") as dataset:
        for _, position in dataset.positions():
            position.zattrs["normalization"] = norm_meta
    return zarr_path


@pytest.fixture
def tracks_dir(tmp_path):
    """Write per-FOV tracking CSVs in the ultrack column layout."""
    import pandas as pd

    root = tmp_path / "tracks"
    for fov in ("0", "1"):
        rows = []
        for track_id, length in TRACK_LENGTHS.items():
            for t in range(length):
                # No fov_name column: the loader casts tracking tables to int
                # and derives the FOV from the position it pairs the table with.
                rows.append(
                    {
                        "track_id": track_id,
                        "t": t,
                        "id": track_id * 1000 + t,
                        "parent_track_id": -1,
                        "parent_id": -1,
                        "z": 0,
                        # Walk the centroid so patches genuinely follow the object,
                        # and place track 3 at the border to exercise clamping.
                        "y": (HEIGHT - 4 if track_id == 3 else HEIGHT // 2 + t),
                        "x": (4 if track_id == 3 else WIDTH // 2 - t),
                    }
                )
        directory = root / "A" / "1" / fov
        directory.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(directory / "tracks.csv", index=False)
    return root


@pytest.fixture
def track_spec():
    """Expose the synthetic dataset's shape and channel layout to tests."""
    return {
        "source_channel": SOURCE_CHANNEL,
        "target_channels": TARGET_CHANNELS,
        "mask_channel": MASK_CHANNEL,
        "timepoints": TIMEPOINTS,
        "patch": PATCH,
        "final_patch": FINAL_PATCH,
        "track_lengths": TRACK_LENGTHS,
        "short_length": SHORT_LENGTH,
        "border_track_id": 3,
    }


@pytest.fixture
def make_datamodule(timelapse_plate, tracks_dir):
    """Return a factory building a data module over the synthetic plate."""

    def _make(**overrides) -> TrackWindowDataModule:
        kwargs = {
            "data_path": str(timelapse_plate),
            "tracks_path": str(tracks_dir),
            "source_channel": SOURCE_CHANNEL,
            "target_channel": TARGET_CHANNELS,
            "mask_channel": MASK_CHANNEL,
            "z_range": (0, 1),
            "initial_yx_patch_size": (PATCH, PATCH),
            "final_yx_patch_size": (FINAL_PATCH, FINAL_PATCH),
            "window_size": 8,
            "min_frames": 4,
            "frames_per_batch": 16,
            "num_workers": 0,
            "normalizations": [
                NormalizeSampled(
                    keys=[SOURCE_CHANNEL, *TARGET_CHANNELS],
                    level="fov_statistics",
                    subtrahend="mean",
                    divisor="std",
                )
            ],
        }
        kwargs.update(overrides)
        return TrackWindowDataModule(**kwargs)

    return _make
