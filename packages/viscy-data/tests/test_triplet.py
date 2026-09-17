import pandas as pd
import torch
from iohub import open_ome_zarr
from pytest import mark, raises

from viscy_data import TripletDataModule, TripletDataset
from viscy_data.channel_utils import parse_channel_name


@mark.parametrize("include_wells", [None, ["A/1", "A/2", "B/1"]])
@mark.parametrize("exclude_fovs", [None, ["A/1/0", "A/1/1", "A/2/2", "B/1/3"]])
def test_datamodule_setup_fit(preprocessed_hcs_dataset, tracks_hcs_dataset, include_wells, exclude_fovs):
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    split_ratio = 0.75
    yx_patch_size = [32, 32]
    batch_size = 4
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        total_wells = len(list(dataset.wells()))
        fovs_per_well = len(dataset["A/1"])
    if include_wells is not None:
        total_wells = len(include_wells)
    total_fovs = total_wells * fovs_per_well
    if exclude_fovs is not None:
        total_fovs -= len(exclude_fovs)
    len_total = total_fovs * 2
    len_train = int(len_total * split_ratio)
    len_val = len_total - len_train
    dm = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=(4, 9),
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        split_ratio=split_ratio,
        batch_size=batch_size,
        fit_include_wells=include_wells,
        fit_exclude_fovs=exclude_fovs,
        return_negative=True,
    )
    dm.setup(stage="fit")
    assert len(dm.train_dataset) == len_train
    assert len(dm.val_dataset) == len_val
    all_tracks = pd.concat([dm.train_dataset.tracks, dm.val_dataset.tracks])
    filtered_fov_names = all_tracks["fov_name"].unique()
    for fov_name in filtered_fov_names:
        well_name, _ = fov_name.rsplit("/", 1)
        if include_wells is not None:
            assert well_name in include_wells
        if exclude_fovs is not None:
            assert fov_name not in exclude_fovs
    assert len(all_tracks) == len_total
    for batch in dm.train_dataloader():
        dm.on_after_batch_transfer(batch, 0)
        assert batch["anchor"].shape == (
            batch_size,
            len(channel_names),
            z_window_size,
            *yx_patch_size,
        )
        assert batch["negative"].shape == (
            batch_size,
            len(channel_names),
            z_window_size,
            *yx_patch_size,
        )


@mark.parametrize("z_window_size", [None, 3])
def test_datamodule_z_window_size(preprocessed_hcs_dataset, tracks_hcs_dataset, z_window_size):
    if z_window_size is not None:
        z_range = (4, 4 + z_window_size)
    else:
        z_range = (4, 9)
    yx_patch_size = [32, 32]
    batch_size = 4
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    dm = TripletDataModule(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=z_range,
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=batch_size,
        return_negative=True,
        z_window_size=z_window_size,
    )
    dm.setup(stage="fit")
    expected_z_shape = z_range[1] - z_range[0]
    for batch in dm.train_dataloader():
        dm.on_after_batch_transfer(batch, 0)
        assert batch["anchor"].shape == (
            batch_size,
            len(channel_names),
            expected_z_shape,
            *yx_patch_size,
        )
        assert batch["negative"].shape == (
            batch_size,
            len(channel_names),
            expected_z_shape,
            *yx_patch_size,
        )


def test_z_range_xor_extraction_window(preprocessed_hcs_dataset, tracks_hcs_dataset):
    """Exactly one of z_range / z_extraction_window must be provided."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    common = dict(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        num_workers=0,
    )
    with raises(ValueError, match="exactly one"):
        TripletDataModule(z_range=None, z_extraction_window=None, **common)
    with raises(ValueError, match="exactly one"):
        TripletDataModule(z_range=(4, 9), z_extraction_window=8, **common)


def test_reference_z_sampling_requires_focus_window_and_reduction(preprocessed_hcs_dataset, tracks_hcs_dataset):
    """Physical Z normalization is only valid for a focus window collapsed to 2D."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    common = dict(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        num_workers=0,
        reference_pixel_size_z_um=0.5,
    )
    with raises(ValueError, match="requires 'z_extraction_window'"):
        TripletDataModule(z_range=(4, 9), z_reduction="mip", **common)
    with raises(ValueError, match="requires 'z_reduction'"):
        TripletDataModule(z_extraction_window=5, **common)


def test_reference_z_sampling_converts_mip_window(preprocessed_hcs_dataset, tracks_hcs_dataset):
    """A reference-grid MIP window covers the same depth at another native Z sampling."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        pixel_scale = next(iter(dataset.positions()))[1].scale
        native_pixel_size_z_um = float(pixel_scale[-3])
        native_pixel_size_xy_um = float(pixel_scale[-1])

    reference_window = 3
    reference_pixel_size_z_um = native_pixel_size_z_um * 2
    expected_native_window = 6

    class RecordZ:
        def __init__(self):
            self.seen: list[int] = []

        def __call__(self, data):
            self.seen.append(int(data[channel_names[0]].shape[-3]))
            return data

    record_z = RecordZ()
    dm = TripletDataModule(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_extraction_window=reference_window,
        reference_pixel_size=native_pixel_size_xy_um,
        reference_pixel_size_z_um=reference_pixel_size_z_um,
        z_reduction="mip",
        augmentations=[record_z],
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=4,
    )
    assert dm._z_extraction_window == expected_native_window

    dm.setup(stage="fit")
    resolved = dm.train_dataset.z_range
    assert isinstance(resolved, dict)
    assert {z.stop - z.start for z in resolved.values()} == {expected_native_window}

    batch = next(iter(dm.train_dataloader()))
    dm.on_after_batch_transfer(batch, 0)
    assert set(record_z.seen) == {reference_window}
    assert batch["anchor"].shape[2] == 1


@mark.parametrize("z_focus_offset", [0.5, 0.3])
def test_focus_centered_z_range(tmp_path_factory, preprocessed_hcs_dataset, tracks_hcs_dataset, z_focus_offset):
    """z_extraction_window resolves a per-FOV focus-centered z_range from zattrs.

    Writes a different per-FOV ``z_focus_mean`` to each position's ``focus_slice``
    ``fov_statistics`` (on a private copy of the session-scoped dataset), then
    checks each FOV gets its own window of ``z_extraction_window`` slices centered
    on its focus plane with ``z_focus_offset`` of the window below it.
    """
    import shutil

    z_extraction_window = 5
    # Copy the session-scoped dataset so writing focus_slice does not leak.
    data_path = tmp_path_factory.mktemp("focus") / "data.zarr"
    shutil.copytree(preprocessed_hcs_dataset, data_path)
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
        fov_names = [name for name, _ in dataset.positions()]
        z_total = dataset[fov_names[0]]["0"].shape[2]
    focus_channel = channel_names[0]

    # Give each FOV a distinct focus plane so per-FOV resolution is exercised.
    per_fov_focus = {fov: float(3 + i % (z_total - 4)) for i, fov in enumerate(fov_names)}
    with open_ome_zarr(data_path, mode="r+") as dataset:
        for fov, pos in dataset.positions():
            pos.zattrs["focus_slice"] = {focus_channel: {"fov_statistics": {"z_focus_mean": per_fov_focus[fov]}}}

    def expected_window(z_focus_mean):
        z_center = round(z_focus_mean)
        z_below = int(z_extraction_window * z_focus_offset)
        z_start = max(0, z_center - z_below)
        z_end = min(z_total, z_start + z_extraction_window)
        return slice(max(0, z_end - z_extraction_window), z_end)

    dm = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_extraction_window=z_extraction_window,
        z_focus_offset=z_focus_offset,
        focus_channel=focus_channel,
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=4,
        return_negative=True,
    )
    assert dm.z_range is None  # explicit z_range not given; resolved per-FOV at setup
    dm.setup(stage="fit")
    resolved = dm.train_dataset.z_range
    assert isinstance(resolved, dict)
    # Every resolved FOV window matches its own focus plane and has uniform width.
    for fov, z_slice in resolved.items():
        assert z_slice == expected_window(per_fov_focus[fov.strip("/")]), f"FOV {fov} window mismatch"
        assert z_slice.stop - z_slice.start == z_extraction_window
    for batch in dm.train_dataloader():
        dm.on_after_batch_transfer(batch, 0)
        assert batch["anchor"].shape[2] == z_extraction_window
        break


@mark.parametrize("z_reduction", ["mip", "center"])
def test_datamodule_z_reduction(preprocessed_hcs_dataset, tracks_hcs_dataset, z_reduction):
    """z_reduction collapses the z_range window to a single slice per channel.

    Label-free channels (Phase, Retardance) take the center slice; all other
    channels (GFP, DAPI) are max-projected, regardless of ``z_reduction``,
    which only sets the fallback strategy.
    """
    z_range = (4, 9)
    yx_patch_size = [32, 32]
    batch_size = 4
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    dm = TripletDataModule(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=z_range,
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=batch_size,
        return_negative=True,
        z_reduction=z_reduction,
    )
    dm.setup(stage="fit")
    labelfree = {ch for ch in channel_names if parse_channel_name(ch)["channel_type"] == "labelfree"}
    assert labelfree, "fixture must contain at least one label-free channel"
    assert set(channel_names) - labelfree, "fixture must contain at least one non-label-free channel"
    z_window_size = z_range[1] - z_range[0]
    center = z_window_size // 2
    for batch in dm.train_dataloader():
        # Snapshot the raw extracted patch before transforms reduce it.
        raw = batch["anchor"].clone()
        dm.on_after_batch_transfer(batch, 0)
        reduced = batch["anchor"]
        assert reduced.shape == (batch_size, len(channel_names), 1, *yx_patch_size)
        for ci, ch in enumerate(channel_names):
            mip = raw[:, ci].amax(dim=1, keepdim=True)
            center_slice = raw[:, ci, center : center + 1]
            if ch in labelfree:
                assert torch.equal(reduced[:, ci], center_slice), f"label-free channel {ch} should be center-sliced"
                # Random Z-stack: center slice must differ from MIP, so a strategy
                # swap (center vs mip) would be caught rather than passing silently.
                assert not torch.equal(reduced[:, ci], mip), f"label-free channel {ch} was max-projected, not centered"
            else:
                assert torch.equal(reduced[:, ci], mip), f"non-label-free channel {ch} should be max-projected"
                assert not torch.equal(reduced[:, ci], center_slice), (
                    f"non-label-free channel {ch} was centered, not MIP"
                )


def test_z_reduction_runs_on_normalized_stack(preprocessed_hcs_dataset, tracks_hcs_dataset):
    """Z-reduction must run after normalization (production order: normalize -> reduce).

    The fixture's ``dataset_statistics`` normalization is a fixed monotone-increasing
    affine ``(x - 0.5) / (1/sqrt(12))``, which commutes with both center-slice and
    MIP. So the datamodule output (normalize-then-reduce) must equal the same affine
    applied to the reduced raw stack. A bug that reduced *before* normalizing, or
    skipped normalization, would change the values and fail this check.
    """
    import numpy as np

    from viscy_transforms import NormalizeSampled

    z_range = (4, 9)
    batch_size = 4
    mean, std = 0.5, 1 / np.sqrt(12)  # matches preprocessed_hcs_dataset fixture
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    normalizations = [
        NormalizeSampled(keys=list(channel_names), level="dataset_statistics", subtrahend="mean", divisor="std")
    ]
    dm = TripletDataModule(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=z_range,
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=batch_size,
        return_negative=True,
        normalizations=normalizations,
        z_reduction="mip",
    )
    dm.setup(stage="fit")
    labelfree = {ch for ch in channel_names if parse_channel_name(ch)["channel_type"] == "labelfree"}
    center = (z_range[1] - z_range[0]) // 2
    for batch in dm.train_dataloader():
        raw = batch["anchor"].clone()
        dm.on_after_batch_transfer(batch, 0)
        reduced = batch["anchor"]
        for ci, ch in enumerate(channel_names):
            if ch in labelfree:
                reduced_raw = raw[:, ci, center : center + 1]
            else:
                reduced_raw = raw[:, ci].amax(dim=1, keepdim=True)
            expected = (reduced_raw - mean) / (std + 1e-8)
            assert torch.allclose(reduced[:, ci], expected, atol=1e-5), f"channel {ch} not reduced on normalized stack"


def test_timepoint_statistics_resolved_in_triplet_dataset(
    tmp_path_factory, preprocessed_hcs_dataset, tracks_hcs_dataset
):
    """Triplet dataset must pre-resolve timepoint_statistics to each sample's t.

    ``NormalizeSampled(level='timepoint_statistics')`` expects a flat
    ``{stat: Tensor}`` dict, but zattrs store ``{tp_idx: {stat: Tensor}}``. The
    dataset must select the sample's timepoint before the transform runs; without
    that resolution the returned norm_meta stays keyed by timepoint index and
    ``NormalizeSampled`` raises ``KeyError('mean')``. The tracks fixture places
    anchors at t=0 and t=1, given distinct means so each sample must resolve to
    its own timepoint.
    """
    import shutil

    tp_stats = {"0": {"mean": 0.0, "std": 1.0}, "1": {"mean": 0.9, "std": 1.0}}
    data_path = tmp_path_factory.mktemp("timepoint_norm") / "data.zarr"
    shutil.copytree(preprocessed_hcs_dataset, data_path)
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
    norm_meta = {ch: {"timepoint_statistics": tp_stats} for ch in channel_names}
    with open_ome_zarr(data_path, mode="r+") as dataset:
        dataset.zattrs["normalization"] = norm_meta
        for _, fov in dataset.positions():
            fov.zattrs["normalization"] = norm_meta

    dm = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=(4, 9),
        initial_yx_patch_size=(32, 32),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=4,
        return_negative=False,
    )
    dm.setup(stage="fit")
    dataset = dm.train_dataset
    checked = 0
    for i in range(len(dataset)):
        sample = dataset.__getitems__([i])
        t = int(dataset.valid_anchors.iloc[i]["t"])
        for ch in channel_names:
            level = sample["anchor_norm_meta"][0][ch]["timepoint_statistics"]
            # Resolved: keyed by stat name, not by timepoint index.
            assert "mean" in level, "timepoint_statistics not resolved to a flat {stat: value} dict"
            assert abs(float(level["mean"]) - tp_stats[str(t)]["mean"]) < 1e-5, (
                f"sample {i} (t={t}) resolved to the wrong timepoint mean"
            )
        checked += 1
    assert checked > 0


@mark.parametrize("reference_pixel_size", [1.08, 1.3186231])
def test_reference_pixel_size_rescale_output_shape(preprocessed_hcs_dataset, tracks_hcs_dataset, reference_pixel_size):
    """reference_pixel_size rescale must land exactly on final_yx_patch_size.

    The extraction size is kept even for centered crops, while interpolation
    receives ``final_yx_patch_size`` explicitly to avoid scale-factor flooring.
    1.3186 mirrors the SEC61B_DENV run (0.1494 / 0.1133).
    """
    z_range = (4, 9)
    final_yx = (32, 32)
    batch_size = 4
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
    dm = TripletDataModule(
        data_path=preprocessed_hcs_dataset,
        tracks_path=tracks_hcs_dataset,
        source_channel=channel_names,
        z_range=z_range,
        final_yx_patch_size=final_yx,
        reference_pixel_size=reference_pixel_size,
        z_reduction="mip",
        num_workers=0,
        batch_size=batch_size,
        return_negative=True,
    )
    dm.setup(stage="fit")
    # Extraction size is rounded to an even number so the centered window is exact.
    assert all(s % 2 == 0 for s in dm.initial_yx_patch_size)
    for batch in dm.train_dataloader():
        dm.on_after_batch_transfer(batch, 0)
        # z_reduction collapses Z to 1; the rescale must hit final_yx exactly.
        assert batch["anchor"].shape == (batch_size, len(channel_names), 1, *final_yx)


def test_filter_anchors_time_interval_any(preprocessed_hcs_dataset, tracks_with_gaps_dataset):
    """Test that time_interval='any' returns all tracks unchanged."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    # Create dataset with time_interval="any"
    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))).astype(int)
        tracks_tables.append(tracks_df)

    total_tracks = sum(len(df) for df in tracks_tables)

    ds = TripletDataset(
        positions=[pos for _, pos in positions],
        tracks_tables=tracks_tables,
        channel_names=channel_names,
        initial_yx_patch_size=(64, 64),
        z_range=slice(4, 9),
        fit=True,
        time_interval="any",
    )

    # Should return all tracks
    assert len(ds.valid_anchors) == total_tracks


def test_filter_anchors_time_interval_1(preprocessed_hcs_dataset, tracks_with_gaps_dataset):
    """Test filtering with time_interval=1."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))).astype(int)
        tracks_tables.append(tracks_df)

    ds = TripletDataset(
        positions=[pos for _, pos in positions],
        tracks_tables=tracks_tables,
        channel_names=channel_names,
        initial_yx_patch_size=(64, 64),
        z_range=slice(4, 9),
        fit=True,
        time_interval=1,
    )

    # Check expected anchors per FOV/track
    valid_anchors = ds.valid_anchors

    # FOV A/1/0, Track 0: t=[0,1,2,3] -> valid anchors at t=[0,1,2]
    fov_a10_track0 = valid_anchors[(valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 0)]
    assert set(fov_a10_track0["t"]) == {0, 1, 2}

    # FOV A/1/0, Track 1: t=[0,1] -> valid anchor at t=[0]
    fov_a10_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 1)]
    assert set(fov_a10_track1["t"]) == {0}

    # FOV A/1/1, Track 0: t=[0,1,3] -> valid anchor at t=[0] only (t=1 has no t+1=2)
    fov_a11_track0 = valid_anchors[(valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 0)]
    assert set(fov_a11_track0["t"]) == {0}

    # FOV A/1/1, Track 1: t=[0,2,4] -> no valid anchors (gaps of 2, no consecutive t+1)
    fov_a11_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 1)]
    assert len(fov_a11_track1) == 0

    # FOV A/2/0, Track 0: t=[0] -> no valid anchors (no t+1)
    fov_a20_track0 = valid_anchors[(valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 0)]
    assert len(fov_a20_track0) == 0

    # FOV A/2/0, Track 1: t=[0,1,2] -> valid anchors at t=[0,1]
    fov_a20_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 1)]
    assert set(fov_a20_track1["t"]) == {0, 1}


def test_filter_anchors_time_interval_2(preprocessed_hcs_dataset, tracks_with_gaps_dataset):
    """Test filtering with time_interval=2."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))).astype(int)
        tracks_tables.append(tracks_df)

    ds = TripletDataset(
        positions=[pos for _, pos in positions],
        tracks_tables=tracks_tables,
        channel_names=channel_names,
        initial_yx_patch_size=(64, 64),
        z_range=slice(4, 9),
        fit=True,
        time_interval=2,
    )

    valid_anchors = ds.valid_anchors

    # FOV A/1/0, Track 0: t=[0,1,2,3] -> valid anchors at t=[0,1] (t+2 available)
    fov_a10_track0 = valid_anchors[(valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 0)]
    assert set(fov_a10_track0["t"]) == {0, 1}

    # FOV A/1/0, Track 1: t=[0,1] -> no valid anchors (no t+2)
    fov_a10_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 1)]
    assert len(fov_a10_track1) == 0

    # FOV A/1/1, Track 0: t=[0,1,3] -> valid anchor at t=[1] (t=1+2=3 exists)
    fov_a11_track0 = valid_anchors[(valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 0)]
    assert set(fov_a11_track0["t"]) == {1}

    # FOV A/1/1, Track 1: t=[0,2,4] -> valid anchors at t=[0,2]
    fov_a11_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 1)]
    assert set(fov_a11_track1["t"]) == {0, 2}

    # FOV A/2/0, Track 1: t=[0,1,2] -> valid anchor at t=[0]
    fov_a20_track1 = valid_anchors[(valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 1)]
    assert set(fov_a20_track1["t"]) == {0}


def test_filter_anchors_cross_fov_independence(preprocessed_hcs_dataset, tracks_with_gaps_dataset):
    """Test that same track_id in different FOVs are treated independently."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))).astype(int)
        tracks_tables.append(tracks_df)

    ds = TripletDataset(
        positions=[pos for _, pos in positions],
        tracks_tables=tracks_tables,
        channel_names=channel_names,
        initial_yx_patch_size=(64, 64),
        z_range=slice(4, 9),
        fit=True,
        time_interval=1,
    )

    # Check global_track_id format and uniqueness
    assert "global_track_id" in ds.tracks.columns
    global_track_ids = ds.tracks["global_track_id"].unique()

    # Verify format: should be "fov_name_track_id"
    for gid in global_track_ids:
        assert "_" in gid
        fov_part, track_id_part = gid.rsplit("_", 1)
        assert "/" in fov_part  # FOV names contain slashes like "A/1/0"

    # Track 0 exists in multiple FOVs (A/1/0, A/1/1, A/2/0) but should have different global_track_ids
    track0_global_ids = ds.tracks[ds.tracks["track_id"] == 0]["global_track_id"].unique()
    assert len(track0_global_ids) >= 3  # At least 3 different FOVs with track_id=0

    # Verify that filtering is independent per FOV
    # A/1/0 Track 0 (continuous) should have more valid anchors than A/1/1 Track 0 (with gap)
    valid_a10_track0 = ds.valid_anchors[(ds.valid_anchors["fov_name"] == "A/1/0") & (ds.valid_anchors["track_id"] == 0)]
    valid_a11_track0 = ds.valid_anchors[(ds.valid_anchors["fov_name"] == "A/1/1") & (ds.valid_anchors["track_id"] == 0)]
    # A/1/0 Track 0 has t=[0,1,2] valid (3 anchors)
    # A/1/1 Track 0 has t=[0] valid (1 anchor, gap at t=2)
    assert len(valid_a10_track0) == 3
    assert len(valid_a11_track0) == 1


def test_filter_anchors_predict_mode(preprocessed_hcs_dataset, tracks_with_gaps_dataset):
    """Test that predict mode (fit=False) returns all tracks regardless of time_interval."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))).astype(int)
        tracks_tables.append(tracks_df)

    total_tracks = sum(len(df) for df in tracks_tables)

    ds = TripletDataset(
        positions=[pos for _, pos in positions],
        tracks_tables=tracks_tables,
        channel_names=channel_names,
        initial_yx_patch_size=(64, 64),
        z_range=slice(4, 9),
        fit=False,  # Predict mode
        time_interval=1,
    )

    # Should return all tracks even with time_interval=1
    assert len(ds.valid_anchors) == total_tracks
