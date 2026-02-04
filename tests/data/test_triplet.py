import pandas as pd
from iohub import open_ome_zarr
from pytest import mark

from viscy.data.triplet import TripletDataModule, TripletDataset


@mark.parametrize("include_wells", [None, ["A/1", "A/2", "B/1"]])
@mark.parametrize("exclude_fovs", [None, ["A/1/0", "A/1/1", "A/2/2", "B/1/3"]])
def test_datamodule_setup_fit(
    preprocessed_hcs_dataset, tracks_hcs_dataset, include_wells, exclude_fovs
):
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
        initial_yx_patch_size=(64, 64),
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
def test_datamodule_z_window_size(
    preprocessed_hcs_dataset, tracks_hcs_dataset, z_window_size
):
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
        initial_yx_patch_size=(64, 64),
        final_yx_patch_size=(32, 32),
        num_workers=0,
        batch_size=batch_size,
        return_negative=True,
        z_window_size=z_window_size,
    )
    dm.setup(stage="fit")
    if z_window_size is None:
        expected_z_shape = z_range[1] - z_range[0]
    else:
        expected_z_shape = z_window_size
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


def test_filter_anchors_time_interval_any(
    preprocessed_hcs_dataset, tracks_with_gaps_dataset
):
    """Test that time_interval='any' returns all tracks unchanged."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    # Create dataset with time_interval="any"
    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(
            next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))
        ).astype(int)
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


def test_filter_anchors_time_interval_1(
    preprocessed_hcs_dataset, tracks_with_gaps_dataset
):
    """Test filtering with time_interval=1."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(
            next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))
        ).astype(int)
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
    fov_a10_track0 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 0)
    ]
    assert set(fov_a10_track0["t"]) == {0, 1, 2}

    # FOV A/1/0, Track 1: t=[0,1] -> valid anchor at t=[0]
    fov_a10_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 1)
    ]
    assert set(fov_a10_track1["t"]) == {0}

    # FOV A/1/1, Track 0: t=[0,1,3] -> valid anchor at t=[0] only (t=1 has no t+1=2)
    fov_a11_track0 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 0)
    ]
    assert set(fov_a11_track0["t"]) == {0}

    # FOV A/1/1, Track 1: t=[0,2,4] -> no valid anchors (gaps of 2, no consecutive t+1)
    fov_a11_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 1)
    ]
    assert len(fov_a11_track1) == 0

    # FOV A/2/0, Track 0: t=[0] -> no valid anchors (no t+1)
    fov_a20_track0 = valid_anchors[
        (valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 0)
    ]
    assert len(fov_a20_track0) == 0

    # FOV A/2/0, Track 1: t=[0,1,2] -> valid anchors at t=[0,1]
    fov_a20_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 1)
    ]
    assert set(fov_a20_track1["t"]) == {0, 1}


def test_filter_anchors_time_interval_2(
    preprocessed_hcs_dataset, tracks_with_gaps_dataset
):
    """Test filtering with time_interval=2."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(
            next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))
        ).astype(int)
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
    fov_a10_track0 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 0)
    ]
    assert set(fov_a10_track0["t"]) == {0, 1}

    # FOV A/1/0, Track 1: t=[0,1] -> no valid anchors (no t+2)
    fov_a10_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/0") & (valid_anchors["track_id"] == 1)
    ]
    assert len(fov_a10_track1) == 0

    # FOV A/1/1, Track 0: t=[0,1,3] -> valid anchor at t=[1] (t=1+2=3 exists)
    fov_a11_track0 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 0)
    ]
    assert set(fov_a11_track0["t"]) == {1}

    # FOV A/1/1, Track 1: t=[0,2,4] -> valid anchors at t=[0,2]
    fov_a11_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/1/1") & (valid_anchors["track_id"] == 1)
    ]
    assert set(fov_a11_track1["t"]) == {0, 2}

    # FOV A/2/0, Track 1: t=[0,1,2] -> valid anchor at t=[0]
    fov_a20_track1 = valid_anchors[
        (valid_anchors["fov_name"] == "A/2/0") & (valid_anchors["track_id"] == 1)
    ]
    assert set(fov_a20_track1["t"]) == {0}


def test_filter_anchors_cross_fov_independence(
    preprocessed_hcs_dataset, tracks_with_gaps_dataset
):
    """Test that same track_id in different FOVs are treated independently."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(
            next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))
        ).astype(int)
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
    track0_global_ids = ds.tracks[ds.tracks["track_id"] == 0][
        "global_track_id"
    ].unique()
    assert len(track0_global_ids) >= 3  # At least 3 different FOVs with track_id=0

    # Verify that filtering is independent per FOV
    # A/1/0 Track 0 (continuous) should have more valid anchors than A/1/1 Track 0 (with gap)
    valid_a10_track0 = ds.valid_anchors[
        (ds.valid_anchors["fov_name"] == "A/1/0") & (ds.valid_anchors["track_id"] == 0)
    ]
    valid_a11_track0 = ds.valid_anchors[
        (ds.valid_anchors["fov_name"] == "A/1/1") & (ds.valid_anchors["track_id"] == 0)
    ]
    # A/1/0 Track 0 has t=[0,1,2] valid (3 anchors)
    # A/1/1 Track 0 has t=[0] valid (1 anchor, gap at t=2)
    assert len(valid_a10_track0) == 3
    assert len(valid_a11_track0) == 1


def test_filter_anchors_predict_mode(
    preprocessed_hcs_dataset, tracks_with_gaps_dataset
):
    """Test that predict mode (fit=False) returns all tracks regardless of time_interval."""
    with open_ome_zarr(preprocessed_hcs_dataset) as dataset:
        channel_names = dataset.channel_names
        positions = list(dataset.positions())

    tracks_tables = []
    for fov_name, _ in positions:
        tracks_df = pd.read_csv(
            next((tracks_with_gaps_dataset / fov_name).glob("*.csv"))
        ).astype(int)
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
