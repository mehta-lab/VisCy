from iohub import open_ome_zarr
from pytest import mark

from viscy.data.triplet import TripletDataModule


@mark.parametrize("include_wells", [None, ["A/1", "B/2"]])
@mark.parametrize("exclude_fovs", [None, ["0", "1"]])
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
        fovs_per_well = len(dataset["A/2"])
    if include_wells is not None:
        total_wells -= len(include_wells)
    if exclude_fovs is not None:
        fovs_per_well -= len(exclude_fovs)
    total_fovs = total_wells * fovs_per_well
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
    for batch in dm.train_dataloader():
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
