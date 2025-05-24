from iohub import open_ome_zarr
from pytest import mark

from viscy.data.triplet import TripletDataModule


@mark.parametrize("include_wells", [None, ["A/1", "B/2"]])
def test_datamodule_setup_fit(
    preprocessed_hcs_dataset, tracks_hcs_dataset, include_wells
):
    data_path = preprocessed_hcs_dataset
    z_window_size = 5
    split_ratio = 0.75
    yx_patch_size = [32, 32]
    batch_size = 4
    with open_ome_zarr(data_path) as dataset:
        channel_names = dataset.channel_names
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
        return_negative=True,
    )
    dm.setup(stage="fit")
    if include_wells is not None:
        assert len(dm.train_dataset) == 6
        assert len(dm.val_dataset) == 2
    else:
        assert len(dm.train_dataset) == 12
        assert len(dm.val_dataset) == 4
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
