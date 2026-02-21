import pytest
from iohub import open_ome_zarr

from viscy_data import CTMCv1DataModule


def _make_ctmc(train_path, val_path, **kwargs):
    defaults = dict(
        train_cpu_transforms=[],
        val_cpu_transforms=[],
        train_gpu_transforms=[],
        val_gpu_transforms=[],
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    defaults.update(kwargs)
    return CTMCv1DataModule(
        train_data_path=train_path,
        val_data_path=val_path,
        **defaults,
    )


def test_ctmc_v1_setup_fit(single_channel_hcs_pair):
    train_path, val_path = single_channel_hcs_pair
    dm = _make_ctmc(train_path, val_path)
    dm.setup("fit")
    with open_ome_zarr(train_path) as plate:
        train_positions = list(plate.positions())
        expected_train = sum(p["0"].frames for _, p in train_positions)
    with open_ome_zarr(val_path) as plate:
        val_positions = list(plate.positions())
        expected_val_full = sum(p["0"].frames for _, p in val_positions)
    assert len(dm.train_dataset) == expected_train
    # default val_subsample_ratio=30, with 2 timepoints * 16 positions = 32 total
    # subsample indices: range(0, 32, 30) -> [0, 30] -> 2 samples
    expected_val = len(range(0, expected_val_full, 30))
    assert len(dm.val_dataset) == expected_val


def test_ctmc_v1_val_subsample(single_channel_hcs_pair):
    train_path, val_path = single_channel_hcs_pair
    dm = _make_ctmc(train_path, val_path, val_subsample_ratio=2)
    dm.setup("fit")
    with open_ome_zarr(val_path) as plate:
        val_positions = list(plate.positions())
        full_len = sum(p["0"].frames for _, p in val_positions)
    expected = len(range(0, full_len, 2))
    assert len(dm.val_dataset) == expected


def test_ctmc_v1_train_dataloader_batch(single_channel_hcs_pair):
    train_path, val_path = single_channel_hcs_pair
    dm = _make_ctmc(train_path, val_path, batch_size=2)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, dict)
    assert "DIC" in batch
    assert batch["DIC"].ndim == 5  # B, C, Z, Y, X
    assert batch["DIC"].shape[0] <= 2  # batch_size


def test_ctmc_v1_unsupported_stage(single_channel_hcs_pair):
    train_path, val_path = single_channel_hcs_pair
    dm = _make_ctmc(train_path, val_path)
    with pytest.raises(NotImplementedError):
        dm.setup("test")
