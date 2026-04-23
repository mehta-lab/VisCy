"""Tests for CombinedDataModule and ConcatDataModule."""

import pytest
from iohub import open_ome_zarr

from viscy_data import (
    BatchedConcatDataModule,
    CombinedDataModule,
    CombineMode,
    ConcatDataModule,
    HCSDataModule,
    ShardedDistributedSampler,
)


def _fake_ddp(monkeypatch, world_size: int = 2, rank: int = 0) -> None:
    """Monkeypatch ``torch.distributed`` entry points to simulate DDP.

    ``DistributedSampler.__init__`` reads ``is_available``,
    ``is_initialized``, ``get_world_size``, and ``get_rank``. Faking these
    is enough to construct the sampler without a real process group.
    """
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)


def _make_dm(data_path, batch_size=4, num_workers=0):
    with open_ome_zarr(data_path) as dataset:
        ch = dataset.channel_names
    return HCSDataModule(
        data_path=data_path,
        source_channel=ch[:2],
        target_channel=ch[2:],
        z_window_size=5,
        batch_size=batch_size,
        num_workers=num_workers,
        split_ratio=0.5,
        yx_patch_size=(128, 96),
    )


def test_combined_datamodule_fit(preprocessed_hcs_dataset):
    """CombinedDataModule sets up and iterates in fit stage."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    combined = CombinedDataModule(
        data_modules=[dm1, dm2],
        train_mode=CombineMode.MAX_SIZE_CYCLE,
    )
    combined.setup(stage="fit")
    train_dl = combined.train_dataloader()
    for batch_data in train_dl:
        # CombinedLoader with max_size_cycle returns (batch_list, batch_idx, dataloader_idx)
        batch_list = batch_data[0]
        assert isinstance(batch_list, (list, tuple))
        assert len(batch_list) == 2
        break

    val_dl = combined.val_dataloader()
    for batch_data in val_dl:
        # CombinedLoader with sequential mode returns (batch, batch_idx, dataloader_idx)
        batch = batch_data[0]
        assert isinstance(batch, dict)
        assert "source" in batch
        break


def test_combined_datamodule_combine_modes():
    """CombineMode enum maps to valid string values."""
    assert CombineMode.MIN_SIZE.value == "min_size"
    assert CombineMode.MAX_SIZE_CYCLE.value == "max_size_cycle"
    assert CombineMode.MAX_SIZE.value == "max_size"
    assert CombineMode.SEQUENTIAL.value == "sequential"


def test_concat_datamodule_fit(preprocessed_hcs_dataset):
    """ConcatDataModule concatenates datasets and produces correct batch shapes."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    concat = ConcatDataModule(data_modules=[dm1, dm2])
    concat.setup(stage="fit")

    # Concatenated dataset should have combined length
    assert len(concat.train_dataset) == len(dm1.train_dataset) + len(dm2.train_dataset)
    assert len(concat.val_dataset) == len(dm1.val_dataset) + len(dm2.val_dataset)

    for batch in concat.train_dataloader():
        assert "source" in batch
        assert batch["source"].shape[1] == 2  # 2 source channels
        break

    for batch in concat.val_dataloader():
        assert "source" in batch
        break


def test_concat_datamodule_inconsistent_batch_size(preprocessed_hcs_dataset):
    """ConcatDataModule raises on inconsistent batch sizes."""
    dm1 = _make_dm(preprocessed_hcs_dataset, batch_size=4)
    dm2 = _make_dm(preprocessed_hcs_dataset, batch_size=8)
    with pytest.raises(ValueError, match="Inconsistent batch size"):
        ConcatDataModule(data_modules=[dm1, dm2])


def test_concat_datamodule_inconsistent_num_workers(preprocessed_hcs_dataset):
    """ConcatDataModule raises on inconsistent num_workers."""
    dm1 = _make_dm(preprocessed_hcs_dataset, num_workers=0)
    dm2 = _make_dm(preprocessed_hcs_dataset, num_workers=2)
    with pytest.raises(ValueError, match="Inconsistent number of workers"):
        ConcatDataModule(data_modules=[dm1, dm2])


def test_concat_datamodule_only_fit_supported(preprocessed_hcs_dataset):
    """ConcatDataModule raises NotImplementedError for non-fit stages."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    concat = ConcatDataModule(data_modules=[dm1, dm2])
    with pytest.raises(NotImplementedError):
        concat.setup(stage="predict")


def test_batched_concat_datamodule_with_hcs_children(preprocessed_hcs_dataset):
    """BatchedConcatDataModule iterates HCS children via the __getitem__ fallback.

    SlidingWindowDataset only defines ``__getitem__`` (per-sample retry
    logic). Prior to the shim in ``BatchedConcatDataset._batched_get``,
    this combination raised ``AttributeError: __getitems__`` at first
    iteration.
    """
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    batched = BatchedConcatDataModule(data_modules=[dm1, dm2])
    batched.setup(stage="fit")

    loader = batched.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, list)
    assert len(batch) >= 1
    for micro_batch in batch:
        assert isinstance(micro_batch, dict)
        assert "_dataset_idx" in micro_batch
        assert "source" in micro_batch
        assert micro_batch["source"].ndim == 5  # (B, C, Z, Y, X)
    # Grouping must be lossless: total batch dim equals _make_dm batch_size=4.
    assert sum(mb["source"].shape[0] for mb in batch) == 4


def test_batched_concat_ddp_attaches_sharded_sampler(preprocessed_hcs_dataset, monkeypatch):
    """Under DDP, train/val dataloaders attach ShardedDistributedSampler."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    batched = BatchedConcatDataModule(data_modules=[dm1, dm2])
    batched.setup(stage="fit")

    _fake_ddp(monkeypatch, world_size=2, rank=0)

    train_loader = batched.train_dataloader()
    assert isinstance(train_loader.sampler, ShardedDistributedSampler)
    assert train_loader.sampler.shuffle is True

    val_loader = batched.val_dataloader()
    assert isinstance(val_loader.sampler, ShardedDistributedSampler)
    assert val_loader.sampler.shuffle is False


def test_batched_concat_ddp_batch_contract_preserved(preprocessed_hcs_dataset, monkeypatch):
    """Under DDP, train and val loaders still yield the micro-batch contract."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    batched = BatchedConcatDataModule(data_modules=[dm1, dm2])
    batched.setup(stage="fit")

    _fake_ddp(monkeypatch, world_size=2, rank=0)

    for loader in (batched.train_dataloader(), batched.val_dataloader()):
        batch = next(iter(loader))
        assert isinstance(batch, list)
        assert len(batch) >= 1
        for micro_batch in batch:
            assert isinstance(micro_batch, dict)
            assert "_dataset_idx" in micro_batch
            assert "source" in micro_batch
            assert micro_batch["source"].ndim == 5
        assert sum(mb["source"].shape[0] for mb in batch) == 4  # _make_dm batch_size=4


def test_batched_concat_ddp_rank_disjointness(preprocessed_hcs_dataset, monkeypatch):
    """Rank 0 and rank 1 train samplers yield disjoint index sets."""
    dm1 = _make_dm(preprocessed_hcs_dataset)
    dm2 = _make_dm(preprocessed_hcs_dataset)
    batched = BatchedConcatDataModule(data_modules=[dm1, dm2])
    batched.setup(stage="fit")

    _fake_ddp(monkeypatch, world_size=2, rank=0)
    rank0_indices = set(iter(batched.train_dataloader().sampler))

    _fake_ddp(monkeypatch, world_size=2, rank=1)
    rank1_indices = set(iter(batched.train_dataloader().sampler))

    assert rank0_indices.isdisjoint(rank1_indices)
    # Together ranks cover the full concatenated dataset (no gaps).
    assert rank0_indices | rank1_indices == set(range(len(batched.train_dataset)))
