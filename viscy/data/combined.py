from enum import Enum
from typing import Literal, Sequence

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from viscy.data.distributed import ShardedDistributedSampler
from viscy.data.hcs import _collate_samples


class CombineMode(Enum):
    MIN_SIZE = "min_size"
    MAX_SIZE_CYCLE = "max_size_cycle"
    MAX_SIZE = "max_size"
    SEQUENTIAL = "sequential"


class CombinedDataModule(LightningDataModule):
    """Wrapper for combining multiple data modules.
    For supported modes, see ``lightning.pytorch.utilities.combined_loader``.

    Parameters
    ----------
    data_modules : Sequence[LightningDataModule]
        data modules to combine
    train_mode : CombineMode, optional
        mode in training stage, by default CombineMode.MAX_SIZE_CYCLE
    val_mode : CombineMode, optional
        mode in validation stage, by default CombineMode.SEQUENTIAL
    test_mode : CombineMode, optional
        mode in testing stage, by default CombineMode.SEQUENTIAL
    predict_mode : CombineMode, optional
        mode in prediction stage, by default CombineMode.SEQUENTIAL
    """

    def __init__(
        self,
        data_modules: Sequence[LightningDataModule],
        train_mode: CombineMode = CombineMode.MAX_SIZE_CYCLE,
        val_mode: CombineMode = CombineMode.SEQUENTIAL,
        test_mode: CombineMode = CombineMode.SEQUENTIAL,
        predict_mode: CombineMode = CombineMode.SEQUENTIAL,
    ):
        super().__init__()
        self.data_modules = data_modules
        self.train_mode = CombineMode(train_mode).value
        self.val_mode = CombineMode(val_mode).value
        self.test_mode = CombineMode(test_mode).value
        self.predict_mode = CombineMode(predict_mode).value
        self.prepare_data_per_node = True

    def prepare_data(self):
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        for dm in self.data_modules:
            dm.setup(stage)

    def train_dataloader(self):
        return CombinedLoader(
            [dm.train_dataloader() for dm in self.data_modules], mode=self.train_mode
        )

    def val_dataloader(self):
        return CombinedLoader(
            [dm.val_dataloader() for dm in self.data_modules], mode=self.val_mode
        )

    def test_dataloader(self):
        return CombinedLoader(
            [dm.test_dataloader() for dm in self.data_modules], mode=self.test_mode
        )

    def predict_dataloader(self):
        return CombinedLoader(
            [dm.predict_dataloader() for dm in self.data_modules],
            mode=self.predict_mode,
        )


class ConcatDataModule(LightningDataModule):
    """
    Concatenate multiple data modules.

    The concatenated data module will have the same batch size and number of workers
    as the first data module. Each element will be sampled uniformly regardless of
    their original data module.

    Parameters
    ----------
    data_modules : Sequence[LightningDataModule]
        Data modules to concatenate.
    """

    def __init__(self, data_modules: Sequence[LightningDataModule]):
        super().__init__()
        self.data_modules = data_modules
        self.num_workers = data_modules[0].num_workers
        self.batch_size = data_modules[0].batch_size
        for dm in data_modules:
            if dm.num_workers != self.num_workers:
                raise ValueError("Inconsistent number of workers")
            if dm.batch_size != self.batch_size:
                raise ValueError("Inconsistent batch size")
        self.prepare_data_per_node = True

    def prepare_data(self):
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        self.train_patches_per_stack = 0
        for dm in self.data_modules:
            dm.setup(stage)
            if patches := getattr(dm, "train_patches_per_stack", 0):
                if self.train_patches_per_stack == 0:
                    self.train_patches_per_stack = patches
                elif self.train_patches_per_stack != patches:
                    raise ValueError("Inconsistent patches per stack")
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self.train_dataset = ConcatDataset(
            [dm.train_dataset for dm in self.data_modules]
        )
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.data_modules])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.train_patches_per_stack,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=bool(self.num_workers),
            collate_fn=_collate_samples,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=bool(self.num_workers),
        )


class CachedConcatDataModule(LightningDataModule):
    def __init__(self, data_modules: Sequence[LightningDataModule]):
        super().__init__()
        self.data_modules = data_modules
        self.num_workers = data_modules[0].num_workers
        self.batch_size = data_modules[0].batch_size
        for dm in data_modules:
            if dm.num_workers != self.num_workers:
                raise ValueError("Inconsistent number of workers")
            if dm.batch_size != self.batch_size:
                raise ValueError("Inconsistent batch size")
        self.prepare_data_per_node = True

    def prepare_data(self):
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        self.train_patches_per_stack = 0
        for dm in self.data_modules:
            dm.setup(stage)
            if patches := getattr(dm, "train_patches_per_stack", 1):
                if self.train_patches_per_stack == 0:
                    self.train_patches_per_stack = patches
                elif self.train_patches_per_stack != patches:
                    raise ValueError("Inconsistent patches per stack")
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self.train_dataset = ConcatDataset(
            [dm.train_dataset for dm in self.data_modules]
        )
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.data_modules])

    def _maybe_sampler(
        self, dataset: Dataset, shuffle: bool
    ) -> ShardedDistributedSampler | None:
        return (
            ShardedDistributedSampler(dataset, shuffle=shuffle)
            if torch.distributed.is_initialized()
            else None
        )

    def train_dataloader(self) -> DataLoader:
        sampler = self._maybe_sampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self._maybe_sampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=lambda x: x,
        )
