"""Combined and concatenated data modules for multi-dataset training."""

import bisect
import logging
from collections import defaultdict
from enum import Enum
from typing import Literal, Sequence

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from monai.data import ThreadDataLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from viscy_data._utils import _collate_samples
from viscy_data.distributed import ShardedDistributedSampler

_logger = logging.getLogger("lightning.pytorch")


class CombineMode(Enum):
    """Mode for combining multiple data modules."""

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
        """Prepare data for all constituent data modules."""
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up all constituent data modules."""
        for dm in self.data_modules:
            dm.setup(stage)

    def train_dataloader(self):
        """Return combined training data loader."""
        return CombinedLoader([dm.train_dataloader() for dm in self.data_modules], mode=self.train_mode)

    def val_dataloader(self):
        """Return combined validation data loader."""
        return CombinedLoader([dm.val_dataloader() for dm in self.data_modules], mode=self.val_mode)

    def test_dataloader(self):
        """Return combined test data loader."""
        return CombinedLoader([dm.test_dataloader() for dm in self.data_modules], mode=self.test_mode)

    def predict_dataloader(self):
        """Return combined predict data loader."""
        return CombinedLoader(
            [dm.predict_dataloader() for dm in self.data_modules],
            mode=self.predict_mode,
        )


class BatchedConcatDataset(ConcatDataset):
    """Concatenated dataset with batched access by constituent dataset."""

    def __getitem__(self, idx):
        """Not implemented; use __getitems__ for batched access."""
        raise NotImplementedError

    def _get_sample_indices(self, idx: int) -> tuple[int, int]:
        """Map a global index to (dataset_idx, sample_idx)."""
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        """Return micro-batches grouped by constituent dataset."""
        grouped_indices = defaultdict(list)
        for idx in indices:
            dataset_idx, sample_indices = self._get_sample_indices(idx)
            grouped_indices[dataset_idx].append(sample_indices)
        _logger.debug(f"Grouped indices: {grouped_indices}")

        micro_batches = []
        for dataset_idx, sample_indices in grouped_indices.items():
            micro_batch = self.datasets[dataset_idx].__getitems__(sample_indices)
            micro_batch["_dataset_idx"] = dataset_idx
            micro_batches.append(micro_batch)

        return micro_batches


class ConcatDataModule(LightningDataModule):
    """Concatenate multiple data modules.

    The concatenated data module will have the same batch size and number of workers
    as the first data module. Each element will be sampled uniformly regardless of
    their original data module.

    Parameters
    ----------
    data_modules : Sequence[LightningDataModule]
        Data modules to concatenate.
    """

    _ConcatDataset = ConcatDataset

    def __init__(self, data_modules: Sequence[LightningDataModule]):
        super().__init__()
        self.data_modules = data_modules
        self.num_workers = data_modules[0].num_workers
        self.batch_size = data_modules[0].batch_size
        self.persistent_workers = data_modules[0].persistent_workers
        self.prefetch_factor = data_modules[0].prefetch_factor
        self.pin_memory = data_modules[0].pin_memory
        for dm in data_modules:
            if dm.num_workers != self.num_workers:
                raise ValueError("Inconsistent number of workers")
            if dm.batch_size != self.batch_size:
                raise ValueError("Inconsistent batch size")
        self.prepare_data_per_node = True

    def prepare_data(self):
        """Prepare data for all constituent data modules."""
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up constituent data modules and create concatenated datasets."""
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self.train_patches_per_stack = 0
        for dm in self.data_modules:
            dm.setup(stage)
            if patches := getattr(dm, "train_patches_per_stack", 0):
                if self.train_patches_per_stack == 0:
                    self.train_patches_per_stack = patches
                elif self.train_patches_per_stack != patches:
                    raise ValueError("Inconsistent patches per stack")
        self.train_dataset = self._ConcatDataset([dm.train_dataset for dm in self.data_modules])
        self.val_dataset = self._ConcatDataset([dm.val_dataset for dm in self.data_modules])

    def _dataloader_kwargs(self) -> dict:
        """Return shared dataloader keyword arguments."""
        return {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor if self.num_workers else None,
            "pin_memory": self.pin_memory,
        }

    def train_dataloader(self):
        """Return concatenated training data loader."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size // self.train_patches_per_stack,
            collate_fn=_collate_samples,
            drop_last=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        """Return concatenated validation data loader."""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            **self._dataloader_kwargs(),
        )


class BatchedConcatDataModule(ConcatDataModule):
    """Concatenated data module with batched micro-batch GPU transforms."""

    _ConcatDataset = BatchedConcatDataset

    def train_dataloader(self):
        """Return batched concatenated training data loader."""
        return ThreadDataLoader(
            self.train_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        """Return batched concatenated validation data loader."""
        return ThreadDataLoader(
            self.val_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: x,
            **self._dataloader_kwargs(),
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Apply GPU transforms from constituent data modules to micro-batches."""
        if not isinstance(batch, list):
            return batch

        processed_micro_batches = []
        for micro_batch in batch:
            if isinstance(micro_batch, dict) and "_dataset_idx" in micro_batch:
                dataset_idx = micro_batch.pop("_dataset_idx")
                dm = self.data_modules[dataset_idx]
                if hasattr(dm, "on_after_batch_transfer"):
                    processed_micro_batch = dm.on_after_batch_transfer(micro_batch, dataloader_idx)
                else:
                    processed_micro_batch = micro_batch
            else:
                # Handle case where micro_batch doesn't have _dataset_idx
                # (e.g., from model summary)
                processed_micro_batch = micro_batch
            processed_micro_batches.append(processed_micro_batch)
        combined_batch = {}
        for key in processed_micro_batches[0].keys():
            if isinstance(processed_micro_batches[0][key], list):
                combined_batch[key] = []
                for micro_batch in processed_micro_batches:
                    if key in micro_batch:
                        combined_batch[key].extend(micro_batch[key])
            else:
                tensors_to_concat = [micro_batch[key] for micro_batch in processed_micro_batches if key in micro_batch]
                if tensors_to_concat:
                    combined_batch[key] = torch.cat(tensors_to_concat, dim=0)

        return combined_batch


class CachedConcatDataModule(LightningDataModule):
    """Concatenated data module with distributed sampling support.

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
        """Prepare data for all constituent data modules."""
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up constituent data modules and create concatenated datasets."""
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self.train_patches_per_stack = 0
        for dm in self.data_modules:
            dm.setup(stage)
            if patches := getattr(dm, "train_patches_per_stack", 1):
                if self.train_patches_per_stack == 0:
                    self.train_patches_per_stack = patches
                elif self.train_patches_per_stack != patches:
                    raise ValueError("Inconsistent patches per stack")
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.data_modules])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.data_modules])

    def _maybe_sampler(self, dataset: Dataset, shuffle: bool) -> ShardedDistributedSampler | None:
        """Return a distributed sampler if DDP is initialized, else None."""
        return ShardedDistributedSampler(dataset, shuffle=shuffle) if torch.distributed.is_initialized() else None

    def train_dataloader(self) -> DataLoader:
        """Return concatenated training data loader with optional DDP sampling."""
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
        """Return concatenated validation data loader with optional DDP sampling."""
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
