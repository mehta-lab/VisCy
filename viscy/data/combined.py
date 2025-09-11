import bisect
import logging
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Literal

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from monai.data import ThreadDataLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from viscy.data.distributed import ShardedDistributedSampler
from viscy.data.hcs import _collate_samples

_logger = logging.getLogger("lightning.pytorch")


class CombineMode(Enum):
    """Enumeration of data combination modes for CombinedDataModule.

    Defines how multiple data modules should be combined during training,
    validation, and testing phases.
    """

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

    def prepare_data(self) -> None:
        """Prepare data for all constituent data modules.

        Propagates trainer reference and calls prepare_data on each
        data module for dataset downloading and preprocessing.
        """
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        """Set up data modules for specified training stage.

        Parameters
        ----------
        stage : Literal["fit", "validate", "test", "predict"]
            Current training stage for Lightning setup.
        """
        for dm in self.data_modules:
            dm.setup(stage)

    def train_dataloader(self) -> CombinedLoader:
        """Create combined training dataloader.

        Returns
        -------
        CombinedLoader
            Combined dataloader using specified train_mode strategy.
        """
        return CombinedLoader(
            [dm.train_dataloader() for dm in self.data_modules], mode=self.train_mode
        )

    def val_dataloader(self) -> CombinedLoader:
        """Create combined validation dataloader.

        Returns
        -------
        CombinedLoader
            Combined dataloader using specified val_mode strategy.
        """
        return CombinedLoader(
            [dm.val_dataloader() for dm in self.data_modules], mode=self.val_mode
        )

    def test_dataloader(self) -> CombinedLoader:
        """Create combined test dataloader.

        Returns
        -------
        CombinedLoader
            Combined dataloader using specified test_mode strategy.
        """
        return CombinedLoader(
            [dm.test_dataloader() for dm in self.data_modules], mode=self.test_mode
        )

    def predict_dataloader(self) -> CombinedLoader:
        """Create combined prediction dataloader.

        Returns
        -------
        CombinedLoader
            Combined dataloader using specified predict_mode strategy.
        """
        return CombinedLoader(
            [dm.predict_dataloader() for dm in self.data_modules],
            mode=self.predict_mode,
        )


class BatchedConcatDataset(ConcatDataset):
    """Batched concatenated dataset for efficient multi-dataset sampling.

    Extends PyTorch's ConcatDataset to support batched item retrieval
    from multiple datasets with optimized index grouping for ML training.
    """

    def __getitem__(self, idx: int):
        """Retrieve single item by index.

        Parameters
        ----------
        idx : int
            Sample index across concatenated datasets.

        Raises
        ------
        NotImplementedError
            Single item access not implemented; use __getitems__ instead.
        """
        raise NotImplementedError

    def _get_sample_indices(self, idx: int) -> tuple[int, int]:
        """Map global index to dataset and sample indices.

        Parameters
        ----------
        idx : int
            Global index across all concatenated datasets.

        Returns
        -------
        tuple[int, int]
            Dataset index and local sample index within that dataset.

        Raises
        ------
        ValueError
            If absolute index value exceeds dataset length.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitems__(self, indices: list[int]) -> list:
        """Retrieve multiple items by indices with batched dataset access.

        Groups indices by source dataset and performs batched retrieval
        for improved data loading performance during ML training.

        Parameters
        ----------
        indices : list[int]
            List of global indices across concatenated datasets.

        Returns
        -------
        list
            Samples from all requested indices, maintaining order.
        """
        grouped_indices = defaultdict(list)
        for idx in indices:
            dataset_idx, sample_indices = self._get_sample_indices(idx)
            grouped_indices[dataset_idx].append(sample_indices)
        _logger.debug(f"Grouped indices: {grouped_indices}")
        sub_batches = []
        for dataset_idx, sample_indices in grouped_indices.items():
            sub_batch = self.datasets[dataset_idx].__getitems__(sample_indices)
            sub_batches.extend(sub_batch)
        return sub_batches


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
        """Prepare data for all constituent data modules.

        Propagates trainer reference and calls prepare_data on each
        data module for dataset preparation and preprocessing.
        """
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up concatenated datasets for training stage.

        Validates patch configuration consistency across data modules
        and creates concatenated train/validation datasets.

        Parameters
        ----------
        stage : Literal["fit", "validate", "test", "predict"]
            Training stage - only "fit" currently supported.

        Raises
        ------
        ValueError
            If patches per stack are inconsistent across data modules.
        NotImplementedError
            If stage other than "fit" is requested.
        """
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
        self.train_dataset = self._ConcatDataset(
            [dm.train_dataset for dm in self.data_modules]
        )
        self.val_dataset = self._ConcatDataset(
            [dm.val_dataset for dm in self.data_modules]
        )

    def _dataloader_kwargs(self) -> dict:
        """Get common dataloader configuration parameters.

        Returns
        -------
        dict
            Common PyTorch DataLoader configuration parameters including
            worker settings, memory pinning, and prefetch configuration.
        """
        return {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor if self.num_workers else None,
            "pin_memory": self.pin_memory,
        }

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader for concatenated datasets.

        Returns
        -------
        DataLoader
            PyTorch DataLoader with shuffling enabled, batch size adjusted
            for patch stacking, and sample collation for ML training.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size // self.train_patches_per_stack,
            collate_fn=_collate_samples,
            drop_last=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader for concatenated datasets.

        Returns
        -------
        DataLoader
            PyTorch DataLoader without shuffling for deterministic
            validation evaluation.
        """
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            **self._dataloader_kwargs(),
        )


class BatchedConcatDataModule(ConcatDataModule):
    """Concatenated data module with batched dataset access.

    Extends ConcatDataModule to use BatchedConcatDataset and
    ThreadDataLoader for optimized multi-dataset training performance.
    """

    _ConcatDataset = BatchedConcatDataset

    def train_dataloader(self) -> ThreadDataLoader:
        """Create threaded training dataloader for batched access.

        Returns
        -------
        ThreadDataLoader
            MONAI ThreadDataLoader with thread-based workers for
            optimized batched dataset access during training.
        """
        return ThreadDataLoader(
            self.train_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> ThreadDataLoader:
        """Create threaded validation dataloader for batched access.

        Returns
        -------
        ThreadDataLoader
            MONAI ThreadDataLoader with thread-based workers for
            optimized validation data loading.
        """
        return ThreadDataLoader(
            self.val_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._dataloader_kwargs(),
        )


class CachedConcatDataModule(LightningDataModule):
    """Cached concatenated data module for distributed training.

    Concatenates multiple data modules with support for distributed
    sampling and caching optimizations for large-scale ML training.
    # TODO: MANUAL_REVIEW - Verify caching behavior and memory usage
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
        """Prepare data for all constituent data modules.

        Propagates trainer reference and calls prepare_data on each
        data module for dataset preparation and caching setup.
        """
        for dm in self.data_modules:
            dm.trainer = self.trainer
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up cached concatenated datasets for distributed training.

        Validates patch configuration and creates concatenated datasets
        with caching optimizations for efficient distributed access.

        Parameters
        ----------
        stage : Literal["fit", "validate", "test", "predict"]
            Training stage - only "fit" currently supported.

        Raises
        ------
        ValueError
            If patches per stack are inconsistent across data modules.
        NotImplementedError
            If stage other than "fit" is requested.
        """
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
        """Create distributed sampler if in distributed training mode.

        Parameters
        ----------
        dataset : Dataset
            PyTorch dataset to create sampler for.
        shuffle : bool
            Whether to shuffle samples across distributed processes.

        Returns
        -------
        ShardedDistributedSampler | None
            Distributed sampler if PyTorch distributed is initialized,
            None otherwise for single-process training.
        """
        return (
            ShardedDistributedSampler(dataset, shuffle=shuffle)
            if torch.distributed.is_initialized()
            else None
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with distributed sampling support.

        Returns
        -------
        DataLoader
            PyTorch DataLoader with distributed sampler if available,
            configured for cached dataset access during training.
        """
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
        """Create validation dataloader with distributed sampling support.

        Returns
        -------
        DataLoader
            PyTorch DataLoader with distributed sampler if available,
            configured for deterministic validation evaluation.
        """
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
