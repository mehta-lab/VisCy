"""Lightning DataModule for multi-experiment DynaCLR training.

Composes :class:`~dynaclr.index.MultiExperimentIndex`,
:class:`~dynaclr.dataset.MultiExperimentTripletDataset`,
:class:`~viscy_data.sampler.FlexibleBatchSampler`,
:class:`~viscy_data.channel_dropout.ChannelDropout`, and
:class:`~monai.data.thread_buffer.ThreadDataLoader` into a fully
configurable training pipeline with experiment-level train/val split.
"""

from __future__ import annotations

import logging

from lightning.pytorch import LightningDataModule
from monai.data.thread_buffer import ThreadDataLoader
from monai.transforms import Compose, MapTransform
from torch import Tensor

from viscy_data._utils import BatchedCenterSpatialCropd, _transform_channel_wise
from viscy_data.channel_dropout import ChannelDropout
from viscy_data.sampler import FlexibleBatchSampler

from dynaclr.dataset import MultiExperimentTripletDataset
from dynaclr.experiment import ExperimentRegistry
from dynaclr.index import MultiExperimentIndex

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentDataModule"]


class MultiExperimentDataModule(LightningDataModule):
    """Lightning DataModule for multi-experiment DynaCLR training.

    Composes MultiExperimentIndex, MultiExperimentTripletDataset,
    FlexibleBatchSampler, ChannelDropout, and ThreadDataLoader into
    a fully configurable training pipeline.

    Parameters
    ----------
    experiments_yaml : str
        Path to YAML config for ExperimentRegistry.from_yaml().
    z_range : tuple[int, int]
        Z-slice range (start, stop) for data loading.
    yx_patch_size : tuple[int, int]
        Initial YX patch size for cell patch extraction.
    final_yx_patch_size : tuple[int, int]
        Final YX patch size after cropping (output size).
    val_experiments : list[str]
        Experiment names to use for validation (rest are training).
    tau_range : tuple[float, float]
        (min_hours, max_hours) for temporal positive sampling.
    tau_decay_rate : float
        Exponential decay rate for tau sampling. Default: 2.0.
    batch_size : int
        Batch size. Default: 128.
    num_workers : int
        Thread workers for ThreadDataLoader. Default: 1.
    experiment_aware : bool
        Restrict each batch to a single experiment. Default: True.
    condition_balanced : bool
        Balance conditions within each batch. Default: True.
    leaky : float
        Fraction of cross-experiment samples. Default: 0.0.
    temporal_enrichment : bool
        Concentrate around focal HPI. Default: False.
    temporal_window_hours : float
        Half-width of focal window. Default: 2.0.
    temporal_global_fraction : float
        Global fraction for temporal enrichment. Default: 0.3.
    experiment_weights : dict[str, float] | None
        Per-experiment sampling weights. Default: None (proportional).
    condition_ratio : dict[str, float] | None
        Per-condition target ratio. Default: None (equal).
    channel_dropout_channels : list[int]
        Channel indices to dropout. Default: [1] (fluorescence).
    channel_dropout_prob : float
        Dropout probability. Default: 0.5.
    normalizations : list[MapTransform]
        Normalization transforms. Default: [].
    augmentations : list[MapTransform]
        Augmentation transforms. Default: [].
    hcl_beta : float
        Hard-negative concentration beta. Default: 0.5.
        NOTE: Stored for YAML discoverability but the actual
        NTXentHCL instance is configured on ContrastiveModule, not here.
    cache_pool_bytes : int
        Tensorstore cache pool size. Default: 0.
    seed : int
        RNG seed for FlexibleBatchSampler. Default: 0.
    include_wells : list[str] | None
        Only include these wells. Default: None.
    exclude_fovs : list[str] | None
        Exclude these FOVs. Default: None.
    """

    def __init__(
        self,
        experiments_yaml: str,
        z_range: tuple[int, int],
        yx_patch_size: tuple[int, int],
        final_yx_patch_size: tuple[int, int],
        val_experiments: list[str],
        tau_range: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        batch_size: int = 128,
        num_workers: int = 1,
        # Sampling hyperparameters (passed to FlexibleBatchSampler)
        experiment_aware: bool = True,
        condition_balanced: bool = True,
        leaky: float = 0.0,
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        experiment_weights: dict[str, float] | None = None,
        condition_ratio: dict[str, float] | None = None,
        # Augmentation hyperparameters
        channel_dropout_channels: list[int] | None = None,
        channel_dropout_prob: float = 0.5,
        normalizations: list[MapTransform] | None = None,
        augmentations: list[MapTransform] | None = None,
        # Loss hyperparameters (informational for CLI discoverability)
        hcl_beta: float = 0.5,
        # Other
        cache_pool_bytes: int = 0,
        seed: int = 0,
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
    ) -> None:
        super().__init__()

        # Core parameters
        self.experiments_yaml = experiments_yaml
        self.z_range = z_range
        self.yx_patch_size = yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.val_experiments = val_experiments
        self.tau_range = tau_range
        self.tau_decay_rate = tau_decay_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Sampling hyperparameters
        self.experiment_aware = experiment_aware
        self.condition_balanced = condition_balanced
        self.leaky = leaky
        self.temporal_enrichment = temporal_enrichment
        self.temporal_window_hours = temporal_window_hours
        self.temporal_global_fraction = temporal_global_fraction
        self.experiment_weights = experiment_weights
        self.condition_ratio = condition_ratio

        # Augmentation hyperparameters
        self.channel_dropout_channels = (
            channel_dropout_channels if channel_dropout_channels is not None else [1]
        )
        self.channel_dropout_prob = channel_dropout_prob
        self.normalizations = normalizations if normalizations is not None else []
        self.augmentations = augmentations if augmentations is not None else []

        # Loss hyperparameters (informational)
        self.hcl_beta = hcl_beta

        # Other
        self.cache_pool_bytes = cache_pool_bytes
        self.seed = seed
        self.include_wells = include_wells
        self.exclude_fovs = exclude_fovs

        # Create ChannelDropout module
        self.channel_dropout = ChannelDropout(
            channels=self.channel_dropout_channels,
            p=self.channel_dropout_prob,
        )

        # Datasets (populated in setup)
        self.train_dataset: MultiExperimentTripletDataset | None = None
        self.val_dataset: MultiExperimentTripletDataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Set up train and val datasets with experiment-level split.

        Parameters
        ----------
        stage : str or None
            Lightning stage: ``"fit"``, ``"predict"``, etc.
        """
        if stage == "fit" or stage is None:
            registry = ExperimentRegistry.from_yaml(self.experiments_yaml)

            # Split by experiment name
            train_configs = [
                exp
                for exp in registry.experiments
                if exp.name not in self.val_experiments
            ]
            val_configs = [
                exp
                for exp in registry.experiments
                if exp.name in self.val_experiments
            ]

            if not train_configs:
                raise ValueError(
                    "No training experiments remaining after splitting. "
                    f"val_experiments={self.val_experiments} covers all experiments."
                )
            if not val_configs:
                _logger.warning(
                    "No validation experiments found. "
                    "val_experiments=%s not present in registry.",
                    self.val_experiments,
                )

            z_slice = slice(*self.z_range)

            # Build separate registries and indices
            train_registry = ExperimentRegistry(experiments=train_configs)
            train_index = MultiExperimentIndex(
                registry=train_registry,
                z_range=z_slice,
                yx_patch_size=self.yx_patch_size,
                tau_range_hours=self.tau_range,
                include_wells=self.include_wells,
                exclude_fovs=self.exclude_fovs,
            )
            self.train_dataset = MultiExperimentTripletDataset(
                index=train_index,
                fit=True,
                tau_range_hours=self.tau_range,
                tau_decay_rate=self.tau_decay_rate,
                cache_pool_bytes=self.cache_pool_bytes,
            )

            if val_configs:
                val_registry = ExperimentRegistry(experiments=val_configs)
                val_index = MultiExperimentIndex(
                    registry=val_registry,
                    z_range=z_slice,
                    yx_patch_size=self.yx_patch_size,
                    tau_range_hours=self.tau_range,
                    include_wells=self.include_wells,
                    exclude_fovs=self.exclude_fovs,
                )
                self.val_dataset = MultiExperimentTripletDataset(
                    index=val_index,
                    fit=True,
                    tau_range_hours=self.tau_range,
                    tau_decay_rate=self.tau_decay_rate,
                    cache_pool_bytes=self.cache_pool_bytes,
                )

            # Build channel names for transforms (generic since experiments
            # may have different names but same count)
            n_ch = train_registry.num_source_channels
            self._channel_names = [f"ch_{i}" for i in range(n_ch)]

            # Build transform pipelines
            self._augmentation_transform = Compose(
                self.normalizations + self.augmentations + [self._final_crop()]
            )
            self._no_augmentation_transform = Compose(
                self.normalizations + [self._final_crop()]
            )

            _logger.info(
                "MultiExperimentDataModule setup: "
                "%d train experiments (%d anchors), "
                "%d val experiments (%d anchors)",
                len(train_configs),
                len(self.train_dataset) if self.train_dataset else 0,
                len(val_configs),
                len(self.val_dataset) if self.val_dataset else 0,
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> ThreadDataLoader:
        """Return training data loader with FlexibleBatchSampler."""
        sampler = FlexibleBatchSampler(
            valid_anchors=self.train_dataset.index.valid_anchors,
            batch_size=self.batch_size,
            experiment_aware=self.experiment_aware,
            leaky=self.leaky,
            experiment_weights=self.experiment_weights,
            condition_balanced=self.condition_balanced,
            condition_ratio=self.condition_ratio,
            temporal_enrichment=self.temporal_enrichment,
            temporal_window_hours=self.temporal_window_hours,
            temporal_global_fraction=self.temporal_global_fraction,
            seed=self.seed,
        )
        return ThreadDataLoader(
            self.train_dataset,
            use_thread_workers=True,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self) -> ThreadDataLoader | None:
        """Return validation data loader (deterministic, no FlexibleBatchSampler)."""
        if self.val_dataset is None:
            return None
        return ThreadDataLoader(
            self.val_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: x,
        )

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _final_crop(self) -> BatchedCenterSpatialCropd:
        """Create center crop from initial to final patch size."""
        z_window = self.z_range[1] - self.z_range[0]
        return BatchedCenterSpatialCropd(
            keys=self._channel_names,
            roi_size=(z_window, self.final_yx_patch_size[0], self.final_yx_patch_size[1]),
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Apply normalizations, augmentations, final crop, and ChannelDropout.

        Parameters
        ----------
        batch : dict or Tensor
            Batch from dataloader. If Tensor (example_input_array), return as-is.
        dataloader_idx : int
            Index of the dataloader.

        Returns
        -------
        dict or Tensor
            Transformed batch.
        """
        if isinstance(batch, Tensor):
            return batch

        # Determine transform: augmentation for training, no-aug for val
        if self.trainer and self.trainer.validating:
            transform = self._no_augmentation_transform
        else:
            transform = self._augmentation_transform

        for key in ["anchor", "positive", "negative"]:
            if key in batch:
                norm_meta_key = f"{key}_norm_meta"
                norm_meta = batch.get(norm_meta_key)
                # _scatter_channels expects NormMeta | None, not list[None].
                # When all entries are None, pass None to avoid collation errors.
                if isinstance(norm_meta, list) and all(m is None for m in norm_meta):
                    norm_meta = None
                transformed = _transform_channel_wise(
                    transform=transform,
                    channel_names=self._channel_names,
                    patch=batch[key],
                    norm_meta=norm_meta,
                )
                batch[key] = transformed
                if norm_meta_key in batch:
                    del batch[norm_meta_key]

        # Apply ChannelDropout to anchor and positive
        for key in ["anchor", "positive"]:
            if key in batch:
                batch[key] = self.channel_dropout(batch[key])

        return batch
