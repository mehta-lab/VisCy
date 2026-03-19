"""Lightning DataModule for multi-experiment DynaCLR training.

Composes :class:`~dynaclr.index.MultiExperimentIndex`,
:class:`~dynaclr.dataset.MultiExperimentTripletDataset`,
:class:`~viscy_data.sampler.FlexibleBatchSampler`,
:class:`~viscy_data.channel_dropout.ChannelDropout`, and
:class:`~monai.data.thread_buffer.ThreadDataLoader` into a fully
configurable training pipeline with experiment-level or FOV-level
train/val split.
"""

from __future__ import annotations

import logging

import numpy as np
from lightning.pytorch import LightningDataModule
from monai.data.thread_buffer import ThreadDataLoader
from monai.transforms import Compose, MapTransform
from torch import Tensor

from dynaclr.data.dataset import MultiExperimentTripletDataset
from dynaclr.data.experiment import ExperimentRegistry
from dynaclr.data.index import MultiExperimentIndex
from viscy_data._utils import BatchedCenterSpatialCropd, _transform_channel_wise
from viscy_data.channel_dropout import ChannelDropout
from viscy_data.sampler import FlexibleBatchSampler

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentDataModule"]


class MultiExperimentDataModule(LightningDataModule):
    """Lightning DataModule for multi-experiment DynaCLR training.

    Composes MultiExperimentIndex, MultiExperimentTripletDataset,
    FlexibleBatchSampler, ChannelDropout, and ThreadDataLoader into
    a fully configurable training pipeline.

    Supports two split modes:

    * **Experiment-level split** (``val_experiments`` is non-empty):
      entire experiments are held out for validation.
    * **FOV-level split** (``val_experiments`` is empty, ``split_ratio`` < 1.0):
      FOVs within each experiment are randomly split into train/val.

    Parameters
    ----------
    collection_path : str
        Path to collection YAML for ExperimentRegistry.from_collection().
    z_window : int
        Number of Z slices the model consumes.  Per-experiment Z
        centering is resolved from ``focus_slice`` zattrs or explicit
        ``z_range`` in the experiment config.
    yx_patch_size : tuple[int, int]
        Initial YX patch size for cell patch extraction.
    final_yx_patch_size : tuple[int, int]
        Final YX patch size after cropping (output size).
    val_experiments : list[str]
        Experiment names to use for validation (rest are training).
        Default: [] (no experiment-level holdout).
    split_ratio : float
        Fraction of FOVs to use for training when ``val_experiments`` is
        empty. E.g. 0.8 means 80% train, 20% val. Ignored when
        ``val_experiments`` is non-empty. Default: 0.8.
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
    stratify_by : str | list[str] | None
        Column name(s) to stratify batches by (e.g. ``"condition"``,
        ``["condition", "marker"]``, ``["condition", "organelle"]``). Default: ``"condition"``.
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
    bag_of_channels : bool
        If ``True``, randomly select one source channel per sample.
        Output shape becomes ``(B, 1, Z, Y, X)``. Pair with
        ``in_channels: 1`` on the encoder. Default: False.
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
    cell_index_path : str | None
        Optional path to a pre-built cell index parquet for faster startup.
        When provided, both train and val indices load from this parquet
        (filtered by their respective registries). Default: None.
    focus_channel : str | None
        Channel name for ``focus_slice`` lookup when auto-resolving z_range.
        Default: None (uses first source_channel).
    num_workers_index : int
        Number of parallel processes for building the cell index. Default: 1
        (sequential). When > 1, one process is spawned per experiment.
        Ignored when ``cell_index_path`` is provided.
    reference_pixel_size_xy_um : float or None
        Reference pixel size in XY (micrometers) for physical-scale normalization.
        None = no rescaling. Default: None.
    reference_pixel_size_z_um : float or None
        Reference voxel size in Z (micrometers) for physical-scale normalization.
        None = no rescaling. Default: None.
    cross_scope_fraction : float
        Fraction of positives sampled as cross-microscope positives.
        0.0 = pure temporal positives. Default: 0.0.
    hpi_window : float
        Half-width of HPI window (hours) for cross-scope positive matching. Default: 1.0.
    """

    def __init__(
        self,
        collection_path: str,
        z_window: int,
        yx_patch_size: tuple[int, int],
        final_yx_patch_size: tuple[int, int],
        val_experiments: list[str] | None = None,
        split_ratio: float = 0.8,
        tau_range: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        batch_size: int = 128,
        num_workers: int = 1,
        # Sampling hyperparameters (passed to FlexibleBatchSampler)
        experiment_aware: bool = True,
        stratify_by: str | list[str] | None = "condition",
        leaky: float = 0.0,
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        experiment_weights: dict[str, float] | None = None,
        # Bag of channels
        bag_of_channels: bool = False,
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
        cell_index_path: str | None = None,
        focus_channel: str | None = None,
        num_workers_index: int = 1,
        reference_pixel_size_xy_um: float | None = None,
        reference_pixel_size_z_um: float | None = None,
        cross_scope_fraction: float = 0.0,
        hpi_window: float = 1.0,
    ) -> None:
        super().__init__()

        # Core parameters
        self.collection_path = collection_path
        self.z_window = z_window
        self.yx_patch_size = yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.val_experiments = val_experiments if val_experiments is not None else []
        self.split_ratio = split_ratio
        self.tau_range = tau_range
        self.tau_decay_rate = tau_decay_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Sampling hyperparameters
        self.experiment_aware = experiment_aware
        self.stratify_by = stratify_by
        self.leaky = leaky
        self.temporal_enrichment = temporal_enrichment
        self.temporal_window_hours = temporal_window_hours
        self.temporal_global_fraction = temporal_global_fraction
        self.experiment_weights = experiment_weights

        # Bag of channels
        self.bag_of_channels = bag_of_channels

        # Augmentation hyperparameters
        self.channel_dropout_channels = channel_dropout_channels if channel_dropout_channels is not None else [1]
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
        self.cell_index_path = cell_index_path
        self.focus_channel = focus_channel
        self.num_workers_index = num_workers_index
        self.reference_pixel_size_xy_um = reference_pixel_size_xy_um
        self.reference_pixel_size_z_um = reference_pixel_size_z_um
        self.cross_scope_fraction = cross_scope_fraction
        self.hpi_window = hpi_window

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
        """Set up train and val datasets.

        Two split modes are supported:

        * **Experiment-level** (``val_experiments`` is non-empty):
          whole experiments are held out for validation.
        * **FOV-level** (``val_experiments`` is empty, ``split_ratio`` < 1.0):
          FOVs within each experiment are randomly split into train/val
          by ``split_ratio``.

        Parameters
        ----------
        stage : str or None
            Lightning stage: ``"fit"``, ``"predict"``, etc.
        """
        if stage == "fit" or stage is None:
            registry = ExperimentRegistry.from_collection(
                self.collection_path,
                z_window=self.z_window,
                focus_channel=getattr(self, "focus_channel", None),
                reference_pixel_size_xy_um=self.reference_pixel_size_xy_um,
                reference_pixel_size_z_um=self.reference_pixel_size_z_um,
            )

            if self.val_experiments:
                self._setup_experiment_split(registry)
            else:
                self._setup_fov_split(registry)

            if self.bag_of_channels:
                self._channel_names = ["channel"]
            else:
                self._channel_names = registry.source_channel_labels

            # Build transform pipelines
            self._augmentation_transform = Compose(self.normalizations + self.augmentations + [self._final_crop()])
            self._no_augmentation_transform = Compose(self.normalizations + [self._final_crop()])

            _logger.info(
                "MultiExperimentDataModule setup: %d train anchors, %d val anchors",
                len(self.train_dataset) if self.train_dataset else 0,
                len(self.val_dataset) if self.val_dataset else 0,
            )

    def _setup_experiment_split(self, registry: ExperimentRegistry) -> None:
        """Split by whole experiments into train/val."""
        train_names = [e.name for e in registry.experiments if e.name not in self.val_experiments]
        val_names = [e.name for e in registry.experiments if e.name in self.val_experiments]

        if not train_names:
            raise ValueError(
                "No training experiments remaining after splitting. "
                f"val_experiments={self.val_experiments} covers all experiments."
            )
        if not val_names:
            _logger.warning(
                "No validation experiments found. val_experiments=%s not present in registry.",
                self.val_experiments,
            )

        train_registry = registry.subset(train_names)
        train_index = MultiExperimentIndex(
            registry=train_registry,
            yx_patch_size=self.yx_patch_size,
            tau_range_hours=self.tau_range,
            include_wells=self.include_wells,
            exclude_fovs=self.exclude_fovs,
            cell_index_path=self.cell_index_path,
            num_workers=self.num_workers_index,
        )
        self.train_dataset = MultiExperimentTripletDataset(
            index=train_index,
            fit=True,
            tau_range_hours=self.tau_range,
            tau_decay_rate=self.tau_decay_rate,
            cache_pool_bytes=self.cache_pool_bytes,
            bag_of_channels=self.bag_of_channels,
            cross_scope_fraction=self.cross_scope_fraction,
            hpi_window=self.hpi_window,
        )

        if val_names:
            val_registry = registry.subset(val_names)
            val_index = MultiExperimentIndex(
                registry=val_registry,
                yx_patch_size=self.yx_patch_size,
                tau_range_hours=self.tau_range,
                include_wells=self.include_wells,
                exclude_fovs=self.exclude_fovs,
                cell_index_path=self.cell_index_path,
                num_workers=self.num_workers_index,
            )
            self.val_dataset = MultiExperimentTripletDataset(
                index=val_index,
                fit=True,
                tau_range_hours=self.tau_range,
                tau_decay_rate=self.tau_decay_rate,
                cache_pool_bytes=self.cache_pool_bytes,
                bag_of_channels=self.bag_of_channels,
                cross_scope_fraction=self.cross_scope_fraction,
                hpi_window=self.hpi_window,
            )

    def _setup_fov_split(self, registry: ExperimentRegistry) -> None:
        """Split FOVs within each experiment by split_ratio."""
        # Build a full index first, then split its tracks by FOV
        full_index = MultiExperimentIndex(
            registry=registry,
            yx_patch_size=self.yx_patch_size,
            tau_range_hours=self.tau_range,
            include_wells=self.include_wells,
            exclude_fovs=self.exclude_fovs,
            cell_index_path=self.cell_index_path,
            num_workers=self.num_workers_index,
        )

        # Split FOVs per experiment to maintain proportional representation
        rng = np.random.default_rng(self.seed)
        train_fovs: list[str] = []
        val_fovs: list[str] = []

        for exp_name, group in full_index.tracks.groupby("experiment"):
            fovs = sorted(group["fov_name"].unique())
            n_train = max(1, int(len(fovs) * self.split_ratio))
            rng.shuffle(fovs)
            train_fovs.extend(fovs[:n_train])
            val_fovs.extend(fovs[n_train:])

        _logger.info(
            "FOV split (ratio=%.2f): %d train FOVs, %d val FOVs",
            self.split_ratio,
            len(train_fovs),
            len(val_fovs),
        )

        # Build train index by excluding val FOVs
        train_exclude = (self.exclude_fovs or []) + val_fovs
        train_index = MultiExperimentIndex(
            registry=registry,
            yx_patch_size=self.yx_patch_size,
            tau_range_hours=self.tau_range,
            include_wells=self.include_wells,
            exclude_fovs=train_exclude,
            cell_index_path=self.cell_index_path,
            num_workers=self.num_workers_index,
        )
        self.train_dataset = MultiExperimentTripletDataset(
            index=train_index,
            fit=True,
            tau_range_hours=self.tau_range,
            tau_decay_rate=self.tau_decay_rate,
            cache_pool_bytes=self.cache_pool_bytes,
            bag_of_channels=self.bag_of_channels,
            cross_scope_fraction=self.cross_scope_fraction,
            hpi_window=self.hpi_window,
        )

        if val_fovs:
            val_exclude = (self.exclude_fovs or []) + train_fovs
            val_index = MultiExperimentIndex(
                registry=registry,
                yx_patch_size=self.yx_patch_size,
                tau_range_hours=self.tau_range,
                include_wells=self.include_wells,
                exclude_fovs=val_exclude,
                cell_index_path=self.cell_index_path,
                num_workers=self.num_workers_index,
            )
            self.val_dataset = MultiExperimentTripletDataset(
                index=val_index,
                fit=True,
                tau_range_hours=self.tau_range,
                tau_decay_rate=self.tau_decay_rate,
                cache_pool_bytes=self.cache_pool_bytes,
                bag_of_channels=self.bag_of_channels,
                cross_scope_fraction=self.cross_scope_fraction,
                hpi_window=self.hpi_window,
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
            stratify_by=self.stratify_by,
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
        return BatchedCenterSpatialCropd(
            keys=self._channel_names,
            roi_size=(self.z_window, self.final_yx_patch_size[0], self.final_yx_patch_size[1]),
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
                if isinstance(norm_meta, list):
                    non_none = [m for m in norm_meta if m is not None]
                    if len(non_none) == 0:
                        norm_meta = None
                    elif len(non_none) != len(norm_meta):
                        raise ValueError(
                            f"Mixed None/non-None norm_meta in batch for '{key}'. "
                            "All FOVs must have normalization metadata or none of them."
                        )
                    # else: all non-None, pass through as list
                transformed = _transform_channel_wise(
                    transform=transform,
                    channel_names=self._channel_names,
                    patch=batch[key],
                    norm_meta=norm_meta,
                )
                batch[key] = transformed
                if norm_meta_key in batch:
                    del batch[norm_meta_key]

        # Apply ChannelDropout to anchor and positive (training only)
        if not (self.trainer and self.trainer.validating):
            for key in ["anchor", "positive"]:
                if key in batch:
                    batch[key] = self.channel_dropout(batch[key])

        return batch
