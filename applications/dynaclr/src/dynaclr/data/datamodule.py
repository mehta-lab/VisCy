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
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from monai.data.thread_buffer import ThreadDataLoader
from monai.transforms import Compose, MapTransform
from torch import Tensor

from dynaclr.data.dataset import MultiExperimentTripletDataset
from dynaclr.data.experiment import ExperimentRegistry
from dynaclr.data.index import MultiExperimentIndex
from viscy_data._utils import BatchedCenterSpatialCropd, _transform_channel_wise
from viscy_data.channel_dropout import ChannelDropout
from viscy_data.channel_utils import parse_channel_name
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
    cell_index_path : str
        Path to preprocessed cell index parquet (from ``build-cell-index``
        + ``preprocess-cell-index``). Contains all metadata needed for
        training: TCZYX shape, normalization stats, focus slice.
    z_window : int
        Number of Z slices the model consumes (final crop size).
    z_extraction_window : int or None
        Number of Z slices to extract from zarr before cropping.
        Must be >= ``z_window``. When None (default), falls back to
        ``z_window`` (deterministic Z, no random crop). When larger,
        enables random Z cropping during training for focus-plane
        invariance. Works for both 2D (``z_window=1``) and 3D.
    z_focus_offset : float
        Fraction of extraction window placed below focus plane.
        0.5 = symmetric (default). 0.3 = 30% below, 70% above.
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
    batch_group_by : str or list[str] or None
        Column(s) to group batches by (e.g. ``"experiment"``). Default: None.
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
    group_weights : dict[str, float] | None
        Per-group sampling weights. Default: None (proportional).
    channels_per_sample : int | list[str] | None
        Controls how many source channels to read per sample.
        ``None`` (default) — read all source channels.
        ``1`` — randomly select one channel per sample.
        List of labels — read those specific channels.
    channel_dropout_channels : list[int]
        Channel indices to dropout. Default: [1] (fluorescence).
    channel_dropout_prob : float
        Dropout probability. Default: 0.5.
    normalizations : list[MapTransform]
        Normalization transforms. Default: [].
    augmentations : list[MapTransform]
        Augmentation transforms. Default: [].
    cache_pool_bytes : int
        Tensorstore cache pool size. Default: 0.
    seed : int
        RNG seed for FlexibleBatchSampler. Default: 0.
    include_wells : list[str] | None
        Only include these wells. Default: None.
    exclude_fovs : list[str] | None
        Exclude these FOVs. Default: None.
    focus_channel : str | None
        Channel name for ``focus_slice`` lookup when auto-resolving z_range.
        Default: None (uses first source_channel).
    reference_pixel_size_xy_um : float or None
        Reference pixel size in XY (micrometers) for physical-scale normalization.
        None = no rescaling. Default: None.
    reference_pixel_size_z_um : float or None
        Reference voxel size in Z (micrometers) for physical-scale normalization.
        None = no rescaling. Default: None.
    positive_cell_source : str
        ``"self"`` — SimCLR: anchor and positive are the same crop.
        ``"lookup"`` (default) — find a different cell via ``positive_match_columns``.
    positive_match_columns : list[str] | None
        Columns defining "same identity" for positive lookup.
        Defaults to ``["lineage_id"]`` (temporal matching).
        For OPS perturbation: ``["gene_name", "reporter"]``.
    positive_channel_source : str
        ``"same"`` (default) — anchor and positive share channel index.
        ``"any"`` — positive draws independently. Only affects bag-of-channels mode.
    label_columns : dict[str, str] | None
        Mapping from ``batch_key`` (used by classification heads) to
        dataframe column name.  E.g. ``{"gene_label": "condition"}``.
        Default: ``None``.
    """

    def __init__(
        self,
        cell_index_path: str,
        z_window: int,
        z_extraction_window: int | None = None,
        z_focus_offset: float = 0.5,
        yx_patch_size: tuple[int, int] = (192, 192),
        final_yx_patch_size: tuple[int, int] = (160, 160),
        val_experiments: list[str] | None = None,
        split_ratio: float = 0.8,
        tau_range: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        batch_size: int = 128,
        num_workers: int = 1,
        # Sampling hyperparameters (passed to FlexibleBatchSampler)
        batch_group_by: str | list[str] | None = None,
        stratify_by: str | list[str] | None = "perturbation",
        leaky: float = 0.0,
        temporal_enrichment: bool = False,
        temporal_window_hours: float = 2.0,
        temporal_global_fraction: float = 0.3,
        group_weights: dict[str, float] | None = None,
        # Bag of channels
        channels_per_sample: int | list[str] | None = None,
        # Augmentation hyperparameters
        channel_dropout_channels: list[int] | None = None,
        channel_dropout_prob: float = 0.0,
        normalizations: list[MapTransform] | None = None,
        augmentations: list[MapTransform] | None = None,
        # Other
        cache_pool_bytes: int = 0,
        seed: int = 0,
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        focus_channel: str | None = None,
        reference_pixel_size_xy_um: float | None = None,
        reference_pixel_size_z_um: float | None = None,
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
        positive_channel_source: str = "same",
        label_columns: dict[str, str] | None = None,
        max_border_shift: int = -1,
        shuffle_val: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        buffer_size: int = 1,
    ) -> None:
        super().__init__()

        # Core parameters
        self.cell_index_path = cell_index_path
        self.z_window = z_window
        self.z_extraction_window = z_extraction_window
        self.z_focus_offset = z_focus_offset
        self.yx_patch_size = yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.val_experiments = val_experiments if val_experiments is not None else []
        self.split_ratio = split_ratio
        self.tau_range = tau_range
        self.tau_decay_rate = tau_decay_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Sampling hyperparameters
        if isinstance(batch_group_by, str):
            batch_group_by = [batch_group_by]
        self.batch_group_by = batch_group_by
        self.stratify_by = stratify_by
        self.leaky = leaky
        self.temporal_enrichment = temporal_enrichment
        self.temporal_window_hours = temporal_window_hours
        self.temporal_global_fraction = temporal_global_fraction
        self.group_weights = group_weights

        # Channel selection
        self.channels_per_sample = channels_per_sample

        # Augmentation hyperparameters
        self.channel_dropout_channels = channel_dropout_channels if channel_dropout_channels is not None else [1]
        self.channel_dropout_prob = channel_dropout_prob
        self.normalizations = normalizations if normalizations is not None else []
        self.augmentations = augmentations if augmentations is not None else []

        # Loss hyperparameters (informational)
        # Other
        self.cache_pool_bytes = cache_pool_bytes
        self.seed = seed
        self.include_wells = include_wells
        self.exclude_fovs = exclude_fovs
        self.focus_channel = focus_channel
        self.reference_pixel_size_xy_um = reference_pixel_size_xy_um
        self.reference_pixel_size_z_um = reference_pixel_size_z_um
        self.positive_cell_source = positive_cell_source
        self.positive_match_columns = positive_match_columns
        self.positive_channel_source = positive_channel_source
        self.label_columns = label_columns
        self.max_border_shift = max_border_shift
        self.shuffle_val = shuffle_val
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.buffer_size = buffer_size

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
            registry = ExperimentRegistry.from_cell_index(
                self.cell_index_path,
                z_window=self.z_window,
                z_extraction_window=self.z_extraction_window,
                z_focus_offset=self.z_focus_offset,
                focus_channel=self.focus_channel,
                reference_pixel_size_xy_um=self.reference_pixel_size_xy_um,
                reference_pixel_size_z_um=self.reference_pixel_size_z_um,
            )

            if self.val_experiments:
                self._setup_experiment_split(registry)
            else:
                self._setup_fov_split(registry)

            if self.channels_per_sample is None:
                self._channel_names = registry.source_channel_labels
            elif isinstance(self.channels_per_sample, int):
                self._channel_names = [f"channel_{i}" for i in range(self.channels_per_sample)]
            else:
                self._channel_names = list(self.channels_per_sample)

            # Build transform pipelines
            # Training: random crop (enables Z-focus invariance when z_extraction_window > z_window)
            # Validation: deterministic center crop
            self._augmentation_transform = Compose(
                self.normalizations + self.augmentations + [self._train_final_crop()]
            )

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
            positive_cell_source=self.positive_cell_source,
            positive_match_columns=self.positive_match_columns,
            max_border_shift=self.max_border_shift,
        )
        self.train_dataset = MultiExperimentTripletDataset(
            index=train_index,
            fit=True,
            tau_range_hours=self.tau_range,
            tau_decay_rate=self.tau_decay_rate,
            cache_pool_bytes=self.cache_pool_bytes,
            channels_per_sample=self.channels_per_sample,
            positive_cell_source=self.positive_cell_source,
            positive_match_columns=self.positive_match_columns,
            positive_channel_source=self.positive_channel_source,
            label_columns=self.label_columns,
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
                positive_cell_source=self.positive_cell_source,
                positive_match_columns=self.positive_match_columns,
                max_border_shift=self.max_border_shift,
            )
            self.val_dataset = MultiExperimentTripletDataset(
                index=val_index,
                fit=True,
                tau_range_hours=self.tau_range,
                tau_decay_rate=self.tau_decay_rate,
                cache_pool_bytes=self.cache_pool_bytes,
                channels_per_sample=self.channels_per_sample,
                positive_cell_source=self.positive_cell_source,
                positive_match_columns=self.positive_match_columns,
                positive_channel_source=self.positive_channel_source,
                label_columns=self.label_columns,
            )

    def _setup_fov_split(self, registry: ExperimentRegistry) -> None:
        """Split FOVs within each experiment by split_ratio.

        Uses experiment-qualified keys ``(experiment, fov_name)`` so that
        experiments sharing the same tile names (e.g. OPS ``A/1/0``) are
        split independently.
        """
        full_index = MultiExperimentIndex(
            registry=registry,
            yx_patch_size=self.yx_patch_size,
            tau_range_hours=self.tau_range,
            include_wells=self.include_wells,
            exclude_fovs=self.exclude_fovs,
            cell_index_path=self.cell_index_path,
            positive_cell_source=self.positive_cell_source,
            positive_match_columns=self.positive_match_columns,
        )

        rng = np.random.default_rng(self.seed)
        train_keys: set[tuple[str, str]] = set()
        val_keys: set[tuple[str, str]] = set()

        for exp_name, group in full_index.tracks.groupby("experiment"):
            fovs = sorted(group["fov_name"].unique())
            n_train = max(1, int(len(fovs) * self.split_ratio))
            rng.shuffle(fovs)
            for f in fovs[:n_train]:
                train_keys.add((exp_name, f))
            for f in fovs[n_train:]:
                val_keys.add((exp_name, f))

        _logger.info(
            "FOV split (ratio=%.2f): %d train FOVs, %d val FOVs",
            self.split_ratio,
            len(train_keys),
            len(val_keys),
        )

        full_qual = list(zip(full_index.tracks["experiment"], full_index.tracks["fov_name"]))
        train_mask = pd.Series([k in train_keys for k in full_qual], index=full_index.tracks.index)

        train_tracks = full_index.tracks[train_mask].reset_index(drop=True)
        val_tracks = full_index.tracks[~train_mask].reset_index(drop=True)

        train_index = full_index.clone_with_subset(
            train_tracks,
            positive_cell_source=self.positive_cell_source,
            positive_match_columns=self.positive_match_columns,
            max_border_shift=self.max_border_shift,
        )
        self.train_dataset = MultiExperimentTripletDataset(
            index=train_index,
            fit=True,
            tau_range_hours=self.tau_range,
            tau_decay_rate=self.tau_decay_rate,
            cache_pool_bytes=self.cache_pool_bytes,
            channels_per_sample=self.channels_per_sample,
            positive_cell_source=self.positive_cell_source,
            positive_match_columns=self.positive_match_columns,
            positive_channel_source=self.positive_channel_source,
            label_columns=self.label_columns,
        )

        if val_keys:
            val_index = full_index.clone_with_subset(
                val_tracks,
                positive_cell_source=self.positive_cell_source,
                positive_match_columns=self.positive_match_columns,
            )
            self.val_dataset = MultiExperimentTripletDataset(
                index=val_index,
                fit=True,
                tau_range_hours=self.tau_range,
                tau_decay_rate=self.tau_decay_rate,
                cache_pool_bytes=self.cache_pool_bytes,
                channels_per_sample=self.channels_per_sample,
                positive_cell_source=self.positive_cell_source,
                positive_match_columns=self.positive_match_columns,
                positive_channel_source=self.positive_channel_source,
                label_columns=self.label_columns,
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> ThreadDataLoader:
        """Return training data loader with FlexibleBatchSampler."""
        sampler = FlexibleBatchSampler(
            valid_anchors=self.train_dataset.index.valid_anchors,
            batch_size=self.batch_size,
            batch_group_by=self.batch_group_by,
            leaky=self.leaky,
            group_weights=self.group_weights,
            stratify_by=self.stratify_by,
            temporal_enrichment=self.temporal_enrichment,
            temporal_window_hours=self.temporal_window_hours,
            temporal_global_fraction=self.temporal_global_fraction,
            seed=self.seed,
        )
        return ThreadDataLoader(
            self.train_dataset,
            use_thread_workers=True,
            buffer_size=self.buffer_size,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self) -> ThreadDataLoader | None:
        """Return validation data loader."""
        if self.val_dataset is None:
            return None
        return ThreadDataLoader(
            self.val_dataset,
            use_thread_workers=True,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_val,
            drop_last=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: x,
        )

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _train_final_crop(self) -> BatchedCenterSpatialCropd:
        """Center crop from extraction size to model input size (training)."""
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
                extra = None
                if isinstance(self.channels_per_sample, int):
                    meta = batch.get(f"{key}_meta")
                    if meta is not None:
                        extra = {
                            "_is_labelfree": torch.tensor(
                                [parse_channel_name(m.get("marker", ""))["channel_type"] == "labelfree" for m in meta],
                                dtype=torch.bool,
                                device=batch[key].device,
                            )
                        }
                transformed = _transform_channel_wise(
                    transform=transform,
                    channel_names=self._channel_names,
                    patch=batch[key],
                    norm_meta=norm_meta,
                    extra=extra,
                )
                batch[key] = transformed
                if norm_meta_key in batch:
                    del batch[norm_meta_key]

        # Apply ChannelDropout to anchor and positive
        for key in ["anchor", "positive"]:
            if key in batch:
                batch[key] = self.channel_dropout(batch[key])

        return batch
