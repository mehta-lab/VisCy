import logging
import random
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from monai.transforms import Compose, MapTransform
from torch import Tensor
from torch.utils.data import Dataset

from viscy.data.hcs import HCSDataModule
from viscy.data.triplet import (
    _transform_channel_wise,
)
from viscy.data.typing import DictTransform, TripletSample

_logger = logging.getLogger("lightning.pytorch")


class CellDivisionTripletDataset(Dataset):
    """Dataset for triplet sampling of cell division data from npy files.

    For the dataset from the paper:
    https://arxiv.org/html/2502.02182v1
    """

    # NOTE: Hardcoded channel mapping for .npy files
    CHANNEL_MAPPING = {
        # Channel 0 aliases (brightfield)
        "bf": 0,
        "brightfield": 0,
        # Channel 1 aliases (h2b)
        "h2b": 1,
        "nuclei": 1,
    }

    def __init__(
        self,
        data_paths: list[Path],
        channel_names: list[str],
        anchor_transform: DictTransform | None = None,
        positive_transform: DictTransform | None = None,
        negative_transform: DictTransform | None = None,
        fit: bool = True,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
        output_2d: bool = False,
    ) -> None:
        """Dataset for triplet sampling of cell division data from npy files.

        Parameters
        ----------
        data_paths : list[Path]
            List of paths to npy files containing cell division tracks (T,C,Y,X format)
        channel_names : list[str]
            Input channel names
        anchor_transform : DictTransform | None, optional
            Transforms applied to the anchor sample, by default None
        positive_transform : DictTransform | None, optional
            Transforms applied to the positive sample, by default None
        negative_transform : DictTransform | None, optional
            Transforms applied to the negative sample, by default None
        fit : bool, optional
            Fitting mode in which the full triplet will be sampled,
            only sample anchor if False, by default True
        time_interval : Literal["any"] | int, optional
            Future time interval to sample positive and anchor from,
            by default "any"
        return_negative : bool, optional
            Whether to return the negative sample during the fit stage, by default True
        output_2d : bool, optional
            Whether to return 2D tensors (C,Y,X) instead of 3D (C,1,Y,X), by default False
        """
        self.channel_names = channel_names
        self.anchor_transform = anchor_transform
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform
        self.fit = fit
        self.time_interval = time_interval
        self.return_negative = return_negative
        self.output_2d = output_2d

        # Load and process all data files
        self.cell_tracks = self._load_data(data_paths)
        self.valid_anchors = self._filter_anchors()

        # Create arrays for vectorized operations
        self.track_ids = np.array([t["track_id"] for t in self.cell_tracks])
        self.cell_tracks_array = np.array(self.cell_tracks)

        # Map channel names to indices using CHANNEL_MAPPING
        self.channel_indices = self._map_channel_indices(channel_names)

    def _map_channel_indices(self, channel_names: list[str]) -> list[int]:
        """Map channel names to their corresponding indices in the data array."""
        channel_indices = []
        for name in channel_names:
            if name in self.CHANNEL_MAPPING:
                channel_indices.append(self.CHANNEL_MAPPING[name])
            else:
                # Try to parse as integer if not in mapping
                try:
                    channel_indices.append(int(name))
                except ValueError:
                    raise ValueError(
                        f"Channel '{name}' not found in CHANNEL_MAPPING and is not a valid integer"
                    )
        return channel_indices

    def _select_channels(self, patch: Tensor) -> Tensor:
        """Select only the requested channels from the patch."""
        return patch[self.channel_indices]

    def _load_data(self, data_paths: list[Path]) -> list[dict]:
        """Load npy files."""
        all_tracks = []

        for path in data_paths:
            data = np.load(path)  # Shape: (T, C, Y, X)
            T, C, Y, X = data.shape

            # Create track info for this file
            # NOTE: using the filename as track ID as UID.
            track_info = {
                "data": torch.from_numpy(data.astype(np.float32)),
                "file_path": str(path),
                "track_id": path.stem,
                "num_timepoints": T,
                "shape": (T, C, Y, X),
            }
            all_tracks.append(track_info)

        _logger.info(f"Loaded {len(all_tracks)} tracks")
        return all_tracks

    def _filter_anchors(self) -> list[dict]:
        """Create valid anchor points based on time interval constraints."""
        valid_anchors = []

        for track in self.cell_tracks:
            num_timepoints = track["num_timepoints"]

            if self.time_interval == "any" or not self.fit:
                valid_timepoints = list(range(num_timepoints))
            else:
                # Only timepoints that have a future timepoint at the specified interval
                valid_timepoints = list(range(num_timepoints - self.time_interval))

            for t in valid_timepoints:
                anchor_info = {
                    "track": track,
                    "timepoint": t,
                    "track_id": track["track_id"],
                    "file_path": track["file_path"],
                }
                valid_anchors.append(anchor_info)

        return valid_anchors

    def __len__(self) -> int:
        return len(self.valid_anchors)

    def _sample_positive(self, anchor_info: dict) -> Tensor:
        """Select a positive sample from the same track."""
        track = anchor_info["track"]
        anchor_t = anchor_info["timepoint"]

        if self.time_interval == "any":
            # Use the same anchor patch (will be augmented differently)
            positive_t = anchor_t
        else:
            # Use future timepoint
            positive_t = anchor_t + self.time_interval

        positive_patch = track["data"][positive_t]
        positive_patch = self._select_channels(positive_patch)
        if not self.output_2d:
            positive_patch = positive_patch.unsqueeze(1)
        return positive_patch

    def _sample_negative(self, anchor_info: dict) -> Tensor:
        """Select a negative sample from a different track."""
        anchor_track_id = anchor_info["track_id"]

        # Vectorized filtering using boolean indexing
        mask = self.track_ids != anchor_track_id
        negative_candidates = self.cell_tracks_array[mask].tolist()

        if not negative_candidates:
            # Fallback: use different timepoint from same track
            track = anchor_info["track"]
            anchor_t = anchor_info["timepoint"]
            available_times = [
                t for t in range(track["num_timepoints"]) if t != anchor_t
            ]
            if available_times:
                neg_t = random.choice(available_times)
                negative_patch = track["data"][neg_t]
                negative_patch = self._select_channels(negative_patch)
            else:
                # Ultimate fallback: use same patch (transforms will differentiate)
                negative_patch = track["data"][anchor_t]
                negative_patch = self._select_channels(negative_patch)
        else:
            # Sample from different track
            neg_track = random.choice(negative_candidates)

            if self.time_interval == "any":
                neg_t = random.randint(0, neg_track["num_timepoints"] - 1)
            else:
                # Try to use same relative timepoint, fallback to random
                anchor_t = anchor_info["timepoint"]
                target_t = anchor_t + self.time_interval
                if target_t < neg_track["num_timepoints"]:
                    neg_t = target_t
                else:
                    neg_t = random.randint(0, neg_track["num_timepoints"] - 1)

            negative_patch = neg_track["data"][neg_t]
            negative_patch = self._select_channels(negative_patch)

        # Add depth dimension only if not output_2d: (C, Y, X) -> (C, D=1, Y, X)
        if not self.output_2d:
            negative_patch = negative_patch.unsqueeze(1)  # Shape: (C, 1, Y, X)
        return negative_patch

    def __getitem__(self, index: int) -> TripletSample:
        anchor_info = self.valid_anchors[index]
        track = anchor_info["track"]
        anchor_t = anchor_info["timepoint"]

        # Get anchor patch and select requested channels
        anchor_patch = track["data"][anchor_t]  # Shape: (C, Y, X)
        anchor_patch = self._select_channels(anchor_patch)
        if not self.output_2d:
            anchor_patch = anchor_patch.unsqueeze(1)

        sample = {"anchor": anchor_patch}

        if self.fit:
            positive_patch = self._sample_positive(anchor_info)

            if self.positive_transform:
                positive_patch = _transform_channel_wise(
                    transform=self.positive_transform,
                    channel_names=self.channel_names,
                    patch=positive_patch,
                    norm_meta=None,
                )

            if self.return_negative:
                negative_patch = self._sample_negative(anchor_info)

                if self.negative_transform:
                    negative_patch = _transform_channel_wise(
                        transform=self.negative_transform,
                        channel_names=self.channel_names,
                        patch=negative_patch,
                        norm_meta=None,
                    )

                sample.update({"positive": positive_patch, "negative": negative_patch})
            else:
                sample.update({"positive": positive_patch})
        else:
            # For prediction mode, include index information
            index_dict = {
                "fov_name": anchor_info["track_id"],
                "id": anchor_t,
            }
            sample.update({"index": index_dict})

        if self.anchor_transform:
            sample["anchor"] = _transform_channel_wise(
                transform=self.anchor_transform,
                channel_names=self.channel_names,
                patch=sample["anchor"],
                norm_meta=None,
            )

        return sample


class CellDivisionTripletDataModule(HCSDataModule):
    def __init__(
        self,
        data_path: str,
        source_channel: str | Sequence[str],
        final_yx_patch_size: tuple[int, int] = (64, 64),  # Match dataset size
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        augment_validation: bool = True,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
        output_2d: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
    ):
        """Lightning data module for cell division triplet sampling.

        Parameters
        ----------
        data_path : str
            Path to directory containing npy files
        source_channel : str | Sequence[str]
            List of input channel names
        final_yx_patch_size : tuple[int, int], optional
            Output patch size, by default (64, 64)
        split_ratio : float, optional
            Ratio of training samples, by default 0.8
        batch_size : int, optional
            Batch size, by default 16
        num_workers : int, optional
            Number of data-loading workers, by default 8
        normalizations : list[MapTransform], optional
            Normalization transforms, by default []
        augmentations : list[MapTransform], optional
            Augmentation transforms, by default []
        augment_validation : bool, optional
            Apply augmentations to validation data, by default True
        time_interval : Literal["any"] | int, optional
            Future time interval to sample positive and anchor from, by default "any"
        return_negative : bool, optional
            Whether to return the negative sample during the fit stage, by default True
        output_2d : bool, optional
            Whether to return 2D tensors (C,Y,X) instead of 3D (C,1,Y,X), by default False
        persistent_workers : bool, optional
            Whether to keep worker processes alive between iterations, by default False
        prefetch_factor : int | None, optional
            Number of batches loaded in advance by each worker, by default None
        pin_memory : bool, optional
            Whether to pin memory in CPU for faster GPU transfer, by default False
        """
        # Initialize parent class with minimal required parameters
        super().__init__(
            data_path=data_path,
            source_channel=source_channel,
            target_channel=[],
            z_window_size=1,
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            target_2d=False,  # Set to False since we're adding depth dimension
            yx_patch_size=final_yx_patch_size,
            normalizations=normalizations,
            augmentations=augmentations,
            caching=False,  # NOTE: Not applicable for npy files
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
        self.split_ratio = split_ratio
        self.data_path = Path(data_path)
        self.time_interval = time_interval
        self.return_negative = return_negative
        self.output_2d = output_2d
        self.augment_validation = augment_validation

        # Find all npy files in the data directory
        self.npy_files = list(self.data_path.glob("*.npy"))
        if not self.npy_files:
            raise ValueError(f"No .npy files found in {data_path}")

        _logger.info(f"Found {len(self.npy_files)} .npy files in {data_path}")

    @property
    def _base_dataset_settings(self) -> dict:
        return {
            "channel_names": self.source_channel,
            "time_interval": self.time_interval,
            "output_2d": self.output_2d,
        }

    def _setup_fit(self, dataset_settings: dict):
        augment_transform, no_aug_transform = self._fit_transform()

        # Shuffle and split the npy files
        shuffled_indices = self._set_fit_global_state(len(self.npy_files))
        npy_files = [self.npy_files[i] for i in shuffled_indices]

        # Set the train and eval positions
        num_train_files = int(len(self.npy_files) * self.split_ratio)
        train_npy_files = npy_files[:num_train_files]
        val_npy_files = npy_files[num_train_files:]

        _logger.debug(f"Number of training files: {len(train_npy_files)}")
        _logger.debug(f"Number of validation files: {len(val_npy_files)}")

        # Determine anchor transform based on time interval
        anchor_transform = (
            no_aug_transform
            if (self.time_interval == "any" or self.time_interval == 0)
            else augment_transform
        )

        # Create training dataset
        self.train_dataset = CellDivisionTripletDataset(
            data_paths=train_npy_files,
            anchor_transform=anchor_transform,
            positive_transform=augment_transform,
            negative_transform=augment_transform,
            fit=True,
            return_negative=self.return_negative,
            **dataset_settings,
        )

        # Choose transforms for validation based on augment_validation parameter
        val_positive_transform = (
            augment_transform if self.augment_validation else no_aug_transform
        )
        val_negative_transform = (
            augment_transform if self.augment_validation else no_aug_transform
        )
        val_anchor_transform = (
            anchor_transform if self.augment_validation else no_aug_transform
        )

        # Create validation dataset
        self.val_dataset = CellDivisionTripletDataset(
            data_paths=val_npy_files,
            anchor_transform=val_anchor_transform,
            positive_transform=val_positive_transform,
            negative_transform=val_negative_transform,
            fit=True,
            return_negative=self.return_negative,
            **dataset_settings,
        )

        _logger.info(f"Training dataset size: {len(self.train_dataset)}")
        _logger.info(f"Validation dataset size: {len(self.val_dataset)}")

    def _setup_predict(self, dataset_settings: dict):
        self._set_predict_global_state()

        # For prediction, use all data
        self.predict_dataset = CellDivisionTripletDataset(
            data_paths=self.npy_files,
            anchor_transform=Compose(self.normalizations),
            fit=False,
            **dataset_settings,
        )

    def _setup_test(self, *args, **kwargs):
        raise NotImplementedError("Self-supervised model does not support testing")
