"""Batch-aware spatial cropping transforms."""

import torch
import torch.nn.functional as F
from monai.transforms import (
    CenterSpatialCrop,
    Cropd,
    MapTransform,
    RandCropd,
    RandSpatialCrop,
)
from typing_extensions import Sequence

__all__ = [
    "BatchedRandSpatialCrop",
    "BatchedRandSpatialCropd",
    "BatchedRandWeightedCropd",
    "BatchedCenterSpatialCrop",
    "BatchedCenterSpatialCropd",
]


class BatchedRandSpatialCrop(RandSpatialCrop):
    """
    Batched version of RandSpatialCrop that applies random spatial cropping to a batch of images.

    Each image in the batch gets its own random crop parameters. When random_size=True,
    all crops use the same randomly chosen size to ensure consistent output tensor shapes.

    Parameters
    ----------
    roi_size : Sequence[int] | int
        Expected ROI size to crop. e.g. [224, 224, 128]. If int, same size used for all dimensions.
    max_roi_size : Sequence[int] | int | None, optional
        Maximum ROI size when random_size=True. If None, defaults to input image size.
    random_center : bool, optional
        Whether to crop at random position (True) or image center (False). Default is True.
    random_size : bool, optional
        Not supported in batched mode, must be False.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
    ) -> None:
        if random_size:
            raise ValueError("Batched transform does not support random size.")
        super().__init__(roi_size, max_roi_size, random_center, random_size, lazy=False)
        self._batch_sizes: list[Sequence[int]] = []
        self._batch_slices: list[tuple[slice, ...]] = []

    def randomize(self, img_size: Sequence[int]) -> None:
        """Generate random crop parameters for each image in the batch."""
        self._batch_sizes = []
        self._batch_slices = []

        # Skip batch and channel dimensions for spatial cropping
        spatial_size = img_size[2:]

        for _ in range(img_size[0]):
            super().randomize(spatial_size)
            if self._size is not None:
                self._batch_sizes.append(tuple(self._size))
            if hasattr(self, "_slices"):
                self._batch_slices.append(self._slices)

    def __call__(
        self,
        img: torch.Tensor,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Apply batched random spatial crop to input tensor.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (B, C, H, W, D) or (B, C, H, W).
        randomize : bool, optional
            Whether to generate new random parameters. Default is True.
        lazy : bool | None, optional
            Not used in batched version. Default is None.

        Returns
        -------
        torch.Tensor
            Cropped tensor with same batch size. When random_size=True, all crops
            use the same randomly chosen size to ensure consistent output shapes.
        """
        if randomize:
            self.randomize(img.shape)

        # Only support 3D cropping
        if len(self._batch_slices[0]) != 3:
            raise ValueError("BatchedRandSpatialCrop only supports 3D data")
        first_slice = self._batch_slices[0]

        crop_depth, crop_height, crop_width = (
            first_slice[0].stop - first_slice[0].start,
            first_slice[1].stop - first_slice[1].start,
            first_slice[2].stop - first_slice[2].start,
        )
        batch_size = img.shape[0]

        start_positions = torch.tensor(
            [[s[0].start, s[1].start, s[2].start] for s in self._batch_slices],
            dtype=torch.long,
            device=img.device,
        )

        windows = img.contiguous().unfold(2, crop_depth, 1).unfold(3, crop_height, 1).unfold(4, crop_width, 1)

        batch_indices = torch.arange(batch_size, device=img.device)
        return windows[
            batch_indices,
            :,
            start_positions[:, 0],
            start_positions[:, 1],
            start_positions[:, 2],
        ].contiguous()


class BatchedRandSpatialCropd(RandCropd):
    def __init__(
        self,
        keys: Sequence[str],
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        cropper = BatchedRandSpatialCrop(roi_size, max_roi_size, random_center, random_size)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict[str, torch.Tensor], lazy: bool | None = None) -> dict[str, torch.Tensor]:
        first_item = data[self.first_key(data)]
        self.randomize(first_item.shape)
        for key in self.key_iterator(data):
            data[key] = self.cropper(data[key], randomize=False)
        return data


class BatchedCenterSpatialCrop(CenterSpatialCrop):
    """
    Batched version of CenterSpatialCrop.

    Parameters
    ----------
    roi_size : Sequence[int] | int
        Expected ROI size to crop. e.g. [224, 224, 128]. If int, same size used for all dimensions.
    """

    def __init__(self, roi_size: Sequence[int] | int) -> None:
        super().__init__(roi_size, lazy=False)

    def __call__(
        self,
        img: torch.Tensor,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Apply batched center spatial crop to input tensor.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (B, C, H, W, D) or (B, C, H, W).
        lazy : bool | None, optional
            Not used in batched version. Default is None.

        Returns
        -------
        torch.Tensor
            Cropped tensor with same batch size and consistent crop size across batch.
        """
        spatial_size = img.shape[2:]
        crop_slices = self.compute_slices(spatial_size)
        slices = (slice(None), slice(None)) + crop_slices
        return img[slices]


class BatchedCenterSpatialCropd(Cropd):
    """
    Batched version of CenterSpatialCropd.

    Parameters
    ----------
    keys : Sequence[str]
        Keys to pick data for transformation.
    roi_size : Sequence[int] | int
        Expected ROI size to crop. e.g. [224, 224, 128]. If int, same size used for all dimensions.
    allow_missing_keys : bool, optional
        Don't raise exception if key is missing. Default is False.
    """

    def __init__(
        self,
        keys: Sequence[str],
        roi_size: Sequence[int] | int,
        allow_missing_keys: bool = False,
    ) -> None:
        cropper = BatchedCenterSpatialCrop(roi_size)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)


class BatchedRandWeightedCropd(MapTransform):
    """Randomly crop regions weighted by a spatial importance map.

    Samples crop positions with probability proportional to the weight map
    values, then extracts crops at those positions. Each sample in the batch
    gets an independently sampled crop position. All keys share the same crop
    coordinates so paired inputs (e.g. source/target) remain aligned.

    The weight map is reduced to ``(B, Y, X)`` by summing over all channels
    and Z slices. For single-channel targets (the common case), this is
    equivalent to using the channel directly. For multi-channel targets, all
    channels contribute equally.

    Parameters
    ----------
    keys : Sequence[str] | str
        Keys of the data dictionary to crop.
    w_key : str
        Key of the weight map tensor in the data dictionary.
    spatial_size : Sequence[int]
        Size of the crop region as ``(Z, Y, X)``.
    allow_missing_keys : bool
        Whether to skip missing keys. Default: False.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        w_key: str,
        spatial_size: Sequence[int],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.w_key = w_key
        self._spatial_size = tuple(spatial_size)

    def _sample_crop_starts(self, weight_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample per-batch crop start positions from the weight map.

        Parameters
        ----------
        weight_map : torch.Tensor
            Weight map tensor of shape ``(B, C, Z, Y, X)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(z_starts, y_starts, x_starts)`` each of shape ``(B,)``.
        """
        B, _C, Z, Y, X = weight_map.shape
        crop_z, crop_y, crop_x = self._spatial_size
        device = weight_map.device

        # Reduce to 2D: sum over channels and Z
        w = weight_map.sum(dim=(1, 2))  # (B, Y, X)
        w = w.clamp(min=0).float()

        # Pool over crop windows to get per-window aggregate weight
        w_4d = w.unsqueeze(1)  # (B, 1, Y, X)
        w_pooled = F.avg_pool2d(w_4d, (crop_y, crop_x), stride=1)  # (B, 1, valid_y, valid_x)
        valid_x = w_pooled.shape[3]

        # Flatten and handle all-zero maps (uniform fallback)
        w_flat = w_pooled.view(B, -1)  # (B, valid_y * valid_x)
        zero_mask = w_flat.sum(dim=1) == 0
        w_flat[zero_mask] = 1.0

        # Sample YX start positions
        indices = torch.multinomial(w_flat, 1).squeeze(1)  # (B,)
        y_starts = indices // valid_x
        x_starts = indices % valid_x

        # Z start positions: uniform random, or 0 if crop covers full Z
        if crop_z >= Z:
            z_starts = torch.zeros(B, dtype=torch.long, device=device)
        else:
            z_starts = torch.randint(0, Z - crop_z + 1, (B,), device=device)

        return z_starts, y_starts, x_starts

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply weighted spatial crop to all specified keys.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary with tensors of shape ``(B, C, Z, Y, X)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with cropped tensors.
        """
        d = dict(data)
        weight_map = d[self.w_key]

        if weight_map.ndim != 5:
            raise ValueError(f"BatchedRandWeightedCropd requires 5D input (B, C, Z, Y, X), got {weight_map.ndim}D.")
        B, _C, Z, Y, X = weight_map.shape
        crop_z, crop_y, crop_x = self._spatial_size
        if crop_z > Z:
            raise ValueError(f"spatial_size Z ({crop_z}) exceeds input Z ({Z}).")
        if crop_y > Y or crop_x > X:
            raise ValueError(f"spatial_size YX ({crop_y}, {crop_x}) exceeds input YX ({Y}, {X}).")

        z_starts, y_starts, x_starts = self._sample_crop_starts(weight_map)

        for key in self.key_iterator(d):
            img = d[key]
            d[key] = torch.stack(
                [
                    img[
                        b,
                        :,
                        z_starts[b] : z_starts[b] + crop_z,
                        y_starts[b] : y_starts[b] + crop_y,
                        x_starts[b] : x_starts[b] + crop_x,
                    ]
                    for b in range(B)
                ]
            )
        return d
