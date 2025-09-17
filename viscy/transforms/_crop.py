import torch
from monai.transforms import (
    CenterSpatialCrop,
    Cropd,
    RandCropd,
    RandSpatialCrop,
)
from typing_extensions import Sequence


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

        windows = (
            img.contiguous()
            .unfold(2, crop_depth, 1)
            .unfold(3, crop_height, 1)
            .unfold(4, crop_width, 1)
        )

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
        cropper = BatchedRandSpatialCrop(
            roi_size, max_roi_size, random_center, random_size
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)

    def __call__(
        self, data: dict[str, torch.Tensor], lazy: bool | None = None
    ) -> dict[str, torch.Tensor]:
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
