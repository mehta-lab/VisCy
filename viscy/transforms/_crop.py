import torch
from monai.transforms import RandSpatialCrop
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
        super().__init__(
            roi_size, max_roi_size, random_center, random_size, lazy=False
        )
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
            
        # Use gather-based cropping for 3D batched operation
        B, C, D, H, W = img.shape
        d, h, w = self.roi_size
        
        # Extract start coordinates  
        z_starts = torch.tensor([slices[0].start for slices in self._batch_slices], device=img.device)
        y_starts = torch.tensor([slices[1].start for slices in self._batch_slices], device=img.device)
        x_starts = torch.tensor([slices[2].start for slices in self._batch_slices], device=img.device)
        
        # Clamp to valid ranges
        z_starts = z_starts.clamp(0, D - d)
        y_starts = y_starts.clamp(0, H - h)  
        x_starts = x_starts.clamp(0, W - w)
        
        # Create coordinate grids
        zs = (z_starts[:, None, None, None] + torch.arange(d, device=img.device)[None, :, None, None]).long()
        ys = (y_starts[:, None, None, None] + torch.arange(h, device=img.device)[None, None, :, None]).long()
        xs = (x_starts[:, None, None, None] + torch.arange(w, device=img.device)[None, None, None, :]).long()
        
        # Apply gather operations
        xz = img.gather(2, zs.view(B, 1, d, 1, 1).expand(B, C, d, H, W))
        xzy = xz.gather(3, ys.view(B, 1, 1, h, 1).expand(B, C, d, h, W))
        out = xzy.gather(4, xs.view(B, 1, 1, 1, w).expand(B, C, d, h, w))
        return out
