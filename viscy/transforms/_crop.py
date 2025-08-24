import torch
from monai.transforms import RandSpatialCrop
from typing_extensions import Sequence


def batched_crop3d_unfold(x, z0, y0, x0, d, h, w):
    # x: (B,C,D,H,W) contiguous; z0,y0,x0: (B,) int
    B, _, D, H, W = x.shape
    z0 = z0.clamp(0, D-d); y0 = y0.clamp(0, H-h); x0 = x0.clamp(0, W-w)

    # Sliding-window view: (B,C, D-d+1, H-h+1, W-w+1, d, h, w)
    win = (x.contiguous()
             .unfold(2, d, 1)
             .unfold(3, h, 1)
             .unfold(4, w, 1))

    b = torch.arange(B, device=x.device)
    out = win[b, :, z0, y0, x0]             # (B,C,d,h,w)
    return out.contiguous()


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
        
        # Extract crop parameters from slices
        B = img.shape[0]
        z0 = torch.zeros(B, dtype=torch.long, device=img.device)
        y0 = torch.zeros(B, dtype=torch.long, device=img.device)
        x0 = torch.zeros(B, dtype=torch.long, device=img.device)
        
        # Get crop dimensions from the first batch (all should be the same)
        d = self._batch_slices[0][0].stop - self._batch_slices[0][0].start
        h = self._batch_slices[0][1].stop - self._batch_slices[0][1].start
        w = self._batch_slices[0][2].stop - self._batch_slices[0][2].start
        
        # Extract start positions for each batch item
        for i, slices in enumerate(self._batch_slices):
            z0[i] = slices[0].start
            y0[i] = slices[1].start
            x0[i] = slices[2].start
        
        # Apply batched cropping using the unfold method
        return batched_crop3d_unfold(img, z0, y0, x0, d, h, w)