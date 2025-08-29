"""3D Gaussian blur transform for batched data."""

from typing import Iterable

import torch
from kornia.constants import BorderType
from kornia.filters import filter3d, get_gaussian_erf_kernel1d
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor


def filter3d_separable(
    input: Tensor,
    kernel_z: Tensor | None,
    kernel_y: Tensor | None,
    kernel_x: Tensor | None,
    border_type: str,
) -> Tensor:
    """Apply separable 3D filtering, skipping dimensions with None kernels.

    Parameters
    ----------
    input : Tensor
        Input tensor with shape (B, C, D, H, W).
    kernel_z : Tensor | None
        1D kernel for Z dimension with shape (B, kZ) or None to skip filtering.
    kernel_y : Tensor | None
        1D kernel for Y dimension with shape (B, kY) or None to skip filtering.
    kernel_x : Tensor | None
        1D kernel for X dimension with shape (B, kX) or None to skip filtering.
    border_type : str
        Border padding mode.

    Returns
    -------
    Tensor
        Filtered tensor with same shape as input.
    """
    out = input
    for kernel, shape in [
        (kernel_z, (slice(None), None, None, slice(None))),  # (B, 1, 1, kZ)
        (kernel_y, (slice(None), None, slice(None), None)),  # (B, 1, kY, 1)
        (kernel_x, (slice(None), slice(None), None, None)),  # (B, kX, 1, 1)
    ]:
        if kernel is not None:
            kernel_3d = kernel[shape]
            out = filter3d(out, kernel_3d, border_type)
    return out


class BatchedRandGaussianSmooth(RandomizableTransform):
    """Randomly apply 3D Gaussian blur to a batch of images.

    Uses separable 3D filtering for efficiency. Dimensions with sigma=0.0 are
    skipped to avoid NaN values from degenerate Gaussian kernels.

    Parameters
    ----------
    sigma_x : tuple[float, float] | float
        Standard deviation range for x-axis blur. If tuple, samples between min and max.
        If float, uses fixed value. Use 0.0 to skip filtering in this dimension.
    sigma_y : tuple[float, float] | float
        Standard deviation range for y-axis blur. If tuple, samples between min and max.
        If float, uses fixed value. Use 0.0 to skip filtering in this dimension.
    sigma_z : tuple[float, float] | float
        Standard deviation range for z-axis blur. If tuple, samples between min and max.
        If float, uses fixed value. Use 0.0 to skip filtering in this dimension.
    truncated : float
        Factor for automatic kernel size estimation: size = sigma * truncated.
        Default is 4.0.
    prob : float
        Probability of applying the transform. Default is 0.5.
    border_type : str
        Border mode for padding. Default is "constant" to match MONAI.
    """

    def __init__(
        self,
        sigma_x: tuple[float, float] | float = (1.0, 3.0),
        sigma_y: tuple[float, float] | float = (1.0, 3.0),
        sigma_z: tuple[float, float] | float = (1.0, 3.0),
        truncated: float = 4.0,
        prob: float = 0.5,
        border_type: str = "constant",
    ) -> None:
        RandomizableTransform.__init__(self, prob)

        self.truncated = truncated

        self.border_type = BorderType.get(border_type)

        # Handle sigma parameters
        self.sigma_params = []
        for sigma in [sigma_x, sigma_y, sigma_z]:
            if isinstance(sigma, (int, float)):
                self.sigma_params.append((sigma, sigma, False))  # (min, max, is_range)
            elif len(sigma) == 2:
                self.sigma_params.append((sigma[0], sigma[1], True))
            else:
                raise ValueError(
                    f"sigma must be float or tuple of 2 values, got {sigma}"
                )

    def randomize(self, data: Tensor) -> None:
        self._do_transform = torch.rand(data.shape[0], device=data.device) < self.prob

        n_transform = self._do_transform.sum()
        if n_transform > 0:
            # generate sigma for each dimension and each sample
            self._sigma_samples = torch.zeros(n_transform, 3, device=data.device)
            for i, (sigma_min, sigma_max, is_range) in enumerate(self.sigma_params):
                if is_range:
                    self._sigma_samples[:, i] = (
                        torch.rand(n_transform, device=data.device)
                        * (sigma_max - sigma_min)
                        + sigma_min
                    )
                else:
                    self._sigma_samples[:, i] = sigma_min

    def _estimate_kernel_size(self, sigma: float) -> int:
        """Estimate kernel size based on sigma truncation."""
        tail = int(max(float(sigma) * self.truncated, 0.5) + 0.5)
        return 2 * tail + 1  # Ensure odd size

    def _maybe_get_kernel(self, sigma_values: Tensor) -> Tensor | None:
        """Generate 1D Gaussian kernel if any sigma > 0, otherwise return None.

        Parameters
        ----------
        sigma_values : Tensor
            Sigma values for this dimension.
        """
        if not (sigma_values > 0).any():
            return None

        # Auto-estimate kernel size from maximum sigma in batch
        max_sigma = sigma_values.max().item()
        kernel_size = self._estimate_kernel_size(max_sigma)

        return get_gaussian_erf_kernel1d(kernel_size, sigma_values.view(-1, 1))

    def __call__(self, data: Tensor, randomize: bool = True) -> Tensor:
        if randomize:
            self.randomize(data)
        if not self._do_transform.any():
            return data
        transform_indices = torch.where(self._do_transform)[0]
        if len(transform_indices) == 0:
            return data
        data_to_transform = data[transform_indices]
        sigma_batch = self._sigma_samples

        # Create 1D kernels for each dimension
        kernel_z = self._maybe_get_kernel(sigma_batch[:, 0])  # Z dimension
        kernel_y = self._maybe_get_kernel(sigma_batch[:, 1])  # Y dimension
        kernel_x = self._maybe_get_kernel(sigma_batch[:, 2])  # X dimension
        blurred_data = filter3d_separable(
            data_to_transform,
            kernel_z,
            kernel_y,
            kernel_x,
            border_type=self.border_type.name.lower(),
        )

        out = data.clone()
        out[transform_indices] = blurred_data

        return out


class BatchedRandGaussianSmoothd(MapTransform, RandomizableTransform):
    """Apply random Gaussian blur to dictionary data.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to apply the transform to.
    sigma_x : tuple[float, float] | float
        Standard deviation range for x-axis blur.
    sigma_y : tuple[float, float] | float
        Standard deviation range for y-axis blur.
    sigma_z : tuple[float, float] | float
        Standard deviation range for z-axis blur.
    truncated : float
        Factor for automatic kernel size estimation.
        Default is 4.0.
    prob : float
        Probability of applying the transform. Default is 0.5.
    border_type : str
        Border mode for padding. Default is "constant" to match MONAI.
    allow_missing_keys : bool
        Whether to allow missing keys. Default is False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        sigma_x: tuple[float, float] | float = (1.0, 3.0),
        sigma_y: tuple[float, float] | float = (1.0, 3.0),
        sigma_z: tuple[float, float] | float = (1.0, 3.0),
        truncated: float = 4.0,
        prob: float = 0.5,
        border_type: str = "constant",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.blur_transform = BatchedRandGaussianSmooth(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            truncated=truncated,
            prob=prob,
            border_type=border_type,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        # Use the first tensor to randomize, then apply same random state to all keys
        first_key = next(iter(sample.keys()))
        self.blur_transform.randomize(sample[first_key])

        for key in self.key_iterator(sample):  # type: ignore[arg-type]
            sample[key] = self.blur_transform(sample[key], randomize=False)

        return sample
