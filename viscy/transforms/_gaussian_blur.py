"""3D Gaussian blur transform for batched data."""

from typing import Iterable

import torch
from kornia.constants import BorderType
from kornia.filters import filter3d, get_gaussian_kernel3d
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor


class BatchedRandGaussianSmooth(RandomizableTransform):
    """Randomly apply 3D Gaussian blur to a batch of images.

    Parameters
    ----------
    sigma_x : tuple[float, float] | float
        Standard deviation range for x-axis blur. If tuple, samples between min and max.
        If float, uses fixed value.
    sigma_y : tuple[float, float] | float
        Standard deviation range for y-axis blur. If tuple, samples between min and max.
        If float, uses fixed value.
    sigma_z : tuple[float, float] | float
        Standard deviation range for z-axis blur. If tuple, samples between min and max.
        If float, uses fixed value.
    kernel_size : int | tuple[int, int, int]
        Size of the Gaussian kernel. Default is 3.
    prob : float
        Probability of applying the transform. Default is 0.5.
    border_type : str
        Border mode for padding. Default is "reflect".
    """

    def __init__(
        self,
        sigma_x: tuple[float, float] | float = (1.0, 3.0),
        sigma_y: tuple[float, float] | float = (1.0, 3.0),
        sigma_z: tuple[float, float] | float = (1.0, 3.0),
        kernel_size: tuple[int, int, int] | int = 3,
        prob: float = 0.5,
        border_type: str = "reflect",
    ) -> None:
        RandomizableTransform.__init__(self, prob)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
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
            # Sample sigma for each dimension and each batch element that will be transformed
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

    def __call__(self, data: Tensor, randomize: bool = True) -> Tensor:
        if randomize:
            self.randomize(data)

        if not self._do_transform.any():
            return data

        out = data.clone()
        transform_indices = torch.where(self._do_transform)[0]

        # Use the sampled sigma values
        sigma_batch = self._sigma_samples

        # Create Gaussian kernel for all samples at once
        kernel = get_gaussian_kernel3d(self.kernel_size, sigma_batch)

        # Apply blur to each sample
        for i, idx in enumerate(transform_indices):
            out[idx] = filter3d(
                data[idx : idx + 1],
                kernel[i : i + 1],
                border_type=self.border_type.name.lower(),
            ).squeeze(0)

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
    kernel_size : int | tuple[int, int, int]
        Size of the Gaussian kernel. Default is 3.
    prob : float
        Probability of applying the transform. Default is 0.5.
    border_type : str
        Border mode for padding. Default is "reflect".
    allow_missing_keys : bool
        Whether to allow missing keys. Default is False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        sigma_x: tuple[float, float] | float = (1.0, 3.0),
        sigma_y: tuple[float, float] | float = (1.0, 3.0),
        sigma_z: tuple[float, float] | float = (1.0, 3.0),
        kernel_size: tuple[int, int, int] | int = 3,
        prob: float = 0.5,
        border_type: str = "reflect",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.blur_transform = BatchedRandGaussianSmooth(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            kernel_size=kernel_size,
            prob=prob,
            border_type=border_type,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        # Use the first tensor to randomize, then apply same random state to all keys
        first_key = next(iter(sample.keys()))
        self.blur_transform.randomize(sample[first_key])

        for key in self.key_iterator(sample):
            sample[key] = self.blur_transform(sample[key], randomize=False)

        return sample
