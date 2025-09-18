"""3D version of `kornia.augmentation._2d.intensity.gaussian_blur`."""

from collections.abc import Iterable
from typing import Any

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.intensity.base import IntensityAugmentationBase3D
from kornia.constants import BorderType
from kornia.filters import filter3d, get_gaussian_kernel3d
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor


class RandomGaussianBlur(IntensityAugmentationBase3D):
    """
    Random Gaussian Blur.

    Parameters
    ----------
    kernel_size : tuple[int, int, int] | int
        Kernel size.
    sigma : tuple[float, float, float] | Tensor
        Sigma.
    border_type : str, optional
        Border type. By default, "reflect".
    same_on_batch : bool, optional
        Whether to apply the same transformation to all batches. By default, False.
    p : float, optional
        Probability of applying the transformation. By default, 0.5.
    keepdim : bool, optional
        Whether to keep the dimensions of the input tensor. By default, False.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int, int] | int,
        sigma: tuple[float, float, float] | Tensor,
        border_type: str = "reflect",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.flags = {
            "kernel_size": kernel_size,
            "border_type": BorderType.get(border_type),
        }
        self._param_generator = rg.RandomGaussianBlurGenerator(sigma)

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        sigma = params["sigma"].unsqueeze(-1).expand(-1, 2)
        kernel = get_gaussian_kernel3d(
            kernel_size=self.flags["kernel_size"], sigma=sigma
        )
        return filter3d(input, kernel, border_type=self.flags["border_type"])


class BatchedRandGaussianBlurd(MapTransform, RandomizableTransform):
    """
    Batched Random Gaussian Blur.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to apply the transformation to.
    kernel_size : tuple[int, int] | int
        Kernel size.
    sigma : tuple[float, float]
        Sigma.
    border_type : str, optional
        Border type. By default, "reflect".
    same_on_batch : bool, optional
        Whether to apply the same transformation to all batches. By default, False.
    prob : float, optional
        Probability of applying the transformation. By default, 0.1.
    allow_missing_keys : bool, optional
        Whether to allow missing keys. By default, False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        kernel_size: tuple[int, int] | int,
        sigma: tuple[float, float],
        border_type: str = "reflect",
        same_on_batch: bool = False,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.filter = RandomGaussianBlur(
            kernel_size=kernel_size,
            sigma=sigma,
            border_type=border_type,
            same_on_batch=same_on_batch,
            p=prob,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.randomize(None)
        if not self._do_transform:
            return sample
        for key in self.keys:
            if key in sample:
                sample[key] = -sample[key]
        return sample
