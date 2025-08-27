import numpy as np
import torch
from monai.transforms import (
    MapTransform,
    RandGaussianNoise,
    RandGaussianNoised,
    RandomizableTransform,
)
from numpy.typing import DTypeLike
from torch import Tensor
from typing_extensions import Iterable


class RandGaussianNoiseTensor(RandGaussianNoise):
    def randomize(self, img: Tensor, mean: float | None = None) -> None:
        self._do_transform = self.R.rand() < self.prob
        if not self._do_transform:
            return None
        std = self.R.uniform(0, self.std) if self.sample_std else self.std
        self.noise = torch.normal(
            self.mean if mean is None else mean,
            std,
            size=img.shape,
            device=img.device,
            dtype=img.dtype,
        )


class RandGaussianNoiseTensord(RandGaussianNoised):
    def __init__(
        self,
        keys: str | Iterable[str],
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        dtype: DTypeLike = np.float32,
        allow_missing_keys: bool = False,
        sample_std: bool = True,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_gaussian_noise = RandGaussianNoiseTensor(
            mean=mean, std=std, prob=1.0, dtype=dtype, sample_std=sample_std
        )


class BatchedRandGaussianNoise(RandGaussianNoiseTensor):
    def randomize(self, img: Tensor, mean: float | None = None) -> None:
        self._do_transform = torch.rand(img.shape[0], device=img.device) < self.prob
        self._noise_batch_indices = torch.where(self._do_transform)[0]
        if self.sample_std:
            std = (
                torch.rand(len(self._noise_batch_indices), device=img.device) * self.std
            ).view(-1, *([1] * (img.ndim - 1)))
        else:
            std = self.std
        if len(self._noise_batch_indices) == 0:
            return None
        noise_single = torch.normal(
            self.mean if mean is None else mean,
            self.std,
            size=img.shape[1:],
            device=img.device,
            dtype=img.dtype,
        )
        self.noise_batch = (
            noise_single.expand(len(self._noise_batch_indices), *img.shape[1:]) * std
        )

    def __call__(
        self, img: Tensor, mean: float | None = None, randomize: bool = True
    ) -> Tensor:
        if randomize:
            self.randomize(img, mean)
        if len(self._noise_batch_indices) > 0:
            return img.index_add(0, self._noise_batch_indices, self.noise_batch)
        else:
            return img
