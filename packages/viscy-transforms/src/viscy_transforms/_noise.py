"""Batch-aware Gaussian noise transforms.

This module provides GPU-efficient Gaussian noise transforms that operate
directly on PyTorch tensors, avoiding unnecessary CPU-GPU transfers.
"""

from collections.abc import Iterable

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

__all__ = [
    "RandGaussianNoiseTensor",
    "RandGaussianNoiseTensord",
    "BatchedRandGaussianNoise",
    "BatchedRandGaussianNoised",
]


class RandGaussianNoiseTensor(RandGaussianNoise):
    """Add Gaussian noise directly to PyTorch tensors.

    Extends MONAI's RandGaussianNoise to generate noise on the same device
    as the input tensor, avoiding CPU-GPU data transfers.

    Parameters
    ----------
    prob : float
        Probability of applying noise. Inherited from parent class.
    mean : float
        Mean of the Gaussian distribution. Inherited from parent class.
    std : float
        Standard deviation (or max std if sample_std=True). Inherited.
    dtype : DTypeLike
        Output data type. Inherited from parent class.
    sample_std : bool
        If True, samples std uniformly from [0, std]. Inherited.

    Returns
    -------
    Tensor
        Input tensor with added Gaussian noise.

    See Also
    --------
    monai.transforms.RandGaussianNoise : Parent MONAI transform.
    BatchedRandGaussianNoise : Batched version with per-sample randomization.
    """

    def randomize(self, img: Tensor, mean: float | None = None) -> None:
        self._do_transform = self.R.rand() < self.prob
        if not self._do_transform:
            return None
        std = self.R.uniform(0, self.std) if self.sample_std else self.std
        self.noise: Tensor = torch.normal(
            self.mean if mean is None else mean,
            std,
            size=img.shape,
            device=img.device,
            dtype=img.dtype,
        )


class RandGaussianNoiseTensord(RandGaussianNoised):
    """Dictionary wrapper for tensor-based Gaussian noise.

    Applies RandGaussianNoiseTensor to specified keys in a data dictionary.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to add noise to.
    prob : float
        Probability of applying noise. Default: 0.1.
    mean : float
        Mean of the Gaussian distribution. Default: 0.0.
    std : float
        Standard deviation (or max std if sample_std=True). Default: 0.1.
    dtype : DTypeLike
        Output data type. Default: np.float32.
    allow_missing_keys : bool
        Whether to allow missing keys in data dictionary. Default: False.
    sample_std : bool
        If True, samples std uniformly from [0, std]. Default: True.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with noisy tensors for specified keys.

    See Also
    --------
    RandGaussianNoiseTensor : Underlying noise transform.
    BatchedRandGaussianNoised : Batched version for GPU efficiency.
    """

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
    """Add Gaussian noise to batched tensors with per-sample randomization.

    GPU-optimized noise transform that applies independent noise decisions
    and noise realizations to each sample in the batch. Shares a single
    noise pattern across selected samples for memory efficiency.

    Parameters
    ----------
    prob : float
        Probability of applying noise to each sample. Inherited.
    mean : float
        Mean of the Gaussian distribution. Inherited from parent class.
    std : float
        Standard deviation (or max std if sample_std=True). Inherited.
    dtype : DTypeLike
        Output data type. Inherited from parent class.
    sample_std : bool
        If True, samples std uniformly from [0, std] per batch. Inherited.

    Returns
    -------
    Tensor
        Input tensor with added Gaussian noise for selected samples.

    Notes
    -----
    Unlike the single-sample version, this transform makes independent
    random decisions for each sample in the batch, enabling efficient
    GPU training with diverse augmentations across the batch.

    See Also
    --------
    RandGaussianNoiseTensor : Single-sample version.
    BatchedRandGaussianNoised : Dictionary wrapper for this transform.
    """

    def randomize(self, img: Tensor, mean: float | None = None) -> None:
        self._do_transform = torch.rand(img.shape[0], device=img.device) < self.prob
        self._noise_batch_indices = torch.where(self._do_transform)[0]
        if self.sample_std:
            std = (torch.rand(len(self._noise_batch_indices), device=img.device) * self.std).view(
                -1, *([1] * (img.ndim - 1))
            )
        else:
            std = torch.tensor(self.std, device=img.device, dtype=img.dtype)
        if len(self._noise_batch_indices) == 0:
            return None
        if mean is None:
            mean = self.mean
        mean = torch.tensor(mean, device=img.device, dtype=img.dtype)
        noise_single = torch.normal(
            mean=0.0,
            std=1.0,
            size=img.shape[1:],
            device=img.device,
            dtype=img.dtype,
        )
        noise_single = torch.addcmul(mean, noise_single, std)
        self.noise_batch = noise_single.expand(len(self._noise_batch_indices), *img.shape[1:])

    def __call__(self, img: Tensor, mean: float | None = None, randomize: bool = True) -> Tensor:
        """Add Gaussian noise to the input tensor.

        Parameters
        ----------
        img : Tensor
            Input tensor with shape (B, C, D, H, W).
        mean : float | None
            Override mean of noise distribution. If None, uses instance mean.
        randomize : bool
            Whether to randomize noise parameters. Default: True.

        Returns
        -------
        Tensor
            Input tensor with added noise for selected batch samples.
        """
        if randomize:
            self.randomize(img, mean=self.mean if mean is None else mean)
        if len(self._noise_batch_indices) > 0:
            return img.index_add(0, self._noise_batch_indices, self.noise_batch)
        else:
            return img


class BatchedRandGaussianNoised(RandGaussianNoiseTensord):
    """Dictionary wrapper for batched Gaussian noise transform.

    GPU-optimized noise transform for dictionary data with per-sample
    randomization across the batch.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to add noise to.
    prob : float
        Probability of applying noise to each sample. Default: 0.1.
    mean : float
        Mean of the Gaussian distribution. Default: 0.0.
    std : float
        Standard deviation (or max std if sample_std=True). Default: 0.1.
    dtype : DTypeLike
        Output data type. Default: np.float32.
    allow_missing_keys : bool
        Whether to allow missing keys in data dictionary. Default: False.
    sample_std : bool
        If True, samples std uniformly from [0, std] per batch. Default: True.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with noisy tensors for specified keys.

    See Also
    --------
    BatchedRandGaussianNoise : Underlying batched noise transform.
    RandGaussianNoiseTensord : Single-sample version.
    """

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
        self.rand_gaussian_noise = BatchedRandGaussianNoise(
            mean=mean, std=std, prob=1.0, dtype=dtype, sample_std=sample_std
        )
