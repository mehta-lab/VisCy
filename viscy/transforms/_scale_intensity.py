from collections.abc import Iterable

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor


class BatchedRandScaleIntensity(RandomizableTransform):
    """Randomly scale intensity of a batch of images.

    The transform multiplies the input by (1 + factor), where factor is randomly
    sampled from the specified range.

    Parameters
    ----------
    factors : tuple[float, float] | float
        Range of scaling factors. If tuple, samples between (min, max).
        If single float, uses range (-factors, +factors). Default is 0.1.
    prob : float
        Probability of applying the transform to each sample. Default is 0.1.
    channel_wise : bool
        If True, scale each channel separately. Default is False.
    """

    def __init__(
        self,
        factors: tuple[float, float] | float = 0.1,
        prob: float = 0.1,
        channel_wise: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.channel_wise = channel_wise
        if isinstance(factors, (int, float)):
            self.factors_range = (-abs(factors), abs(factors))
        else:
            self.factors_range = (min(factors), max(factors))

    def randomize(self, data: Tensor) -> None:
        batch_size = data.shape[0]
        do_transform = torch.rand(batch_size, device=data.device) < self.prob
        factors_min, factors_max = self.factors_range
        if self.channel_wise and data.ndim > 2:
            shape = (batch_size, data.shape[1])
        else:
            shape = (batch_size,)
        factors = torch.empty(shape, device=data.device).uniform_(
            factors_min, factors_max
        )
        factors[~do_transform] = 0.0
        scale_factors = 1.0 + factors
        if self.channel_wise and data.ndim > 2:
            factors_shape = [scale_factors.shape[0], scale_factors.shape[1]] + [1] * (
                data.ndim - 2
            )
            self._broadcast_factors = scale_factors.view(*factors_shape)
        else:
            factors_shape = [scale_factors.shape[0]] + [1] * (data.ndim - 1)
            self._broadcast_factors = scale_factors.view(*factors_shape)

    def __call__(self, data: Tensor, randomize: bool = True) -> Tensor:
        """Apply intensity scaling to batch of images.

        Parameters
        ----------
        data : Tensor
            Input batch with shape (B, C, D, H, W) or (B, C, H, W).
        randomize : bool
            Whether to randomize parameters. Default is True.

        Returns
        -------
        Tensor
            Intensity-scaled batch with same shape as input.
        """
        if randomize:
            self.randomize(data)
        return data * self._broadcast_factors


class BatchedRandScaleIntensityd(MapTransform, RandomizableTransform):
    """Dictionary version of BatchedRandScaleIntensity.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys in the dictionary to apply the transform to.
    factors : tuple[float, float] | float
        Range of scaling factors. If tuple, samples between (min, max).
        If single float, uses range (-factors, +factors). Default is 0.1.
    prob : float
        Probability of applying the transform to each sample. Default is 0.1.
    channel_wise : bool
        If True, scale each channel separately. Default is False.
    allow_missing_keys : bool
        Whether to allow missing keys in the dictionary. Default is False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        factors: tuple[float, float] | float = 0.1,
        prob: float = 0.1,
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.intensity_transform = BatchedRandScaleIntensity(
            factors=factors,
            prob=prob,
            channel_wise=channel_wise,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply intensity scaling to dictionary data.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors to transform.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with transformed tensors.
        """
        first_key = next(iter(sample.keys()))
        self.intensity_transform.randomize(sample[first_key])

        for key in self.key_iterator(sample):
            sample[key] = self.intensity_transform(sample[key], randomize=False)

        return sample
