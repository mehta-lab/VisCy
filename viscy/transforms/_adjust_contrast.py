import torch
from monai.transforms import MapTransform, RandomizableTransform
from monai.transforms.intensity.array import AdjustContrast
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandAdjustContrast(RandomizableTransform):
    """Randomly adjust contrast of a batch of images using gamma transform.

    Parameters
    ----------
    gamma : tuple[float, float] | float
        Range of gamma values for contrast adjustment. If tuple, samples between min and max.
        If single float, uses that exact value. Must be positive. Default is (0.5, 4.5).
    prob : float
        Probability of applying the transform to each sample. Default is 0.1.
    invert_image : bool
        Whether to invert the image before applying gamma correction. Default is False.
    retain_stats : bool
        If True, applies scaling and offset to maintain original mean and std. Default is False.
    """

    def __init__(
        self,
        gamma: tuple[float, float] | float = (0.5, 4.5),
        prob: float = 0.1,
        invert_image: bool = False,
        retain_stats: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.gamma = gamma
        self.invert_image = invert_image
        self.retain_stats = retain_stats
        if isinstance(gamma, (int, float)):
            self.gamma_range = (gamma, gamma)
        elif isinstance(gamma, tuple) and len(gamma) == 2:
            self.gamma_range = (min(gamma), max(gamma))
        else:
            raise ValueError("Gamma must be a float or a tuple of two floats.")
        if self.gamma_range[0] <= 0.0:
            raise ValueError("Gamma must be a positive value.")

    def randomize(self, data: Tensor) -> None:
        batch_size = data.shape[0]
        self._do_transform = torch.rand(batch_size) < self.prob
        gamma_min, gamma_max = self.gamma_range
        self._gamma_values = torch.empty(batch_size).uniform_(gamma_min, gamma_max)

    def __call__(self, data: Tensor, randomize: bool = True) -> Tensor:
        """Apply contrast adjustment to batch of images.

        Parameters
        ----------
        data : Tensor
            Input batch with shape (B, C, D, H, W) or (B, C, H, W).
        randomize : bool
            Whether to randomize parameters. Default is True.

        Returns
        -------
        Tensor
            Contrast-adjusted batch with same shape as input.
        """
        if randomize:
            self.randomize(data)

        out = torch.empty_like(data)
        for i in range(data.shape[0]):
            sample = data[i]
            if self._do_transform[i]:
                # Create MONAI transform with random gamma for this sample
                gamma_val = self._gamma_values[i].item()
                monai_transform = AdjustContrast(
                    gamma=gamma_val,
                    invert_image=self.invert_image,
                    retain_stats=self.retain_stats,
                )
                out[i] = monai_transform(sample)
            else:
                out[i] = sample
        return out


class BatchedRandAdjustContrastd(MapTransform, RandomizableTransform):
    """Dictionary version of BatchedRandAdjustContrast.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys in the dictionary to apply the transform to.
    gamma : tuple[float, float] | float
        Range of gamma values for contrast adjustment. If tuple, samples between min and max.
        If single float, uses that exact value. Must be positive. Default is (0.5, 4.5).
    prob : float
        Probability of applying the transform to each sample. Default is 0.1.
    invert_image : bool
        Whether to invert images before applying gamma correction. Default is False.
    retain_stats : bool
        If True, applies scaling and offset to maintain original mean and std. Default is False.
    allow_missing_keys : bool
        Whether to allow missing keys in the dictionary. Default is False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        gamma: tuple[float, float] | float = (0.5, 4.5),
        prob: float = 0.1,
        invert_image: bool = False,
        retain_stats: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.contrast_transform = BatchedRandAdjustContrast(
            gamma=gamma,
            prob=prob,
            invert_image=invert_image,
            retain_stats=retain_stats,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply contrast adjustment to dictionary data.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors to transform.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with transformed tensors.
        """
        # Use the first tensor to generate random parameters, then apply
        # the same random state to all specified keys for consistency
        first_key = next(iter(sample.keys()))
        self.contrast_transform.randomize(sample[first_key])

        for key in self.key_iterator(sample):
            sample[key] = self.contrast_transform(sample[key], randomize=False)

        return sample
