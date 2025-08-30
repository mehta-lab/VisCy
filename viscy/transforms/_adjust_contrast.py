import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandAdjustContrast(RandomizableTransform):
    """Randomly adjust contrast of a batch of images using gamma transform.

    Each pixel/voxel intensity is updated as:
        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Parameters
    ----------
    gamma : tuple[float, float] | float
        Range of gamma values for contrast adjustment. If tuple, samples between min and max.
        If single number, value is picked from (0.5, gamma). Default is (0.5, 4.5).
    prob : float
        Probability of applying the transform. Default is 0.1.
    invert_image : bool
        Whether to invert the image before applying gamma augmentation. Default is False.
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

        # Handle gamma parameter validation and setup
        if isinstance(gamma, (int, float)):
            if gamma <= 0.0:
                raise ValueError(
                    f"gamma must be positive, got {gamma}"
                )
            # Use the provided gamma as both min and max for fixed gamma
            self.gamma_range = (gamma, gamma)
        elif len(gamma) != 2:
            raise ValueError("gamma should be a number or pair of numbers.")
        else:
            if any(g <= 0.0 for g in gamma):
                raise ValueError(
                    f"all gamma values must be positive, got {gamma}"
                )
            self.gamma_range = (min(gamma), max(gamma))

        self.invert_image = invert_image
        self.retain_stats = retain_stats

    def randomize(self, data: Tensor) -> None:
        """Generate random parameters for the batch."""
        self._do_transform = torch.rand(data.shape[0], device=data.device) < self.prob

        n_transform = self._do_transform.sum()
        if n_transform > 0:
            # Generate random gamma values for samples that will be transformed
            self._gamma_values = torch.empty(
                int(n_transform.item()), device=data.device, dtype=data.dtype
            )
            gamma_min, gamma_max = self.gamma_range
            self._gamma_values.uniform_(gamma_min, gamma_max)

    def __call__(self, data: Tensor, randomize: bool = True) -> Tensor:
        """Apply contrast adjustment to batch of images.

        Uses individual processing in a loop since it's faster than batched operations
        for this transform, but maintains the batched API for consistency.

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

        if not self._do_transform.any():
            return data

        # Get indices of samples to transform
        transform_indices = torch.where(self._do_transform)[0]
        if len(transform_indices) == 0:
            return data

        # Process each sample individually - this is faster than batched operations
        # for large 5D tensors due to memory bandwidth limitations
        out = data.clone()
        
        for i, sample_idx in enumerate(transform_indices):
            gamma_val = self._gamma_values[i].item()  # Convert tensor to float
            
            # Apply transform to individual sample using fastest approach
            sample = data[sample_idx]
            transformed_sample = self._adjust_contrast_individual(sample, gamma_val)
            out[sample_idx] = transformed_sample

        return out

    def _adjust_contrast_individual(self, data: Tensor, gamma: float) -> Tensor:
        """Apply contrast adjustment to a single image using direct PyTorch operations.
        
        This is faster than kornia for individual samples and avoids external dependencies.

        Parameters
        ----------
        data : Tensor
            Input image with shape (C, D, H, W) or (C, H, W).
        gamma : float
            Gamma value for contrast adjustment.

        Returns
        -------
        Tensor
            Contrast-adjusted image with same shape as input.
        """
        # Invert image if requested
        if self.invert_image:
            data = -data

        # Store original stats if retain_stats is True
        if self.retain_stats:
            original_mean = data.mean()
            original_std = data.std()

        # Normalize to [0, 1] range
        data_min = data.min()
        data_max = data.max()
        epsilon = 1e-7
        data_range = data_max - data_min
        normalized = (data - data_min) / (data_range + epsilon)

        # Apply gamma correction with direct PyTorch operations
        result = torch.pow(normalized, gamma)

        # Denormalize back to original range
        result = result * data_range + data_min

        # Restore original statistics if requested
        if self.retain_stats:
            # Zero mean and normalize
            result_mean = result.mean()
            result_std = result.std()
            result = (result - result_mean) / (result_std + 1e-8)
            # Restore old mean and standard deviation
            result = original_std * result + original_mean

        # Invert back if requested
        if self.invert_image:
            result = -result

        return result


class BatchedRandAdjustContrastd(MapTransform, RandomizableTransform):
    """Apply random contrast adjustment to dictionary data.

    This transform applies random contrast adjustment using gamma transform to batched data
    with shape [B, C, D, H, W] or [B, C, H, W].

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys to apply the transform to.
    gamma : tuple[float, float] | float
        Range of gamma values for contrast adjustment. If tuple, samples between min and max.
        If single number, value is picked from (0.5, gamma). Default is (0.5, 4.5).
    prob : float
        Probability of applying the transform. Default is 0.1.
    invert_image : bool
        Whether to invert the image before applying gamma augmentation. Default is False.
    retain_stats : bool
        If True, applies scaling and offset to maintain original mean and std. Default is False.
    allow_missing_keys : bool
        Whether to allow missing keys. Default is False.
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
        """Apply transform to sample dictionary."""
        # Use the first tensor to randomize, then apply same random state to all keys
        first_key = next(iter(sample.keys()))
        self.contrast_transform.randomize(sample[first_key])

        for key in self.key_iterator(sample):  # type: ignore[arg-type]
            sample[key] = self.contrast_transform(sample[key], randomize=False)

        return sample
