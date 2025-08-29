from collections.abc import Sequence

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor


class BatchedRandFlip(RandomizableTransform):
    """Randomly flip a batch of images along specified spatial axes.

    Parameters
    ----------
    spatial_axes : int | Sequence[int]
        The spatial axes along which to flip the images.
    prob : float
        The probability of applying the flip.
    """

    def __init__(self, spatial_axes: int | Sequence[int], prob=0.5) -> None:
        RandomizableTransform.__init__(self, prob)
        if isinstance(spatial_axes, int):
            spatial_axes = [spatial_axes]
        self.spatial_axes = spatial_axes

    def randomize(self, img: Tensor) -> None:
        self._flip_spatial_dims = (
            torch.rand(img.shape[0], len(self.spatial_axes)) < self.prob
        )

    def __call__(self, data: Tensor, randomize: bool = True):
        if randomize:
            self.randomize(data)
        if not self._flip_spatial_dims.any():
            return data

        out = torch.zeros_like(data)
        for i in range(data.shape[0]):
            flip_mask = self._flip_spatial_dims[i]
            if flip_mask.any():
                axis_indices = torch.where(flip_mask)[0]
                flip_dims = [self.spatial_axes[idx] + 1 for idx in axis_indices]
                out[i] = data[i].flip(dims=flip_dims)
            else:
                # NOTE: Copying one-by-one is slightly faster than vectorized indexing
                # possibly due to memory access pattern
                out[i] = data[i]

        return out


class BatchedRandFlipd(MapTransform, RandomizableTransform):
    """Apply random flips to batched data.

    This transform applies random flips along specified spatial axes to batched data
    with shape [B, C, D, H, W].

    Parameters
    ----------
    keys : list
        Keys to apply flipping to.
    spatial_axes : list of int, optional
        List of spatial axes to randomly flip (0=D, 1=H, 2=W). Default is [0, 1, 2].
    prob : float, optional
        Probability of applying each flip. Default is 0.5.
    allow_missing_keys : bool, optional
        Whether to allow missing keys. Default is False.
    """

    def __init__(
        self, keys, spatial_axes=[0, 1, 2], prob=0.5, allow_missing_keys=False
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.random_flip = BatchedRandFlip(spatial_axes=spatial_axes, prob=prob)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.random_flip.randomize(next(iter(sample.values())))
        for key in self.key_iterator(sample):
            sample[key] = self.random_flip(sample[key], randomize=False)
        return sample
