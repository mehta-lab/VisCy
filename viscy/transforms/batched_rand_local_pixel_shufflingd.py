import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandLocalPixelShufflingd(MapTransform, RandomizableTransform):
    """Apply random local pixel shuffling to simulate texture variations.

    Shuffles pixels within small local patches to add texture noise.

    Parameters
    ----------
    keys : str or Iterable[str]
        Keys of the corresponding items to be transformed.
    patch_size : int, optional
        Size of local patches for pixel shuffling, by default 3.
    shuffle_prob : float, optional
        Probability of shuffling within patches, by default 0.1.
    prob : float, optional
        Probability of applying the transform, by default 0.1.
    allow_missing_keys : bool, optional
        Whether to ignore missing keys, by default False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        patch_size: int = 3,
        shuffle_prob: float = 0.1,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.patch_size = patch_size
        self.shuffle_prob = shuffle_prob

    def _shuffle_patches(self, data: Tensor) -> Tensor:
        """Randomly shuffle pixels within local patches."""
        batch_size, channels, d, h, w = data.shape
        result = data.clone()

        # Compute maximum number of patches
        max_patches = int(self.shuffle_prob * (d * h * w) / (self.patch_size**3))

        for b in range(batch_size):
            if self.R.rand() < self.prob:
                # Generate random patch locations
                num_patches = min(max_patches, 100)
                z_coords = torch.randint(
                    0, max(1, d - self.patch_size + 1), (num_patches,)
                )
                y_coords = torch.randint(
                    0, max(1, h - self.patch_size + 1), (num_patches,)
                )
                x_coords = torch.randint(
                    0, max(1, w - self.patch_size + 1), (num_patches,)
                )

                for i in range(num_patches):
                    z, y, x = z_coords[i].item(), y_coords[i].item(), x_coords[i].item()

                    # Extract and shuffle patch
                    patch = result[
                        b,
                        :,
                        z : z + self.patch_size,
                        y : y + self.patch_size,
                        x : x + self.patch_size,
                    ]
                    original_shape = patch.shape
                    flat_patch = patch.reshape(channels, -1)

                    # Create random permutation
                    perm = torch.randperm(flat_patch.shape[1], device=data.device)
                    shuffled_patch = flat_patch[:, perm].reshape(original_shape)

                    # Put back shuffled patch
                    result[
                        b,
                        :,
                        z : z + self.patch_size,
                        y : y + self.patch_size,
                        x : x + self.patch_size,
                    ] = shuffled_patch

        return result

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply pixel shuffling to sample data.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing image tensors to transform.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with pixel-shuffled tensors.
        """
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            d[key] = self._shuffle_patches(data)

        return d
