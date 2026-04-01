"""Batched 3D elastic deformation transform.

This module provides a GPU-efficient batched elastic deformation transform
for 3D microscopy data, simulating natural tissue deformations.
"""

import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from torch.nn.functional import conv3d
from typing_extensions import Iterable

__all__ = ["BatchedRand3DElasticd"]


class BatchedRand3DElasticd(MapTransform, RandomizableTransform):
    """Randomly apply 3D elastic deformation to batched data.

    Simulates natural tissue deformations commonly seen in biological
    structures. Uses a smoothed random displacement field to warp the
    input volume while preserving local structure.

    Parameters
    ----------
    keys : str | Iterable[str]
        Keys of the data dictionary to apply deformation to.
    sigma_range : tuple[float, float]
        Range for the Gaussian smoothing sigma applied to the displacement
        field. Higher values produce smoother, more gradual deformations.
    magnitude_range : tuple[float, float]
        Range for the displacement magnitude. Higher values produce
        stronger deformations.
    prob : float
        Probability of applying the transform to the entire batch.
        When triggered, all samples in the batch are deformed with
        independent random fields. Default: 0.1.
    mode : str
        Interpolation mode for grid sampling. Options: "bilinear", "nearest".
        Default: "bilinear".
    padding_mode : str
        Padding mode for out-of-bounds values. Options: "zeros", "border",
        "reflection". Default: "reflection".
    allow_missing_keys : bool
        Whether to allow missing keys in the data dictionary. Default: False.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with deformed tensors for specified keys.

    Examples
    --------
    >>> elastic = BatchedRand3DElasticd(
    ...     keys=["image", "label"],
    ...     sigma_range=(3.0, 5.0),
    ...     magnitude_range=(50.0, 100.0),
    ...     prob=0.5,
    ... )
    >>> sample = {"image": torch.randn(2, 1, 32, 64, 64)}
    >>> output = elastic(sample)
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        prob: float = 0.1,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.mode = mode
        self.padding_mode = padding_mode

    def _generate_elastic_field(self, shape: torch.Size, device: torch.device) -> Tensor:
        """Generate batched elastic deformation field."""
        batch_size = shape[0]
        spatial_dims = shape[2:]
        displacement_fields = []

        for _ in range(batch_size):
            sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
            magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])

            random_field = torch.randn((3,) + spatial_dims, device=device) * magnitude

            # Smooth with box filter approximation of Gaussian.
            kernel_size = int(sigma * 6) | 1  # Ensure odd
            if kernel_size > 1:
                kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=device) / (kernel_size**3)
                for dim in range(3):
                    random_field[dim : dim + 1] = conv3d(
                        random_field[dim : dim + 1].unsqueeze(0),
                        kernel,
                        padding=kernel_size // 2,
                    ).squeeze(0)

            displacement_fields.append(random_field)

        return torch.stack(displacement_fields)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply elastic deformation to the sample.

        The displacement field and sampling grid are generated once from the
        first key and reused for all keys so that paired inputs (e.g.
        source/target) receive identical spatial deformations.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing tensors with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with deformed tensors for specified keys.
        """
        d = dict(sample)
        self.randomize(None)
        if not self._do_transform:
            return d
        # Find the first present key; return unchanged if none match.
        first_key = self.first_key(d)
        if first_key not in d:
            return d
        ref = d[first_key]

        # Generate displacement field and sampling grid once.
        # displacement has shape (B, 3, D, H, W) where axis-1 is (D, H, W) order.
        displacement = self._generate_elastic_field(ref.shape, ref.device)
        coords = torch.meshgrid(
            torch.linspace(-1, 1, ref.shape[2], device=ref.device),
            torch.linspace(-1, 1, ref.shape[3], device=ref.device),
            torch.linspace(-1, 1, ref.shape[4], device=ref.device),
            indexing="ij",
        )
        # grid_sample expects grid[..., 0]=X(W), [..1]=Y(H), [..2]=Z(D).
        grid = torch.stack([coords[2], coords[1], coords[0]], dim=-1)
        grid = grid.unsqueeze(0).repeat(ref.shape[0], 1, 1, 1, 1)
        # Map displacement (D=0, H=1, W=2) → grid (X=0, Y=1, Z=2).
        grid[..., 0] += displacement[:, 2] / ref.shape[4] * 2  # W disp → X grid
        grid[..., 1] += displacement[:, 1] / ref.shape[3] * 2  # H disp → Y grid
        grid[..., 2] += displacement[:, 0] / ref.shape[2] * 2  # D disp → Z grid

        # Apply the same grid to every key.
        for key in self.key_iterator(d):
            d[key] = torch.nn.functional.grid_sample(
                d[key],
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=True,
            )

        return d
