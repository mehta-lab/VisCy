import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRand3DElasticd(MapTransform, RandomizableTransform):
    """Batched 3D elastic deformation for biological structures."""

    def __init__(
        self,
        keys: str | Iterable[str],
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        spatial_size: tuple[int, int, int] | int | None = None,
        prob: float = 0.1,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode

    def _generate_elastic_field(
        self, shape: torch.Size, device: torch.device
    ) -> Tensor:
        """Generate batched elastic deformation field."""
        batch_size = shape[0]
        spatial_dims = shape[2:]  # Skip batch and channel

        # Generate random displacement field for each sample in batch
        displacement_fields = []

        for b in range(batch_size):
            if self.R.rand() < self.prob:
                # Random sigma and magnitude for this sample
                sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
                magnitude = self.R.uniform(
                    self.magnitude_range[0], self.magnitude_range[1]
                )

                # Generate random field
                random_field = (
                    torch.randn((3,) + spatial_dims, device=device) * magnitude
                )

                # Smooth with Gaussian kernel (simplified version)
                # In practice, you'd use proper Gaussian smoothing
                kernel_size = int(sigma * 6) | 1  # Ensure odd
                if kernel_size > 1:
                    from torch.nn.functional import conv3d

                    # Simple box filter approximation
                    kernel = torch.ones(
                        1, 1, kernel_size, kernel_size, kernel_size, device=device
                    ) / (kernel_size**3)
                    for dim in range(3):
                        random_field[dim : dim + 1] = conv3d(
                            random_field[dim : dim + 1].unsqueeze(0),
                            kernel,
                            padding=kernel_size // 2,
                        ).squeeze(0)

                displacement_fields.append(random_field)
            else:
                # No deformation
                displacement_fields.append(
                    torch.zeros((3,) + spatial_dims, device=device)
                )

        return torch.stack(displacement_fields)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            if self.R.rand() < self.prob:
                # Apply elastic deformation using grid_sample
                displacement = self._generate_elastic_field(data.shape, data.device)

                # Create sampling grid
                coords = torch.meshgrid(
                    torch.linspace(-1, 1, data.shape[2], device=data.device),
                    torch.linspace(-1, 1, data.shape[3], device=data.device),
                    torch.linspace(-1, 1, data.shape[4], device=data.device),
                    indexing="ij",
                )
                grid = torch.stack(
                    [coords[2], coords[1], coords[0]], dim=-1
                )  # xyz order
                grid = grid.unsqueeze(0).repeat(data.shape[0], 1, 1, 1, 1)

                # Add displacement (normalize to [-1, 1] range)
                for i in range(3):
                    grid[..., i] += displacement[:, i] / data.shape[2 + i] * 2

                # Apply transformation
                d[key] = torch.nn.functional.grid_sample(
                    data,
                    grid,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=True,
                )

        return d
