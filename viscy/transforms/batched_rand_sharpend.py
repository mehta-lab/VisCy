import torch
import torch.nn.functional as F
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandSharpend(MapTransform, RandomizableTransform):
    """Batched random sharpening for microscopy images."""

    def __init__(
        self,
        keys: str | Iterable[str],
        alpha_range: tuple[float, float] = (0.1, 0.5),
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.alpha_range = alpha_range
        # Cached kernel for reuse
        self._cached_kernel = None
        self._cached_device = None

    def _get_sharpen_kernel(self, device: torch.device, channels: int) -> Tensor:
        """Get 3D sharpening kernel for all channels."""
        if self._cached_kernel is None or self._cached_device != device:
            # Create kernel for all channels
            kernel = torch.zeros((channels, 1, 3, 3, 3), device=device)
            kernel[:, 0, 1, 1, 1] = 7.0  # Center
            # Neighbors
            kernel[:, 0, 0, 1, 1] = -1.0
            kernel[:, 0, 2, 1, 1] = -1.0
            kernel[:, 0, 1, 0, 1] = -1.0
            kernel[:, 0, 1, 2, 1] = -1.0
            kernel[:, 0, 1, 1, 0] = -1.0
            kernel[:, 0, 1, 1, 2] = -1.0
            self._cached_kernel = kernel
            self._cached_device = device
        return self._cached_kernel

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]  # Shape: [B, C, D, H, W]

            if self.R.rand() < self.prob:
                batch_size, channels = data.shape[:2]

                # Generate random alphas for the batch
                alphas = torch.empty(batch_size, device=data.device)
                for b in range(batch_size):
                    alphas[b] = self.R.uniform(self.alpha_range[0], self.alpha_range[1])

                # Get kernel
                kernel = self._get_sharpen_kernel(data.device, channels)

                # Reshape data for group convolution
                data_reshaped = data.reshape(-1, 1, *data.shape[2:])

                # Apply convolution to all batch*channel at once
                filtered = F.conv3d(
                    data_reshaped,
                    kernel.reshape(-1, 1, 3, 3, 3),
                    padding=1,
                    groups=channels,
                )

                # Reshape back to original batch format
                filtered = filtered.reshape(batch_size, channels, *data.shape[2:])

                # Apply alpha blending
                alphas = alphas.view(batch_size, 1, 1, 1, 1)
                d[key] = (1 - alphas) * data + alphas * filtered

        return d
