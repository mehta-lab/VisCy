import torch
from torch import Tensor, nn


class ChannelDropout(nn.Module):
    """Randomly zero out entire channels during training.

    Designed for (B, C, Z, Y, X) tensors in the GPU augmentation pipeline.
    Applied after the scatter/gather augmentation chain in on_after_batch_transfer.

    Parameters
    ----------
    channels : list[int]
        Channel indices to potentially drop.
    p : float
        Probability of dropping each specified channel per sample. Default: 0.5.
    """

    def __init__(self, channels: list[int], p: float = 0.5) -> None:
        super().__init__()
        self.channels = channels
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        out = x.clone()
        B = out.shape[0]
        for ch in self.channels:
            # Per-sample dropout mask
            mask = torch.rand(B, device=out.device) < self.p
            # Zero out channel ch for selected samples
            # mask shape: (B,), index into batch dimension
            out[mask, ch] = 0.0
        return out
