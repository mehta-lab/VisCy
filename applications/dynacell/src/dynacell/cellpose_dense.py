"""Differentiable dense outputs of a frozen Cellpose-SAM model (flows + cell-probability logit).

Ported from the G4 segmenter gate (``experiments/2026-10-15_conditional-generation-review/
gates/G4/segmenters.py``), which verified it against stock ``CellposeModel.eval``: flows and
cell-probability logits within 2.9e-6 max abs on real iPSC frames, and central
finite-difference gradients within 2.5e-3 relative error (cpsam).

The input may be in any positive-affine intensity units (raw fluorescence, or a
normalized target or prediction): cellpose's ``normalize99`` (1st/99th percentile ->
[0, 1], no clip, flat image -> 0) is applied per image with its percentiles computed
under ``no_grad``, so gradients flow through the pixels and not through the statistics.
This mirrors how the eval feeds cpdino (``normalize=True`` on each side independently).

The network is the stock checkpoint in cellpose's model directory
(``CELLPOSE_LOCAL_MODELS_PATH`` or ``~/.cellpose/models``), loaded by absolute path so a
missing file raises instead of falling back to another model or a download.
"""

import numpy as np
import torch
import torch.nn.functional as F
from cellpose import models as cp_models
from cellpose import transforms as cp_transforms
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

__all__ = ["CellposeDense", "normalize99_diff"]


def normalize99_diff(x: Tensor) -> Tensor:
    """Apply cellpose ``normalize99`` per image, with the percentiles under ``no_grad``.

    Parameters
    ----------
    x : Tensor
        Images, shape ``(B, ...)``.

    Returns
    -------
    Tensor
        ``(x - p1) / (p99 - p1)`` per image, unclipped; 0 where the percentile
        range is below 1e-3 (cellpose's flat-image rule).
    """
    with torch.no_grad():
        flat = x.reshape(x.shape[0], -1).float()
        lo, hi = torch.quantile(flat, 0.01, dim=1), torch.quantile(flat, 0.99, dim=1)
        ok = (hi - lo) > 1e-3
        scale = torch.where(ok, 1.0 / (hi - lo).clamp_min(1e-12), torch.zeros_like(hi))
    shape = (-1, *([1] * (x.ndim - 1)))
    return (x - lo.view(shape)) * scale.view(shape)


class CellposeDense(nn.Module):
    """Frozen Cellpose-SAM network returning differentiable dense outputs.

    Reproduces ``core.run_net`` (non-augmented): the image in channel 0 of 3 (the
    others zero, as ``convert_image`` does), padded by ``get_pad_yx``, cut into
    256-px tiles at ``tile_overlap=0.1`` and blended with cellpose's taper mask. No
    diameter rescale, matching the eval's ``diameter=None``.

    It is loaded on the CPU, in eval mode, and follows the module's ``.to(device)``.
    The mode matters: cellpose builds the network in train mode, whose flows differ
    from stock inference by up to 1.7 (measured), so ``train()`` keeps it in eval even
    if this wrapper is ever a registered submodule. Parameter gradients are the caller's
    to switch off; ``SegConsistencyLoss`` does it.

    Parameters
    ----------
    model_name : str
        File name of a Cellpose-SAM checkpoint in cellpose's model directory, e.g.
        ``"cpsam"``. The cpdino family is the evaluation segmenter and is refused.
    use_bfloat16 : bool
        Load the weights in bfloat16, cellpose's default.
    checkpoint_tiles : int or None
        When the input needs a gradient, run the network on this many tiles at a time
        under activation checkpointing: the backward pass then holds one such group's
        activations (~5.7 GB per 256-px tile in bfloat16, measured on an A40) instead of
        every tile's, for one extra forward pass. ``None`` keeps all activations.
    """

    BSIZE = 256
    TILE_OVERLAP = 0.1

    def __init__(self, model_name: str, use_bfloat16: bool = True, checkpoint_tiles: int | None = 1) -> None:
        super().__init__()
        if checkpoint_tiles is not None and checkpoint_tiles < 1:
            raise ValueError(f"checkpoint_tiles must be >= 1 or None, got {checkpoint_tiles}")
        if model_name.startswith("cpdino"):
            raise ValueError("cpdino is the evaluation segmenter and must not be used as a training segmenter")
        path = cp_models.MODEL_DIR / model_name
        if not path.is_file():
            raise FileNotFoundError(f"cellpose model {model_name!r} not found at {path}")
        # Building the net random-initializes it before loading the checkpoint, which would
        # advance the global RNG that HCSDataModule's train/val split draws from later; fork it
        # so a segcons arm and its same-seed control split identically.
        with torch.random.fork_rng(devices=[]):
            cpm = cp_models.CellposeModel(
                device=torch.device("cpu"), pretrained_model=str(path), use_bfloat16=use_bfloat16
            )
        if cpm.backbone != "sam_vitl":
            raise ValueError(f"{model_name}: expected a SAM backbone, got {cpm.backbone}")
        self.net = cpm.net.eval()
        self.model_name = model_name
        self.checkpoint_tiles = checkpoint_tiles

    def train(self, mode: bool = True) -> "CellposeDense":
        """Keep the network in eval mode whatever ``mode`` is."""
        return super().train(False)

    @classmethod
    def tiling(cls, ly: int, lx: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
        """Return cellpose's padding and tile corners for a ``(ly, lx)`` image.

        ``get_pad_yx`` then ``make_tiles`` (non-augmented): the image is padded to a
        multiple of 16 of at least ``BSIZE`` plus a margin, so a 240-px image pads to
        exactly one 256-px tile while a 256-px one pads to 272 and needs 2 x 2 tiles.

        Returns
        -------
        tuple
            ``(yp1, yp2, xp1, xp2)`` padding and the ``(y, x)`` corner of every tile.
        """
        bs = cls.BSIZE
        yp1, yp2, xp1, xp2 = cp_transforms.get_pad_yx(ly, lx, min_size=(bs, bs))
        py, px = ly + yp1 + yp2, lx + xp1 + xp2
        ny = 1 if py <= bs else int(np.ceil((1.0 + 2 * cls.TILE_OVERLAP) * py / bs))
        nx = 1 if px <= bs else int(np.ceil((1.0 + 2 * cls.TILE_OVERLAP) * px / bs))
        ys = np.linspace(0, py - bs, ny).astype(int)
        xs = np.linspace(0, px - bs, nx).astype(int)
        return (yp1, yp2, xp1, xp2), [(int(a), int(c)) for a in ys for c in xs]

    def _net_flows(self, tiles: Tensor) -> Tensor:
        """Run the network on ``(N, 3, 256, 256)`` tiles; return its ``(N, 3, 256, 256)`` output."""
        return self.net(tiles)[0]

    def _taper(self, device: torch.device) -> Tensor:
        """Return cellpose's ``transforms._taper_mask`` for one tile."""
        bsize = self.BSIZE
        xm = torch.arange(bsize, device=device, dtype=torch.float32)
        xm = (xm - xm.mean()).abs()
        mask = 1 / (1 + torch.exp((xm - (bsize / 2 - 20)) / 7.5))
        return mask[:, None] * mask[None, :]

    def forward(self, x: Tensor, batch_size: int = 8) -> dict[str, Tensor]:
        """Compute the dense outputs of ``(B, 1, Y, X)`` images in any positive-affine units.

        Parameters
        ----------
        x : Tensor
            Images, shape ``(B, 1, Y, X)``.
        batch_size : int
            Tiles per network call without a gradient (see ``checkpoint_tiles`` with one).

        Returns
        -------
        dict of str to Tensor
            ``"flow"`` ``(B, 2, Y, X)`` (dY, dX) and ``"cellprob_logit"`` ``(B, 1, Y, X)``,
            float32.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected (B, 1, Y, X), got {tuple(x.shape)}")
        return self.dense_normalized(normalize99_diff(x.float()), batch_size=batch_size)

    def dense_normalized(self, xn: Tensor, batch_size: int = 8) -> dict[str, Tensor]:
        """Compute the dense outputs of already ``normalize99``-ed images.

        Exposed for finite-difference checks: the normalization statistics are
        stop-gradient by design, so only this path has an exact finite difference.

        Parameters
        ----------
        xn : Tensor
            Normalized images, shape ``(B, 1, Y, X)``.
        batch_size : int
            Tiles per network call without a gradient.

        Returns
        -------
        dict of str to Tensor
            As :meth:`forward`.
        """
        b, _, ly0, lx0 = xn.shape
        net_dtype = next(self.net.parameters()).dtype
        xn = torch.cat([xn, torch.zeros_like(xn), torch.zeros_like(xn)], dim=1)
        bs = self.BSIZE
        pads, coords = self.tiling(ly0, lx0)
        yp1, yp2, xp1, xp2 = pads
        xn = F.pad(xn, (xp1, xp2, yp1, yp2))
        ly, lx = xn.shape[-2:]
        tiles = torch.stack([xn[:, :, a : a + bs, c : c + bs] for a, c in coords], dim=1).reshape(-1, 3, bs, bs)
        checkpointed = self.checkpoint_tiles is not None and tiles.requires_grad
        step = self.checkpoint_tiles if checkpointed else batch_size
        outs = []
        for j in range(0, tiles.shape[0], step):
            chunk = tiles[j : j + step].to(net_dtype)
            y = checkpoint(self._net_flows, chunk, use_reentrant=False) if checkpointed else self._net_flows(chunk)
            outs.append(y.float())
        y = torch.cat(outs).reshape(b, len(coords), 3, bs, bs)
        taper = self._taper(xn.device)
        acc = xn.new_zeros((b, 3, ly, lx), dtype=torch.float32)
        wsum = xn.new_zeros((ly, lx), dtype=torch.float32)
        for i, (a, c) in enumerate(coords):
            acc = acc + F.pad(y[:, i] * taper, (c, lx - c - bs, a, ly - a - bs))
            wsum = wsum + F.pad(taper, (c, lx - c - bs, a, ly - a - bs))
        yf = (acc / wsum)[:, :, yp1 : ly - yp2, xp1 : lx - xp2]
        return {"flow": yf[:, :2], "cellprob_logit": yf[:, 2:3]}
