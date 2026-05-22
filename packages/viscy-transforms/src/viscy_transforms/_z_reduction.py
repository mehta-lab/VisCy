"""Channel-wise Z-reduction transforms for 2D training from 3D z-stacks."""

from __future__ import annotations

from collections.abc import Hashable

import torch
from monai.transforms import MapTransform
from torch import Tensor

__all__ = ["BatchedChannelWiseZReduction", "BatchedChannelWiseZReductiond"]


class BatchedChannelWiseZReduction:
    """Reduce the Z dimension of a ``(B, C, Z, Y, X)`` tensor.

    Label-free samples get the center z-slice; fluorescence samples get a
    max-intensity projection (MIP).  A per-sample boolean mask selects the
    strategy when the batch mixes both types.

    Parameters
    ----------
    default_strategy : str
        Strategy when no mask is provided: ``"mip"`` or ``"center"``.
    """

    def __init__(self, default_strategy: str = "mip") -> None:
        if default_strategy not in ("mip", "center"):
            raise ValueError(f"default_strategy must be 'mip' or 'center', got '{default_strategy}'")
        self.default_strategy = default_strategy

    def __call__(self, img: Tensor, is_labelfree: Tensor | None = None) -> Tensor:
        """Apply z-reduction.

        Parameters
        ----------
        img : Tensor
            Shape ``(B, C, Z, Y, X)``.
        is_labelfree : Tensor or None
            Boolean tensor of shape ``(B,)``.  ``True`` → center-slice,
            ``False`` → MIP.  When ``None``, ``default_strategy`` is used
            uniformly.

        Returns
        -------
        Tensor
            Shape ``(B, C, 1, Y, X)``.
        """
        z = img.shape[2]
        if z == 1:
            return img

        if is_labelfree is None:
            if self.default_strategy == "center":
                return img[:, :, z // 2 : z // 2 + 1]
            return img.amax(dim=2, keepdim=True)

        center = img[:, :, z // 2 : z // 2 + 1]
        mip = img.amax(dim=2, keepdim=True)
        mask = is_labelfree.view(-1, 1, 1, 1, 1)
        return torch.where(mask, center, mip)


class BatchedChannelWiseZReductiond(MapTransform):
    """Dict transform that applies channel-wise Z-reduction.

    In **bag-of-channels mode** each sample may represent a different channel.
    The transform reads a ``_is_labelfree`` boolean tensor from the data dict
    (injected by the datamodule) to decide per-sample strategy.

    In **all-channels mode** the dict keys identify channel type.  Pass
    ``labelfree_keys`` to specify which keys should use center-slice; all
    others get MIP.

    Parameters
    ----------
    keys : KeysCollection
        Keys of the image tensors to transform.
    labelfree_keys : list[str] or None
        Channel keys that should use center-slice (all-channels mode).
        When set, ``_is_labelfree`` in the data dict is ignored.
    default_strategy : str
        Fallback strategy when neither ``labelfree_keys`` nor
        ``_is_labelfree`` can determine the channel type.
    allow_missing_keys : bool
        If ``True``, skip keys not present in the data dict.
    """

    def __init__(
        self,
        keys,
        labelfree_keys: list[str] | None = None,
        default_strategy: str = "mip",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.labelfree_keys = set(labelfree_keys) if labelfree_keys is not None else None
        self.reducer = BatchedChannelWiseZReduction(default_strategy=default_strategy)

    def __call__(self, data: dict[Hashable, Tensor]) -> dict[Hashable, Tensor]:
        is_labelfree = data.pop("_is_labelfree", None)

        for key in self.key_iterator(data):
            if self.labelfree_keys is not None:
                # All-channels mode: strategy determined by key name.
                img = data[key]
                z = img.shape[2]
                if z == 1:
                    continue
                if key in self.labelfree_keys:
                    data[key] = img[:, :, z // 2 : z // 2 + 1]
                else:
                    data[key] = img.amax(dim=2, keepdim=True)
            else:
                data[key] = self.reducer(data[key], is_labelfree=is_labelfree)

        return data
