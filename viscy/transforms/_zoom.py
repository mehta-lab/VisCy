from typing import Sequence

import torch
from monai.transforms import MapTransform, Transform
from torch import Tensor
from typing_extensions import Literal


class BatchedRescaleYX(Transform):
    """Rescale the YX spatial dimensions of a batched 5D tensor (B, C, Z, Y, X).

    Merges the batch and Z dimensions before calling
    ``torch.nn.functional.interpolate`` (bilinear, 4-D) and then restores the
    original shape.  This avoids trilinear interpolation so that ``antialias``
    can be used, which is important for downscaling.

    Parameters
    ----------
    target_yx_size : tuple[int, int]
        Target (Y, X) output size in pixels.
    mode : str, optional
        Interpolation mode passed to ``F.interpolate``, by default ``"bilinear"``.
    antialias : bool, optional
        Apply an anti-aliasing filter before downscaling, by default ``True``.
    """

    def __init__(
        self,
        target_yx_size: tuple[int, int],
        mode: str = "bilinear",
        antialias: bool = True,
    ) -> None:
        self.target_yx_size = target_yx_size
        self.mode = mode
        self.antialias = antialias

    def __call__(self, sample: Tensor) -> Tensor:
        b, c, z, y, x = sample.shape
        # Merge batch and Z for 4-D bilinear interpolation
        flat = sample.reshape(b * z, c, y, x)
        resized = torch.nn.functional.interpolate(
            flat.float(),
            size=self.target_yx_size,
            mode=self.mode,
            align_corners=False,
            antialias=self.antialias,
        )
        return resized.view(b, c, z, *self.target_yx_size)


class BatchedRescaleYXd(MapTransform):
    """Dictionary wrapper of :py:class:`BatchedRescaleYX`.

    Parameters
    ----------
    keys : Sequence[str]
        Keys to apply the transform to.
    target_yx_size : tuple[int, int]
        Target (Y, X) output size in pixels.
    mode : str, optional
        Interpolation mode, by default ``"bilinear"``.
    antialias : bool, optional
        Apply anti-aliasing filter before downscaling, by default ``True``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        target_yx_size: tuple[int, int],
        mode: str = "bilinear",
        antialias: bool = True,
    ) -> None:
        super().__init__(keys)
        self.transform = BatchedRescaleYX(
            target_yx_size=target_yx_size,
            mode=mode,
            antialias=antialias,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d


class BatchedZoom(Transform):
    "Batched zoom transform using ``torch.nn.functional.interpolate``."

    def __init__(
        self,
        scale_factor: float | tuple[float, float, float],
        mode: Literal[
            "nearest",
            "nearest-exact",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
        ],
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def __call__(self, sample: Tensor) -> Tensor:
        return torch.nn.functional.interpolate(
            sample,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )


class BatchedZoomd(MapTransform):
    "Dictionary wrapper of :py:class:`BatchedZoom`."

    def __init__(
        self,
        keys: Sequence[str],
        scale_factor: float | tuple[float, float, float],
        mode: Literal[
            "nearest",
            "nearest-exact",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
        ],
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> None:
        super().__init__(keys)
        self.transform = BatchedZoom(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d
