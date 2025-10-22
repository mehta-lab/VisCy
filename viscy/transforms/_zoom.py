from typing import Sequence

import torch
from monai.transforms import MapTransform, Transform
from torch import Tensor
from typing_extensions import Literal


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
