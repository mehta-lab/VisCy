"""Katamari adapter for collection-based, single-marker inference.

The canonical Katamari API consumes ``[B, T, C, Z, Y, X]`` tensors.  VisCy's
triplet loader emits ``[B, C, Z, Y, X]`` tensors and the model matrix writes one
AnnData store per marker, so this adapter deliberately accepts exactly one
channel and one reduced Z plane.  It never pools or concatenates channels.

Intensity normalization is intentionally not performed here.  The triplet data
module has the source-FOV metadata required for Katamari's ``p1/p99_p1``
normalization; the checkpoint-owned inference transform applies the residual
``[-1, 2]`` clamp inside :meth:`KatamariModel.embed`.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence

import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.transforms import MapTransform
from torch import Tensor, nn


def mask_incompatible_flash_attn() -> bool:
    """Hide a broken optional FlashAttention wheel from downstream imports.

    Kornia treats an absent ``flash_attn`` package as optional and falls back
    to PyTorch SDPA, but some cluster wheels fail with ``ImportError`` (rather
    than ``ModuleNotFoundError``) when their glibc requirement is unmet. Keep a
    working installation available; mask only a package that cannot import.
    """
    try:
        import flash_attn  # noqa: F401
    except (ImportError, OSError) as error:
        sys.modules["flash_attn"] = None
        warnings.warn(
            f"Disabling incompatible optional flash_attn; using PyTorch attention fallback: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    return False


class KatamariFovNormalize(MapTransform):
    """Apply Katamari's per-FOV ``(x - p1) / (p99 - p1)`` scaling.

    This small inference-only transform avoids importing the full
    :mod:`viscy_transforms` public API. That API eagerly imports Kornia's
    augmentation stack, which is unrelated to Katamari inference and can pull
    optional FlashAttention binaries into otherwise portable environments.
    """

    is_spatial = False

    def __init__(self, keys: Sequence[str]) -> None:
        super().__init__(keys, allow_missing_keys=False)

    def __call__(self, sample: dict) -> dict:
        for key in self.keys:
            image = sample[key]
            stats = sample["norm_meta"][key]["fov_statistics"]
            p1 = stats["p1"].reshape(stats["p1"].shape + (1,) * (image.ndim - stats["p1"].ndim)).to(image)
            span = stats["p99_p1"].reshape(stats["p99_p1"].shape + (1,) * (image.ndim - stats["p99_p1"].ndim)).to(image)
            sample[key] = (image - p1) / (span + 1e-8)
        return sample


def prepare_single_channel_input(images: Tensor, model_input_size: Sequence[int]) -> Tensor:
    """Convert VisCy ``BCZYX`` crops to Katamari ``BTCZYX`` input.

    Parameters
    ----------
    images
        Float tensor shaped ``[B, 1, 1, Y, X]`` after the model-matrix Z
        reduction and source-FOV normalization.
    model_input_size
        Final ``(Y, X)`` size presented to Katamari.  The sampled physical crop
        remains controlled independently by ``--yx-patch-size``.
    """
    if images.ndim != 5:
        raise ValueError(f"Katamari expects VisCy BCZYX input, got shape {tuple(images.shape)}")
    if images.shape[1] != 1:
        raise ValueError(
            "Katamari model-matrix inference is single-channel; "
            f"received C={images.shape[1]}. Run one marker at a time."
        )
    if images.shape[2] != 1:
        raise ValueError(
            f"Katamari model-matrix inference requires Z reduction to one plane; received Z={images.shape[2]}."
        )
    if len(model_input_size) != 2 or any(int(size) <= 0 for size in model_input_size):
        raise ValueError("model_input_size must contain two positive integers")

    target_size = tuple(int(size) for size in model_input_size)
    images_2d = images[:, :, 0]
    if tuple(images_2d.shape[-2:]) != target_size:
        images_2d = F.interpolate(
            images_2d,
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    # BCYX -> B T C Z Y X, with singleton time and Z axes.
    return images_2d[:, None, :, None]


class KatamariSingleChannelEncoder(nn.Module):
    """Load a Katamari checkpoint and emit one embedding per marker crop."""

    def __init__(
        self,
        checkpoint: str,
        *,
        model_input_size: Sequence[int] = (224, 224),
        readout: str = "avg",
        use_teacher: bool = False,
    ) -> None:
        super().__init__()
        from projects.katamari.katamari.inference import KatamariModel

        self.model_input_size = tuple(int(size) for size in model_input_size)
        self.readout = readout
        self.use_teacher = use_teacher
        self.model = KatamariModel.from_checkpoint(
            checkpoint,
            device="cpu",
            precision="32-true",
            readout=readout,
            use_teacher=use_teacher,
        )
        if self.model.input_domain != "pre_normalized":
            raise ValueError(
                "This adapter applies Katamari's source-FOV p1/p99_p1 normalization "
                f"upstream, but checkpoint input_domain={self.model.input_domain!r}."
            )

    def forward(self, images: Tensor) -> Tensor:
        """Return ``[B, D]`` global embeddings for one marker."""
        katamari_input = prepare_single_channel_input(images.float(), self.model_input_size)
        return self.model.embed(katamari_input).global_embeddings


class KatamariPredictModule(LightningModule):
    """Minimal Lightning prediction wrapper compatible with ``EmbeddingWriter``."""

    def __init__(
        self,
        checkpoint: str,
        *,
        model_input_size: Sequence[int] = (224, 224),
        readout: str = "avg",
        use_teacher: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = KatamariSingleChannelEncoder(
            checkpoint,
            model_input_size=model_input_size,
            readout=readout,
            use_teacher=use_teacher,
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Embed the anchor crop while preserving its source index."""
        features = self.encoder(batch["anchor"])
        return {
            "features": features,
            "projections": features,
            "index": batch["index"],
        }
