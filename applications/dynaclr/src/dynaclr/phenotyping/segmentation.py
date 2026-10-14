"""Strict CP-DINO ViT-L adapter; model/working-grid identities are recorded."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dynaclr.phenotyping.config import Segmentation


def sha256_file(path: str | Path) -> str:
    """Hash a checkpoint file in bounded-memory blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(model, checkpoint: str | Path) -> None:
    """Verify the checkpoint AND the instantiated network, never set a backbone."""
    from cellpose.models import get_backbone

    if get_backbone(str(checkpoint)) != "dino_vitl" or getattr(model, "backbone", None) != "dino_vitl":
        raise RuntimeError("Loaded checkpoint/network is not CP-DINO DINOv3 ViT-L")
    from cellpose.vit import CPDINO

    if not isinstance(model.net, CPDINO):
        raise RuntimeError("Cellpose loaded an unexpected network class")
    # get_backbone checks encoder.cls_token and encoder.patch_embed's 1024 features.
    # Checking loaded weights as well rejects an incorrectly substituted encoder.
    state = model.net.state_dict()
    if "encoder.cls_token" not in state or state["encoder.patch_embed.proj.weight"].shape[0] != 1024:
        raise RuntimeError("Loaded DINO encoder does not have ViT-L weights")


class CPDinoSegmenter:
    """Run verified DINOv3 ViT-L inference on raw grayscale membrane projections."""

    def __init__(self, config: Segmentation):
        import torch
        from cellpose import models

        if not torch.cuda.is_available():
            raise RuntimeError("CP-DINO inference requires an allocated CUDA node")
        checkpoint = models.cache_model_path("cpdino") if config.checkpoint == "cpdino" else config.checkpoint
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)  # prevent Cellpose's silent SAM fallback
        self.model = models.CellposeModel(gpu=True, pretrained_model=str(checkpoint))
        verify_model(self.model, checkpoint)
        self.config = config
        self.identity = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "backbone": self.model.backbone,
            "network_class": type(self.model.net).__name__,
            "encoder": "DINOv3 ViT-L",
        }

    def __call__(self, membrane: np.ndarray, pixel_size_yx_um: tuple[float, float]) -> np.ndarray:
        """Segment raw grayscale using Cellpose normalization and restore native labels."""
        from skimage.transform import resize

        from dynaclr.phenotyping.algorithms import FrameSkip

        if membrane.ndim != 2 or not np.isfinite(membrane).all():
            raise FrameSkip("invalid_membrane_projection")
        native = membrane.shape
        target = self.config.target_pixel_size_um
        shape = (
            native
            if target is None
            else tuple(max(32, int(round(n * size / target))) for n, size in zip(native, pixel_size_yx_um))
        )
        if np.ptp(membrane) == 0:
            raise FrameSkip("uninformative_membrane_projection")
        # Faithful DynaCell CP-DINO recipe: raw grayscale, Cellpose's own
        # normalization, no external clipping/CLAHE/channel replication.
        image = membrane.astype(np.float32, copy=False)
        if shape != native:
            image = resize(image, shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        normalize = (
            True
            if tuple(self.config.normalize_percentiles) == (1.0, 99.0)
            else {"percentile": list(self.config.normalize_percentiles)}
        )
        masks, _, _ = self.model.eval(
            image,
            channel_axis=None,
            do_3D=False,
            normalize=normalize,
            diameter=None,
            batch_size=self.config.batch_size,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
        )
        if masks.shape != shape or not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0):
            raise RuntimeError("CP-DINO returned invalid working-grid labels")
        if shape == native:
            return masks.astype(np.uint32)
        return resize(masks, native, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
