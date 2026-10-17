"""Vision foundation models as the per-frame encoder E of ``e_t = E(p_t)``, frozen unless trained with the model.

Every model sees the same single-channel phase crop, z-scored with whole-mosaic statistics, and gets the intensity
handling it was pretrained with:

- MorphEm and CELL-DINO were trained on single-channel microscopy with per-image normalization, so they keep it
  (MorphEm: instance norm; CELL-DINO: per-image z-score).
- DINOv3 and C-RADIO are RGB models trained on natural images. Phase is mapped to ``[0, 1]`` through a **fixed**
  window (``rgb_window``, in z-score units) and replicated to three channels, so the absolute phase, which tracks dry
  mass and grows through the cycle, survives into their input. DINOv3 then applies ImageNet statistics; C-RADIO
  normalizes internally.

The frame embedding is the model's global token (class token, or C-RADIO's summary) concatenated with the mean of the
patch tokens in the central ``center_patches`` x ``center_patches`` window. The crop is centred on the tracked nucleus,
so the second half describes that nucleus rather than the whole field.

``vit-untrained`` is the control without a foundation model: MorphEm's ViT-S/16 architecture and input handling with
the model's own random initialization (seeded) instead of the pretrained weights, so pretraining is the only
difference from ``morphem``. With ``trainable=True`` the same initialization is trained with the rest of the model
instead of kept frozen: a ViT learned from scratch on this task.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

from viscy_models.foundation.cell_dino import CellDinoModel
from viscy_models.foundation.morphem import MorphEmModel

__all__ = ["FOUNDATION_MODELS", "FoundationFrameEncoder"]

FoundationName = Literal["morphem", "cell-dino", "dinov3", "c-radio", "vit-untrained"]

#: Default weights per model: HuggingFace identifiers, or a local state dict for CELL-DINO.
FOUNDATION_MODELS: dict[str, str] = {
    "morphem": "CaicedoLab/MorphEm",
    "cell-dino": "/hpc/projects/organelle_phenotyping/models/CELL-DINO/model_weights/weights/"
    "channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "c-radio": "nvidia/C-RADIOv4-H",
    # Architecture source only: built with random weights.
    "vit-untrained": "CaicedoLab/MorphEm",
}
#: Seed of the untrained ViT's random initialization.
UNTRAINED_SEED = 0
#: Input statistics of the DINOv3 web-image (LVD-1689M) models, as in their image processor configuration.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: The encoders that can be trained: MorphEm's ViT-S, pretrained or not.
TRAINABLE = ("morphem", "vit-untrained")


class _Checkpointed(nn.Module):
    """Run a module under activation checkpointing: its activations are recomputed in the backward pass."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        """Apply the wrapped module to ``x``."""
        return checkpoint(self.module, x, use_reentrant=False)


class FoundationFrameEncoder(nn.Module):
    """A foundation model (or its untrained control) mapping one phase frame to one embedding, frozen by default.

    Parameters
    ----------
    name : {"morphem", "cell-dino", "dinov3", "c-radio", "vit-untrained"}
        Which foundation model; ``vit-untrained`` is MorphEm's architecture with random weights.
    weights : str | None
        HuggingFace identifier, or a local state dict for CELL-DINO. ``None`` uses
        :data:`FOUNDATION_MODELS`, by default ``None``.
    img_size : int
        Side length the frame is resized to before the backbone. Must be divisible by the patch size (16 for all four
        models), by default 224.
    center_patches : int
        Side of the central window of patch tokens averaged into the second half of the embedding, by default 4
        (64 px at 16 px patches, about one nucleus here).
    rgb_window : tuple[float, float]
        z-score range mapped to ``[0, 1]`` for the RGB models, by default ``(-4.0, 8.0)``: phase halos sit around
        -3 and dense nuclear regions reach about 8 at the mosaic's 0.21 rad standard deviation.
    trainable : bool
        Train the backbone with the rest of the model instead of freezing it, by default ``False``. MorphEm's ViT only
        (``morphem``, ``vit-untrained``). A trained encoder cannot be cached: it runs on every frame of every batch,
        with each transformer block under activation checkpointing so that a batch's activations fit in memory.

    Attributes
    ----------
    token_dim : int
        Width of the global token.
    patch_dim : int
        Width of a patch token.
    embed_dim : int
        Width of the embedding, ``token_dim + patch_dim``.
    """

    def __init__(
        self,
        name: FoundationName,
        weights: str | None = None,
        img_size: int = 224,
        center_patches: int = 4,
        rgb_window: tuple[float, float] = (-4.0, 8.0),
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if name not in FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model {name!r}, expected one of {sorted(FOUNDATION_MODELS)}.")
        if trainable and name not in TRAINABLE:
            raise ValueError(f"Only {TRAINABLE} can be trained, not {name!r}.")
        self.name = name
        self.weights = weights or FOUNDATION_MODELS[name]
        self.img_size = img_size
        self.center_patches = center_patches
        self.rgb_window = tuple(rgb_window)
        self.trainable = trainable
        if name in ("morphem", "vit-untrained"):
            # Seeded without touching the caller's random state, so the untrained weights are the same every run.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(UNTRAINED_SEED)
                self.backbone = MorphEmModel(
                    model_name=self.weights, freeze=not trainable, img_size=img_size, pretrained=name == "morphem"
                )
            if trainable:
                vit = self.backbone.model
                vit.blocks = nn.ModuleList(_Checkpointed(block) for block in vit.blocks)
            self.token_dim = self.patch_dim = self.backbone.embed_dim
        elif name == "cell-dino":
            self.backbone = CellDinoModel(weights_path=self.weights, img_size=img_size, freeze=True)
            self.token_dim = self.patch_dim = self.backbone.model.embed_dim
        elif name == "dinov3":
            self.backbone = AutoModel.from_pretrained(self.weights)
            self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
            self.n_prefix = 1 + self.backbone.config.num_register_tokens
            self.token_dim = self.patch_dim = self.backbone.config.hidden_size
        else:
            self.backbone = AutoModel.from_pretrained(self.weights, trust_remote_code=True)
            self.token_dim = int(self.backbone.radio_model.summary_dim)
            self.patch_dim = int(self.backbone.radio_model.model.embed_dim)
        self.patch_size = 16
        if img_size % self.patch_size:
            raise ValueError(f"img_size {img_size} is not divisible by the patch size {self.patch_size}.")
        self.grid = img_size // self.patch_size
        if not 0 < center_patches <= self.grid:
            raise ValueError(f"center_patches {center_patches} must lie in [1, {self.grid}].")
        self.embed_dim = self.token_dim + self.patch_dim
        if not trainable:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True) -> "FoundationFrameEncoder":
        """Keep a frozen backbone in eval mode."""
        return super().train(mode and self.trainable)

    def _to_unit_rgb(self, x: Tensor) -> Tensor:
        """Map z-scored phase to ``[0, 1]`` through the fixed window and replicate it to RGB."""
        low, high = self.rgb_window
        x = ((x - low) / (high - low)).clamp(0, 1)
        return x.expand(-1, 3, -1, -1)

    def tokens(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the backbone on z-scored phase frames.

        Parameters
        ----------
        x : Tensor
            ``(N, 1, Y, X)`` phase, z-scored with whole-mosaic statistics.

        Returns
        -------
        tuple[Tensor, Tensor]
            Global token ``(N, token_dim)`` and patch tokens ``(N, grid, grid, patch_dim)``.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected (N, 1, Y, X) phase frames, got {tuple(x.shape)}.")
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", antialias=True)
        if self.name in ("morphem", "vit-untrained"):
            token, patches = self.backbone(self.backbone.preprocess_2d(x))
        elif self.name == "cell-dino":
            features = self.backbone.model.forward_features(self.backbone.preprocess_2d(x))
            token, patches = features["x_norm_clstoken"], features["x_norm_patchtokens"]
        elif self.name == "dinov3":
            rgb = (self._to_unit_rgb(x) - self.image_mean) / self.image_std
            hidden = self.backbone(pixel_values=rgb).last_hidden_state
            token, patches = hidden[:, 0], hidden[:, self.n_prefix :]
        else:
            token, patches = self.backbone(self._to_unit_rgb(x))
        return token, patches.reshape(len(x), self.grid, self.grid, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Embed z-scored phase frames.

        Parameters
        ----------
        x : Tensor
            ``(N, 1, Y, X)`` phase, z-scored with whole-mosaic statistics.

        Returns
        -------
        Tensor
            ``(N, embed_dim)``: the global token, then the mean of the central patch tokens.
        """
        token, patches = self.tokens(x)
        start = (self.grid - self.center_patches) // 2
        window = patches[:, start : start + self.center_patches, start : start + self.center_patches]
        return torch.cat([token, window.mean(dim=(1, 2))], dim=-1)
