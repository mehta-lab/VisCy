"""MorphEm foundation model wrapper for per-frame microscopy feature extraction."""

import torch.nn.functional as F
from torch import Tensor, nn

MORPHEM_DEFAULT = "CaicedoLab/MorphEm"


def _load_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Load a MorphEm checkpoint published with transformers 4.x remote code.

    With ``pretrained=False`` the same architecture is built from the published
    configuration with the model's own random initialization and no weights are
    loaded: the untrained baseline for the pretrained one.

    ``AutoModel.from_pretrained`` cannot load these checkpoints under
    transformers 5: weight loading reads ``all_tied_weights_keys``, which
    ``PreTrainedModel.post_init`` populates, and the published remote code
    predates that call. The model ties no weights, so the empty mapping
    ``post_init`` would have produced is supplied on the class instead. An
    instance attribute from a future ``post_init`` still takes precedence, so
    this stays correct once the checkpoint is republished.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_name)
    if not pretrained:
        return model_class(config)
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    return model_class.from_pretrained(model_name)


class MorphEmModel(nn.Module):
    """Wrap the pretrained MorphEm ViT-S for single-channel microscopy images.

    MorphEm is a DINOv2-style vision transformer trained on microscopy crops:
    one input channel, patch size 16, image size 224, embedding dimension 384.
    Unlike :class:`~viscy_models.foundation.openphenom.OpenPhenomModel`, both the
    class token and the patch tokens are returned, so a decoder can reconstruct
    an image from the patch grid while a temporal model consumes the class token.

    The backbone expects per-image standardized input, so :meth:`forward` applies
    an :class:`torch.nn.InstanceNorm2d` and resizes to ``img_size`` regardless of
    any dataset-level normalization applied upstream.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, by default ``"CaicedoLab/MorphEm"``.
    freeze : bool
        If ``True``, all backbone parameters are frozen and the module is kept
        in eval mode, by default ``False``.
    img_size : int
        Side length the input is resized to, by default 224 (the pretraining
        resolution). Must be divisible by the backbone patch size.
    pretrained : bool
        Load the published weights, by default ``True``. ``False`` keeps the
        architecture with a random initialization (seed it first for
        reproducible weights).

    Attributes
    ----------
    embed_dim : int
        Width of the returned tokens.
    grid_size : int
        Number of patches per side, i.e. ``img_size // patch_size``.
    """

    def __init__(
        self, model_name: str = MORPHEM_DEFAULT, freeze: bool = False, img_size: int = 224, pretrained: bool = True
    ) -> None:
        super().__init__()

        self.model = _load_backbone(model_name, pretrained=pretrained)
        self.embed_dim = int(getattr(self.model.config, "embed_dim", 384))
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        if img_size % self.patch_size != 0:
            raise ValueError(f"img_size {img_size} is not divisible by patch size {self.patch_size}.")
        self.img_size = int(img_size)
        self.grid_size = self.img_size // self.patch_size
        self.instance_norm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)

        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        else:
            # ``from_pretrained`` returns the backbone in eval mode. Leaving it
            # that way makes Lightning report the whole backbone as frozen at
            # the start of training even though it is being fine-tuned.
            self.model.train()

    def train(self, mode: bool = True) -> "MorphEmModel":
        """Override train to keep the backbone in eval when frozen."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocess_2d(self, x: Tensor) -> Tensor:
        """Standardize and resize a raw dataloader tensor for the backbone.

        Squeezes a singleton Z dimension (or takes the center slice), applies
        per-image instance normalization, and resizes to ``img_size``.

        Parameters
        ----------
        x : Tensor
            ``(B, 1, Z, Y, X)`` or ``(B, 1, Y, X)``.

        Returns
        -------
        Tensor
            ``(B, 1, img_size, img_size)`` ready for :meth:`forward`.
        """
        if x.ndim == 5:
            x = x[:, :, x.shape[2] // 2]
        if x.shape[1] != 1:
            raise ValueError(f"MorphEm is a single-channel backbone, got {x.shape[1]} channels.")
        x = self.instance_norm(x)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True
            )
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the MorphEm backbone on a preprocessed image batch.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, 1, img_size, img_size)``, see :meth:`preprocess_2d`.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(class_token, patch_tokens)`` of shape ``(B, embed_dim)`` and
            ``(B, grid_size ** 2, embed_dim)``.
        """
        features = self.model.forward_features(x)
        return features["x_norm_clstoken"], features["x_norm_patchtokens"]
