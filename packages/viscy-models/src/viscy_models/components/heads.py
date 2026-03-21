"""Head modules for VisCy models.

Includes spatial heads (2D→3D projection) and embedding-space heads
(projection MLPs, cosine classifiers).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.utils import normal_init
from torch import Tensor, nn

from viscy_models.components.blocks import icnr_init
from viscy_models.schedule import cosine_anneal

__all__ = [
    "BaseHead",
    "ClassificationHead",
    "CosineClassifier",
    "MLP",
    "PixelToVoxelHead",
    "PixelToVoxelShuffleHead",
    "UnsqueezeHead",
]


class BaseHead(ABC, nn.Module):
    """Abstract base class for pluggable task heads in multi-task models.

    Each head is self-contained: it knows its batch key, computes its own
    loss, and can be registered in an ``nn.ModuleDict`` for generic iteration
    in the training loop.

    Call :meth:`get_weight` with the current epoch to get the (possibly scheduled)
    loss weight. Call :meth:`step` at the start of each epoch to advance the schedule.

    Parameters
    ----------
    head_name : str
        Name used for logging (e.g. ``"gene_ko"`` → ``loss/aux/gene_ko/train``).
    batch_key : str
        Key used to look up targets in the batch dict (``batch[batch_key]``).
        Decoupled from ``head_name`` so the logging name stays stable while
        the dataset key can vary across experiments.
    loss_weight : float
        Final (or constant) scalar weight applied to this head's loss.
    weight_schedule : {"cosine", "constant"}
        Schedule for loss weight. ``"cosine"`` ramps from ``weight_start``
        up to ``loss_weight`` over ``weight_warmup_epochs``. Default: ``"constant"``.
    weight_start : float
        Initial weight when using ``"cosine"`` schedule. Default: 0.0.
    weight_warmup_epochs : int
        Epochs over which to ramp up the weight. Default: 50.

    Notes
    -----
    Subclasses must implement :meth:`forward`, :meth:`compute_loss`, and
    :meth:`log_metrics`.
    """

    def __init__(
        self,
        head_name: str,
        batch_key: str,
        loss_weight: float = 1.0,
        weight_schedule: Literal["cosine", "constant"] = "constant",
        weight_start: float = 0.0,
        weight_warmup_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.head_name = head_name
        self.batch_key = batch_key
        self.loss_weight = loss_weight
        self.weight_schedule = weight_schedule
        self.weight_start = weight_start
        self.weight_warmup_epochs = weight_warmup_epochs
        self._current_weight = weight_start if weight_schedule == "cosine" else loss_weight

    def step(self, epoch: int) -> None:
        """Update the loss weight for the given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).
        """
        if self.weight_schedule == "cosine":
            self._current_weight = cosine_anneal(
                self.weight_start,
                self.loss_weight,
                epoch,
                self.weight_warmup_epochs,
            )

    def get_weight(self) -> float:
        """Return the current loss weight.

        Returns
        -------
        float
            Loss weight for the current epoch.
        """
        return self._current_weight

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Map backbone features to head output (logits, embeddings, etc.).

        Parameters
        ----------
        x : Tensor
            Backbone feature tensor of shape ``(B, embedding_dim)``.

        Returns
        -------
        Tensor
            Head output tensor.
        """

    @abstractmethod
    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Compute the head-specific loss.

        Parameters
        ----------
        y_hat : Tensor
            Head output from :meth:`forward`.
        y : Tensor
            Target tensor from the batch.

        Returns
        -------
        Tensor
            Scalar loss tensor.
        """

    @abstractmethod
    def log_metrics(self, out: dict, log_fn: callable, stage: str) -> None:
        """Log head-specific metrics via the Lightning module's log function.

        Parameters
        ----------
        out : dict
            Output dict containing at minimum ``"loss"``, ``"logits"``, and ``"y"``.
        log_fn : callable
            The Lightning module's ``self.log`` or ``self.log_dict``.
        stage : str
            One of ``"train"`` or ``"val"``.
        """


class ClassificationHead(BaseHead):
    """Classification head built on :class:`MLP` with top-k accuracy logging.

    Parameters
    ----------
    head_name : str
        Name used for logging (e.g. ``"gene_ko"``).
    batch_key : str
        Key to look up integer class labels in the batch dict (e.g. ``"gene_label"``).
    in_dims : int
        Input feature dimension (backbone embedding size).
    hidden_dims : int | list[int]
        Hidden layer width(s) passed to :class:`MLP`.
    num_classes : int
        Number of output classes.
    cosine_classifier : bool
        Use :class:`CosineClassifier` instead of a plain linear head.
    loss_weight : float
        Weight applied to this head's loss in the combined loss.
    top_k : int
        Compute top-k accuracy in addition to top-1. Default 5.
    weight_schedule : {"cosine", "constant"}
        Schedule for loss weight. ``"cosine"`` ramps from ``weight_start``
        up to ``loss_weight`` over ``weight_warmup_epochs``.
    weight_start : float
        Initial weight when using ``"cosine"`` schedule. Default: 0.0.
    weight_warmup_epochs : int
        Epochs over which to ramp up the weight. Default: 50.
    """

    def __init__(
        self,
        head_name: str,
        batch_key: str,
        in_dims: int,
        hidden_dims: int | list[int],
        num_classes: int,
        cosine_classifier: bool = True,
        loss_weight: float = 1.0,
        top_k: int = 5,
        weight_schedule: Literal["cosine", "constant"] = "constant",
        weight_start: float = 0.0,
        weight_warmup_epochs: int = 50,
    ) -> None:
        super().__init__(
            head_name=head_name,
            batch_key=batch_key,
            loss_weight=loss_weight,
            weight_schedule=weight_schedule,
            weight_start=weight_start,
            weight_warmup_epochs=weight_warmup_epochs,
        )
        self.mlp = MLP(
            in_dims=in_dims,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            cosine_classifier=cosine_classifier,
        )
        self.top_k = top_k

    def forward(self, x: Tensor) -> Tensor:
        """Map backbone features to class logits.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes)``.
        """
        return self.mlp(x)

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Cross-entropy loss.

        Parameters
        ----------
        y_hat : Tensor
            Logits of shape ``(B, num_classes)``.
        y : Tensor
            Integer class labels of shape ``(B,)``.

        Returns
        -------
        Tensor
            Scalar cross-entropy loss.
        """
        return F.cross_entropy(y_hat, y)

    def log_metrics(self, out: dict, log_fn: callable, stage: str) -> None:
        """Log loss, top-1, and top-k accuracy.

        Parameters
        ----------
        out : dict
            Must contain ``"loss"``, ``"logits"`` ``(B, num_classes)``, and ``"y"`` ``(B,)``.
        log_fn : callable
            Lightning module's ``self.log``.
        stage : str
            ``"train"`` or ``"val"``.
        """
        logits = out["logits"]
        y = out["y"]

        top1 = (logits.argmax(dim=1) == y).float().mean()
        topk = (logits.topk(self.top_k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean()

        log_fn(f"loss/aux/{self.head_name}/{stage}", out["loss"])
        log_fn(f"metrics/acc_top1/{self.head_name}/{stage}", top1)
        log_fn(f"metrics/acc_top{self.top_k}/{self.head_name}/{stage}", topk)


class CosineClassifier(nn.Module):
    """L2-normalised linear head with learnable temperature.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    num_classes : int
        Number of output classes.
    init_scale : float
        Initial value of the temperature scale (before log).
    learn_scale : bool
        Whether to make the temperature a learnable parameter.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        init_scale: float = 20.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.normal_(self.weight, std=0.01)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer("log_scale", torch.tensor(math.log(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return torch.exp(self.log_scale) * (x @ w.t())


class MLP(nn.Module):
    """Configurable MLP with optional classification head and penultimate-layer extraction.

    Supports two modes:

    - **Projection mode** (``num_classes=None``, default): maps embeddings to a
      projection space for contrastive loss. Output norm is applied to the final
      layer via ``norm``.
    - **Classification mode** (``num_classes`` set): adds a classification head
      (linear or cosine) on top of the backbone. Use :meth:`encode` to extract
      L2-normalised penultimate-layer representations.

    Parameters
    ----------
    in_dims : int
        Input feature dimension.
    hidden_dims : int | list[int]
        Hidden layer width. A single ``int`` gives one hidden layer (matching
        the legacy two-layer behaviour); a ``list`` gives one layer per element.
    out_dims : int
        Output dimension of the final projection layer (ignored in classification
        mode — the backbone output feeds directly into ``head``).
    norm : Literal["bn", "ln"]
        Normalization applied after each hidden layer. ``"bn"`` = BatchNorm1d,
        ``"ln"`` = LayerNorm.
    activation : Literal["relu", "gelu", "silu"]
        Hidden activation function.
    dropout : float
        Dropout rate applied after each hidden layer. ``0.0`` disables dropout.
    num_classes : int or None
        When set, adds a classification head and enables :meth:`encode`.
        When ``None`` (default), the MLP acts as a projection head.
    cosine_classifier : bool
        Use :class:`CosineClassifier` instead of a plain linear head.
        Only used when ``num_classes`` is set.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int | list[int],
        out_dims: int | None = None,
        norm: Literal["bn", "ln"] = "bn",
        activation: Literal["relu", "gelu", "silu"] = "relu",
        dropout: float = 0.0,
        num_classes: int | None = None,
        cosine_classifier: bool = True,
    ) -> None:
        if num_classes is None and out_dims is None:
            raise ValueError("out_dims is required in projection mode (num_classes=None).")
        super().__init__()
        self.input_dim = in_dims

        hidden_list = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)

        def _norm(dim: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm1d(dim)
            elif norm == "ln":
                return nn.LayerNorm(dim)
            raise ValueError(f"norm must be 'bn' or 'ln', got '{norm}'")

        def _act() -> nn.Module:
            if activation == "relu":
                return nn.ReLU(inplace=True)
            elif activation == "gelu":
                return nn.GELU()
            elif activation == "silu":
                return nn.SiLU(inplace=True)
            raise ValueError(f"activation must be 'relu', 'gelu', or 'silu', got '{activation}'")

        layers: list[nn.Module] = []
        prev_dim = in_dims
        for h in hidden_list:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(_norm(h))
            layers.append(_act())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        if num_classes is None:
            # Projection mode: final linear + norm
            layers.append(nn.Linear(prev_dim, out_dims))
            layers.append(_norm(out_dims))
            self.backbone = nn.Sequential(*layers)
            self.head: nn.Module | None = None
        else:
            # Classification mode: backbone stops before head
            self.backbone = nn.Sequential(*layers)
            if cosine_classifier:
                self.head = CosineClassifier(prev_dim, num_classes)
            else:
                self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone and optional head.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            Projected or classified output.
        """
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        """Return L2-normalised penultimate-layer representations.

        Only valid when ``num_classes`` was set at construction.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, in_dims)``.

        Returns
        -------
        Tensor
            L2-normalised backbone output of shape ``(B, hidden_dims[-1])``.

        Raises
        ------
        RuntimeError
            If called on a projection-mode MLP (``num_classes=None``).
        """
        if self.head is None:
            raise RuntimeError("encode() is only available in classification mode (num_classes != None).")
        return F.normalize(self.backbone(x), dim=1)


class PixelToVoxelHead(nn.Module):
    """Pixel-shuffle head that upsamples 2D features to 3D voxel output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int,
        expansion_ratio: int,
        pool: bool,
    ) -> None:
        super().__init__()
        first_scale = 2
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels // first_scale**2,
            scale_factor=first_scale,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )
        mid_channels = out_channels * expansion_ratio * 2**2
        self.conv = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_channels // first_scale**2 // (out_stack_depth + 2),
                out_channels=mid_channels,
                kernel_size=3,
                padding=(0, 1, 1),
            ),
            nn.Conv3d(mid_channels, out_channels * 2**2, 1),
        )
        normal_init(self.conv[0])
        icnr_init(self.conv[-1], 2, upsample_dims=2)
        self.out = nn.PixelShuffle(2)
        self.out_stack_depth = out_stack_depth

    def forward(self, x: Tensor) -> Tensor:
        """Upsample 2D feature map and reshape to 3D voxel output."""
        x = self.upsample(x)
        d = self.out_stack_depth + 2
        b, c, h, w = x.shape
        x = x.reshape((b, c // d, d, h, w))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.out(x)
        return x.transpose(1, 2)


class UnsqueezeHead(nn.Module):
    """Unsqueeze 2D (B, C, H, W) feature map to 3D (B, C, 1, H, W) output."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Add a singleton depth dimension to convert 2D to 3D."""
        x = x.unsqueeze(2)
        return x


class PixelToVoxelShuffleHead(nn.Module):
    """Pixel-shuffle head that reshapes 2D features into a 3D volume."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int = 5,
        xy_scaling: int = 4,
        pool: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_stack_depth * out_channels,
            scale_factor=xy_scaling,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample 2D feature map and reshape to 3D voxel output."""
        x = self.upsample(x)
        b, _, h, w = x.shape
        x = x.reshape(b, self.out_channels, self.out_stack_depth, h, w)
        return x
