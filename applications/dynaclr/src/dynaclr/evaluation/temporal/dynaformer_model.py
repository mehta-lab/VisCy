"""DynaFormer: a mixed-mask temporal head on frozen embeddings with a contrastive future objective."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dynaclr.evaluation.temporal.model import ContinuousRotaryEmbedding, RoPETransformerLayer


class DynaFormerHead(nn.Module):
    """Transformer over per-frame embeddings with continuous-time rotary positions.

    Whether a token sees only the past or also the future is decided entirely by
    the ``visible`` mask, so one set of weights serves causal and bidirectional
    inference.

    Parameters
    ----------
    in_dim : int
        Frozen embedding dimension.
    d_model, n_heads, n_layers, feedforward_multiplier, dropout : int or float
        Transformer hyperparameters.
    rope_min_period_hours, rope_max_period_hours : float
        Rotation period range for :class:`ContinuousRotaryEmbedding`.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        feedforward_multiplier: int = 2,
        dropout: float = 0.1,
        rope_min_period_hours: float = 1.0 / 6.0,
        rope_max_period_hours: float = 96.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_dim, d_model)
        self.aux_projection = nn.Linear(2, d_model)
        rope = ContinuousRotaryEmbedding(d_model // n_heads, rope_min_period_hours, rope_max_period_hours)
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(d_model, n_heads, feedforward_multiplier, dropout, rope=rope)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        z: torch.Tensor,
        hours: torch.Tensor,
        delta_t: torch.Tensor,
        division: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Contextualize every token.

        Parameters
        ----------
        z : torch.Tensor
            Standardized embeddings, ``(B, T, D)``.
        hours : torch.Tensor
            Acquisition time in hours, ``(B, T)``; only differences matter.
        delta_t : torch.Tensor
            Hours since the previous visible frame, ``(B, T)``.
        division : torch.Tensor
            Division-boundary flag, ``(B, T)``.
        visible : torch.Tensor
            Boolean ``(B, T)``; hidden and padded frames are ``False``.

        Returns
        -------
        torch.Tensor
            Belief states ``(B, T, d_model)``; read the anchor position.
        """
        aux = torch.stack((torch.log1p(delta_t.clamp_min(0)), division), dim=-1)
        x = self.input_projection(z) + self.aux_projection(aux)
        for layer in self.layers:
            x = layer(x, positions=hours, key_padding_mask=visible)
        return self.final_norm(x)


class HorizonDeltaMLP(nn.Module):
    """Predict the embedding change ``z_{p+k} - z_p`` for each horizon from ``h_p``."""

    def __init__(self, d_model: int, n_horizons: int, out_dim: int) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        hidden = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model + n_horizons, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return predicted deltas ``(B, K, out_dim)``."""
        batch = h.shape[0]
        onehot = torch.eye(self.n_horizons, device=h.device, dtype=h.dtype)
        expanded = torch.cat(
            (h[:, None, :].expand(batch, self.n_horizons, -1), onehot[None].expand(batch, -1, -1)),
            dim=-1,
        )
        return self.net(expanded)


def multi_horizon_infonce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    negatives: torch.Tensor,
    negatives_valid: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrastive future prediction with same-frame and in-batch negatives.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted future embeddings ``z_p + delta_hat``, ``(B, K, D)``.
    target : torch.Tensor
        True future embeddings, ``(B, K, D)``.
    target_valid : torch.Tensor
        Boolean ``(B, K)``.
    negatives : torch.Tensor
        Same-field same-frame negatives, ``(B, K, N, D)``.
    negatives_valid : torch.Tensor
        Boolean ``(B, K, N)``.
    temperature : float
        Softmax temperature.

    Returns
    -------
    tuple
        Scalar loss, top-1 accuracy per horizon ``(K,)``, valid count per horizon ``(K,)``.
    """
    pred = F.normalize(prediction, dim=-1)
    tgt = F.normalize(target, dim=-1)
    neg = F.normalize(negatives, dim=-1)
    positive = (pred * tgt).sum(-1, keepdim=True) / temperature
    hard = torch.einsum("bkd,bknd->bkn", pred, neg) / temperature
    hard = hard.masked_fill(~negatives_valid, float("-inf"))
    batch = torch.einsum("bkd,ckd->bkc", pred, tgt) / temperature
    B = pred.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=pred.device)[:, None, :]
    batch_valid = target_valid.permute(1, 0)[None, :, :] & ~eye
    batch = batch.masked_fill(~batch_valid, float("-inf"))
    logits = torch.cat((positive, hard, batch), dim=-1)
    log_prob = positive.squeeze(-1) - torch.logsumexp(logits, dim=-1)
    valid = target_valid & torch.isfinite(log_prob)
    loss = -(log_prob[valid]).mean() if valid.any() else pred.sum() * 0.0
    others = torch.cat((hard, batch), dim=-1).max(dim=-1).values
    top1 = ((positive.squeeze(-1) > others) & valid).sum(0).float() / valid.sum(0).clamp_min(1)
    return loss, top1.detach(), valid.sum(0)


def batch_to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a sampler/eval batch to torch on ``device``."""
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        out[key] = tensor.to(device)
    return out


def anchor_states(
    head: DynaFormerHead,
    batch: dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    """Gather window embeddings from ``features`` on device, run the head, and return the anchor state ``(B, d_model)``."""
    z = features[batch["window_rows"].clamp_min(0)]
    hidden = head(z, batch["hours"], batch["delta_t"], batch["division"].float(), batch["visible"])
    index = batch["p_index"].view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(1, index).squeeze(1)


@torch.no_grad()
def encode_rows(
    head: DynaFormerHead,
    features: torch.Tensor,
    rows: np.ndarray,
    window_builder: Callable[[np.ndarray], dict[str, np.ndarray]],
    *,
    device: torch.device,
    batch_size: int = 1024,
    delta_mlp: HorizonDeltaMLP | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode ``rows`` into anchor states, optionally with predicted future embeddings.

    ``features`` is the standardized embedding matrix already on ``device``.

    Returns
    -------
    tuple
        ``h`` of shape ``(rows, d_model)`` and, when ``delta_mlp`` is given, the
        predicted future embeddings ``z_p + delta_hat`` of shape ``(rows, K, D)``.
    """
    head.eval()
    rows = np.asarray(rows, dtype=int)
    states = np.empty((len(rows), head.d_model), dtype=np.float32)
    predicted = None
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = batch_to_tensors(window_builder(chunk), device)
        h = anchor_states(head, batch, features)
        states[start : start + len(chunk)] = h.cpu().numpy()
        if delta_mlp is not None:
            delta_mlp.eval()
            anchor_z = features[torch.from_numpy(chunk).to(device)]
            future = anchor_z[:, None, :] + delta_mlp(h)
            if predicted is None:
                predicted = np.empty((len(rows), *future.shape[1:]), dtype=np.float32)
            predicted[start : start + len(chunk)] = future.cpu().numpy()
    return states, predicted
