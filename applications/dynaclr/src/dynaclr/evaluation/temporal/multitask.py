"""Reusable compact multi-task TCN training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dynaclr.evaluation.temporal.model import CausalTCN


@dataclass(frozen=True)
class CompactTCNConfig:
    """Validated five-frame compact TCN training defaults."""

    hidden: int = 64
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.3
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    random_seed: int = 42
    class_balance_loss: bool = False


@dataclass
class MultitaskTCNFit:
    """Fitted TCN with a frozen training-only feature standardizer."""

    model: CausalTCN
    center: np.ndarray
    scale: np.ndarray
    config: CompactTCNConfig
    device: torch.device

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """Apply the frozen training-row z-score contract."""
        return ((np.asarray(sequences, dtype=np.float32) - self.center) / self.scale).astype(np.float32)

    def predict_proba(
        self,
        sequences: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return one sigmoid probability per sequence and target head."""
        values = self.transform(sequences)
        batch_size = batch_size or self.config.batch_size
        output = np.empty(
            (len(values), self.model.head.out_features),
            dtype=np.float32,
        )
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                logits = self.model(torch.from_numpy(values[start:stop]).to(self.device))
                if logits.ndim == 1:
                    logits = logits[:, None]
                output[start:stop] = torch.sigmoid(logits).cpu().numpy()
        return output


def fit_multitask_tcn(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_rows_by_head: Sequence[np.ndarray],
    *,
    config: CompactTCNConfig | None = None,
    device: str | torch.device | None = None,
) -> MultitaskTCNFit:
    """Fit a shared-trunk, independent-head causal TCN.

    Targets is rows by heads and may contain NaN for unlabeled endpoints.
    train_rows_by_head selects rows contributing to each head loss. The feature
    z-score is fit only on the union of those training rows.
    """
    config = config or CompactTCNConfig()
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape (rows, time, features)")
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.shape[0] != len(sequences):
        raise ValueError("targets and sequences must have the same rows")
    if len(train_rows_by_head) != targets.shape[1]:
        raise ValueError("Need exactly one training-row array per target head")

    selected: list[np.ndarray] = []
    for head, rows in enumerate(train_rows_by_head):
        rows = np.asarray(rows, dtype=int)
        if len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(sequences)):
            raise ValueError(f"Head {head} has empty or out-of-range training rows")
        if not np.isfinite(targets[rows, head]).all():
            raise ValueError(f"Head {head} training rows contain non-finite targets")
        selected.append(rows)
    union = np.unique(np.concatenate(selected))

    flat = sequences[union].reshape(-1, sequences.shape[-1])
    center = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        flat.std(axis=0, dtype=np.float64),
        1e-6,
    ).astype(np.float32)

    mask = np.zeros(
        (len(union), targets.shape[1]),
        dtype=np.float32,
    )
    for head, rows in enumerate(selected):
        mask[:, head] = np.isin(union, rows)
    train_targets = targets[union].copy()
    train_targets[~np.isfinite(train_targets)] = 0.0

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CausalTCN(
        in_dim=sequences.shape[-1],
        hidden=config.hidden,
        dilations=config.dilations,
        out_dim=targets.shape[1],
        dropout=config.dropout,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    positive_weight = None
    if config.class_balance_loss:
        positive = ((train_targets == 1) * mask).sum(axis=0)
        negative = ((train_targets == 0) * mask).sum(axis=0)
        if np.any(positive == 0) or np.any(negative == 0):
            raise ValueError("Every active head must contain both classes")
        positive_weight = torch.from_numpy((negative / positive).astype(np.float32)).to(resolved_device)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=positive_weight,
    )

    x_tensor = torch.from_numpy((sequences[union] - center) / scale)
    y_tensor = torch.from_numpy(train_targets)
    mask_tensor = torch.from_numpy(mask)
    generator = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor, mask_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y, batch_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(resolved_device))
            if logits.ndim == 1:
                logits = logits[:, None]
            losses = loss_fn(logits, batch_y.to(resolved_device))
            active = batch_mask.to(resolved_device)
            per_head = (losses * active).sum(dim=0) / active.sum(dim=0).clamp_min(1)
            per_head.mean().backward()
            optimizer.step()

    return MultitaskTCNFit(
        model=model,
        center=center,
        scale=scale,
        config=config,
        device=resolved_device,
    )
