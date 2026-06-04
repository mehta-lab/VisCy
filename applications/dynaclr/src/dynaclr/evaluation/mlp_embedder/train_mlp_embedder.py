"""CLI for training an MLP embedder on cell embeddings.

Usage:
    dynaclr train-mlp-embedder -c path/to/config.yaml
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import anndata as ad
import click
import numpy as np
import torch
import torch.nn as nn
import wandb
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from viscy_models.components.heads import MLP
from viscy_utils.cli_utils import load_config


class MlpEmbedderTrainConfig(BaseModel):
    """Configuration for MLP embedder training.

    Parameters
    ----------
    embeddings_path : str
        Path to the AnnData zarr store with embeddings in ``.X``.
    target_col : str
        Column in ``adata.obs`` to use as classification target.
        Must already be present (e.g. appended by apply-linear-classifier).
    hidden_dims : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout rate after each hidden layer.
    cosine_classifier : bool
        Use cosine similarity head instead of plain linear.
    batch_norm : bool
        Include BatchNorm1d after each hidden layer.
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    batch_size : int
        Mini-batch size.
    val_fraction : float
        Fraction of samples held out for validation.
    seed : int
        Random seed.
    wandb_project : str
        W&B project name.
    wandb_entity : Optional[str]
        W&B entity (username or team).
    output_path : str
        Path to save the trained model weights (``.pt``).
    """

    embeddings_path: str = Field(..., min_length=1)
    target_col: str = Field(..., min_length=1)
    hidden_dims: list[int] = Field(default=[512, 512, 512])
    dropout: float = Field(default=0.4, ge=0.0, lt=1.0)
    cosine_classifier: bool = Field(default=True)
    batch_norm: bool = Field(default=True)
    num_epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=256, gt=0)
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = Field(default=42)
    wandb_project: str = Field(default="dynaclr")
    wandb_entity: Optional[str] = Field(default=None)
    output_path: str = Field(..., min_length=1)


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> int:
    k = min(k, logits.size(1))
    topk_preds = logits.topk(k, dim=1).indices
    return topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()  # type: ignore[return-value]


def _train_loop(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    wandb_run: wandb.sdk.wandb_run.Run | None,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Top1':>10} "
        f"| {'Train Top5':>10} | {'Val Loss':>10} | {'Val Top1':>10} "
        f"| {'Val Top5':>10} | {'Time':>6}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = run_top1 = run_top5 = run_total = 0

        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(labels)
            run_top1 += (logits.argmax(1) == labels).sum().item()
            run_top5 += _topk_accuracy(logits, labels, k=5)
            run_total += len(labels)

        train_loss = run_loss / run_total
        train_acc = run_top1 / run_total
        train_acc5 = run_top5 / run_total

        model.eval()
        val_loss = val_top1 = val_top5 = val_total = 0
        with torch.no_grad():
            for embs, labels in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_top1 += (logits.argmax(1) == labels).sum().item()
                val_top5 += _topk_accuracy(logits, labels, k=5)
                val_total += len(labels)

        v_loss = val_loss / val_total
        v_acc = val_top1 / val_total
        v_acc5 = val_top5 / val_total

        click.echo(
            f"{epoch:5d} | {train_loss:10.4f} | {train_acc:10.4%} "
            f"| {train_acc5:10.4%} | {v_loss:10.4f} | {v_acc:10.4%} "
            f"| {v_acc5:10.4%} | {time.time() - t0:5.1f}s"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/top1": train_acc,
                    "train/top5": train_acc5,
                    "val/loss": v_loss,
                    "val/top1": v_acc,
                    "val/top5": v_acc5,
                },
                step=epoch,
            )

    click.echo("Training complete.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Train an MLP embedder on cell embeddings."""
    click.echo("=" * 60)
    click.echo("MLP EMBEDDER TRAINING")
    click.echo("=" * 60)

    config_dict = load_config(config)
    cfg = MlpEmbedderTrainConfig(**config_dict)

    click.echo(f"\n  Embeddings: {cfg.embeddings_path}")
    click.echo(f"  Target col: {cfg.target_col}")
    click.echo(f"  Hidden dims: {cfg.hidden_dims}")
    click.echo(f"  Output: {cfg.output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"  Device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    adata = ad.read_zarr(cfg.embeddings_path)
    click.echo(f"\n  Loaded embeddings: {adata.shape}")

    if cfg.target_col not in adata.obs.columns:
        raise click.ClickException(f"Column '{cfg.target_col}' not found in obs. Available: {list(adata.obs.columns)}")

    mask = adata.obs[cfg.target_col].notna()
    adata = adata[mask]
    click.echo(f"  Samples with label: {len(adata):,}")

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    le = LabelEncoder()
    y = le.fit_transform(adata.obs[cfg.target_col].to_numpy())
    num_classes = len(le.classes_)
    input_dim = X.shape[1]

    click.echo(f"  Classes ({num_classes}): {list(le.classes_)}")

    perm = np.random.permutation(len(X))
    n_val = max(1, int(len(X) * cfg.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    click.echo(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    def _make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        embs = torch.tensor(X[idx], dtype=torch.float32)
        labels = torch.tensor(y[idx], dtype=torch.long)
        return DataLoader(
            TensorDataset(embs, labels),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx, shuffle=False)

    model = MLP(
        in_dims=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        norm="bn" if cfg.batch_norm else "ln",
        num_classes=num_classes,
        cosine_classifier=cfg.cosine_classifier,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    click.echo(f"\n  Model: {total_params:,} params")
    click.echo(str(model))

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        job_type="train-mlp-embedder",
        config=cfg.model_dump(),
    )

    _train_loop(
        model,
        train_loader,
        val_loader,
        cfg.num_epochs,
        cfg.learning_rate,
        cfg.weight_decay,
        device,
        run,
    )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "classes": list(le.classes_),
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "cosine_classifier": cfg.cosine_classifier,
            "batch_norm": cfg.batch_norm,
        },
        output_path,
    )
    click.echo(f"\n  Saved model to {output_path}")

    artifact = wandb.Artifact(
        name="mlp-embedder",
        type="model",
        metadata=cfg.model_dump(),
    )
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)
    run.finish()

    click.echo("Done.")


if __name__ == "__main__":
    main()
