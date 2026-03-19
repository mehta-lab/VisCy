"""Demo: WandB logging for DynaCLR contrastive training.

Runs 4 epochs on synthetic triplet data and logs:
  - Image grids: anchor | positive | negative, one row per (sample, channel)
  - PCA pairplot: 8 PCs colored by condition, logged every 2 epochs

Usage
-----
Set WANDB_API_KEY in your environment, then:

    uv run python applications/dynaclr/scripts/demo_wandb_image_logging.py

Optional flags:
    --n-channels 2        Use multichannel inputs (default: 1)
    --log-batches 5       Batches to accumulate per epoch (default: 5)
    --log-samples 2       Samples per batch to include (default: 2)
    --offline             Run wandb in offline mode (no network needed)
"""

import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dynaclr.engine import ContrastiveModule
from viscy_data._typing import TripletSample

# ── Synthetic data ──────────────────────────────────────────────────────────

BATCH_SIZE = 4
N_SAMPLES = 32
Z, H, W = 5, 32, 32  # spatial dims — middle Z slice will be displayed


class SyntheticTripletDataset(Dataset):
    """Random triplet dataset with configurable channel count."""

    def __init__(self, n_channels: int = 1, size: int = N_SAMPLES):
        self.n_channels = n_channels
        self.size = size

    def __len__(self) -> int:  # noqa: D105
        return self.size

    def __getitems__(self, indices: list[int]) -> TripletSample:  # noqa: D105
        b = len(indices)

        def make_patch() -> Tensor:
            # ch0: white noise (mimics label-free phase) — shape (B, 1, Z, H, W)
            ch0 = torch.randn(b, 1, Z, H, W)
            if self.n_channels == 1:
                return ch0
            # ch1: smooth blobs via avg-pooling (mimics fluorescence reporter)
            ch1 = torch.nn.functional.avg_pool3d(torch.randn(b, 1, Z, H, W) * 3, kernel_size=5, stride=1, padding=2)
            return torch.cat([ch0, ch1], dim=1)

        conditions = ["control", "treated_low", "treated_high"]
        experiments = ["exp_A", "exp_B"]
        return {
            "anchor": make_patch(),
            "positive": make_patch(),
            "negative": make_patch(),
            "anchor_meta": [
                {
                    "condition": conditions[idx % len(conditions)],
                    "experiment": experiments[idx % len(experiments)],
                    "hours_post_perturbation": float(idx % 24),
                }
                for idx in indices
            ],
        }


# ── Minimal encoder ─────────────────────────────────────────────────────────


class TinyEncoder(nn.Module):
    """Flatten → linear encoder that returns (features, projections)."""

    def __init__(self, in_features: int, feature_dim: int = 32, projection_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_features, feature_dim)
        self.proj = nn.Linear(feature_dim, projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # noqa: D102
        features = self.fc(x.flatten(1))
        return features, self.proj(features)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--log-batches", type=int, default=5, help="Batches to accumulate per epoch")
    parser.add_argument("--log-samples", type=int, default=2, help="Samples per batch to include")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    args = parser.parse_args()

    n_ch = args.n_channels
    in_features = n_ch * Z * H * W
    n_rows = args.log_batches * args.log_samples * n_ch

    dataset = SyntheticTripletDataset(n_channels=n_ch)
    indices = list(range(N_SAMPLES))
    train_batches = [indices[i : i + BATCH_SIZE] for i in range(0, N_SAMPLES, BATCH_SIZE)]
    train_dl = DataLoader(dataset, batch_sampler=train_batches, collate_fn=lambda x: x)
    val_dl = DataLoader(dataset, batch_sampler=train_batches, collate_fn=lambda x: x)

    encoder = TinyEncoder(in_features)
    module = ContrastiveModule(
        encoder=encoder,
        log_batches_per_epoch=args.log_batches,
        log_samples_per_batch=args.log_samples,
        log_embeddings_every_n_epochs=2,
        example_input_array_shape=(1, n_ch, Z, H, W),
    )

    wandb_logger = WandbLogger(
        project="dynaclr-demo",
        name=f"image-logging-ch{n_ch}-b{args.log_batches}-s{args.log_samples}",
        mode="offline" if args.offline else "online",
    )

    trainer = Trainer(
        max_epochs=4,
        logger=wandb_logger,
        accelerator="cpu",
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(
        f"\nDone! Check WandB project 'dynaclr-demo'.\n"
        f"  Image grid : 'train_samples', 'val_samples'\n"
        f"               {n_rows} rows × 3 cols (anchor | positive | negative)\n"
        f"               {args.log_batches} batches × {args.log_samples} samples × {n_ch} channel(s)\n"
        f"  PCA plot   : 'val_pca' — 8 PCs colored by condition, logged every 2 epochs\n"
        f"  Input shape: (B={BATCH_SIZE}, C={n_ch}, Z={Z}, H={H}, W={W})\n"
    )


if __name__ == "__main__":
    main()
