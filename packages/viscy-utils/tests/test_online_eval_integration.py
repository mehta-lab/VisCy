"""Integration test: run a 2-epoch training loop with OnlineEvalCallback."""

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from viscy_utils.callbacks.online_eval import OnlineEvalCallback


class _SimpleEncoder(nn.Module):
    def __init__(self, feature_dim: int = 32, projection_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(16, feature_dim)
        self.proj = nn.Linear(feature_dim, projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.flatten(1)
        features = self.fc(x)
        projections = self.proj(features)
        return features, projections


class _SyntheticDataset(Dataset):
    """Synthetic dataset using __getitems__ to match MultiExperimentTripletDataset contract.

    Returns pre-batched dicts with anchor_meta as a flat list[dict].
    Must be used with ``collate_fn=lambda x: x``.
    """

    def __init__(self, n_samples: int = 40, n_markers: int = 4, n_tracks: int = 5):
        self.n_samples = n_samples
        self.n_markers = n_markers
        self.n_tracks = n_tracks

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("Use __getitems__ with collate_fn=lambda x: x")

    def __getitems__(self, indices: list[int]) -> dict:
        b = len(indices)
        meta = []
        for idx in indices:
            marker_id = idx % self.n_markers
            track_id = idx % self.n_tracks
            t = idx // self.n_tracks
            meta.append(
                {
                    "experiment": "exp_a",
                    "perturbation": "control",
                    "marker": f"marker_{marker_id}",
                    "global_track_id": track_id,
                    "t": t,
                    "labels": {"marker": marker_id},
                }
            )
        return {
            "anchor": torch.randn(b, 1, 1, 4, 4),
            "positive": torch.randn(b, 1, 1, 4, 4),
            "negative": torch.randn(b, 1, 1, 4, 4),
            "anchor_meta": meta,
        }


class _ContrastiveModule(LightningModule):
    """Minimal contrastive module that matches the real forward() API."""

    def __init__(self):
        super().__init__()
        self.model = _SimpleEncoder()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        _, proj_a = self(batch["anchor"])
        _, proj_p = self(batch["positive"])
        return nn.functional.mse_loss(proj_a, proj_p)

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        _, proj_a = self(batch["anchor"])
        _, proj_p = self(batch["positive"])
        return nn.functional.mse_loss(proj_a, proj_p)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class _DataModule(LightningDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(_SyntheticDataset(40), batch_size=8, collate_fn=lambda x: x)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(_SyntheticDataset(40), batch_size=8, collate_fn=lambda x: x)


def test_online_eval_callback_does_not_crash():
    """Run 2 epochs with OnlineEvalCallback and verify it logs without crashing."""
    callback = OnlineEvalCallback(
        every_n_epochs=1,
        label_key="marker",
        k=5,
        track_id_key="global_track_id",
        timepoint_key="t",
    )
    model = _ContrastiveModule()
    dm = _DataModule()
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(model, datamodule=dm)

    logged = trainer.logged_metrics
    assert "metrics/effective_rank/val" in logged, f"Missing effective_rank in {logged.keys()}"
    assert "metrics/knn_acc/marker/val" in logged, f"Missing knn_acc in {logged.keys()}"
    assert "metrics/temporal_smoothness/val" in logged, f"Missing temporal_smoothness in {logged.keys()}"

    erank = logged["metrics/effective_rank/val"]
    assert erank > 0, f"Effective rank should be positive, got {erank}"

    knn_acc = logged["metrics/knn_acc/marker/val"]
    assert 0 <= knn_acc <= 1, f"k-NN accuracy out of range: {knn_acc}"
