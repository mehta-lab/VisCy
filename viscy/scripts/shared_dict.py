from multiprocessing.managers import DictProxy

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
from torch.distributed import get_rank
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, Dataset, Subset

from viscy.data.distributed import ShardedDistributedSampler


class CachedDataset(Dataset):
    def __init__(self, shared_dict: DictProxy, length: int):
        self.rank = get_rank()
        print(f"=== Initializing cache pool for rank {self.rank} ===")
        self.shared_dict = shared_dict
        self.length = length

    def __getitem__(self, index):
        if index not in self.shared_dict:
            print(f"* Adding {index} to cache dict on rank {self.rank}")
            self.shared_dict[index] = torch.tensor(index).float()[None]
        return self.shared_dict[index]

    def __len__(self):
        return self.length


class CachedDataModule(LightningDataModule):
    def __init__(
        self,
        length: int,
        split_ratio: float,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__()
        self.length = length
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage):
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported.")
        shared_dict = Manager().dict()
        dataset = CachedDataset(shared_dict, self.length)
        split_idx = int(self.length * self.split_ratio)
        self.train_dataset = Subset(dataset, range(0, split_idx))
        self.val_dataset = Subset(dataset, range(split_idx, self.length))

    def train_dataloader(self):
        sampler = ShardedDistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        sampler = ShardedDistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            sampler=sampler,
        )


class DummyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def on_train_start(self):
        rank_zero_info("=== Starting training ===")

    def on_train_epoch_start(self):
        rank_zero_info(f"=== Starting training epoch {self.current_epoch} ===")

    def training_step(self, batch, batch_idx):
        loss = torch.nn.functional.mse_loss(self.layer(batch), torch.zeros_like(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.nn.functional.mse_loss(self.layer(batch), torch.zeros_like(batch))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


trainer = Trainer(
    max_epochs=5,
    strategy="ddp",
    accelerator="cpu",
    devices=3,
    use_distributed_sampler=False,
    enable_progress_bar=False,
    logger=False,
    enable_checkpointing=False,
)

data_module = CachedDataModule(
    length=50, batch_size=2, split_ratio=0.6, num_workers=4, persistent_workers=False
)
model = DummyModel()
trainer.fit(model, data_module)
