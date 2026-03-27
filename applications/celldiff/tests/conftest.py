"""Test fixtures for celldiff application tests."""

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# Synthetic data dimensions.
# Chosen to satisfy UNetViT3D with num_res_block=[1], patch_size=4:
#   latent_size      = [D, H//2, W//2] = [4, 8, 8]
#   latent_grid_size = [4//4, 8//4, 8//4] = [1, 2, 2]  (all > 0)
SYNTH_B = 2
SYNTH_C = 1
SYNTH_D = 4
SYNTH_H = 16
SYNTH_W = 16

# Lightweight UNetViT3D config for fast CPU tests.
tiny_model_config = {
    "input_spatial_size": [SYNTH_D, SYNTH_H, SYNTH_W],
    "in_channels": SYNTH_C,
    "out_channels": SYNTH_C,
    "dims": [8, 16],
    "num_res_block": [1],
    "hidden_size": 32,
    "num_heads": 2,
    "dim_head": 16,
    "num_hidden_layers": 1,
    "patch_size": 4,
}

# Lightweight CellDiffNet config for fast CPU tests.
tiny_celldiff_fm_net_config = {
    "input_spatial_size": [SYNTH_D, SYNTH_H, SYNTH_W],
    "in_channels": SYNTH_C,
    "dims": [8, 16],
    "num_res_block": [1],
    "hidden_size": 32,
    "num_heads": 2,
    "dim_head": 16,
    "num_hidden_layers": 1,
    "patch_size": 4,
}


class SyntheticE2EDataset(Dataset):
    """Synthetic dataset returning source/target dict for CellDiffE2E."""

    def __init__(self, size: int = 8):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "source": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
            "target": torch.randn(SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
        }


class SyntheticE2EDataModule(LightningDataModule):
    """DataModule wrapping SyntheticE2EDataset for train and val."""

    def __init__(self, batch_size: int = SYNTH_B, num_samples: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(SyntheticE2EDataset(self.num_samples), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(SyntheticE2EDataset(self.num_samples), batch_size=self.batch_size)
