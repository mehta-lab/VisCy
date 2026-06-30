"""
# Distributed training

Demonstrate how to train a model using distributed data parallel (DDP) with PyTorch Lightning.
"""

import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from viscy.data.hcs import HCSDataModule
from viscy.scripts.infection_phenotyping.classify_infection_2D import SemanticSegUNet2D
from viscy.transforms import NormalizeSampled, RandWeightedCropd


def main():
    dm = HCSDataModule(
        data_path="/hpc/mydata/ziwen.liu/demo/Exp_2024_02_13_DENV_3infMarked_trainVal.zarr",
        source_channel=["Sensor", "Phase"],
        target_channel=["Inf_mask"],
        yx_patch_size=(128, 128),
        split_ratio=0.5,
        z_window_size=1,
        architecture="2D",
        num_workers=8,
        batch_size=128,
        normalizations=[
            NormalizeSampled(
                keys=["Sensor", "Phase"],
                level="fov_statistics",
                subtrahend="median",
                divisor="iqr",
            )
        ],
        augmentations=[
            RandWeightedCropd(
                num_samples=8,
                spatial_size=[-1, 128, 128],
                keys=["Sensor", "Phase", "Inf_mask"],
                w_key="Inf_mask",
            )
        ],
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    model = SemanticSegUNet2D(
        in_channels=2,
        out_channels=3,
        loss_function=torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.25, 0.7])),
    )
    log_dir = Path(os.getenv("MYDATA", "")) / "torch_demo"
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        precision=32,
        num_nodes=1,
        devices=2,
        fast_dev_run=True,
        max_epochs=100,
        logger=TensorBoardLogger(save_dir=log_dir, version="interactive_demo"),
        log_every_n_steps=10,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="loss/validate", save_top_k=5, every_n_epochs=1, save_last=True
            ),
        ],
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
