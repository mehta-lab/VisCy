from viscy.data.hcs import ContrastiveDataModule, PredictDataset
from viscy.light.engine import ContrastiveModule
import os
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.optim import Adam
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import wandb
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd


def main(hparams):
    # Set paths
    top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
    timesteps_csv_path = (
        top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/predict_timesteps.csv"
    )
    predict_base_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/all_annotations_patch.zarr"
    checkpoint_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/infection_score/updated_multiple_channels/contrastive_model-test-epoch=97-val_loss=0.00.ckpt"

    # Data parameters
    channels = 2
    x = 200
    y = 200
    z_range = (28, 43)
    batch_size = 11
    channel_names = ["RFP", "Phase3D"]

    # Initialize the data module for prediction
    data_module = ContrastiveDataModule(
        base_path=str(predict_base_path),
        channels=channels,
        x=x,
        y=y,
        timesteps_csv_path=timesteps_csv_path,
        channel_names=channel_names,
        batch_size=batch_size,
        z_range=z_range,
        predict_base_path=predict_base_path,
    )

    data_module.setup(stage="predict")

    # Load the model from checkpoint
    model = ContrastiveModule.load_from_checkpoint(str(checkpoint_path), predict=True)
    model.eval()
    model.encoder.predict = True

    # Initialize the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=DDPStrategy(),
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )

    # Run prediction
    predictions = trainer.predict(model, datamodule=data_module)

    # Collect features and projections
    features_list = []
    projections_list = []

    for batch_idx, batch in enumerate(predictions):
        features, projections = batch
        features_list.append(features.cpu().numpy())
        projections_list.append(projections.cpu().numpy())

    all_features = np.concatenate(features_list, axis=0)
    all_projections = np.concatenate(projections_list, axis=0)

    # Save features and projections
    np.save("updated_epoch97_predicted_features.npy", all_features)
    np.save("updated_epoch97_predicted_projections.npy", all_projections)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", type=str, default="Constant")
    parser.add_argument("--log_steps_per_epoch", type=int, default=10)
    parser.add_argument("--embedding_len", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    args = parser.parse_args()

    main(args)
