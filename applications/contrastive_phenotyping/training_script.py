# %% Imports and paths.
import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torchview
from torch.optim import Adam

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
#from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar
import wandb
from tqdm import tqdm

from viscy.light.engine import ContrastiveModule
from viscy.representation.contrastive import ContrastiveEncoder
from viscy.data.hcs import ContrastiveDataModule  

# %% Paths and constants
os.environ["WANDB_DIR"] = "/hpc/mydata/alishba.imran/wandb_logs/"

#wandb.init(project="contrastive_model", dir="/hpc/mydata/alishba.imran/wandb_logs/")

top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
input_zarr = top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/full_patch.zarr"
model_dir = top_dir / "infection_classification/models/infection_score"
timesteps_csv_path = top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/final_track_timesteps.csv"

# Data parameters
base_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/full_patch.zarr"
channels = 2
x = 200
y = 200
z = 15
z_range = (28, 43)
batch_size = 32

# %% Initialize the model and log the graph
#contra_model = ContrastiveEncoder(backbone="convnext_tiny")
# print(contra_model)

# model_graph = torchview.draw_graph(
#     contra_model,
#     torch.randn(1, 2, 15, 224, 224),
#     depth=3,
#     device="cpu",
# )
# model_graph.visual_graph

#contrastive_module = ContrastiveModule()
# print(contrastive_module.encoder)

# model_graph = torchview.draw_graph(
#     contrastive_module.encoder,
#     torch.randn(1, 2, 15, 200, 200),
#     depth=3,
#     device="cpu",
# )
# model_graph.visual_graph

# %% Progress bar 

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Running validation...")
        return bar
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training...")
        return bar
    
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.set_description("Testing...")
        return bar

# %% Define the main function for training
def main(hparams):
    # Seed for reproducibility
    # seed_everything(42, workers=True)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    print("Starting data module..")
    # Initialize the data module
    data_module = ContrastiveDataModule(
        base_path=base_path,
        channels=channels,
        x=x,
        y=y,
        timesteps_csv_path=timesteps_csv_path,
        batch_size=batch_size,
        z_range=z_range,
    )

    print("data module set up!")
    
    # Setup the data module for training, val and testing
    data_module.setup(stage='fit')
    
    print(f"Total dataset size: {len(data_module.train_dataset) + len(data_module.val_dataset) + len(data_module.test_dataset)}")
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    

    # Initialize the model
    model = ContrastiveModule(
        backbone=hparams.backbone,
        loss_function=torch.nn.TripletMarginLoss(),
        margin=hparams.margin,
        lr=hparams.lr,
        schedule=hparams.schedule,
        log_batches_per_epoch=hparams.log_batches_per_epoch,
        log_samples_per_batch=hparams.log_samples_per_batch,
        in_channels=channels,
        example_input_yx_shape=(x, y),
        in_stack_depth=z,
        stem_kernel_size=(5, 3, 3),
        embedding_len=hparams.embedding_len,
    )

    # Initialize logger
    wandb_logger = WandbLogger(project="contrastive_model", log_model="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="contrastive_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        monitor="val/loss_epoch",
    )

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        num_nodes=hparams.num_nodes,
        strategy="ddp",
        log_every_n_steps=hparams.log_every_n_steps,
        num_sanity_val_steps=0
    )

    # Fetches batches from the training dataloader, 
    # Calls the training_step method on the model for each batch
    # Aggregates the losses and performs optimization steps
    trainer.fit(model, datamodule=data_module)

    # Validate the model
    trainer.validate(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    import sys
    if "ipykernel_launcher" in sys.argv[0]:
        # Jupyter Notebook environment
        args = {
            "backbone": "convnext_tiny",
            "margin": 0.5,
            "lr": 1e-3,
            "schedule": "Constant",
            "log_batches_per_epoch": 8,
            "log_samples_per_batch": 1,
            "embedding_len": 256,
            "max_epochs": 100,
            "accelerator": "gpu",
            "devices": 1,  # Set to 4 GPUs
            "num_nodes": 2,
            "log_every_n_steps": 1,
        }
        class HParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        hparams = HParams(**args)
        main(hparams)
    else:
        parser = ArgumentParser()
        parser.add_argument("--backbone", type=str, default="convnext_tiny")
        parser.add_argument("--margin", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--schedule", type=str, default="Constant")
        parser.add_argument("--log_batches_per_epoch", type=int, default=8)
        parser.add_argument("--log_samples_per_batch", type=int, default=1)
        parser.add_argument("--embedding_len", type=int, default=256)
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--accelerator", type=str, default="gpu")
        parser.add_argument("--devices", type=int, default=1)  # 4 GPUs
        parser.add_argument("--num_nodes", type=int, default=2)
        parser.add_argument("--log_every_n_steps", type=int, default=1)
        args = parser.parse_args()

        main(args)

# %%
