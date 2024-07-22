# %% Imports and initialization.
import os
import time
import warnings
from pathlib import Path

import wandb
from tqdm import tqdm

from viscy.data.hcs import ContrastiveDataModule

warnings.filterwarnings("ignore")
os.environ["WANDB_DIR"] = f"/hpc/mydata/{os.environ['USER']}/"
data_on_lustre = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
data_on_vast = Path("/hpc/projects/virtual_staining/viral_sensor_test_dataio/")
wandb.init(project="contrastive_model", entity="alishba_imran-CZ Biohub")

# %% Method that iterates over two epochs and logs the resource usage.


def profile_dataio(top_dir, num_epochs=1):

    channels = 2
    x = 200
    y = 200
    z_range = (0, 10)
    batch_size = 16
    base_path = (
        top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/full_patch.zarr"
    )
    timesteps_csv_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/final_track_timesteps.csv"

    data_module = ContrastiveDataModule(
        base_path=base_path,
        channels=channels,
        x=x,
        y=y,
        timesteps_csv_path=timesteps_csv_path,
        batch_size=batch_size,
        num_workers=8,
        z_range=z_range,
    )

    # for train and val
    data_module.setup()

    print(
        f"Total dataset size: {len(data_module.train_dataset) + len(data_module.val_dataset) + len(data_module.test_dataset)}"
    )
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")

    start_time = time.time()
    total_bytes_transferred = 0  # Track the total number of bytes transferred

    # Profile the data i/o
    for i in range(num_epochs):
        # Train dataloader
        train_dataloader = data_module.train_dataloader()
        train_dataloader = tqdm(
            train_dataloader, desc=f"Epoch {i+1}/{num_epochs} - Train"
        )
        for batch in train_dataloader:
            anchor_batch, positive_batch, negative_batch = batch
            total_bytes_transferred += (
                anchor_batch.nbytes + positive_batch.nbytes + negative_batch.nbytes
            )
            # print("Anchor batch shape:", anchor_batch.shape)
            # print("Positive batch shape:", positive_batch.shape)
            # print("Negative batch shape:", negative_batch.shape)

        # Validation dataloader
        val_dataloader = data_module.val_dataloader()
        val_dataloader = tqdm(
            val_dataloader, desc=f"Epoch {i+1}/{num_epochs} - Validation"
        )
        for batch in val_dataloader:
            anchor_batch, positive_batch, negative_batch = batch
            total_bytes_transferred += (
                anchor_batch.nbytes + positive_batch.nbytes + negative_batch.nbytes
            )
            # print("Anchor batch shape:", anchor_batch.shape)
            # print("Positive batch shape:", positive_batch.shape)
            # print("Negative batch shape:", negative_batch.shape)

    end_time = time.time()
    elapsed_time = end_time - start_time
    data_transfer_speed = (total_bytes_transferred / elapsed_time) / (
        1024 * 1024
    )  # Calculate data transfer speed in MBPS

    print("Anchor batch shape:", anchor_batch.shape)
    print("Positive batch shape:", positive_batch.shape)
    print("Negative batch shape:", negative_batch.shape)

    print(f"Elapsed time for {num_epochs} iterations: {elapsed_time} seconds")
    print(f"Average time per iteration: {elapsed_time/num_epochs} seconds")
    print(f"Data transfer speed: {data_transfer_speed} MBPS")


# %% Testing the data i/o with data stored on Vast
print(f"Profiling data i/o with data stored on VAST\n{data_on_vast}\n")
profile_dataio(data_on_vast)


# %%  Testing the data i/o with data stored on Lustre
print(f"Profiling data i/o with data stored on Lustre\n{data_on_lustre}\n")

profile_dataio(data_on_lustre)

# %%
wandb.finish()
