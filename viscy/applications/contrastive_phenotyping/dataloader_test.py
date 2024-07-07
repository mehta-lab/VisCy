import warnings
import os
from pathlib import Path
from viscy.data.hcs import ContrastiveDataModule
import time

warnings.filterwarnings("ignore")
data_on_lustre = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
data_on_vast = Path("/hpc/projects/virtual_staining/viral_sensor_test_dataio/")


def profile_dataio(top_dir, num_epochs=2):
    channels = 2
    x = 200
    y = 200
    z_range = (0, 10)
    batch_size = 128
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

    total_data_size = os.path.getsize(base_path)  # Get the file size in bytes
    total_data_size_mb = total_data_size / (1024 * 1024)  # Convert to MB

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
        for batch in data_module.train_dataloader():
            anchor_batch, positive_batch, negative_batch = batch
            total_bytes_transferred += (
                anchor_batch.nbytes + positive_batch.nbytes + negative_batch.nbytes
            )
            print("Anchor batch shape:", anchor_batch.shape)
            print("Positive batch shape:", positive_batch.shape)
            print("Negative batch shape:", negative_batch.shape)
            break
        for batch in data_module.val_dataloader():
            anchor_batch, positive_batch, negative_batch = batch
            total_bytes_transferred += (
                anchor_batch.nbytes + positive_batch.nbytes + negative_batch.nbytes
            )
            print("Anchor batch shape:", anchor_batch.shape)
            print("Positive batch shape:", positive_batch.shape)
            print("Negative batch shape:", negative_batch.shape)
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    data_transfer_speed = (total_bytes_transferred / elapsed_time) / (
        1024 * 1024
    )  # Calculate data transfer speed in MBPS

    print(f"Elapsed time for {num_epochs} iterations: {elapsed_time} seconds")
    print(f"Average time per iteration: {elapsed_time/num_epochs} seconds")
    print(f"Data transfer speed: {data_transfer_speed} MBPS")


# %%  Testing the data i/o with data stored on Lustre
profile_dataio(data_on_lustre)

# %% Testing the data i/o with data stored on Vast
profile_dataio(data_on_vast)
