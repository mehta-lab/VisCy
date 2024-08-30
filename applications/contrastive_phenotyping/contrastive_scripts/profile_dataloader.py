# %% Imports and initialization.
import time

from monai.transforms import NormalizeIntensityd, ScaleIntensityRangePercentilesd
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule

# %% Setup parameters for dataloader
# rechunked data
data_path = "/hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr"

# updated tracking data
tracks_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"
source_channel = ["RFP", "Phase3D"]
z_range = (28, 43)
batch_size = 32
num_workers = 16

# Updated normalizations
normalizations = [
    NormalizeIntensityd(
        keys=["Phase3D"],
        subtrahend=None,
        divisor=None,
        nonzero=False,
        channel_wise=False,
        dtype=None,
        allow_missing_keys=False,
    ),
    ScaleIntensityRangePercentilesd(
        keys=["RFP"],
        lower=50,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=False,
        relative=False,
        channel_wise=False,
        dtype=None,
        allow_missing_keys=False,
    ),
]


# %% Initialize the data module
data_module = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=source_channel,
    z_range=z_range,
    initial_yx_patch_size=(512, 512),
    final_yx_patch_size=(224, 224),
    batch_size=batch_size,
    num_workers=num_workers,
    normalizations=normalizations,
)
# for train and val
data_module.setup("fit")

print(
    f"Total dataset size: {len(data_module.train_dataset) + len(data_module.val_dataset)}"
)
print(f"Training dataset size: {len(data_module.train_dataset)}")
print(f"Validation dataset size: {len(data_module.val_dataset)}")

# %% Profile the data i/o
num_epochs = 1
start_time = time.time()
total_bytes_transferred = 0  # Track the total number of bytes transferred

# Profile the data i/o
for i in range(num_epochs):
    # Train dataloader
    train_dataloader = data_module.train_dataloader()
    train_dataloader = tqdm(train_dataloader, desc=f"Epoch {i+1}/{num_epochs} - Train")
    for batch in train_dataloader:
        anchor_batch = batch["anchor"]
        positive_batch = batch["positive"]
        negative_batch = batch["negative"]
        total_bytes_transferred += (
            anchor_batch.nbytes + positive_batch.nbytes + negative_batch.nbytes
        )
        # print("Anchor batch shape:", anchor_batch.shape)
        # print("Positive batch shape:", positive_batch.shape)
        # print("Negative batch shape:", negative_batch.shape)

    # Validation dataloader
    val_dataloader = data_module.val_dataloader()
    val_dataloader = tqdm(val_dataloader, desc=f"Epoch {i+1}/{num_epochs} - Validation")
    for batch in val_dataloader:
        anchor_batch = batch["anchor"]
        positive_batch = batch["positive"]
        negative_batch = batch["negative"]
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

# %%
