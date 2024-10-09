import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).",
)

# %% Paths and constants
save_dir = (
    "/hpc/mydata/alishba.imran/VisCy/applications/contrastive_phenotyping/embeddings4"
)

# rechunked data
data_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2.2-register_annotations/updated_all_annotations.zarr"

# updated tracking data
tracks_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"

source_channel = ["background_mask", "uninfected_mask", "infected_mask"]
z_range = (0, 1)
batch_size = 1  # match the number of fovs being processed such that no data is left
# set to 15 for full, 12 for infected, and 8 for uninfected

# non-rechunked data
data_path_1 = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"

# updated tracking data
tracks_path_1 = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"

source_channel_1 = ["Nuclei_prediction_labels"]


# %% Define the main function for training
def main(hparams):
    # Initialize the data module for prediction, re-do embeddings but with size 224 by 224
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        source_channel=source_channel,
        z_range=z_range,
        initial_yx_patch_size=(224, 224),
        final_yx_patch_size=(224, 224),
        batch_size=batch_size,
        num_workers=hparams.num_workers,
    )

    data_module.setup(stage="predict")

    print(f"Total prediction dataset size: {len(data_module.predict_dataset)}")

    dataloader = DataLoader(
        data_module.predict_dataset,
        batch_size=batch_size,
        num_workers=hparams.num_workers,
    )

    # Initialize the second data module for segmentation masks
    seg_data_module = TripletDataModule(
        data_path=data_path_1,
        tracks_path=tracks_path_1,
        source_channel=source_channel_1,
        z_range=z_range,
        initial_yx_patch_size=(224, 224),
        final_yx_patch_size=(224, 224),
        batch_size=batch_size,
        num_workers=hparams.num_workers,
    )

    seg_data_module.setup(stage="predict")

    seg_dataloader = DataLoader(
        seg_data_module.predict_dataset,
        batch_size=batch_size,
        num_workers=hparams.num_workers,
    )

    # Initialize lists to store average values
    background_avg = []
    uninfected_avg = []
    infected_avg = []

    for batch, seg_batch in tqdm(
        zip(dataloader, seg_dataloader),
        desc="Processing batches",
        total=len(data_module.predict_dataset),
    ):
        anchor = batch["anchor"]
        seg_anchor = seg_batch["anchor"].int()

        # Extract the fov_name and id from the batch
        fov_name = batch["index"]["fov_name"][0]
        cell_id = batch["index"]["id"].item()

        fov_dirs = fov_name.split("/")
        # Construct the path to the CSV file
        csv_path = os.path.join(
            tracks_path, *fov_dirs, f"tracks{fov_name.replace('/', '_')}.csv"
        )

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Find the row with the specified id and extract the track_id
        track_id = df.loc[df["id"] == cell_id, "track_id"].values[0]

        # Create a boolean mask where segmentation values are equal to the track_id
        mask = seg_anchor == track_id
        # mask = (seg_anchor > 0)

        # Find the most frequent non-zero value in seg_anchor
        # unique, counts = np.unique(seg_anchor[seg_anchor > 0], return_counts=True)
        # most_frequent_value = unique[np.argmax(counts)]

        # # Create a boolean mask where segmentation values are equal to the most frequent value
        # mask = (seg_anchor == most_frequent_value)

        # Expand the mask to match the anchor tensor shape
        mask = mask.expand(1, 3, 1, 224, 224)

        # Calculate average values for each channel (background, uninfected, infected) using the mask
        background_avg.append(anchor[:, 0, :, :, :][mask[:, 0]].mean().item())
        uninfected_avg.append(anchor[:, 1, :, :, :][mask[:, 1]].mean().item())
        infected_avg.append(anchor[:, 2, :, :, :][mask[:, 2]].mean().item())

    # Convert lists to numpy arrays
    background_avg = np.array(background_avg)
    uninfected_avg = np.array(uninfected_avg)
    infected_avg = np.array(infected_avg)

    print("Average values per cell for each mask calculated.")
    print("Background average shape:", background_avg.shape)
    print("Uninfected average shape:", uninfected_avg.shape)
    print("Infected average shape:", infected_avg.shape)

    # Save the averages as .npy files
    np.save(os.path.join(save_dir, "background_avg.npy"), background_avg)
    np.save(os.path.join(save_dir, "uninfected_avg.npy"), uninfected_avg)
    np.save(os.path.join(save_dir, "infected_avg.npy"), infected_avg)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", type=str, default="Constant")
    parser.add_argument("--log_steps_per_epoch", type=int, default=10)
    parser.add_argument("--embedding_len", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
