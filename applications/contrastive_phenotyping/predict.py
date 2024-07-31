from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from viscy.data.triplet import TripletDataModule, TripletDataset
from viscy.light.engine import ContrastiveModule
import os 
from torch.multiprocessing import Manager
from viscy.transforms import (
    NormalizeSampled,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)

normalizations = [
            # Normalization for Phase3D using mean and std
            NormalizeSampled(
                keys=["Phase3D"],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            ),
            # Normalization for RFP using median and IQR
            NormalizeSampled(
                keys=["RFP"],
                level="fov_statistics",
                subtrahend="median",
                divisor="iqr",
            ),
]

def main(hparams):
    # Set paths
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/expanded_final_track_timesteps.csv
    # /hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/uninfected_cells.csv
    # /hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/expanded_transitioning_cells_metadata.csv
    checkpoint_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/infection_score/contrastive_model-test-epoch=09-val_loss=0.00.ckpt"
    
    # non-rechunked data 
    data_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2.1-register/registered.zarr"

    # updated tracking data
    tracks_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/5-finaltrack/track_labels_final.zarr"
    
    source_channel = ["RFP", "Phase3D"]
    z_range = (26, 38)
    batch_size = 15 # match the number of fovs being processed such that no data is left
    # set to 15 for full, 12 for infected, and 8 for uninfected

    # Initialize the data module for prediction
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        source_channel=source_channel,
        z_range=z_range,
        initial_yx_patch_size=(512, 512),
        final_yx_patch_size=(224, 224),
        batch_size=batch_size,
        num_workers=hparams.num_workers,
        normalizations=normalizations,
    )

    data_module.setup(stage="predict")

    print(f"Total prediction dataset size: {len(data_module.predict_dataset)}")
    
    # Load the model from checkpoint
    backbone = "resnet50"
    in_stack_depth = 12
    stem_kernel_size = (5, 3, 3)
    model = ContrastiveModule.load_from_checkpoint(
    str(checkpoint_path), 
    predict=True, 
    backbone=backbone,
    in_channels=len(source_channel),
    in_stack_depth=in_stack_depth,
    stem_kernel_size=stem_kernel_size,
    tracks_path = tracks_path,
    )
    
    model.eval()

    # Initialize the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )

    # Run prediction
    trainer.predict(model, datamodule=data_module)
    
    # # Collect features and projections
    # features_list = []
    # projections_list = []

    # for batch_idx, batch in enumerate(predictions):
    #     features, projections = batch
    #     features_list.append(features.cpu().numpy())
    #     projections_list.append(projections.cpu().numpy())
    # all_features = np.concatenate(features_list, axis=0)
    # all_projections = np.concatenate(projections_list, axis=0)

    # # for saving visualizations embeddings 
    # base_dir = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/5-finaltrack/test_visualizations"
    # features_path = os.path.join(base_dir, 'B', '4', '2', 'before_projected_embeddings', 'test_epoch88_predicted_features.npy')
    # projections_path = os.path.join(base_dir, 'B', '4', '2', 'projected_embeddings', 'test_epoch88_predicted_projections.npy')

    # np.save("/hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/embeddings/resnet_uninf_rfp_epoch99_predicted_features.npy", all_features)
    # np.save("/hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/embeddings/resnet_uninf_rfp_epoch99_predicted_projections.npy", all_projections)

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
    parser.add_argument("--num_workers", type=int, default=15)
    args = parser.parse_args()
    main(args)
