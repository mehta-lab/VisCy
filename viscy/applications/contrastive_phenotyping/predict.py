from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from viscy.data.hcs import ContrastiveDataModule
from viscy.light.engine import ContrastiveModule
import os 
from torch.multiprocessing import Manager

def main(hparams):
    # Set paths
    # this CSV defines the order in which embeddings should be processed. Currently using num_workers = 1 to keep order
    top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
    timesteps_csv_path = "/hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/uninfected_cells.csv"
    predict_base_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/all_annotations_patch.zarr"
    checkpoint_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/infection_score/contrastive_model-test-epoch=26-val_loss=0.00.ckpt"

    channels = 1
    x = 200
    y = 200
    z_range = (26, 38)
    batch_size = 8
    channel_names = ["RFP"]

    manager = Manager()
    embeddings_dict = manager.dict()
    order_dict = manager.dict()

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
        analysis=True,  # for self-supervised results
    )

    data_module.setup(stage="predict")
    position_to_timesteps = data_module.position_to_timesteps
    
    # Load the model from checkpoint
    backbone = "resnet50"
    in_stack_depth = 12
    stem_kernel_size = (5, 3, 3)
    model = ContrastiveModule.load_from_checkpoint(
    str(checkpoint_path), 
    predict=True, 
    backbone=backbone,
    in_stack_depth=in_stack_depth,
    stem_kernel_size=stem_kernel_size,
    position_to_timesteps=position_to_timesteps
    )
    
    model.embeddings_dict = embeddings_dict
    model.order_dict = order_dict
    model.eval()

    # Initialize the trainer
    trainer = Trainer(
        accelerator="cpu",
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
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    args = parser.parse_args()
    main(args)
