from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor


from viscy.data.triplet import TripletDataModule
from viscy.light.engine import ContrastiveModule


def main():
    dm = TripletDataModule(
        data_path="/hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr",
        source_channel=["Phase3D", "RFP"],
        z_range=(20, 35),
        batch_size=16,
        num_workers=10,
        initial_yx_patch_size=(384, 384),
        final_yx_patch_size=(224, 224),
    )
    model = ContrastiveModule(
        backbone="convnext_tiny",
        in_channels=2,
        log_batches_per_epoch=2,
        log_samples_per_batch=3,
    )
    trainer = Trainer(
        max_epochs=5,
        limit_train_batches=10,
        limit_val_batches=5,
        logger=TensorBoardLogger(
            "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/test_tb",
            log_graph=True,
            default_hp_metric=True,
        ),
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint()],
        profiler="simple",  # other options: "advanced" uses cprofiler, "pytorch" uses pytorch profiler.
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
