# script to profile dataloading
# use with a sampling profiler like py-spy


import logging

import torch
from lightning.pytorch import LightningModule, Trainer

from viscy.data.combined import BatchedConcatDataModule
from viscy.data.triplet import TripletDataModule
from viscy.transforms import (
    BatchedCenterSpatialCropd,
    BatchedRandAdjustContrastd,
    BatchedRandAffined,
    BatchedRandGaussianNoised,
    BatchedRandGaussianSmoothd,
    BatchedRandScaleIntensityd,
    BatchedScaleIntensityRangePercentilesd,
    NormalizeSampled,
)

_logger = logging.getLogger(__name__)


class DummyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def training_step(self, batch, batch_idx):
        img = batch["anchor"]
        _logger.info(img.shape)
        return (img * self.a).mean()

    def validation_step(self, batch, batch_idx):
        img = batch["anchor"]
        _logger.info(img.shape)
        return (img * self.a).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def channel_augmentations(processing_channel: str):
    return [
        BatchedRandAffined(
            keys=[processing_channel],
            prob=0.8,
            scale_range=((1.0, 1.0), (0.8, 1.2), (0.8, 1.2)),
            rotate_range=[1.0, 0.0, 0.0],
            shear_range=(0.2, 0.2, 0.0, 0.2, 0.0, 0.2),
        ),
        BatchedCenterSpatialCropd(keys=[processing_channel], roi_size=(32, 192, 192)),
        BatchedRandAdjustContrastd(
            keys=[processing_channel],
            prob=0.5,
            gamma=(0.8, 1.2),
        ),
        BatchedRandScaleIntensityd(
            keys=[processing_channel],
            prob=0.5,
            factors=0.5,
        ),
        BatchedRandGaussianSmoothd(
            keys=[processing_channel],
            prob=0.5,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.0, 0.0),
        ),
        BatchedRandGaussianNoised(
            keys=[processing_channel],
            prob=0.5,
            mean=0.0,
            std=0.2,
        ),
    ]


def channel_normalization(
    phase_channel: str = None,
    fl_channel: str = None,
):
    if phase_channel:
        return [
            NormalizeSampled(
                keys=[phase_channel],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            )
        ]
    elif fl_channel:
        return [
            BatchedScaleIntensityRangePercentilesd(
                keys=[fl_channel],
                lower=50,
                upper=99,
                b_min=0.0,
                b_max=1.0,
            ),
        ]
    else:
        raise NotImplementedError("Either phase_channel or fl_channel must be provided")


if __name__ == "__main__":
    num_workers = 1
    batch_size = 128
    persistent_workers = True
    cache_pool_bytes = 32 << 30
    dm1 = TripletDataModule(
        data_path="/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61B/2024_10_16_A549_SEC61_ZIKV_DENV/2024_10_16_A549_SEC61_ZIKV_DENV_2.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_16_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_10_16_A549_SEC61_ZIKV_DENV_cropped.zarr",
        source_channel=["raw GFP EX488 EM525-45"],
        z_range=[5, 35],
        initial_yx_patch_size=(384, 384),
        final_yx_patch_size=(192, 192),
        batch_size=batch_size,
        num_workers=num_workers,
        time_interval=1,
        augmentations=channel_augmentations("raw GFP EX488 EM525-45"),
        normalizations=channel_normalization(
            phase_channel=None, fl_channel="raw GFP EX488 EM525-45"
        ),
        fit_include_wells=["B/3", "B/4", "C/3", "C/4"],
        return_negative=False,
        persistent_workers=persistent_workers,
        cache_pool_bytes=cache_pool_bytes,
    )
    dm2 = TripletDataModule(
        data_path="/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61B/2024_10_16_A549_SEC61_ZIKV_DENV/2024_10_16_A549_SEC61_ZIKV_DENV_2.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_16_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_10_16_A549_SEC61_ZIKV_DENV_cropped.zarr",
        source_channel=["Phase3D"],
        z_range=[5, 35],
        initial_yx_patch_size=(384, 384),
        final_yx_patch_size=(192, 192),
        batch_size=batch_size,
        num_workers=num_workers,
        time_interval=1,
        augmentations=channel_augmentations("Phase3D"),
        normalizations=channel_normalization(phase_channel="Phase3D", fl_channel=None),
        fit_include_wells=["B/3", "B/4", "C/3", "C/4"],
        return_negative=False,
        persistent_workers=persistent_workers,
        cache_pool_bytes=cache_pool_bytes,
    )
    dm = BatchedConcatDataModule(data_modules=[dm1, dm2])
    model = DummyModel()
    trainer = Trainer(max_epochs=4, limit_train_batches=8, limit_val_batches=8)
    trainer.fit(model, dm)
