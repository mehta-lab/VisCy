# script to profile dataloading
# use with a sampling profiler like py-spy
from monai.transforms import (
    Decollated,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ToDeviced,
)
from pytorch_metric_learning.losses import NTXentLoss

from viscy.data.combined import BatchedConcatDataModule
from viscy.data.triplet import TripletDataModule
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.transforms import (
    NormalizeSampled,
)
from viscy.transforms._transforms import (
    BatchedRandAffined,
    BatchedScaleIntensityRangePercentilesd,
    RandGaussianNoiseTensord,
)


def model(
    input_channel_number: int = 1,
    z_stack_depth: int = 30,
    patch_size: int = 192,
    temperature: float = 0.5,
):
    return ContrastiveModule(
        encoder=ContrastiveEncoder(
            backbone="convnext_tiny",
            in_channels=input_channel_number,
            in_stack_depth=z_stack_depth,
            stem_kernel_size=(5, 4, 4),
            embedding_dim=768,
            projection_dim=32,
            drop_path_rate=0.0,
        ),
        loss_function=NTXentLoss(temperature=temperature),
        lr=0.00002,
        log_batches_per_epoch=3,
        log_samples_per_batch=3,
        example_input_array_shape=[
            1,
            input_channel_number,
            z_stack_depth,
            patch_size,
            patch_size,
        ],
    )


def channel_augmentations(processing_channel: str):
    return [
        BatchedRandAffined(
            keys=[processing_channel],
            prob=0.8,
            scale_range=((1.0, 1.0), (0.8, 1.2), (0.8, 1.2)),
            rotate_range=[1.0, 0.0, 0.0],
            shear_range=(0.2, 0.2, 0.0, 0.2, 0.0, 0.2),
        ),
        Decollated(keys=[processing_channel]),
        RandAdjustContrastd(
            keys=[processing_channel],
            prob=0.5,
            gamma=[0.8, 1.2],
        ),
        RandScaleIntensityd(
            keys=[processing_channel],
            prob=0.5,
            factors=0.5,
        ),
        RandGaussianSmoothd(
            keys=[processing_channel],
            prob=0.5,
            sigma_x=[0.25, 0.75],
            sigma_y=[0.25, 0.75],
            sigma_z=[0.0, 0.0],
        ),
        RandGaussianNoiseTensord(
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
            ToDeviced(keys=[fl_channel], device="cuda"),
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
    dm1 = TripletDataModule(
        data_path="/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61B/2024_10_16_A549_SEC61_ZIKV_DENV/2024_10_16_A549_SEC61_ZIKV_DENV_2.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_10_16_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_10_16_A549_SEC61_ZIKV_DENV_cropped.zarr",
        source_channel=["raw GFP EX488 EM525-45"],
        z_range=[5, 35],
        initial_yx_patch_size=(384, 384),
        final_yx_patch_size=(192, 192),
        batch_size=16,
        num_workers=4,
        time_interval=1,
        augmentations=channel_augmentations("raw GFP EX488 EM525-45"),
        normalizations=channel_normalization(
            phase_channel=None, fl_channel="raw GFP EX488 EM525-45"
        ),
        fit_include_wells=["B/3", "B/4", "C/3", "C/4"],
        return_negative=False,
    )
    dm2 = TripletDataModule(
        data_path="/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61B/2024_10_16_A549_SEC61_ZIKV_DENV/2024_10_16_A549_SEC61_ZIKV_DENV_2.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_10_16_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_10_16_A549_SEC61_ZIKV_DENV_cropped.zarr",
        source_channel=["raw mCherry EX561 EM600-37"],
        z_range=[5, 35],
        initial_yx_patch_size=(384, 384),
        final_yx_patch_size=(192, 192),
        batch_size=16,
        num_workers=4,
        time_interval=1,
        augmentations=channel_augmentations("raw mCherry EX561 EM600-37"),
        normalizations=channel_normalization(
            phase_channel=None, fl_channel="raw mCherry EX561 EM600-37"
        ),
        fit_include_wells=["B/3", "B/4", "C/3", "C/4"],
        return_negative=False,
    )
    dm = BatchedConcatDataModule(data_modules=[dm1, dm2])
    dm.setup("fit")

    print(len(dm1.train_dataset), len(dm2.train_dataset), len(dm.train_dataset))
    n = 1

    print("Training batches:")
    for i, batch in enumerate(dm.train_dataloader()):
        print(i, batch["anchor"].shape, batch["positive"].device)
        if i == n - 1:
            break
    print("Validation batches:")
    for i, batch in enumerate(dm.val_dataloader()):
        print(i, batch["anchor"].shape, batch["positive"].device)
        if i == n - 1:
            break
