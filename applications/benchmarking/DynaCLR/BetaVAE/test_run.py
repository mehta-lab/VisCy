# %%
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ScaleIntensityRangePercentilesd,
)
from monai.transforms.spatial.dictionary import RandAffined

from viscy.data.triplet import TripletDataModule
from viscy.representation.engine import VaeModule
from viscy.representation.vae import VaeDecoder, VaeEncoder
from viscy.trainer import VisCyTrainer
from viscy.transforms import (
    NormalizeSampled,
)


# %%
def channel_augmentations(processing_channel: str):
    return [
        RandAffined(
            keys=[processing_channel],
            prob=0.8,
            scale_range=[0, 0.2, 0.2],
            rotate_range=[3.14, 0.0, 0.0],
            shear_range=[0.0, 0.01, 0.01],
            padding_mode="zeros",
        ),
        RandAdjustContrastd(
            keys=[processing_channel],
            prob=0.5,
            gamma=(0.8, 1.2),
        ),
        RandScaleIntensityd(
            keys=[processing_channel],
            prob=0.5,
            factors=0.5,
        ),
        RandGaussianSmoothd(
            keys=[processing_channel],
            prob=0.5,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.0, 0.0),
        ),
        RandGaussianNoised(
            keys=[processing_channel],
            prob=0.5,
            mean=0.0,
            std=0.2,
        ),
    ]


# %%
def channel_normalization(
    phase_channel: str | None = None,
    fl_channel: str | None = None,
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
            ScaleIntensityRangePercentilesd(
                keys=[fl_channel],
                lower=50,
                upper=99,
                b_min=0.0,
                b_max=1.0,
            )
        ]
    else:
        raise NotImplementedError("Either phase_channel or fl_channel must be provided")


if __name__ == "__main__":
    seed_everything(42)

    # use tensor cores on Ampere GPUs (24-bit tensorfloat matmul)
    torch.set_float32_matmul_precision("high")

    initial_yx_patch_size = (384, 384)
    final_yx_patch_size = (192, 192)
    batch_size = 64
    num_workers = 12
    time_interval = 1
    z_stack_depth = 32

    print("Creating model components...")

    # Create encoder with debug info
    encoder = VaeEncoder(
        backbone="resnet50",
        in_channels=1,
        in_stack_depth=z_stack_depth,
        embedding_dim=256,
        stem_kernel_size=(8, 4, 4),
        stem_stride=(8, 4, 4),
    )
    print(f"Encoder created successfully")

    # Test encoder forward pass
    test_input = torch.randn(1, 1, z_stack_depth, 192, 192)
    try:
        encoder_output = encoder(test_input)
        print(f"Encoder test passed: {encoder_output.embedding.shape}")
    except Exception as e:
        print(f"Encoder test failed: {e}")
        exit(1)

    # Create decoder
    decoder = VaeDecoder(
        decoder_channels=[1024, 512, 256, 128],
        latent_dim=256,
        out_channels=1,
        out_stack_depth=z_stack_depth,
        latent_spatial_size=3,
        head_expansion_ratio=2,
        head_pool=False,
        upsample_mode="deconv",
        conv_blocks=2,
        norm_name="batch",
        upsample_pre_conv=None,
        strides=[2, 2, 2, 2],
    )
    print(f"Decoder created successfully")

    # Create VaeModule
    model = VaeModule(
        encoder=encoder,
        decoder=decoder,
        example_input_array_shape=(1, 1, z_stack_depth, 192, 192),
        latent_dim=256,
        beta=3.0,
        lr=2e-4,
    )
    print(f"VaeModule created successfully")

    # Test model forward pass
    try:
        model_output = model(test_input)
        print(f"Model test passed: loss={model_output['loss']}")
    except Exception as e:
        print(f"Model test failed: {e}")
        exit(1)

    # Create data module
    print("Creating data module...")
    dm = TripletDataModule(
        data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_10_16_A549_SEC61_ZIKV_DENV/2-assemble/2024_10_16_A549_SEC61_ZIKV_DENV.zarr",
        tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2024_10_16_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_10_16_A549_SEC61_ZIKV_DENV_cropped.zarr",
        source_channel=["Phase3D"],
        z_range=(5, 37),
        initial_yx_patch_size=initial_yx_patch_size,
        final_yx_patch_size=final_yx_patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        time_interval=time_interval,
        augmentations=channel_augmentations("Phase3D"),
        normalizations=channel_normalization(phase_channel="Phase3D"),
        fit_include_wells=["B/3", "B/4", "C/3", "C/4"],
    )
    print(f"DataModule created successfully")

    # Create trainer
    trainer = VisCyTrainer(
        accelerator="gpu",
        strategy="ddp",
        devices=4,
        num_nodes=1,
        precision="16-mixed",
        # fast_dev_run=True,
        max_epochs=100,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=TensorBoardLogger(
            save_dir="/hpc/projects/organelle_phenotyping/models/SEC61B/vae",
            name="betavae_phase3D_ddp",
            version="beta_3_16slice",
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="loss/val", save_top_k=5, save_last=True, every_n_epochs=1
            ),
        ],
        use_distributed_sampler=True,
    )

    print("Starting training...")
    trainer.fit(model, dm)

# %%
