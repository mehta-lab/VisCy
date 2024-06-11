# %%

from viscy.light.predict_writer import HCSPredictionWriter
from viscy.data.hcs import HCSDataModule
import lightning.pytorch as pl
from viscy.transforms import NormalizeSampled
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled
from viscy.light.engine import VSUNet
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from datetime import datetime

# Example usage
import os

seed_everything(42)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%

if __name__ == "__main__":
    data_path = "/hpc/projects/comp.micro/zebrafish/20240528_test_pipeline_AA/xxx-ed_test/a549_tomm20_lysotracker_w1_1_biref_toy_data_xyrect.zarr"
    model_ckpt = "/hpc/projects/comp.micro/mantis/2023_11_01_OpenCell_infection/5.1-VS_training/lightning_logs/20231130-120039/checkpoints/epoch=145-step=114902.ckpt"
    noise_std = 2e-4
    ch = ["Phase3D", "DummyChan", "DummyChan2"]

    data_module = HCSDataModule(
        data_path=data_path,
        batch_size=3,
        source_channel=ch[0],
        target_channel=ch[1:3],
        z_window_size=5,
        split_ratio=0.8,
        num_workers=8,  # TODO: MP for debugging
        architecture="3D",
        yx_patch_size=[384, 384],
        augmentations=[],
        normalizations=[
            NormalizeSampled(
                [ch[0]], level="fov_statistics", subtrahend="median", divisor="iqr"
            ),
            # NormalizeSampled(
            #     [ch[1]], level="fov_statistics", subtrahend="median", divisor="iqr"
            # ),
        ],
    )
    data_module.prepare_data()
    data_module.setup(stage="predict")

    model = VSUNet.load_from_checkpoint(
        model_ckpt,
        architecture="UNeXt2",
        model_config=dict(
            in_channels=1,
            out_channels=2,
            in_stack_depth=5,
            backbone="convnextv2_tiny",
            pretrained=False,
            stem_kernel_size=[5, 4, 4],
            decoder_mode="pixelshuffle",
            decoder_conv_blocks=2,
            head_pool=True,
            head_expansion_ratio=4,
            drop_path_rate=0.0,
        ),
        test_time_augmentations=True,
        tta_type="product",
    )
    model.eval()

    # %%
    # Test DataModule
    f, ax = plt.subplots(1, 2, figsize=(8, 8))

    dl = data_module.predict_dataloader()
    for batch in dl:
        for i, a in enumerate(ax.ravel()):
            a.imshow(batch["source"][i, 0, 2], cmap="gray")
        break

    for a in ax.flatten():
        a.axis("off")

    plt.tight_layout()
    plt.show()

    # %% perform prediction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./tmp/test_tta_{timestamp}.zarr"

    trainer = VSTrainer(
        accelerator="gpu",
        callbacks=[HCSPredictionWriter(output_path)],
    )
    # %%
    trainer.predict(
        model=model,
        datamodule=data_module,
        return_predictions=False,
    )

    # %%
