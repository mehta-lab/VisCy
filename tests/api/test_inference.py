import torch
from viscy.api.inference import VS_inference_t2t


def test_vs_inference_t2t():
    in_stack_depth = 21
    dims = [24, 48, 96, 192]  # dims[0] must be divisible by ratio (24/3=8)

    cfg = {
        "model": {
            "class_path": "viscy.translation.engine.VSUNet",
            "init_args": {
                "architecture": "fcmae",
                "model_config": {
                    "in_channels": 1,
                    "out_channels": 2,
                    "in_stack_depth": in_stack_depth,
                    "encoder_blocks": [1, 1, 1, 1],
                    "dims": dims,
                    "stem_kernel_size": [7, 4, 4],
                    "pretraining": False,
                    "decoder_conv_blocks": 1,
                    "head_conv": True,
                    "head_conv_expansion_ratio": 2,
                    "head_conv_pool": False,
                },
                "test_time_augmentations": False,
                "tta_type": "none",
                "ckpt_path": None,
            },
        }
    }

    x = torch.rand(1, 1, in_stack_depth, 64, 64)
    pred = VS_inference_t2t(x, cfg)

    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (1, 2, in_stack_depth, 64, 64), (
        f"Unexpected shape: {pred.shape}"
    )
