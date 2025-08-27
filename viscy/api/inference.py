import importlib

import torch

from viscy.translation.engine import AugmentedPredictionVSUNet


@torch.no_grad()
def VS_inference_t2t(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Run virtual staining using a config dictionary and 5D input tensor (B, C, Z, Y, X).
    Returns predicted tensor of shape (B, C_out, Z, Y, X).
    """

    # Extract model info
    model_cfg = cfg["model"].copy()
    init_args = model_cfg["init_args"]
    class_path = model_cfg["class_path"]

    # Inject ckpt_path from top-level config if needed
    if "ckpt_path" in cfg:
        init_args["ckpt_path"] = cfg["ckpt_path"]

    # Import model class dynamically
    module_path, class_name = class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)

    # Instantiate model
    model = model_class(**init_args).to(x.device).eval()

    # Wrap with augmentation logic
    wrapper = (
        AugmentedPredictionVSUNet(
            model=model.model,
            forward_transforms=[lambda t: t],
            inverse_transforms=[lambda t: t],
        )
        .to(x.device)
        .eval()
    )

    wrapper.on_predict_start()
    return wrapper.inference_tiled(x)
