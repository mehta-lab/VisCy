# %%
"""
Demo: In-memory volume prediction using predict_sliding_windows.

This API provides the same results as the `viscy predict` CLI (HCSPredictionWriter)
since both use the same linear feathering blending algorithm for overlapping windows.
"""

from pathlib import Path

import napari
import numpy as np
import torch
from iohub import open_ome_zarr

from viscy.translation.engine import AugmentedPredictionVSUNet, VSUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model manually
model = (
    VSUNet(
        architecture="fcmae",
        model_config={
            "in_channels": 1,
            "out_channels": 2,
            "in_stack_depth": 21,
            "encoder_blocks": [3, 3, 9, 3],
            "dims": [96, 192, 384, 768],
            "decoder_conv_blocks": 2,
            "stem_kernel_size": [7, 4, 4],
            "pretraining": False,
            "head_conv": True,
            "head_conv_expansion_ratio": 4,
            "head_conv_pool": False,
        },
        ckpt_path="/path/to/checkpoint.ckpt",
    )
    .to(DEVICE)
    .eval()
)

vs = (
    AugmentedPredictionVSUNet(
        model=model.model,
        forward_transforms=[lambda t: t],
        inverse_transforms=[lambda t: t],
    )
    .to(DEVICE)
    .eval()
)

# Load data
path = Path("/path/to/your.zarr/0/1/000000")
with open_ome_zarr(path) as ds:
    vol_np = np.asarray(ds.data[0:1, 0:1])  # (1, 1, Z, Y, X)

vol = torch.from_numpy(vol_np).float().to(DEVICE)

# Run inference with sliding windows and linear feathering blending
# step=1 gives maximum overlap; increase step for faster inference
with torch.inference_mode():
    pred = vs.predict_sliding_windows(vol, out_channel=2, step=1)

# Visualize
pred_np = pred.cpu().numpy()
nuc, mem = pred_np[0, 0], pred_np[0, 1]

viewer = napari.Viewer()
viewer.add_image(vol_np, name="phase_input", colormap="gray")
viewer.add_image(nuc, name="virt_nuclei", colormap="magenta")
viewer.add_image(mem, name="virt_membrane", colormap="cyan")
napari.run()
