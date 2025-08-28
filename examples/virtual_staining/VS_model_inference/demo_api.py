
# %%
import time
from pathlib import Path
import numpy as np
import torch
from iohub import open_ome_zarr
import napari

from viscy.translation.inference import VS_inference_t2t

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration dictionary (from CLI .yaml)
config = {
    "model": {
        "class_path": "viscy.translation.engine.VSUNet",
        "init_args": {
            "architecture": "fcmae",
            "model_config": {
                "in_channels": 1,
                "out_channels": 2,
                "in_stack_depth": 21,
                "encoder_blocks": [3, 3, 9, 3],
                "dims": [96, 192, 384, 768],
                "encoder_drop_path_rate": 0.0,
                "stem_kernel_size": [7, 4, 4],
                "decoder_conv_blocks": 2,
                "pretraining": False,
                "head_conv": True,
                "head_conv_expansion_ratio": 4,
                "head_conv_pool": False,
            },
        },
        "test_time_augmentations": True,
        "tta_type": "median",
    },
    "ckpt_path": "/path/to/checkpoint.ckpt"
}

# Load Phase3D input volume
path = Path("/path/to/your.zarr/0/1/000000")
with open_ome_zarr(path) as ds:
    vol_np = np.asarray(ds.data[0, 0])  # (Z, Y, X)

vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (B=1, C=1, Z, Y, X)

# Run model
start = time.time()
pred = VS_inference_t2t(vol, config)
torch.cuda.synchronize()
print(f"Inference time: {time.time() - start:.2f} sec")

# Visualize
pred_np = pred.cpu().numpy()
nuc, mem = pred_np[0, 0], pred_np[0, 1]

viewer = napari.Viewer()
viewer.add_image(vol_np, name="phase_input", colormap="gray")
viewer.add_image(nuc, name="virt_nuclei", colormap="magenta")
viewer.add_image(mem, name="virt_membrane", colormap="cyan")
napari.run()


#%%

# examples/inference_manual.py
import time
from pathlib import Path
import numpy as np
import torch
from iohub import open_ome_zarr
import napari

from viscy.translation.engine import VSUNet, AugmentedPredictionVSUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model manually
vs = VSUNet(
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
    test_time_augmentations=True,
    tta_type="median",
).to(DEVICE).eval()

wrapper = AugmentedPredictionVSUNet(
    model=vs.model,
    forward_transforms=[lambda t: t],
    inverse_transforms=[lambda t: t],
).to(DEVICE).eval()
wrapper.on_predict_start()

# Load data
path = Path("/path/to/your.zarr/0/1/000000")
with open_ome_zarr(path) as ds:
    vol_np = np.asarray(ds.data[0, 0])  # (Z, Y, X)

vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

# Run inference
with torch.no_grad():
    pred = wrapper.inference_tiled(vol)
torch.cuda.synchronize()

# Visualize
pred_np = pred.cpu().numpy()
nuc, mem = pred_np[0, 0], pred_np[0, 1]

viewer = napari.Viewer()
viewer.add_image(vol_np, name="phase_input", colormap="gray")
viewer.add_image(nuc, name="virt_nuclei", colormap="magenta")
viewer.add_image(mem, name="virt_membrane", colormap="cyan")
napari.run()
