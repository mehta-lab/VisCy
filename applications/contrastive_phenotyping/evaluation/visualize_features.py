# %%
"""
Modified from viscy/scripts/visualize_features.py
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.representation.evaluation import feature_map_to_pca_rgb
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

# %%
fov = "/B/4/6"
track = 4

save_dir = (
    Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/test"
    )
    / "pca"
    / (fov.strip("/s").replace("/", "_") + "layernorm")
    / str(track)
)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr",
    source_channel=["Phase3D", "RFP"],
    z_range=[25, 40],
    batch_size=48,
    num_workers=0,
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        NormalizeSampled(
            keys=["Phase3D"], level="fov_statistics", subtrahend="mean", divisor="std"
        ),
        ScaleIntensityRangePercentilesd(
            keys=["RFP"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
    ],
    predict_cells=True,
    include_fov_names=[fov],
    include_track_ids=[track],
)
dm.setup("predict")
len(dm.predict_dataset)

# %%
# load model
model = ContrastiveModule.load_from_checkpoint(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/epoch=178-step=16826.ckpt",
    encoder=ContrastiveEncoder(
        backbone="convnext_tiny",
        in_channels=2,
        in_stack_depth=15,
        stem_kernel_size=(5, 4, 4),
        stem_stride=(5, 4, 4),
        embedding_dim=768,
        projection_dim=32,
    ),
).eval()


# %%
def plot_feature_maps_video(
    img: torch.Tensor,
    rgb_features: list[np.ndarray],
    save_path: Path,
    rescale_per_timepoint: bool = True,
    rescale_per_color: bool = True,
) -> None:
    """Render one frame showing the input image and the feature maps.

    Parameters
    ----------
    img : torch.Tensor
        2-channel input image
    rgb_features : list[np.ndarray]
        RGB feature maps
    save_path : Path
        save path for matplotlib figure
    rescale_per_color : bool, optional
        Rescale intensity per channel for feature maps, by default True
    """
    input_ = img.cpu().numpy()

    f, ax = plt.subplots(2, 4, figsize=(8, 4))
    ax[0, 0].imshow(input_[0, 0, 7], cmap="gray", vmin=-3, vmax=3)
    ax[0, 0].set_title("Phase")
    ax[1, 0].imshow(input_[0, 1, 7], cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title("RFP")

    for rgb, a, name in zip(
        rgb_features,
        ax[:, 1:].flatten(),
        ["Stem", *[f"Encoder stage {i}" for i in range(4)], "Pooled features"],
    ):
        if rescale_per_timepoint:
            rgb = rescale_intensity(rgb, out_range=(0, 1))
        if rescale_per_color:
            rgb = np.stack(
                [rescale_intensity(rgb[..., c], out_range=(0, 1)) for c in range(3)],
                axis=-1,
            )
        a.imshow(rgb)
        a.set_title(name)

    legend_table = {"PC1": (1, 0, 0), "PC2": (0, 1, 0), "PC3": (0, 0, 1)}
    handles = [Rectangle((0, 0), 1, 1, color=v) for v in legend_table.values()]
    f.legend(
        handles,
        legend_table.keys(),
        title="feature\nprincipal\ncomponents",
        bbox_to_anchor=(1.001, 0.6),
        loc="upper left",
    )
    for a in ax.flatten():
        a.axis("off")
    timepoint = int(save_path.stem)
    f.suptitle(f"{(timepoint * 0.5 + 3):.1f} HPI")
    f.tight_layout()
    f.savefig(save_path, dpi=300)
    plt.close()


# %%
# load the entire track in one batch
for i, sample in enumerate(dm.predict_dataloader()):
    print(sample["index"]["t"])
    img = sample["anchor"]

# %%
feat = model.extract_features(img)
for f in feat:
    print(f.shape)


# %%
rgbs = [
    rescale_intensity(feature_map_to_pca_rgb(f, normalize=True), out_range=(0, 1))
    for f in feat
]

# %%
for i in tqdm(range(feat[0].shape[0])):
    im = img[i : i + 1]
    rgb = [rgbs[j][i] for j in range(len(feat))]
    plot_feature_maps_video(
        im,
        rgb,
        save_dir / f"{i}.png",
        rescale_per_timepoint=False,
        rescale_per_color=False,
    )

# %%
