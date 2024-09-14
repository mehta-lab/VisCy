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
from sklearn.decomposition import PCA
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
from viscy.representation.engine import ContrastiveEncoder, ContrastiveModule
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

# %%
fov = "/B/4/6"
track = 4

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
)
model = model.eval()


# %%
# extract features
def feature_map_to_pca_rgb(feature_map: np.ndarray) -> np.ndarray:
    """Compute first 3 principal components of feature map over (T, Z, Y, X).
    Or in other words, reduce the channel dimension to 3 while preserving the most variance.

    Parameters
    ----------
    feature_map : np.ndarray
        feature map of shape (N, C, H, W)

    Returns
    -------
    np.ndarray
        (N, H, W, 3)
    """
    out_shape = (feature_map.shape[0], 3, *feature_map.shape[-2:])
    feature_map = feature_map.reshape(feature_map.shape[1], -1)
    pca = PCA(n_components=3)
    pca.fit(feature_map)
    pc_first_3 = pca.components_.reshape(out_shape)
    pc_first_3 = np.stack([pc_first_3[:, i] for i in range(3)], axis=-1)
    return pc_first_3


def extract_features(img: torch.Tensor) -> list[np.ndarray]:
    with torch.inference_mode():
        features = model.model.stem(img.to(model.device))
        feature_maps = [features.clone()]
        for stage in model.model.encoder.stages:
            features = stage(features)
            feature_maps.append(features)
        feature_maps.append(model.model.encoder.head(features)[..., None, None])
        return [f.detach().cpu().numpy() for f in feature_maps]


# %%
def plot_feature_maps(
    img: torch.Tensor, rgb_features: list[np.ndarray], save_path: Path
) -> None:
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
        rescaled = rescale_intensity(rgb, out_range=(0, 1))
        a.imshow(
            np.stack(
                [
                    rescale_intensity(rescaled[..., c], out_range=(0, 1))
                    for c in range(3)
                ],
                axis=-1,
            )
        )
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
save_dir = (
    Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/test"
    )
    / (fov.strip("/s").replace("/", "_") + "norm_per_ch")
    / str(track)
)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
# load the entire track in one batch
for i, sample in enumerate(dm.predict_dataloader()):
    print(sample["index"]["t"])
    img = sample["anchor"]

# %%
feat = extract_features(img)
for f in feat:
    print(f.shape)

# %%
rgbs = [feature_map_to_pca_rgb(f) for f in feat]

# %%
for i in tqdm(range(feat[0].shape[0])):
    im = img[i : i + 1]
    rgb = [rgbs[j][i] for j in range(len(feat))]
    plot_feature_maps(im, rgb, save_dir / f"{i}.png")
# %%
