"""Use pre-trained ImageNet models to extract features from images."""

# %%
import pandas as pd
import seaborn as sns
import timm
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from viscy.data.triplet import TripletDataModule
from viscy.transforms import ScaleIntensityRangePercentilesd

# %%
model = timm.create_model("convnext_tiny", pretrained=True).eval().to("cuda")

# %%
dm = TripletDataModule(
    data_path="/hpc/reference/imaging/ALFI_dataset/Analysis/float_phase_ome_zarr_output_test.zarr",
    tracks_path="/hpc/reference/imaging/ALFI_dataset/Analysis/track_phase_ome_zarr_output_test.zarr",
    source_channel=["DIC"],
    z_range=(0, 1),
    batch_size=128,
    num_workers=8,
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        ScaleIntensityRangePercentilesd(
            keys=["DIC"], lower=50, upper=99, b_min=0.0, b_max=1.0
        )
    ],
)
dm.prepare_data()
dm.setup("predict")

# %%
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        image = batch["anchor"][:, :, 0]
        rgb_image = image.repeat(1, 3, 1, 1).to("cuda")
        features.append(model.forward_features(rgb_image))
        indices.append(batch["index"])

# %%
pooled = torch.cat(features).mean(dim=(2, 3)).cpu().numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])

# %%
scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# %%
ax = sns.kdeplot(
    x=pca_features[:, 0],
    y=pca_features[:, 1],
    hue=tracks["fov_name"].rename("FOV name"),
    legend="full",
)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# %%
