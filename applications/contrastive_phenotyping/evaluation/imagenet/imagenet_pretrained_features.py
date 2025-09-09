"""Use pre-trained ImageNet models to extract features from images."""

# %%
import pandas as pd
import seaborn as sns
import timm
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from viscy.data.triplet import TripletDataModule
from viscy.transforms import ScaleIntensityRangePercentilesd

# %%
model = timm.create_model("convnext_tiny", pretrained=True).eval().to("cuda")

# %%
dm = TripletDataModule(
    data_path="/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_test.zarr",
    tracks_path="/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/track_phase_ome_zarr_output_test.zarr",
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

# %% add pooled to dataframe naming each column with feature_i
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# add pca features to dataframe naming each column with pca_i
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature

# # save the dataframe as csv
# tracks.to_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv", index=False)

# %% load the dataframe
# tracks = pd.read_csv("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/code/ALFI/imagenet_pretrained_features.csv")

# %% load annotations

ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)
ann_path = ann_root / "test_annotations.csv"
annotation = pd.read_csv(ann_path)

# add division column from annotation to tracks
tracks["division"] = annotation["division"]

# %%
ax = sns.scatterplot(
    x=tracks["pca_0"],
    y=tracks["pca_1"],
    hue=tracks["division"],
    legend="full",
)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# %% compute the accuracy of the model using a linear classifier

# remove rows with division = -1
tracks = tracks[tracks["division"] != -1]

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = tracks[
    tracks["fov_name"].str.contains("/0/0/0")
    | tracks["fov_name"].str.contains("/0/1/0")
    | tracks["fov_name"].str.contains("/0/2/0")
]

data_test = tracks[
    tracks["fov_name"].str.contains("/0/3/0")
    | tracks["fov_name"].str.contains("/0/4/0")
]

x_train = data_train_val.drop(
    columns=[
         "division",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_train = data_train_val["division"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "division",
        "fov_name",
        "t",
        "track_id",
        "id",
        "parent_id",
        "parent_track_id",
        "pca_0",
        "pca_1",
    ]
)
y_test = data_test["division"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# compute the accuracy of the classifier

accuracy = np.mean(y_pred == y_test)
# save the accuracy for final ploting
print(f"Accuracy of model: {accuracy}")

# %%
