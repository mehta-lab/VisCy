# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cmap import Colormap
from lightning.pytorch import seed_everything
from skimage.exposure import rescale_intensity
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.data.triplet import TripletDataModule
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd

plt.style.use("../evaluation/figure.mplstyle")
seed_everything(42, workers=True)

# %% Paths and parameters.

features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/2-register/registered_chunked.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr"
)

# %%
embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
features = features[features["fov_name"].str.contains(r"/0/[36]")]
features

# %%
scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP(random_state=42)
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)


# %%
# Add UMAP coordinates to the dataset

features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

# %%
ax = sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], hue=features["t"], s=7, alpha=0.8
)
fmt = mpl.ticker.StrMethodFormatter("{x}")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

# %%
fovs = ["/0/3/002000", "/0/6/000000", "/0/6/000002", "/0/6/001000"]
tracks = [24, 14, 34, 38]


track_features = xr.concat(
    [features.sel(fov_name=fov, track_id=track) for fov, track in zip(fovs, tracks)],
    dim="sample",
)

# %%
dm = TripletDataModule(
    data_path=data_path,
    tracks_path=tracks_path,
    source_channel=[
        "Phase3D",
        "MultiCam_GFP_mCherry_BF-Prime BSI Express",
        "MultiCam_GFP_mCherry_BF-Andor EMCCD",
    ],
    z_range=[10, 55],
    batch_size=48,
    num_workers=0,
    initial_yx_patch_size=(128, 128),
    final_yx_patch_size=(128, 128),
    normalizations=[
        NormalizeSampled(
            keys=["Phase3D"], level="fov_statistics", subtrahend="mean", divisor="std"
        ),
        ScaleIntensityRangePercentilesd(
            keys=[
                "MultiCam_GFP_mCherry_BF-Prime BSI Express",
                "MultiCam_GFP_mCherry_BF-Andor EMCCD",
            ],
            lower=50,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            channel_wise=True,
        ),
    ],
    predict_cells=True,
    include_fov_names=fovs,
    include_track_ids=tracks,
)
dm.setup("predict")
ds = dm.predict_dataset
len(ds)


# %%
def render(img, cmaps: list[str]):
    channels = []
    for ch, cmap in zip(img, cmaps):
        lo, hi = np.percentile(ch, [1, 99])
        rescaled = rescale_intensity(ch.clip(lo, hi), out_range=(0, 1))
        rendered = Colormap(cmap)(rescaled)
        channels.append(rendered)
    return np.sum(channels, axis=0).clip(0, 1)


renders = []

f, ax = plt.subplots(4, 12, figsize=(12, 4))
for sample, a in zip(ds, ax.flatten()):
    img = sample["anchor"][1:].numpy().max(1)
    rend = render(img, ["magenta", "green"])
    renders.append(rend)
    a.imshow(rend, cmap="gray")
    idx = sample["index"]
    name = "-".join([str(idx["track_id"]), str(idx["t"])])
    a.set_title(name)
    a.axis("off")

# %%
track_df = ds.tracks
selected_times = [2, 6, 8]
track_df = track_df[track_df["t"].isin(selected_times)]
selected_features = track_features[track_features["t"].isin(selected_times)]
selected_renders = [renders[i] for i in track_df.index]


# %%
fig = plt.figure(layout="constrained", figsize=(5.5, 2.7))
subfigs = fig.subfigures(1, 2, wspace=0.02, width_ratios=[4, 7])

umap_fig = subfigs[0]
umap_fig.suptitle("a", horizontalalignment="left", x=0, y=1)
umap_ax = umap_fig.subplots(1, 1)
umap_ax.invert_xaxis()

sns.scatterplot(
    x=features["UMAP1"], y=features["UMAP2"], s=40, alpha=0.01, ax=umap_ax, color="k"
)

sns.scatterplot(
    x=track_features["UMAP1"],
    y=track_features["UMAP2"],
    ax=umap_ax,
    hue=track_features["fov_name"],
    s=5,
    legend=False,
)

sns.lineplot(
    x=track_features["UMAP1"],
    y=track_features["UMAP2"],
    ax=umap_ax,
    hue=track_features["fov_name"],
    legend=False,
    size=0.5,
)

hpi = (track_df["t"].reset_index(0, drop=True) * 2 + 2.5).astype(str) + " HPI"
track_names = pd.Series(
    np.concatenate([[t] * 3 for t in ["Track 1", "Track 2", "Track 3", "Track 4"]]),
    name="track",
)
sns.scatterplot(
    x=selected_features["UMAP1"],
    y=selected_features["UMAP2"],
    ax=umap_ax,
    style=hpi,
    markers=["P", "s", "D"],
    s=20,
    hue=track_names,
    # legend=False,
)
handles, labels = umap_ax.get_legend_handles_labels()
umap_ax.legend(
    handles=handles[1:5] + handles[6:],
    labels=labels[1:5] + labels[6:],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.2),
    labelspacing=0.2,
    handletextpad=0,
    fontsize=8,
)

img_fig = subfigs[1]
img_fig.suptitle("b", horizontalalignment="left", x=-0, y=1)
img_axes = img_fig.subplots(3, 4, sharex=True, sharey=True)

for i, (ax, rend, time, track_name) in enumerate(
    zip(img_axes.T.flatten(), selected_renders, hpi.to_list(), track_names)
):
    ax.imshow(rend)
    if i % 3 == 0:
        ax.set_title(track_name)
    if i < 3:
        ax.set_ylabel(f"{time}")
    ax.set_xticks([])
    ax.set_yticks([])

for sf in subfigs:
    for a in sf.get_axes():
        fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
        a.xaxis.set_major_formatter(fmt)
        a.yaxis.set_major_formatter(fmt)

# %%
fig.savefig(
    Path.home()
    / "gdrive/publications/learning_impacts_of_infection/fig_manuscript/fig_organelle_dynamics/fig_organelle_dynamics.pdf",
    dpi=300,
)

# %%
