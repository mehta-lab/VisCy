# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cmap import Colormap
from iohub import open_ome_zarr
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_umap

# %%
t_slice = slice(18, 33)
y_slice = slice(16, 144)
x_slice = slice(0, 224)

phase = open_ome_zarr(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr/B/4/8"
)["0"][t_slice, 3, 31, y_slice, x_slice]

segments = open_ome_zarr(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr/B/4/8"
)["0"][t_slice, 0, 0, y_slice, x_slice]

# %%
features = read_embedding_dataset(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)

# %%
_, _, umap_df = compute_umap(features)
umap_df

# %%
track_ids = np.unique(segments)[1:]
track_ids

# %%
selected_umap = umap_df[
    (umap_df["fov_name"] == "/B/4/8")
    & umap_df["track_id"].isin(track_ids)
    & (umap_df["t"] >= t_slice.start)
    & (umap_df["t"] < t_slice.stop)
]

selected_umap["HPI"] = selected_umap["t"] * 0.5 + 3

# %%
plt.style.use("../evaluation/figure.mplstyle")
fig = plt.figure(figsize=(5.5, 4.5), layout="constrained")
subfigs = fig.subfigures(2, 1, wspace=0.02, height_ratios=[3, 2])

img_fig = subfigs[0]
img_fig.suptitle("a", horizontalalignment="left", x=0, y=1)
img_ax = img_fig.subplots(3, 5)

clim = 0.03
cmap = Colormap("tab10")

labels = label2rgb(
    segments,
    image=rescale_intensity(phase, in_range=(-clim, clim), out_range=(0, 1)),
    colors=cmap(range(10)),
)

for t, (a, rgb) in enumerate(zip(img_ax.flatten(), labels)):
    a.imshow(rgb)
    a.set_title(f"{(t+t_slice.start)/2 + 3} HPI")
    a.axis("off")

line_fig = subfigs[1]
line_fig.suptitle("b", horizontalalignment="left", x=0, y=1)
line_ax_1 = line_fig.subplots(1, 1)
line_ax_2 = line_ax_1.twinx()
sns.lineplot(
    data=selected_umap,
    x="HPI",
    y="UMAP1",
    hue="track_id",
    palette=[c for c in cmap([2, 4, 6])],
    ax=line_ax_1,
)
sns.move_legend(line_ax_1, "upper right", title="Track ID")
sns.lineplot(
    data=selected_umap,
    x="HPI",
    y="UMAP2",
    hue="track_id",
    palette=[c for c in cmap([2, 4, 6])],
    ax=line_ax_2,
    linestyle="--",
    legend=False,
)

fmt = mpl.ticker.StrMethodFormatter("{x:.1f}")
for a in [line_ax_1, line_ax_2]:
    a.xaxis.set_major_formatter(fmt)
    a.yaxis.set_major_formatter(fmt)

# %%
fig.savefig(
    Path.home()
    / "gdrive/publications/learning_impacts_of_infection/fig_manuscript/si/appendix_track_smoothness.pdf",
    dpi=300,
)

# %%
