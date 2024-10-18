# %% figures for visualizing the results of cell division
import sys

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
# single channel. with temporal regularizations
# dataset = read_embedding_dataset(
#     "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval_phase/predictions/epoch_186/1chan_128patch_186ckpt_Febtest.zarr"
# )
# dataset

# single cahnnel, without temporal regularizations
# dataset = read_embedding_dataset(
#     "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_1chan_128patch_32projDim/1chan_128patch_63ckpt_FebTest_divGT.zarr"
# )
# dataset

# two channel, with temporal regularizations
# dataset = read_embedding_dataset(
#     "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178_gt_tracks.zarr"
# )
# dataset

# two channel, without temporal regularizations
dataset = read_embedding_dataset(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest_divGT.zarr"
)
dataset

# %%
# load all unprojected features:
features = dataset["features"]
# or select a well:
# features - features[features["fov_name"].str.contains("B/4")]
features

# %% umap with 2 components
scaled_features = StandardScaler().fit_transform(features.values)

umap = UMAP()

embedding = umap.fit_transform(features.values)
features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

# %%


def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    # annotation_columns = annotation.columns.tolist()
    # print(annotation_columns)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


# %%

ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/9-lineage-cell-division/lineages_gt"
)

division = load_annotation(
    features,
    ann_root / "cell_division_state_test_set.csv",
    "division",
    {0: "interphase", 2: "mitosis"},
)

# %%
sns.scatterplot(
    x=features["UMAP1"],
    y=features["UMAP2"],
    hue=division,
    palette={"interphase": "steelblue", 1: "green", "mitosis": "orangered"},
    s=7,
    alpha=0.8,
)
plt.show()
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/UMAP_cellDiv_GTtracking_sc_woT.svg"
# )

# %%
no_inter = division[division == "interphase"].count()
no_div = division[division == "mitosis"].count()

# %% plot the trajectory quiver of one cell on top of the UMAP

from matplotlib.patches import FancyArrowPatch

cell_parent = features[
    (features["fov_name"].str.contains("A/3/7")) & (features["track_id"].isin([13]))
]
cell_daughter1 = features[
    (features["fov_name"].str.contains("A/3/7")) & (features["track_id"].isin([14]))
]
cell_daughter2 = features[
    (features["fov_name"].str.contains("A/3/7")) & (features["track_id"].isin([15]))
]


# %% Plot: Adding arrows to indicate trajectory direction
def add_arrows(df, color):
    for i in range(df.shape[0] - 1):
        start = df.iloc[i]
        end = df.iloc[i + 1]
        arrow = FancyArrowPatch(
            (start["UMAP1"], start["UMAP2"]),
            (end["UMAP1"], end["UMAP2"]),
            color=color,
            arrowstyle="->",
            mutation_scale=20,  # reduce the size of arrowhead by half
            lw=2,
            shrinkA=0,
            shrinkB=0,
        )
        plt.gca().add_patch(arrow)


# tried A/3/7, 8 to 9 & 10
# tried A/3/7, 13 to 14 & 15
# tried A/3/7, 18 to 19 & 20
# tried A/3/8, 23 to 24 & 25

sns.scatterplot(
    x=features["UMAP1"],
    y=features["UMAP2"],
    hue=division,
    palette={"interphase": "steelblue", 1: "green", "mitosis": "orangered"},
    s=7,
    alpha=0.5,
)

# Apply arrows to the trajectories
add_arrows(cell_parent.to_dataframe(), color="black")
add_arrows(cell_daughter1.to_dataframe(), color="red")
add_arrows(cell_daughter2.to_dataframe(), color="blue")

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
# plt.title('UMAP with Trajectory Direction')
# plt.legend(title='Division Phase')
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.legend([], [], frameon=False)
# plt.show()

# single channel, with temporal regularizations
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_singelChannel.png",
    dpi=300,
)

# single channel, without temporal regularizations
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_singelChannel_woT.png",
#     dpi=300
# )

# two channel, with temporal regularizations
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_2Channel.png",
#     dpi=300
# )

# two channel, without temporal regularizations
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_2Channel_woT.png",
#     dpi=300
# )

# %% Plot: display one arrow at end of trajectory of cell overlayed on UMAP

sns.scatterplot(
    x=features["UMAP1"],
    y=features["UMAP2"],
    hue=division,
    palette={"interphase": "steelblue", 1: "green", "mitosis": "orangered"},
    s=27,
    alpha=0.5,
)

sns.lineplot(x=cell_parent["UMAP1"], y=cell_parent["UMAP2"], color="black", linewidth=2)
sns.lineplot(
    x=cell_daughter1["UMAP1"], y=cell_daughter1["UMAP2"], color="blue", linewidth=2
)
sns.lineplot(
    x=cell_daughter2["UMAP1"], y=cell_daughter2["UMAP2"], color="red", linewidth=2
)

parent_arrow = FancyArrowPatch(
    (cell_parent["UMAP1"].values[-2], cell_parent["UMAP2"].values[-2]),
    (cell_parent["UMAP1"].values[-1], cell_parent["UMAP2"].values[-1]),
    color="black",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(parent_arrow)
daughter1_arrow = FancyArrowPatch(
    (cell_daughter1["UMAP1"].values[0], cell_daughter1["UMAP2"].values[0]),
    (cell_daughter1["UMAP1"].values[1], cell_daughter1["UMAP2"].values[1]),
    color="blue",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter1_arrow)
daughter2_arrow = FancyArrowPatch(
    (cell_daughter2["UMAP1"].values[0], cell_daughter2["UMAP2"].values[0]),
    (cell_daughter2["UMAP1"].values[1], cell_daughter2["UMAP2"].values[1]),
    color="red",
    arrowstyle="->",
    mutation_scale=20,  # reduce the size of arrowhead by half
    lw=2,
    shrinkA=0,
    shrinkB=0,
)
plt.gca().add_patch(daughter2_arrow)


# single channel, with temporal regularizations
# plt.xlim(-5, 8)
# plt.ylim(-6, 8)
# plt.legend([], [], frameon=False)
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_singelChannel_arrow.png",
#     dpi=300,
# )

# single channel, without temporal regularizations
# plt.xlim(0, 13)
# plt.ylim(-2, 6)
# plt.legend([], [], frameon=False)
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_singelChannel_woT_arrow.png",
#     dpi=300
# )

# two channel, with temporal regularizations
# plt.xlim(-2, 15)
# plt.ylim(-5, 5)
# plt.legend([], [], frameon=False)
# plt.savefig(
#     "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_2Channel_arrow.png",
#     dpi=300
# )

# two channel, without temporal regularizations
plt.xlim(-3, 12)
plt.ylim(1, 10)
plt.legend([], [], frameon=False)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/cell_division/cellDiv_trajectory_2Channel_woT_arrow.png",
    dpi=300,
)

# %%
