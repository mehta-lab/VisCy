
# %%

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
 
features_path = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/log_alfi_triplet_time_intervals/prediction/ALFI_91mins.zarr"
)
# data_path = Path(
#     "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
# )
# tracks_path = Path(
#     "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/3-track/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
# )

# %%

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

PHATE1 = embedding_dataset["PHATE1"].values
PHATE2 = embedding_dataset["PHATE2"].values

# %% plot PHATE map based on the embedding dataset time points

sns.scatterplot(
    x=embedding_dataset["PHATE1"], y=embedding_dataset["PHATE2"], hue=embedding_dataset["t"], s=7, alpha=0.8
)

# %% color using human annotation for cell cycle state

def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    # annotation_columns = annotation.columns.tolist()
    # print(annotation_columns)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected


# %% load the cell cycle state annotation

ann_root = Path(
    "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
)

division = load_annotation(
    embedding_dataset,
    ann_root / "test_annotations.csv",
    "division",
    {0: "interphase", 1: "mitosis"},
)

# %% plot PHATE map based on the cell cycle annotation

sns.scatterplot(
    x=embedding_dataset["PHATE1"], y=embedding_dataset["PHATE2"], hue=division, s=7, alpha=0.8
)

# %% plot intercative plot to hover over the points on scatter plot and see the fov_name and track_id

import plotly.express as px

fig = px.scatter(
    embedding_dataset.to_dataframe(),
    x="PHATE1",
    y="PHATE2",
    color=division,
    hover_data=["fov_name", "id"],
)

# %% 
# find row index in 'division' where the value is -1
division[division == -1].index
# find the track_id and 't' value of cell instance where 'fov_name' is '/0/0/0' and 'id' is 1000941
embedding_dataset.where(embedding_dataset["fov_name"] == "/0/0/0", drop=True).where(
    embedding_dataset["id"] == 1000942, drop=True
)

# %%
