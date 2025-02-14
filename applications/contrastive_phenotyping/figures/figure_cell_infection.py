# %%
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.dimensionality_reduction import compute_phate

# %% Paths and parameters.


features_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"
)

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

features = embedding_dataset["features"]

# %% compute and store phate components

phate_embedding = compute_phate(
    embedding_dataset=embedding_dataset,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)

# %% OVERLAY INFECTION ANNOTATION

# load the infection state annotation
def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov_name"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected
        
ann_root = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred")

infection = load_annotation(
    features,
    ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0:"background", 1: "uninfected", 2: "infected"},
)

# %% filter background class from the data

infection_npy = infection.cat.codes.values
infection_npy_filtered = infection_npy[infection_npy != 0]

feature_npy = features.values
feature_npy_filtered = feature_npy[infection_npy != 0]

# add time and well info into dataframe
time_npy = features["t"].values
time_npy_filtered = time_npy[infection_npy != 0]

phate1_npy = phate_embedding[1][:,0]
phate1_npy_filtered = phate1_npy[infection_npy != 0]

phate2_npy = phate_embedding[1][:,1]
phate2_npy_filtered = phate2_npy[infection_npy != 0]

fov_name_list = features["fov_name"].values
fov_name_list_filtered = fov_name_list[infection_npy != 0]

data = pd.DataFrame(
    {
        "infection": infection_npy_filtered,
        "time": time_npy_filtered,
        "fov_name": fov_name_list_filtered,
        "PHATE1": phate1_npy_filtered,
        "PHATE2": phate2_npy_filtered,
    }
)
# Add all 768 features to the dataframe
feature_columns = pd.DataFrame(feature_npy_filtered, columns=[f"feature_{i+1}" for i in range(768)])
data = pd.concat([data, feature_columns], axis=1)

# %% manually data the dataset into training and testing set by well name

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = data[
    data["fov_name"].str.contains("/B/4/6")
    | data["fov_name"].str.contains("/B/4/7")
    | data["fov_name"].str.contains("/A/3/")
]

# dataframe for testing set, fov names starts with "/B/4/8" or "/B/4/9" or "/A/4/"
data_test = data[
    data["fov_name"].str.contains("/B/4/8")
    | data["fov_name"].str.contains("/B/4/9")
    | data["fov_name"].str.contains("/B/3/")
]

# %% train a linear classifier to predict infection state from PCA components

x_train = data_train_val.drop(
    columns=[
        "infection",
        "fov_name",
        "time",
        "PHATE1",
        "PHATE2",
    ]
)
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "infection",
        "fov_name",
        "time",
        "PHATE1",
        "PHATE2",
    ]
)
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# %% use the trained classifier to perform prediction on the entire dataset

data_test["predicted_infection"] = y_pred

# plot the predicted infection state over time for /B/3 well and /B/4 well
time_points_test = np.unique(data_test["time"])

infected_test_cntrl = []
infected_test_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["time"] == time)
        & (data_test["predicted_infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["time"] == time)
    ].shape[0]
    infected_test_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["time"] == time)
        & (data_test["predicted_infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["time"] == time)
    ].shape[0]
    infected_test_infected.append(infected_cell * 100 / total_cell)


infected_true_cntrl = []
infected_true_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["time"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["time"] == time)
    ].shape[0]
    infected_true_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["time"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["time"] == time)
    ].shape[0]
    infected_true_infected.append(infected_cell * 100 / total_cell)


# %% perform prediction on the mantis dataset

#  Paths and parameters.
features_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/2024_11_07_NTXent_phase_sensor/sensor_phase_160patch_98ckpt_rev5.zarr"
)

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

mantis_features = embedding_dataset["features"]

# %% plot mantis and Feb test combined UMAP

# add time and well info into dataframe
mantis_time_npy = mantis_features["t"].values
mantis_fov_npy = mantis_features["fov_name"].values
mantis_phate1_npy = mantis_features["PHATE1"].values
mantis_phate2_npy = mantis_features["PHATE2"].values

mantis_data = pd.DataFrame(
    {
        "time": mantis_time_npy,
        "fov_name": mantis_fov_npy,
        "PHATE1": mantis_phate1_npy,
        "PHATE2": mantis_phate2_npy,
    }
)

# Add all 768 features to the dataframe
mantis_features_npy = mantis_features.values
for i in range(768):
    mantis_data[f"feature_{i+1}"] = mantis_features_npy[:, i]

# use wells with mCherry sensor only
mantis_data = mantis_data[
    mantis_data["fov_name"].str.contains("/B/3")
    | mantis_data["fov_name"].str.contains("/C/2")
]

# add the predicted infection state
mantis_pred = clf.predict(
    mantis_data.drop(
        columns=[
            "fov_name",
            "time",
            "PHATE1",
            "PHATE2",
        ]
    )
)
mantis_data["predicted_infection"] = mantis_pred

# %% plot % infected over time

time_points_mantis = np.unique(mantis_data["time"])

infected_mantis_cntrl = []
infected_infected = []
mock_wells = '/B/3' # Create regex pattern for mock wells
infected_wells = '/C/2'  # Create regex pattern for dengue infected wells

for time in time_points_mantis:
    infected_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(mock_wells))
        & (mantis_data["time"] == time)
        & (mantis_data["predicted_infection"] == 2)
    ].shape[0]
    total_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(mock_wells)) & (mantis_data["time"] == time)
    ].shape[0]
    if total_mantis!=0:
        infected_mantis_cntrl.append(infected_mantis * 100 / total_mantis)
    else:
        infected_mantis_cntrl.append(0)

    infected_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(infected_wells))
        & (mantis_data["time"] == time)
        & (mantis_data["predicted_infection"] == 2)
    ].shape[0]
    total_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(infected_wells)) & (mantis_data["time"] == time)
    ].shape[0]
    if total_mantis!=0:
        infected_infected.append(infected_mantis * 100 / total_mantis)
    else:
        infected_infected.append(0)


# %% plot infected percentage over time for both wells
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_cntrl,
    label="mock true",
    color="steelblue",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_test_cntrl,
    label="mock predicted",
    color="blue",
    marker="+",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="MOI true",
    color="orange",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_test_infected,
    label="MOI predicted",
    color="red",
    marker="+",
)
plt.plot(
    time_points_mantis * 0.167 + 4,
    infected_mantis_cntrl,
    label="mock new predicted",
    color="blue",
    marker="o",
)
plt.plot(
    time_points_mantis * 0.167 + 4,
    infected_infected,
    label="MOI infected predicted",
    color="red",
    marker="o",
)
plt.xlabel("HPI")
plt.ylabel("Infected percentage")
plt.legend()
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/infected_percentage_withmantis.svg",
    format="svg",
)

# %% appendix video for infection dynamics umap, Feb test data, colored by human revised annotation

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=data_test[(data_test["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="infection",
        palette={1: "steelblue", 2: "orangered"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {(time*0.5+3):.2f} HPI")
    plt.ylim(-0.05, 0.03)
    plt.xlim(-0.05, 0.04)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Feb_true/phate_feb_true_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% appendix video for infection dynamics umap, Feb test data, colored by predicted infection

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=data_test[(data_test["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {(time*0.5+3):.2f} HPI")
    plt.ylim(-0.05, 0.03)
    plt.xlim(-0.05, 0.03)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Feb_test/phate_feb_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% appendix video for infection dynamics umap, mantis data, colored by predicted infection

for time in range(len(time_points_mantis)):
    plt.clf()
    sns.scatterplot(
        data=mantis_data[(mantis_data["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {(time*0.167+4):.2f} HPI")
    plt.ylim(-0.04, 0.04)
    plt.xlim(-0.04, 0.04)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Mantis/phate_mantis_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% 
