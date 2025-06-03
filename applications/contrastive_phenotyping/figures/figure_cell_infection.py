# %%
import sys
from pathlib import Path

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation

# %% Paths and parameters.


features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)


# %%
embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

# %%
# Compute UMAP over all features
features = embedding_dataset["features"]
# or select a well:
# features = features[features["fov_name"].str.contains("B/4")]


scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)

features = (
    features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
features

pca = PCA(n_components=4)
# scaled_features = StandardScaler().fit_transform(features.values)
# pca_features = pca.fit_transform(scaled_features)
pca_features = pca.fit_transform(features.values)


features = (
    features.assign_coords(PCA1=("sample", pca_features[:, 0]))
    .assign_coords(PCA2=("sample", pca_features[:, 1]))
    .assign_coords(PCA3=("sample", pca_features[:, 2]))
    .assign_coords(PCA4=("sample", pca_features[:, 3]))
    .set_index(sample=["PCA1", "PCA2", "PCA3", "PCA4"], append=True)
)

# %% OVERLAY INFECTION ANNOTATION
ann_root = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
)

infection = load_annotation(
    features,
    ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)

# %% plot the umap

# remove the rows in umap and annotation for background class
# Convert UMAP coordinates to a DataFrame
umap_npy = embedding.copy()
infection_npy = infection.cat.codes.values

# Filter out the background class
umap_npy_filtered = umap_npy[infection_npy != 0]
infection_npy_filtered = infection_npy[infection_npy != 0]

feature_npy = features.values
feature_npy_filtered = feature_npy[infection_npy != 0]

sns.scatterplot(
    x=umap_npy_filtered[:, 0],
    y=umap_npy_filtered[:, 1],
    hue=infection_npy_filtered,
    palette={1: "steelblue", 2: "orangered"},
    hue_order=[1, 2],
    s=7,
    alpha=0.8,
)
plt.legend([], [], frameon=False)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/umap_infection.png",
    format="png",
    dpi=300,
)

# %% plot PCA components with infection hue

pca_npy = pca_features.copy()
pca_npy_filtered = pca_npy[infection_npy != 0]

sns.scatterplot(
    x=pca_npy_filtered[:, 0],
    y=pca_npy_filtered[:, 1],
    hue=infection_npy_filtered,
    palette={1: "steelblue", 2: "orangered"},
    hue_order=[1, 2],
    s=7,
    alpha=0.8,
)
plt.legend([], [], frameon=False)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/pca_infection.png",
    format="png",
    dpi=300,
)

# %% combine the umap, pca and infection annotation in one dataframe

data = pd.DataFrame(
    {
        "UMAP1": umap_npy_filtered[:, 0],
        "UMAP2": umap_npy_filtered[:, 1],
        "PCA1": pca_npy_filtered[:, 0],
        "PCA2": pca_npy_filtered[:, 1],
        "PCA3": pca_npy_filtered[:, 2],
        "PCA4": pca_npy_filtered[:, 3],
        "infection": infection_npy_filtered,
    }
)

# add time and well info into dataframe
time_npy = features["t"].values
time_npy_filtered = time_npy[infection_npy != 0]
data["time"] = time_npy_filtered

fov_name_list = features["fov_name"].values
fov_name_list_filtered = fov_name_list[infection_npy != 0]
data["fov_name"] = fov_name_list_filtered

# Add all 768 features to the dataframe
for i in range(768):
    data[f"feature_{i+1}"] = feature_npy_filtered[:, i]

# %% manually split the dataset into training and testing set by well name

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

from sklearn.linear_model import LogisticRegression

x_train = data_train_val.drop(
    columns=[
        "infection",
        "fov_name",
        "time",
        "UMAP1",
        "UMAP2",
        "PCA1",
        "PCA2",
        "PCA3",
        "PCA4",
    ]
)
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

x_test = data_test.drop(
    columns=[
        "infection",
        "fov_name",
        "time",
        "UMAP1",
        "UMAP2",
        "PCA1",
        "PCA2",
        "PCA3",
        "PCA4",
    ]
)
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# %% construct confusion matrix to compare the true and predicted infection state

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Percentage)")
plt.xticks(ticks=[0.5, 1.5], labels=["uninfected", "infected"])
plt.yticks(ticks=[0.5, 1.5], labels=["uninfected", "infected"])
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/confusion_matrix.svg",
    format="svg",
)

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


# %% perform prediction on the june dataset

#  Paths and parameters.
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
june_features = embedding_dataset["features"]

scaled_features = StandardScaler().fit_transform(june_features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)

june_features = (
    june_features.assign_coords(UMAP1=("sample", embedding[:, 0]))
    .assign_coords(UMAP2=("sample", embedding[:, 1]))
    .set_index(sample=["UMAP1", "UMAP2"], append=True)
)
june_features

pca = PCA(n_components=4)
pca_features = pca.fit_transform(june_features.values)

# %%

# sns.scatterplot(
#    x=june_features["UMAP1"],
#    y=june_features["UMAP2"],
#    hue=june_pred,
#    palette={1: 'blue', 2: 'red'},
#    hue_order=[1, 2],
#    s=7,
#    alpha=0.8,
# )
# plt.legend([], [], frameon=False)
# plt.xlim(0, 15)
# plt.savefig('/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/june_umap_infection.png', format='png', dpi=300)

# %% plot June and Feb test combined UMAP

june_umap_npy = embedding.copy()
june_pca_npy = pca_features.copy()
june_data = pd.DataFrame(
    {
        "UMAP1": june_umap_npy[:, 0],
        "UMAP2": june_umap_npy[:, 1],
        "PCA1": june_pca_npy[:, 0],
        "PCA2": june_pca_npy[:, 1],
        "PCA3": june_pca_npy[:, 2],
        "PCA4": june_pca_npy[:, 3],
        "infection": np.nan,
    }
)

# add time and well info into dataframe
june_data["time"] = june_features["t"].values

june_data["fov_name"] = june_features["fov_name"].values

# Add all 768 features to the dataframe
june_features_npy = june_features.values
for i in range(768):
    june_data[f"feature_{i+1}"] = june_features_npy[:, i]

# use one mock and one dengue infecected well only
june_data = june_data[
    june_data["fov_name"].str.contains("/0/6")
    | june_data["fov_name"].str.contains("/0/2")
]

# add the predicted infection state
june_pred = clf.predict(
    june_data.drop(
        columns=[
            "infection",
            "fov_name",
            "time",
            "UMAP1",
            "UMAP2",
            "PCA1",
            "PCA2",
            "PCA3",
            "PCA4",
        ]
    )
)
june_data["predicted_infection"] = june_pred

# %% combine the june and feb data

combined_data = pd.concat([data_test, june_data])

# perform the umap analysis again with the 768 features
features = combined_data.drop(
    columns=[
        "infection",
        "predicted_infection",
        "fov_name",
        "time",
        "UMAP1",
        "UMAP2",
        "PCA1",
        "PCA2",
        "PCA3",
        "PCA4",
    ]
)
scaled_features = StandardScaler().fit_transform(features.values)
umap = UMAP()
# Fit UMAP on all features
embedding = umap.fit_transform(scaled_features)

# overwrite the umap coordinates on combined data
combined_data["UMAP1"] = embedding[:, 0]
combined_data["UMAP2"] = embedding[:, 1]

# plot the combined data with 'fov_name' starting with '/A and '/B' hue 'infection' and '/0' hue 'predicted_infection'
Feb_split = combined_data[
    combined_data["fov_name"].str.contains("/A")
    | combined_data["fov_name"].str.contains("/B")
]
June_split = combined_data[combined_data["fov_name"].str.contains("/0")]

sns.scatterplot(
    x=June_split["UMAP1"],
    y=June_split["UMAP2"],
    hue=June_split["predicted_infection"],
    palette={1: "blue", 2: "red"},
    hue_order=[1, 2],
    s=7,
    alpha=0.8,
)
sns.scatterplot(
    x=Feb_split["UMAP1"],
    y=Feb_split["UMAP2"],
    hue=Feb_split["infection"],
    palette={1: "steelblue", 2: "orange"},
    hue_order=[1, 2],
    s=7,
    alpha=0.8,
)
plt.legend([], [], frameon=False)
# plt.savefig('/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/combined_umap_infection.png', format='png', dpi=300)

# plot the scatterplot hue well name '/A' and '/B' are blue and '/0' are red
combined_data["color"] = combined_data["fov_name"].apply(
    lambda x: "brown" if x.startswith("/0") else "green"
)

sns.scatterplot(
    x=combined_data["UMAP1"],
    y=combined_data["UMAP2"],
    hue="color",
    palette={"green": "green", "brown": "brown"},
    data=combined_data,
    s=7,
    alpha=0.2,  # Increased transparency
)
plt.xlim(-5, 5)
plt.ylim(-2, 20)
plt.legend([], [], frameon=False)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/combined_umap_well.png",
    format="png",
    dpi=300,
)

# plot the predicted infection state with combined data
sns.scatterplot(
    x=combined_data["UMAP1"],
    y=combined_data["UMAP2"],
    hue=combined_data["predicted_infection"],
    palette={1: "blue", 2: "red"},
    hue_order=[1, 2],
    s=7,
    alpha=0.8,
)
plt.xlim(-5, 5)
plt.ylim(-2, 20)
plt.legend([], [], frameon=False)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/combined_umap_predicted_infection.png",
    format="png",
    dpi=300,
)

# %% plot % infected over time

time_points_june = np.unique(June_split["time"])

infected_june_cntrl = []
infected_june_infected = []

for time in time_points_june:
    infected_june = June_split[
        (June_split["fov_name"].str.startswith("/0/2"))
        & (June_split["time"] == time)
        & (June_split["predicted_infection"] == 2)
    ].shape[0]
    total_june = June_split[
        (June_split["fov_name"].str.startswith("/0/2")) & (June_split["time"] == time)
    ].shape[0]
    infected_june_cntrl.append(infected_june * 100 / total_june)
    infected_june = June_split[
        (June_split["fov_name"].str.startswith("/0/6"))
        & (June_split["time"] == time)
        & (June_split["predicted_infection"] == 2)
    ].shape[0]
    total_june = June_split[
        (June_split["fov_name"].str.startswith("/0/6")) & (June_split["time"] == time)
    ].shape[0]
    infected_june_infected.append(infected_june * 100 / total_june)


# plot infected percentage over time for both wells
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
    time_points_june * 2 + 3,
    infected_june_cntrl,
    label="mock new predicted",
    color="blue",
    marker="o",
)
plt.plot(
    time_points_june * 2 + 3,
    infected_june_infected,
    label="MOI new predicted",
    color="red",
    marker="o",
)
plt.xlabel("HPI")
plt.ylabel("Infected percentage")
plt.legend()
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/infected_percentage_withJune.svg",
    format="svg",
)

# %% appendix video for infection dynamics umap, Feb test data, colored by human revised annotation

for time in range(48):
    plt.clf()
    sns.scatterplot(
        data=data_test[(data_test["time"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="infection",
        palette={1: "steelblue", 2: "orangered"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {time*0.5+3} HPI")
    plt.ylim(-10, 20)
    plt.xlim(2, 18)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/video_umap/umap_feb_true_infection_"
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
        x="UMAP1",
        y="UMAP2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {time*0.5+3} HPI")
    plt.ylim(-10, 18)
    plt.xlim(2, 18)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/video_umap/umap_feb_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% appendix video for infection dynamics umap, June data, colored by predicted infection

for time in range(12):
    plt.clf()
    sns.scatterplot(
        data=June_split[(June_split["time"] == time)],
        x="UMAP1",
        y="UMAP2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=20,
        alpha=0.8,
    )
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["uninfected", "infected"])
    plt.suptitle(f"Time: {time*2+3} HPI")
    plt.ylim(-8, 10)
    plt.xlim(-5, 5)
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/infection/video_umap/umap_june_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %%
