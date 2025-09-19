
# %%
from pathlib import Path

import pandas as pd

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

infection_npy = infection.cat.codes.values

# Filter out the background class
infection_npy_filtered = infection_npy[infection_npy != 0]

feature_npy = features.values
feature_npy_filtered = feature_npy[infection_npy != 0]

# %% combine the umap, pca and infection annotation in one dataframe

data = pd.DataFrame({"infection": infection_npy_filtered})

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
data_train_val = data[data["fov_name"].str.contains("/B/4/6") | data["fov_name"].str.contains("/B/4/7") | data["fov_name"].str.contains("/A/3/")]

# dataframe for testing set, fov names starts with "/B/4/8" or "/B/4/9" or "/A/4/"
data_test = data[data["fov_name"].str.contains("/B/4/8") | data["fov_name"].str.contains("/B/4/9") | data["fov_name"].str.contains("/B/3/")]

# %% train a linear classifier to predict infection state from PCA components

from sklearn.linear_model import LogisticRegression

x_train = data_train_val.drop(columns=["infection", "fov_name", "time"])
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

x_test = data_test.drop(columns=["infection", "fov_name", "time"])
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)

# %%
