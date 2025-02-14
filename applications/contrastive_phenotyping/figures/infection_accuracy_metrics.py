# %% compute accuracy of model from ALFI data using cell infection state classification

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
accuracies = []

features_paths = {
    'timeAware triplet': '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_102ckpt_timeAware_triplet.zarr',
    'cellAware triplet': '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_101ckpt_cellAware_triplet.zarr',
    'classical triplet': '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_101ckpt_classical_triplet.zarr',
    'timeAware ntxent': '/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr',
    'classical ntxent': '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/10-phenotyping/predictions/2chan_192patch_100ckpt_classical_ntxent.zarr',
}

for model_name, path in features_paths.items():
    features_path = Path(path)
    embedding_dataset = read_embedding_dataset(features_path)
    embedding_dataset
    features = embedding_dataset["features"]

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

    ann_root = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred"
    )

    infection = load_annotation(
        embedding_dataset,
        ann_root / "extracted_inf_state.csv",
        "infection_state",
        {1: "uninfected", 2: "infected"},
    )

    # train a linear classifier on half the data

    infection_npy = infection.cat.codes.values
    infection_npy_filtered = infection_npy[infection_npy != 0]

    feature_npy = features.values
    feature_npy_filtered = feature_npy[infection_npy != 0]

    # add time and well info into dataframe
    time_npy = features["t"].values
    time_npy_filtered = time_npy[infection_npy != 0]


    fov_name_list = features["fov_name"].values
    fov_name_list_filtered = fov_name_list[infection_npy != 0]

    data = pd.DataFrame(
        {
            "infection": infection_npy_filtered,
            "time": time_npy_filtered,
            "fov_name": fov_name_list_filtered,
        }
    )
    # Add all 768 features to the dataframe
    feature_columns = pd.DataFrame(feature_npy_filtered, columns=[f"feature_{i+1}" for i in range(768)])
    data = pd.concat([data, feature_columns], axis=1)

    # dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
    data_train_val = data[
        data["fov_name"].str.contains("/A/3/")
        | data["fov_name"].str.contains("/B/4/7")
        | data["fov_name"].str.contains("/B/4/6")
    ]

    data_test = data[
        data["fov_name"].str.contains("/B/4/8")
        | data["fov_name"].str.contains("/B/4/9")
        | data["fov_name"].str.contains("/B/3/")
    ]

    x_train = data_train_val.drop(
        columns=[
            "infection",
            "fov_name",
            "time",
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
        ]
    )
    y_test = data_test["infection"]

    # predict the infection state for the testing set
    y_pred = clf.predict(x_test)

    # compute the accuracy of the classifier

    accuracy = np.mean(y_pred == y_test)*100
    # save the accuracy for final ploting
    print(f"Accuracy of model trained on {model_name} data: {accuracy}")
    accuracies.append(accuracy)

# %% plot the accuracy of the model trained on different time intervals

plt.figure(figsize=(10, 7))
plt.bar(features_paths.keys(), accuracies)
plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel("Accuracy (%)", fontsize=24)
plt.ylim(90, 100)
plt.tight_layout()
plt.show()

# %%
