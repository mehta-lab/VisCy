
# %% compute accuracy of model from ALFI data using cell division state classification

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
accuracies = []

features_paths = {
    '7 min interval': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_7mins.zarr',
    '14 min interval': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_14mins.zarr',
    '28 min interval': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_28mins.zarr',
    '56 min interval': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_56mins.zarr',
    '91 min interval': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_91mins.zarr',
    'classical': '/hpc/projects/organelle_phenotyping/ALFI_ntxent_loss/logs_alfi_ntxent_time_intervals/predictions/ALFI_classical.zarr',
}

for interval_name, path in features_paths.items():
    features_path = Path(path)
    embedding_dataset = read_embedding_dataset(features_path)
    embedding_dataset
    features = embedding_dataset["features"]

    # load the cell cycle state annotation

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

    ann_root = Path(
        "/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets"
    )

    division = load_annotation(
        embedding_dataset,
        ann_root / "test_annotations.csv",
        "division",
        {0: "interphase", 1: "mitosis"},
    )

    # train a linear classifier on half the data

    division_npy = division.cat.codes.values
    division_npy_filtered = division_npy[division_npy != -1]

    feature_npy = features.values
    feature_npy_filtered = feature_npy[division_npy != -1]

    # add time and well info into dataframe
    time_npy = features["t"].values
    time_npy_filtered = time_npy[division_npy != -1]


    fov_name_list = features["fov_name"].values
    fov_name_list_filtered = fov_name_list[division_npy != -1]

    data = pd.DataFrame(
        {
            "division": division_npy_filtered,
            "time": time_npy_filtered,
            "fov_name": fov_name_list_filtered,
        }
    )
    # Add all 768 features to the dataframe
    feature_columns = pd.DataFrame(feature_npy_filtered, columns=[f"feature_{i+1}" for i in range(768)])
    data = pd.concat([data, feature_columns], axis=1)

    # dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
    data_train_val = data[
        data["fov_name"].str.contains("/0/0/0")
        | data["fov_name"].str.contains("/0/1/0")
        | data["fov_name"].str.contains("/0/2/0")
    ]

    data_test = data[
        data["fov_name"].str.contains("/0/3/0")
        | data["fov_name"].str.contains("/0/4/0")
    ]

    x_train = data_train_val.drop(
        columns=[
            "division",
            "fov_name",
            "time",
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
            "time",
        ]
    )
    y_test = data_test["division"]

    # predict the infection state for the testing set
    y_pred = clf.predict(x_test)

    # compute the accuracy of the classifier

    accuracy = np.mean(y_pred == y_test)
    # save the accuracy for final ploting
    print(f"Accuracy of model trained on {interval_name} data: {accuracy}")
    accuracies.append(accuracy)

# %% plot the accuracy of the model trained on different time intervals

plt.figure(figsize=(8, 6))
plt.bar(features_paths.keys(), accuracies)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Time interval", fontsize=14)
plt.ylim(0.9, 1)
plt.show()
# %%
