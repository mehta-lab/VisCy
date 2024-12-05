# %% Imports
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from typing import Mapping
from numpy.typing import NDArray
from xarray import DataArray
from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation.lca import fit_logistic_regression

# %% Constants
TRAIN_FOVS = ["/0/0/0", "/0/1/0", "/0/2/0"]

# Function to load annotations
def load_annotation(da, path, name, categories=None):
    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov ID"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.loc[mi][name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected

# Model embeddings and annotation paths
model_embeddings = {
    "track": Path(
        "/hpc/projects/organelle_phenotyping/ALFI_benchmarking/predictions_final/ALFI_opp_cellaware.zarr"
    )
}

annotation_path = Path("/hpc/reference/imaging/ALFI_dataset/Analysis/train_annotations.csv")

# %% Processing each model embedding
for model_name, path_embedding in model_embeddings.items():
    print(f"Model: {model_name}")
    dataset = read_embedding_dataset(path_embedding)
    features = dataset["features"]

    # Load the division annotations
    division = load_annotation(
        dataset,
        annotation_path,
        "division",
        {0: "interphase", 1: "mitosis", -1: "NoAnnotation"}
    )
    
    # Fit logistic regression model
    log_reg = fit_logistic_regression(
        features,
        division,
        train_fovs=TRAIN_FOVS,
        remove_background_class=True,
        scale_features=False,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    # # Load the division annotations
    # fineclass_mapping = {
    #     0: "EarlyMitosis",
    #     1: "LateMitosis",
    #     -1: "NoAnnotation"
    # }

    # fineclass = load_annotation(
    #     dataset,
    #     annotation_path,
    #     "fineclass",
    #     fineclass_mapping
    # )

    # # print("Class distribution for fineclass before splitting:")
    # # print(fineclass.value_counts())
    
    # # Fit logistic regression model
    # log_reg = fit_logistic_regression(
    #     features,
    #     fineclass,
    #     train_fovs=TRAIN_FOVS,
    #     remove_background_class=False,
    #     scale_features=False,
    #     class_weight="balanced",
    #     solver="liblinear",
    #     random_state=42,
    # )
    

# %%