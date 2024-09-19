# %% Imports
from pathlib import Path

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation
from viscy.representation.lca import fit_logistic_regression

# %%
TRAIN_FOVS = ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/6", "/B/4/7"]

path_embedding = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
)
path_annotations_infection = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)

# %%
dataset = read_embedding_dataset(path_embedding)
features = dataset["features"]
features

# %%
infection = load_annotation(
    dataset,
    path_annotations_infection,
    "infection_state",
    {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
)
infection

# %%
log_reg = fit_logistic_regression(
    features,
    infection,
    train_fovs=TRAIN_FOVS,
    remove_background_class=True,
    scale_features=False,
    class_weight="balanced",
    solver="liblinear",
)

# %%
