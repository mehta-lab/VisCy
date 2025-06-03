# %% Imports
from pathlib import Path

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation
from viscy.representation.evaluation.lca import fit_logistic_regression

# %%
TRAIN_FOVS = ["/A/3/7", "/A/3/8", "/A/3/9", "/B/4/6", "/B/4/7"]


model_embeddings = {
    "no-track": Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_random_sampling2/feb_fixed_test_predict.zarr"
    ),
    "cell-aware-2ch": Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_2chan_128patch_32projDim/2chan_128patch_56ckpt_FebTest.zarr"
    ),
    "cell-aware-1ch": Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/negpair_difcell_randomtime_sampling/Ver2_updateTracking_refineModel/predictions/Feb_1chan_128patch_32projDim/1chan_128patch_63ckpt_FebTest.zarr"
    ),
    "time-cell-aware": Path(
        "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr"
    ),
}
path_annotations_infection = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/extracted_inf_state.csv"
)

# %%
for model_name, path_embedding in model_embeddings.items():
    print(f"Model: {model_name}")
    dataset = read_embedding_dataset(path_embedding)
    features = dataset["features"]

    infection = load_annotation(
        dataset,
        path_annotations_infection,
        "infection_state",
        {0.0: "background", 1.0: "uninfected", 2.0: "infected"},
    )

    log_reg = fit_logistic_regression(
        features,
        infection,
        train_fovs=TRAIN_FOVS,
        remove_background_class=True,
        scale_features=False,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

# %%
