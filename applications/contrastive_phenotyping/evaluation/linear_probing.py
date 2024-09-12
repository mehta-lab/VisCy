# %% Imports
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import load_annotation
from viscy.representation.lca import train_and_test_linear_classifier

# %%
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
temp_dir = TemporaryDirectory()
log_path = Path(temp_dir.name)

train_and_test_linear_classifier(
    features.to_numpy(),
    infection.cat.codes.values,
    num_classes=3,
    trainer=Trainer(max_epochs=60, logger=CSVLogger(log_path), log_every_n_steps=1),
    split_ratio=(0.4, 0.2, 0.4),
    batch_size=2**14,
    lr=0.001,
)

# plot loss curves to check if training converged/overfitted
# adjust number of epochs if necessary
losses = pd.read_csv(
    log_path / "lightning_logs" / "version_0" / "metrics.csv", index_col="epoch"
)
losses = pd.merge(
    losses["loss/train"].dropna(), losses["loss/val"].dropna(), on="epoch"
)
losses.plot()
temp_dir.cleanup()
# %%
