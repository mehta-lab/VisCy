# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from functools import partial
from DynaCell.benchmark import compute_metrics
from monai.transforms import NormalizeIntensityd
from viscy.translation.evaluation import IntensityMetrics

# csv_database_path = Path(
#     "~/mydata/gdrive/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
# ).expanduser()
# output_dir = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/metrics")

csv_database_path = Path(
    "~/gdrive/publications/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
).expanduser()
output_dir = Path("~/Documents/dynacell//metrics/virtual_staining").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

database = pd.read_csv(csv_database_path, dtype={"FOV": str})

# Select test set only
database = database[database["Test Set"] == "x"]

# %% Compute VSCyto3D intensity-based metrics
print("\nComputing VSCyto3D intensity-based metrics...")

# HEK293T cells - Mock condition only
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=database,
    target_channel_name="Organelle",
    prediction_database=database,
    prediction_channel_name="Nuclei-prediction",
    log_output_dir=output_dir,
    log_name="intensity_VSCyto3D_HEK293T_nuclei_mock",
    z_slice=36,
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=database,
    target_channel_name="Membrane",
    prediction_database=database,
    prediction_channel_name="Membrane-prediction",
    log_output_dir=output_dir,
    log_name="intensity_VSCyto3D_HEK293T_membrane_mock",
    z_slice=36,
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

# A549 cells - Both Mock and DENV conditions in single calls
print("Computing A549 nuclei metrics for Mock and DENV conditions...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["A549"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock", "DENV"],  # Multiple conditions in single call
    target_database=database,
    target_channel_name="raw Cy5 EX639 EM698-70",
    prediction_database=database,
    prediction_channel_name="nuclei_prediction",
    log_output_dir=output_dir,
    log_name="intensity_VSCyto3D_A549_nuclei_mock_denv",
    z_slice=10,
    num_workers=6,       # Use parallel processing for speed
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

print("Computing A549 membrane metrics for Mock and DENV conditions...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["A549"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock", "DENV"],  # Multiple conditions in single call
    target_database=database,
    target_channel_name="raw mCherry EX561 EM600-37",
    prediction_database=database,
    prediction_channel_name="membrane_prediction",
    log_output_dir=output_dir,
    log_name="intensity_VSCyto3D_A549_membrane_mock_denv",
    z_slice=10,
    num_workers=6,       # Use parallel processing for speed
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

# %% Compute CellDiff intensity metrics

# Construct required databases - cropped target and CellDiff prediction are saved in target_root and pred_root
hek_h2b_database = database[
    (database["Organelle"] == "HIST2H2BE") & (database["Cell type"] == "HEK293T")
]
pred_database = hek_h2b_database.copy()
target_database = hek_h2b_database.copy()

old_root = Path("/hpc/projects/comp.micro/mantis/mantis_paper_data_release/figure_4.zarr")
pred_root = Path("/hpc/projects/virtual_staining/datasets/huang-lab/prediction/hek/output.zarr")
target_root = Path("/hpc/projects/virtual_staining/datasets/huang-lab/crops/hek/mantis_figure_4.zarr")

def replace_root(path: str, new_root: Path) -> str:
    new_path = new_root / Path(path).relative_to(old_root)
    return str(new_path)

pred_database["Path"] = pred_database["Path"].apply(partial(replace_root, new_root=pred_root))
target_database["Path"] = target_database["Path"].apply(partial(replace_root, new_root=target_root))

print("\nComputing CellDiff intensity-based metrics...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=target_database,
    target_channel_name="Organelle",
    prediction_database=pred_database,
    prediction_channel_name="Nuclei-prediction",
    log_output_dir=output_dir,
    log_name="intensity_CellDiff_HEK293T_nuclei",
    z_slice=36-15,  # Crop start at slice 15
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=target_database,
    target_channel_name="Membrane",
    prediction_database=pred_database,
    prediction_channel_name="Membrane-prediction",
    log_output_dir=output_dir,
    log_name="intensity_CellDiff_HEK293T_membrane",
    z_slice=36-15,  # Crop start at slice 15
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

a549_h2b_database = database[
    (database["Organelle"] == "HIST2H2BE") & (database["Cell type"] == "A549")
]
pred_database = a549_h2b_database.copy()
target_database = a549_h2b_database.copy()

old_root = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/2-assemble/2025_04_17_A549_H2B_CAAX_DENV.zarr")
pred_root = Path("/hpc/projects/virtual_staining/datasets/huang-lab/prediction/a549/output.zarr")
target_root = Path("/hpc/projects/virtual_staining/datasets/huang-lab/crops/a549/2025_04_17_A549_H2B_CAAX_DENV.zarr")

pred_database["Path"] = pred_database["Path"].apply(partial(replace_root, new_root=pred_root))
target_database["Path"] = target_database["Path"].apply(partial(replace_root, new_root=target_root))

print("Computing CellDiff A549 nuclei metrics for Mock and DENV conditions...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["A549"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock", "DENV"],  # Multiple conditions in single call
    target_database=target_database,
    target_channel_name="raw Cy5 EX639 EM698-70",
    prediction_database=pred_database,
    prediction_channel_name="Nuclei-prediction",
    log_output_dir=output_dir,
    log_name="intensity_CellDiff_A549_nuclei_mock_denv",
    z_slice=10,
    num_workers=6,       # Use parallel processing for speed
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

print("Computing CellDiff A549 membrane metrics for Mock and DENV conditions...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["A549"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock", "DENV"],  # Multiple conditions in single call
    target_database=target_database,
    target_channel_name="raw mCherry EX561 EM600-37",
    prediction_database=pred_database,
    prediction_channel_name="Membrane-prediction",
    log_output_dir=output_dir,
    log_name="intensity_CellDiff_A549_membrane_mock_denv",
    z_slice=10,
    num_workers=6,       # Use parallel processing for speed
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

# %%
