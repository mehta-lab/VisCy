# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from functools import partial
from DynaCell.benchmark import compute_metrics
from monai.transforms import NormalizeIntensityd
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

# csv_database_path = Path(
#     "~/mydata/gdrive/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
# ).expanduser()
# output_dir = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/metrics")

csv_database_path = Path(
    "~/gdrive/publications/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
).expanduser()
output_dir = Path("~/gdrive/publications/dynacell/metrics/virtual_staining").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

database = pd.read_csv(csv_database_path, dtype={"FOV": str})

# Select test set only
database = database[database["Test Set"] == "x"]

# %% Compute VSCyto3D intensity-based metrics
print("\nComputing VSCyto3D intensity-based metrics...")

metrics = compute_metrics(
    metrics_module=IntensityMetrics(),  # IntensityMetrics(), or SegmentationMetrics(mode="2D"),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=database,
    target_channel_name="Organelle",
    prediction_database=database,
    prediction_channel_name="Nuclei-prediction",
    log_output_dir=output_dir,
    log_name="intensity_VSCyto3D",
    z_slice=36,
    transforms=[
        NormalizeIntensityd(
            keys=["pred", "target"],
        )
    ],
)

# %% Compute VSCyto3D segmentation-based metrics

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
    log_name="intensity_CellDiff",
    z_slice=36-15,  # Crop start at slice 15
)

# %%
