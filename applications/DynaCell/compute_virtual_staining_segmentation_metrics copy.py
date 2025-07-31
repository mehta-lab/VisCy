# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functools import partial

import pandas as pd
from DynaCell.benchmark import compute_metrics
from monai.transforms import NormalizeIntensityd

from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

# csv_database_path = Path(
#     "~/mydata/gdrive/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
# ).expanduser()
# output_dir = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/metrics")

csv_database_path = Path(
    "~/mydata/gdrive/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
).expanduser()
output_dir = Path("~/repos/viscy/applications/DynaCell/metrics").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

database = pd.read_csv(csv_database_path, dtype={"FOV": str})

# Select test set only
database = database[database["Test Set"] == "x"]

# %% Compute VSCyto3D intensity-based metrics
print("\nComputing VSCyto3D intensity-based metrics...")

# metrics = compute_metrics(
#     metrics_module=IntensityMetrics(),  # IntensityMetrics(), or SegmentationMetrics(mode="2D"),
#     cell_types=["HEK293T"],
#     organelles=["HIST2H2BE"],
#     infection_conditions=["Mock"],
#     target_database=database,
#     target_channel_name="Organelle",
#     prediction_database=database,
#     prediction_channel_name="Nuclei-prediction",
#     log_output_dir=output_dir,
#     log_name="intensity_VSCyto3D",
#     z_slice=36,
#     transforms=[
#         NormalizeIntensityd(
#             keys=["pred", "target"],
#         )
#     ],
# )

# %% Compute VSCyto3D segmentation-based metrics

# Construct required databases - cropped target and CellDiff prediction are saved in target_root and pred_root
# hek_h2b_database = database[
#     (database["Organelle"] == "HIST2H2BE") & (database["Cell type"] == "HEK293T")
# ]

a549_h2b_database = database[
    (database["Organelle"] == "HIST2H2BE") & (database["Cell type"] == "A549")
]
pred_database = a549_h2b_database.copy()
target_database = a549_h2b_database.copy()

old_root = Path('/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/2-assemble/2025_04_17_A549_H2B_CAAX_DENV.zarr')

segmentation_root= Path('/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/4-segmentations-dynacell/2025_04_17_A549_H2B_CAAX_DENV_test_B_2_001000.zarr')
# segmentation_root_2 = Path('/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/4-segmentations-dynacell/2025_04_17_A549_H2B_CAAX_DENV_test_B_1_000001.zarr')

def replace_root(path: str, new_root: Path) -> str:
    new_path = new_root / Path(path).relative_to(old_root)
    return str(new_path)

pred_database["Path"] = pred_database["Path"].apply(partial(replace_root, new_root=segmentation_root))
target_database["Path"] = target_database["Path"].apply(partial(replace_root, new_root=segmentation_root))

print("\nComputing Cytolan segmentation-based metrics...")
metrics = compute_metrics(
    metrics_module=SegmentationMetrics(mode="2D"),
    cell_types=["A549"],
    organelles=["HIST2H2BE"],
    infection_conditions=["DENV"],
    target_database=target_database,
    target_channel_name="fl_membrane_labels",
    prediction_database=pred_database,
    prediction_channel_name="vs_membrane_labels",
    log_output_dir=output_dir,
    log_name="segmentation_2025_04_17_A549_H2B_CAAX_DENV_membrane_only",
    z_slice=51,
)

# %%
