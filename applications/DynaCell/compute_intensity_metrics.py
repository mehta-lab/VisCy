# %%
import pandas as pd
from pathlib import Path
from .benchmark import compute_metrics
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

csv_database_path = Path(
    "~/gdrive/publications/dynacell/summary_table/dynacell_summary_table.csv"
).expanduser()
output_dir = Path(
    "~/gdrive/publications/dynacell/metrics"
).expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

database = pd.read_csv(csv_database_path, dtype={"FOV": str})

# %%
print("\nRunning intensity metrics with z-slice range...")
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),  # IntensityMetrics(), or SegmentationMetrics(mode="2D"),
    cell_types=["HEK293T"],
    organelles=["HIST2H2BE"],
    infection_conditions=["Mock"],
    target_database=database,
    target_channel_name="GFP",
    prediction_database=database,
    prediction_channel_name="nuclei_prediction",
    log_output_dir=output_dir,
    log_name="intensity_metrics",
    z_slice=None,
)