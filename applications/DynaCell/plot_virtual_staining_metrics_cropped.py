# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Path("/home/ivan.ivanov/Documents/dynacell/metrics/virtual_staining")
a549_VSCyto3D_csv_path = root / "intensity_VSCyto3D_A549_nuclei_mock_denv/20250916_171028/metrics.csv"
hek_VSCyto3D_csv_path = root / "intensity_VSCyto3D_HEK293T_nuclei_mock/20250916_170923/metrics.csv"
a549_CellDiff_csv_path = root / "intensity_CellDiff_A549_nuclei_mock_denv/20251002_092905/metrics.csv"
hek_CellDiff_csv_path = root / "intensity_CellDiff_HEK293T_nuclei_mock/20251002_092333/metrics.csv"
a549_VSCyto3D_df = pd.read_csv(a549_VSCyto3D_csv_path)
hek_VSCyto3D_df = pd.read_csv(hek_VSCyto3D_csv_path)
a549_VSCyto3D_df["Model"] = "VSCyto3D"
hek_VSCyto3D_df["Model"] = "VSCyto3D"
a549_CellDiff_df = pd.read_csv(a549_CellDiff_csv_path)
hek_CellDiff_df = pd.read_csv(hek_CellDiff_csv_path)    
a549_CellDiff_df["Model"] = "CellDiff"
hek_CellDiff_df["Model"] = "CellDiff"
df = pd.concat([a549_VSCyto3D_df, hek_VSCyto3D_df, a549_CellDiff_df, hek_CellDiff_df])

plots_dir = root / "plots"
plots_dir.mkdir(exist_ok=True)

hek_time_step = 15  # in minutes
a549_time_step = 10  # in minutes

# Add a new column to df for time in hours, depending on cell_type
def compute_time_hours(row):
    if row["cell_type"].lower() == "hek293t":
        return row["time"] * hek_time_step / 60
    elif row["cell_type"].lower() == "a549":
        return row["time"] * a549_time_step / 60
    else:
        return np.nan

df["time_hours"] = df.apply(compute_time_hours, axis=1)
df["original_position_name"] = df["position_name"].str[:-1]
df["crop_index"] = df["position_name"].str[-1]

# %% A549 VSCyto3D plots
_df_mock = df[(df["infection_condition"] == "Mock") & (df["cell_type"] == "A549") & (df["Model"] == "VSCyto3D")]
_df_mock_mean_vscyto3d = _df_mock.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 Mock VSCyto3D Nuclei")
# plt.legend()
plt.savefig(plots_dir / "A549_Mock_VSCyto3D_Nuclei_pearson.png")

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 Mock VSCyto3D Nuclei")
# plt.legend()
plt.show()
# plt.savefig(plots_dir / "A549_Mock_VSCyto3D_Nuclei_ssim.png")


_df_denv = df[(df["infection_condition"] == "DENV") & (df["Model"] == "VSCyto3D")]
_df_denv_mean_vscyto3d = _df_denv.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_denv.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 DENV VSCyto3D Nuclei")
# plt.legend()
plt.show()
# plt.savefig(plots_dir / "A549_DENV_VSCyto3D_Nuclei_pearson.png")


plt.figure()
for idx, sub_df in _df_denv.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 DENV VSCyto3D Nuclei")
# plt.legend()
plt.show()
# plt.savefig(plots_dir / "A549_DENV_VSCyto3D_Nuclei_ssim.png")


# %% Compare A549 Mock and DENV
plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["pearson"], label="Mock")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["pearson"], label="DENV")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 VSCyto3D Nuclei")
plt.legend()
plt.show()
# plt.savefig(plots_dir / "A549_VSCyto3D_Nuclei_pearson.png")


plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["ssim"], label="Mock")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["ssim"], label="DENV")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 VSCyto3D Nuclei")
plt.legend()
plt.show()
# plt.savefig(plots_dir / "A549_VSCyto3D_Nuclei_ssim.png")


# %% A549 CellDiff plots
_df_mock = df[(df["infection_condition"] == "Mock") & (df["cell_type"] == "A549") & (df["Model"] == "CellDiff")]
_df_mock_mean_celldiff = _df_mock.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 Mock CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_Mock_CellDiff_Nuclei_pearson.png")

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 Mock CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_Mock_CellDiff_Nuclei_ssim.png")

_df_denv = df[(df["infection_condition"] == "DENV") & (df["cell_type"] == "A549") & (df["Model"] == "CellDiff")]
_df_denv_mean_celldiff = _df_denv.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_denv.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 DENV CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_DENV_CellDiff_Nuclei_pearson.png")


plt.figure()
for idx, sub_df in _df_denv.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 DENV CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_DENV_CellDiff_Nuclei_ssim.png")

# %% Compare A549 Mock and DENV
plt.figure()
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["pearson"], label="Mock")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["pearson"], label="DENV")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 CellDiff Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_CellDiff_Nuclei_pearson.png")

plt.figure()
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["ssim"], label="Mock")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["ssim"], label="DENV")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 CellDiff Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_CellDiff_Nuclei_ssim.png")

# %% Compare mean pearson and ssim for all models
plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["pearson"], label="Mock VSCyto3D", color="blue")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["pearson"], label="Mock CellDiff", linestyle="--", color="orange")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["pearson"], label="DENV VSCyto3D", color="blue")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["pearson"], label="DENV CellDiff", linestyle="--", color="orange")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_Nuclei_pearson.png")

plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["ssim"], label="Mock VSCyto3D", color="blue")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["ssim"], label="Mock CellDiff", linestyle="--", color="orange")
plt.plot(_df_denv_mean_vscyto3d["time_hours"], _df_denv_mean_vscyto3d["ssim"], label="DENV VSCyto3D", color="blue")
plt.plot(_df_denv_mean_celldiff["time_hours"], _df_denv_mean_celldiff["ssim"], label="DENV CellDiff", linestyle="--", color="orange")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_Nuclei_ssim.png")


# %% HEK293T VSCyto3D plots
_df_mock = df[(df["infection_condition"] == "Mock") & (df["cell_type"] == "HEK293T") & (df["Model"] == "VSCyto3D")]
_df_hek_mean_vscyto3d = _df_mock.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_hek_mean_vscyto3d["time_hours"], _df_hek_mean_vscyto3d["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("HEK293T Mock VSCyto3D Nuclei")
# plt.legend()
plt.show()
# plt.savefig(plots_dir / "HEK293T_Mock_VSCyto3D_Nuclei_pearson.png")


plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_hek_mean_vscyto3d["time_hours"], _df_hek_mean_vscyto3d["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("HEK293T Mock VSCyto3D Nuclei")
# plt.legend()
plt.show()
# plt.savefig(plots_dir / "HEK293T_Mock_VSCyto3D_Nuclei_ssim.png")

# %% HEK293T CellDiff plots
_df_mock = df[(df["infection_condition"] == "Mock") & (df["cell_type"] == "HEK293T") & (df["Model"] == "CellDiff")]
_df_hek_mean_celldiff = _df_mock.groupby("time")[["pearson", "ssim", "time_hours"]].mean()

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["pearson"], label=f"FOV {idx}")
plt.plot(_df_hek_mean_celldiff["time_hours"], _df_hek_mean_celldiff["pearson"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("HEK293T Mock CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "HEK293T_Mock_CellDiff_Nuclei_pearson.png")

plt.figure()
for idx, sub_df in _df_mock.groupby("position_name"):
    plt.plot(sub_df["time_hours"], sub_df["ssim"], label=f"FOV {idx}")
plt.plot(_df_hek_mean_celldiff["time_hours"], _df_hek_mean_celldiff["ssim"], 'k-', label="Average")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("HEK293T Mock CellDiff Nuclei")
# plt.legend()
# plt.show()
plt.savefig(plots_dir / "HEK293T_Mock_CellDiff_Nuclei_ssim.png")

# %% Compare mean metrics for A549 and HEK293T
plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["pearson"], label="A549 Mock VSCyto3D", color="blue")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["pearson"], label="A549 Mock CellDiff", linestyle="--", color="orange")
plt.plot(_df_hek_mean_vscyto3d["time_hours"], _df_hek_mean_vscyto3d["pearson"], label="HEK293T Mock VSCyto3D", color="green")
plt.plot(_df_hek_mean_celldiff["time_hours"], _df_hek_mean_celldiff["pearson"], label="HEK293T Mock CellDiff", linestyle="--", color="green")
plt.xlabel("Time [hours]")
plt.ylabel("Pearson")
plt.title("A549 and HEK293T Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_HEK293T_Nuclei_pearson.png")

plt.figure()
plt.plot(_df_mock_mean_vscyto3d["time_hours"], _df_mock_mean_vscyto3d["ssim"], label="A549 Mock VSCyto3D", color="blue")
plt.plot(_df_mock_mean_celldiff["time_hours"], _df_mock_mean_celldiff["ssim"], label="A549 Mock CellDiff", linestyle="--", color="orange")
plt.plot(_df_hek_mean_vscyto3d["time_hours"], _df_hek_mean_vscyto3d["ssim"], label="HEK293T Mock VSCyto3D", color="green")
plt.plot(_df_hek_mean_celldiff["time_hours"], _df_hek_mean_celldiff["ssim"], label="HEK293T Mock CellDiff", linestyle="--", color="green")
plt.xlabel("Time [hours]")
plt.ylabel("SSIM")
plt.title("A549 and HEK293T Nuclei")
plt.legend()
# plt.show()
plt.savefig(plots_dir / "A549_HEK293T_Nuclei_ssim.png")

# %% Plot heatmap of pearson with time as columns and cell type as rows, averaging over position name
_df = df[(df["time_hours"] < 7) & (df["infection_condition"] == "Mock") & (df["Model"] == "VSCyto3D")]
# bin time_hours into 1 hour bins
_df["time_hours_binned"] = _df["time_hours"].round(0)

plt.figure()
sns.heatmap(_df.groupby(["time_hours_binned", "cell_type"])["pearson"].mean().unstack(), cmap="Blues")
plt.ylabel("Time [hours]")
plt.xlabel("Cell Type")
plt.title("Pearson Heatmap")
plt.show()
# plt.savefig(plots_dir / "Pearson_Heatmap_Mock.png")

plt.figure()
sns.heatmap(_df.groupby(["time_hours_binned", "cell_type"])["ssim"].mean().unstack(), cmap="Blues")
plt.ylabel("Time [hours]")
plt.xlabel("Cell Type")
plt.title("Pearson Heatmap")
plt.show()
# plt.savefig(plots_dir / "SSIM_Heatmap_Mock.png")



# %%
