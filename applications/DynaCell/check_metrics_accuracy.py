#Compare metric accuracy when computed using scikit-image and torchmetrics as implemented in viscy

# Metrics list: mae, mse, ssim, pearson
# %%
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
import torch

from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
)
from skimage.metrics import (   
    structural_similarity as ssim
)
from skimage.measure import pearson_corr_coeff as pearson
from skimage.exposure import rescale_intensity

from torchmetrics.functional import (
    mean_squared_error as torch_mse,
    mean_absolute_error as torch_mae,
    pearson_corrcoef as torch_pearson,
    structural_similarity_index_measure as torch_ssim
)

from monai.transforms import NormalizeIntensity


# %% Load data
data_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_04_17_A549_H2B_CAAX_DENV/2-assemble/2025_04_17_A549_H2B_CAAX_DENV.zarr/B/1/000001")
t_idx = 0
z_idx = 28
target_channel_name = "raw Cy5 EX639 EM698-70"
prediction_channel_name = "nuclei_prediction"

target_offset = 100
pred_offset = 0

with open_ome_zarr(data_path, mode="r") as dataset:
    channel_names = dataset.channel_names
    target_channel_index = channel_names.index(target_channel_name)
    prediction_channel_index = channel_names.index(prediction_channel_name)
    target_volume_raw = dataset.data[t_idx, target_channel_index]
    pred = dataset.data[t_idx, prediction_channel_index, z_idx]

# Normalize entire volume by median and iqr, then take out slice of interest
# Offset doesn't matter
median = np.median(target_volume_raw)
iqr = np.percentile(target_volume_raw, 75) - np.percentile(target_volume_raw, 25)

target_volume_correctly_normalized = (target_volume_raw - median) / iqr
target = target_volume_correctly_normalized[z_idx]

# Convert to tensor
target_tensor = torch.from_numpy(target)
pred_tensor = torch.from_numpy(pred)

# %% Compare MSE
mse_sk = mse(target, pred)
mse_torch = torch_mse(target_tensor, pred_tensor).item()
print(f"MSE: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}")

# %% Compare MAE
mae_sk = mae(target, pred)
mae_torch = torch_mae(target_tensor, pred_tensor).item()
print(f"MAE: SK: {mae_sk:.3f}, Torch: {mae_torch:.3f}")

# %% Compare Pearson
pearson_sk = pearson(target, pred).statistic
pearson_torch = torch_pearson(target_tensor.flatten(), pred_tensor.flatten()).item()
print(f"Pearson: SK: {pearson_sk:.3f}, Torch: {pearson_torch:.3f}")

# %% Compare SSIM
ssim_sk = ssim(target, pred, data_range=target.max() - target.min())
ssim_torch = torch_ssim(
    target_tensor.unsqueeze(0).unsqueeze(0),
    pred_tensor.unsqueeze(0).unsqueeze(0)
).item()
print(f"SSIM: SK: {ssim_sk:.3f}, Torch: {ssim_torch:.3f}")


# %%
### ------------------ Normalize both target and prediction ------------------
# Note: we are normalizing the 2D slice which is not correct


# Normalize
target_raw = target_volume_raw[z_idx]
target_norm = (target_raw - target_raw.mean()) / target_raw.std()
pred_norm = (pred - pred.mean()) / pred.std()

target_no_offset = target_raw - target_offset
pred_no_offset = pred - pred_offset

target_no_offset_norm = (target_no_offset - target_no_offset.mean()) / target_no_offset.std()
pred_no_offset_norm = (pred_no_offset - pred_no_offset.mean()) / pred_no_offset.std()

# Convert to tensor
target_tensor = torch.from_numpy(target_raw)
pred_tensor = torch.from_numpy(pred)

target_no_offset_tensor = torch.from_numpy(target_no_offset)
pred_no_offset_tensor = torch.from_numpy(pred_no_offset)

intensity_normalizer = NormalizeIntensity()
target_tensor_norm = intensity_normalizer(target_tensor)
pred_tensor_norm = intensity_normalizer(pred_tensor)

target_no_offset_tensor_norm = intensity_normalizer(target_no_offset_tensor)
pred_no_offset_tensor_norm = intensity_normalizer(pred_no_offset_tensor)

# SSIM normalization
target_ssim = rescale_intensity(
    target_no_offset,
    in_range=(np.quantile(target_no_offset, 0.01), np.quantile(target_no_offset, 0.99)),
    out_range=np.uint16
)
pred_ssim = rescale_intensity(
    pred_no_offset,
    in_range=(np.quantile(pred_no_offset, 0.01), np.quantile(pred_no_offset, 0.99)),
    out_range=np.uint16
)

# %% Compare MSE
# Note: MSE is sensitive intensity normalization but not offset
print("Comparing MSE...")

# With offset, no normalization
print("  With offset:")
mse_sk = mse(target_raw, pred)
mse_torch = torch_mse(target_tensor, pred_tensor).item()
print(
    f"    Without normalization: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}"
)

# With offset, with intensity normalization
mse_sk = mse(target_norm, pred_norm)
mse_torch = torch_mse(target_tensor_norm, pred_tensor_norm).item()
print(
    f"    With intensity normalization: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}"
)

# Without offset, no intensity normalization
mse_sk = mse(target_no_offset, pred_no_offset)
mse_torch = torch_mse(target_no_offset_tensor, pred_no_offset_tensor).item()
print("  Without offset:")
print(
    f"    Without normalization: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}"
)

# Without offset, with intensity normalization
mse_sk = mse(target_no_offset_norm, pred_no_offset_norm)
mse_torch = torch_mse(target_no_offset_tensor_norm, pred_no_offset_tensor_norm).item()
print(
    f"    With intensity normalization: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}"
)

mse_sk = mse(target_norm, pred)
mse_torch = torch_mse(target_tensor_norm, pred_tensor).item()
print(
    f"Comparing normalized target with unnormalized prediction: SK: {mse_sk:.3f}, Torch: {mse_torch:.3f}"
)

# %% Compare MAE    
# Note: MAE is sensitive intensity normalization but not offset
print("Comparing MAE...")

# With offset, no normalization
print("  With offset:")
mae_sk = mae(target_raw, pred)
mae_torch = torch_mae(target_tensor, pred_tensor).item()
print(
    f"    Without normalization: SK: {mae_sk:.3f}, Torch: {mae_torch:.3f}"
)

# With offset, with intensity normalization
mae_sk = mae(target_norm, pred_norm)
mae_torch = torch_mae(target_tensor_norm, pred_tensor_norm).item()
print(
    f"    With intensity normalization: SK: {mae_sk:.3f}, Torch: {mae_torch:.3f}"
)

# Without offset, no intensity normalization
mae_sk = mae(target_no_offset, pred_no_offset)
mae_torch = torch_mae(target_no_offset_tensor, pred_no_offset_tensor).item()
print("  Without offset:")
print(
    f"    Without normalization: SK: {mae_sk:.3f}, Torch: {mae_torch:.3f}"
)

# Without offset, with intensity normalization
mae_sk = mae(target_no_offset_norm, pred_no_offset_norm)
mae_torch = torch_mae(target_no_offset_tensor_norm, pred_no_offset_tensor_norm).item()
print(
    f"    With intensity normalization: SK: {mae_sk:.3f}, Torch: {mae_torch:.3f}"
)

# %% Compare SSIM
# Note: SSIM is sensitive to offset if intensity normalization is not applied
# SSIM is sensitive to intensity normalization
print("Comparing SSIM...")

# With offset, no normalization
print("  With offset:")
ssim_sk = ssim(target_raw, pred, data_range=target_raw.max() - target_raw.min())
ssim_torch = torch_ssim(target_tensor.unsqueeze(0).unsqueeze(0), pred_tensor.unsqueeze(0).unsqueeze(0)).item()
print(
    f"    Without normalization: SK: {ssim_sk:.3f}, Torch: {ssim_torch:.3f}"
)

# With offset, with intensity normalization
ssim_sk = ssim(target_norm, pred_norm, data_range=target_norm.max() - target_norm.min())
ssim_torch = torch_ssim(target_tensor_norm.unsqueeze(0).unsqueeze(0), pred_tensor_norm.unsqueeze(0).unsqueeze(0)).item()
print(
    f"    With intensity normalization: SK: {ssim_sk:.3f}, Torch: {ssim_torch:.3f}"
)

# Without offset, no intensity normalization
ssim_sk = ssim(target_no_offset, pred_no_offset, data_range=target_no_offset.max() - target_no_offset.min())
ssim_torch = torch_ssim(target_no_offset_tensor.unsqueeze(0).unsqueeze(0), pred_no_offset_tensor.unsqueeze(0).unsqueeze(0)).item()
print("  Without offset:")
print(
    f"    Without normalization: SK: {ssim_sk:.3f}, Torch: {ssim_torch:.3f}"
)

# Without offset, with intensity normalization
ssim_sk = ssim(target_no_offset_norm, pred_no_offset_norm, data_range=target_no_offset_norm.max() - target_no_offset_norm.min())
ssim_torch = torch_ssim(target_no_offset_tensor_norm.unsqueeze(0).unsqueeze(0), pred_no_offset_tensor_norm.unsqueeze(0).unsqueeze(0)).item()
print(
    f"    With intensity normalization: SK: {ssim_sk:.3f}, Torch: {ssim_torch:.3f}"
)

# %% Compare Pearson
# Note: Pearson is not sensitive to offset or intensity normalization
print("Comparing Pearson...")

# With offset, no normalization
print("  With offset:")
pearson_sk = pearson(target_raw, pred).statistic
pearson_torch = torch_pearson(target_tensor.flatten(), pred_tensor.flatten()).item()
print(
    f"    Without normalization: SK: {pearson_sk:.3f}, Torch: {pearson_torch:.3f}"
)

# With offset, with intensity normalization
pearson_sk = pearson(target_norm, pred_norm).statistic
pearson_torch = torch_pearson(target_tensor_norm.flatten(), pred_tensor_norm.flatten()).item()
print(
    f"    With intensity normalization: SK: {pearson_sk:.3f}, Torch: {pearson_torch:.3f}"
)

# Without offset, no normalization
pearson_sk = pearson(target_no_offset, pred_no_offset).statistic
pearson_torch = torch_pearson(target_no_offset_tensor.flatten(), pred_no_offset_tensor.flatten()).item()
print("  Without offset:")
print(
    f"    Without normalization: SK: {pearson_sk:.3f}, Torch: {pearson_torch:.3f}"
)

# Without offset, with intensity normalization
pearson_sk = pearson(target_no_offset_norm, pred_no_offset_norm).statistic
pearson_torch = torch_pearson(target_no_offset_tensor_norm.flatten(), pred_no_offset_tensor_norm.flatten()).item()
print(
    f"    With intensity normalization: SK: {pearson_sk:.3f}, Torch: {pearson_torch:.3f}"
)

# # %%
# %%
