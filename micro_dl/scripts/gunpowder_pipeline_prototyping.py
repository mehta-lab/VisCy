# %%
import numpy as np
import os
import gunpowder as gp
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

from micro_dl.utils.gunpowder_utils import (
    gpsum,
    multi_zarr_source,
    get_zarr_source_position,
)
from micro_dl.input.gunpowder_nodes import (
    IntensityAugment,
    BlurAugment,
    ShearAugment,
    LogNode,
    NoiseAugment,
    Normalize,
    PrepMaskRoi,
)


# %%
# -------------------------------------------- #
# SOURCE + RAND PROV + RANDOM LOC + SIMPLE AUG #
#                                              #
#       According to William's Suggestions     #
# -------------------------------------------- #

zarr_dir = os.path.join(
    "/hpc/projects/CompMicro/projects/infected_cell_imaging/Image_preprocessing/"
    "Exp_2022_10_25_VeroCellNuclMemStain/VeroCell_40X_11NA",
)
# zarr_dir = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/ome_zarr_reference_dataset/"
zarr_dir = "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/data/2022_11_01_VeroMemNuclStain/output.zarr"
# zarr_dir = "/hpc/projects/CompMicro/projects/infected_cell_imaging/Image_preprocessing/Exp_2022_11_01_VeroCellNuclMemStain_revision/2022_11_01_VeroMemNuclStain/"

print("Taking data from: ", zarr_dir)
print("building sources...", end="")
spec = gp.ArraySpec(interpolatable=True, voxel_size=gp.Coordinate((1, 1, 1)))
multi_source, keys = multi_zarr_source(zarr_dir, array_spec=spec)
raw = keys[0]
mask = keys[1]

# %%
print("done")

print("building nodes...", end="")
random_provider = gp.RandomProvider()
random_location = gp.RandomLocation()
reject = gp.Reject(
    mask=mask,
    min_masked=0.2,
)
prep_mask_roi = PrepMaskRoi(
    array=raw,
    mask=mask,
)
simple_aug = gp.SimpleAugment(
    transpose_only=(1, 2),
    mirror_only=(1, 2),
)
elastic_aug = gp.ElasticAugment(
    control_point_spacing=(1, 1, 1),
    jitter_sigma=(0, 0, 0),
    rotation_interval=(0, 0),
    scale_interval=(1, 1),
    spatial_dims=2,
    subsample=1,
)
blur_aug = BlurAugment(
    array=raw,
    mode="gaussian",
    width_range=(23, 25),
    sigma=1,
    prob=1,
    blur_channels=(0,),
)
shear_aug = ShearAugment(
    array=raw, angle_range=(-15, 15), prob=1, shear_middle_slice_channels=(1, 2)
)
intensity_aug = IntensityAugment(
    array=raw,
    jitter_channels=(0,),
    jitter_demeaned=True,
    shift_range=(-0.15, 0.15),
    scale_range=(0.5, 1.5),
    norm_before_shift=True,
    prob=1,
)
noise_aug = NoiseAugment(
    array=raw,
    noise_channels=(0,),
    mode="gaussian",
    prob=1,
)
normalize = Normalize(
    array=raw,
    scheme="dataset",
    type="median_and_iqr",
)

profiling = gp.PrintProfilingStats(every=1)

cache = gp.PreCache(cache_size=2500, num_workers=16)
stack_num = 5
batch_stack = gp.Stack(stack_num)
print("done")
print("building pipeline...", end="")
batch_pipeline = gpsum(
    [
        multi_source,
        random_provider,
        random_location,
        reject,
        prep_mask_roi,
        normalize,
        simple_aug,
        elastic_aug,
        shear_aug,
        blur_aug,
        intensity_aug,
        noise_aug,
        # cache,  # important to cache upstream of stack
        batch_stack,
    ],
    verbose=True,
)

request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0, 0), (3, 256, 256))
request[mask] = gp.Roi((0, 0, 0), (3, 256, 256))
# %%
with gp.build(batch_pipeline) as pipeline:
    print("done")

    print("requesting batch...")
    import time

    start = time.time()
    for i in range(1):
        sample = pipeline.request_batch(request=request)
        data = sample[raw].data[:, 0, ...]
    print("done")
    print(time.time() - start, " seconds")

    print("returned data shape:", data.shape)
    print(f"max: {np.max(data)}, min: {np.min(data)}")

    vmax = None
    fig, ax = plt.subplots(stack_num, 3, figsize=(15, 5 * stack_num))
    for row in range(min(5, stack_num)):
        for channel in range(3):
            if channel == 0:
                ax[row][channel].imshow(np.mean(data[row][channel], 0), cmap="gray")
                ax[row][channel].set_title(
                    f"max:{np.max(np.mean(data[row][channel],0)):.3f}, "
                    f"min:{np.max(np.mean(data[row][channel],0)):.3f}"
                )
            else:
                ax[row][channel].imshow(data[row][channel][2])
                ax[row][channel].set_title(
                    f"max:{np.max(data[row][channel][2]):.3f}, "
                    f"min:{np.max(data[row][channel][2]):.3f}"
                )
    plt.show()

# %%
