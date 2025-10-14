# %%
import torch
from lightning.pytorch import seed_everything
from monai.data.meta_obj import set_track_meta
from monai.transforms import RandSpatialCrop
from torch.utils.benchmark import Timer

from viscy.transforms import BatchedRandSpatialCrop

seed_everything(42)

# %%
x = torch.rand(32, 2, 15, 512, 512, device="cuda")

# %%
roi_size = [8, 256, 256]

monai_transform = RandSpatialCrop(
    roi_size=roi_size, random_center=True, random_size=False
)
batched_transform = BatchedRandSpatialCrop(roi_size=roi_size, random_center=True)


# %%
def bench_monai(x):
    set_track_meta(False)
    with torch.inference_mode():
        results = []
        for sample in x:
            cropped = monai_transform(sample)
            results.append(cropped)
        return torch.stack(results)


def bench_batched(x):
    with torch.inference_mode():
        return batched_transform(x)


# %%
globals_injection = {
    "x": x,
    "bench_monai": bench_monai,
    "bench_batched": bench_batched,
}

monai_timer = Timer(
    stmt="bench_monai(x)",
    globals=globals_injection,
    label="MONAI (loop)",
    setup="from __main__ import bench_monai",
    # num_threads=16,
)

batched_timer = Timer(
    stmt="bench_batched(x)",
    globals=globals_injection,
    label="Batched (gather)",
    setup="from __main__ import bench_batched",
    # num_threads=16,
)

# %%
monai_timer.timeit(10)

# %%
batched_timer.timeit(10)

# %%
