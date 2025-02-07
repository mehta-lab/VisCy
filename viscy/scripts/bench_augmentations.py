# %%
import torch
from kornia.augmentation import RandomAffine3D
from lightning.pytorch import seed_everything
from monai.data.meta_obj import set_track_meta
from monai.transforms import RandAffine
from torch.utils.benchmark import Timer

seed_everything(42)

# %%
x = torch.rand(32, 2, 15, 512, 512, device="cuda")

# %%
monai_transform = RandAffine(
    prob=1.0,
    rotate_range=(torch.pi, 0, 0),
    scale_range=(0.2, 0.3, 0.3),
    padding_mode="zeros",
)

kornia_transform = RandomAffine3D(
    degrees=(360.0, 0.0, 0.0),
    scale=((0.8, 1.2), (0.7, 1.3), (0.7, 1.3)),
    p=1.0,
)


# %%
def bench_monai(x):
    set_track_meta(False)
    with torch.inference_mode():
        for sample in x:
            _ = monai_transform(sample)


def bench_kornia(x):
    with torch.inference_mode():
        _ = kornia_transform(x)


# %%
globals_injection = {
    "x": x,
    "monai_transform": monai_transform,
    "kornia_transform": kornia_transform,
}

monai_timer = Timer(
    stmt="bench_monai(x)",
    globals=globals_injection,
    label="monai",
    setup="from __main__ import bench_monai",
    # num_threads=16,
)

kornia_timer = Timer(
    stmt="bench_kornia(x)",
    globals=globals_injection,
    label="kornia",
    setup="from __main__ import bench_kornia",
    # num_threads=16,
)

# %%
monai_timer.timeit(10)

# %%
kornia_timer.timeit(10)

# %%
