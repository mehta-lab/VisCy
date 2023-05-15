# script to profile dataloading

from micro_dl.light.data import HCSDataModule
from numcodecs import blosc
from profilehooks import profile

dataset = (
    "/hpc/nodes/gpu-c-1/groups/cmanalysis.grp/microDL_SP/Input/TestData_HEK_2022_04_19/"
    "no_pertubation_Phase1e-4_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50_NGFF.zarr"
)


dm = HCSDataModule(
    dataset,
    "Phase3D",
    "Deconvolved-Nuc",
    5,
    0.8,
    batch_size=32,
    num_workers=32,
    augment=True,
    caching=False,
)

dm.setup("fit")


@profile(immediate=True, sort="time", dirs=True)
def load_batch(n=1):
    for i, batch in enumerate(dm.train_dataloader()):
        print(batch["source"].shape)
        print(dm.on_before_batch_transfer(batch, 0)["target"].shape)
        if i == n - 1:
            break


load_batch(3)
