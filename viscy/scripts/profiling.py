# script to profile dataloading

from profilehooks import profile

from viscy.data.hcs import HCSDataModule

dataset = "/path/to/dataset.zarr"


dm = HCSDataModule(
    dataset,
    "Phase3D",
    "Deconvolved-Nuc",
    5,
    0.8,
    batch_size=32,
    num_workers=32,
    augment=None,
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
