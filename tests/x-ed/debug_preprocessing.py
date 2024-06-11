# %%
from viscy.utils.meta_utils import generate_normalization_metadata
from iohub.ngff import Position, open_ome_zarr
import numpy as np
import matplotlib.pyplot as plt


def create_dummy_zarr(output_folder):
    dataset = open_ome_zarr(
        output_folder,
        mode="w",
        layout="hcs",
        channel_names=["Brightfield", "Nuclei"],
    )

    np.random.seed(42)

    dummy_hcs = []
    img_files = []
    for j in range(2):
        for i in range(5):
            Z = np.random.randint(5, 11)
            dummy_hcs.append((f"row", f"col{j}", f"fov{i}"))
            mu = np.random.uniform(100 + (j * 30), 105 + (j * 30))
            sigma = np.random.uniform(1, 2)
            random_image = sigma * np.random.rand(1, 2, Z, 64, 64) + mu
            # print(random_image.shape)
            img_files.append(random_image)
            print(random_image.mean(), random_image.std())

    for (row, col, fov), img_data in zip(dummy_hcs, img_files):
        pos: Position = dataset.create_position(row, col, fov)
        pos.create_image("0", img_data)


def main():
    # %%
    # Create dummy zarr with varying z to emulate the aics datset
    output_folder = "./test_prepross.zarr"
    create_dummy_zarr(output_folder)
    # %%
    channel_names = ["Brightfield", "Nuclei"]
    with open_ome_zarr(output_folder, layout="hcs", mode="r") as dataset:
        channel_indices = (
            [dataset.channel_names.index(c) for c in channel_names]
            if channel_names != -1
            else channel_names
        )
    # Generate normalization_metadata
    generate_normalization_metadata(
        output_folder, num_workers=3, channel_ids=channel_indices, grid_spacing=16
    )


# %%
if __name__ == "__main__":
    main()
# %%
