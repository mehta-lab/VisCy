import dask.array as da
from iohub.ngff import open_ome_zarr
from numpy.typing import NDArray


def sematic_class_weights(
    dataset_path: str, target_channel: str, num_classes: int = 3
) -> NDArray:
    """Computes class balancing weights for semantic segmentation.
    The weights can be used for cross-entropy loss.

    :param str dataset_path: HCS OME-Zarr dataset path
    :param str target_channel: target channel name
    :param int num_classes: number of classes
    :return NDArray: inverted ratio of background, uninfected and infected pixels
    """
    dataset = open_ome_zarr(dataset_path)
    arrays = [da.from_zarr(pos["0"]) for _, pos in dataset.positions()]
    imgs = da.stack(arrays, axis=0)[:, :, dataset.get_channel_index(target_channel)]
    ratio, _ = da.histogram(imgs, bins=range(num_classes + 1), density=True)
    weights = 1 / ratio
    return weights.compute()
