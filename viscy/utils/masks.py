"""Mask generation and processing utilities."""

from typing import Any

import numpy as np
import scipy.ndimage as ndimage
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian, laplace, threshold_otsu
from skimage.morphology import (
    ball,
    binary_dilation,
    binary_opening,
    disk,
    remove_small_objects,
)


def create_otsu_mask(
    input_image: NDArray[Any], sigma: float = 0.6
) -> NDArray[np.bool_]:
    """Create a binary mask using Otsu thresholding and morphological operations.

    Parameters
    ----------
    input_image : NDArray
        Generate masks from this 3D image.
    sigma : float, optional
        Gaussian blur standard deviation, increase in value increases blur,
        by default 0.6.

    Returns
    -------
    NDArray
        Volume mask of input_image, 3D binary array.
    """
    input_sz = input_image.shape
    mid_slice_id = input_sz[0] // 2

    thresh = threshold_otsu(input_image[mid_slice_id, :, :])
    mask = input_image >= thresh

    return mask


def create_membrane_mask(
    input_image: NDArray[Any],
    str_elem_size: int = 23,
    sigma: float = 0.4,
    k_size: int = 3,
    msize: int = 120,
) -> NDArray[np.bool_]:
    """Create a binary mask using Laplacian of Gaussian (LOG) feature detection.

    Parameters
    ----------
    input_image : NDArray
        Generate masks from this image.
    str_elem_size : int, optional
        Size of the laplacian filter used for contrast enhancement, odd number.
        Increase in value increases sensitivity of contrast enhancement,
        by default 23.
    sigma : float, optional
        Gaussian blur standard deviation, by default 0.4.
    k_size : int, optional
        Disk/ball size for mask dilation, ball for 3D and disk for 2D data,
        by default 3.
    msize : int, optional
        Size of small objects removed to clean segmentation, by default 120.

    Returns
    -------
    NDArray
        Binary mask of input_image.
    """
    input_image_blur = gaussian(input_image, sigma=sigma)

    input_Lapl = laplace(input_image_blur, ksize=str_elem_size)

    thresh = threshold_otsu(input_Lapl)
    mask_bin = input_Lapl >= thresh

    if len(input_image.shape) == 2:
        str_elem = disk(k_size)
    else:
        str_elem = ball(k_size)

    mask_dilated = binary_dilation(mask_bin, str_elem)

    mask = remove_small_objects(mask_dilated, min_size=msize)

    return mask


def get_unimodal_threshold(input_image: NDArray[Any]) -> float:
    """Determine optimal unimodal threshold using Rosin's method.

    References
    ----------
    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding

    Parameters
    ----------
    input_image : NDArray
        Generate mask for this image.

    Returns
    -------
    float
        Optimal lower threshold for the foreground histogram.
    """
    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5)),
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, "Error in unimodal thresholding"
    return best_threshold


def create_unimodal_mask(
    input_image: NDArray[Any], str_elem_size: int = 3, sigma: float = 0.6
) -> NDArray[np.bool_]:
    """Create a mask with unimodal thresholding and morphological operations.

    Unimodal thresholding seems to oversegment, erode it by a fraction.

    Parameters
    ----------
    input_image : NDArray
        Generate masks from this image.
    str_elem_size : int, optional
        Size of the structuring element, typically 3 or 5, by default 3.
    sigma : float, optional
        Gaussian blur standard deviation, by default 0.6.

    Returns
    -------
    NDArray
        Binary mask of input_image.
    """
    input_image = gaussian(input_image, sigma=sigma)

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    mask = input_image >= thr
    mask = binary_opening(mask, str_elem)
    mask = binary_fill_holes(mask)
    return mask


def get_unet_border_weight_map(
    annotation: NDArray[Any], w0: int = 10, sigma: int = 5
) -> NDArray[np.float64]:
    """Return weight map for borders as specified in U-Net paper.

    TODO: Calculate boundaries directly and calculate distance from boundary
    of cells to another. Note: The below method only works for UNet Segmentation only.

    Parameters
    ----------
    annotation : NDArray
        A 2D array of shape (image_height, image_width) containing annotation
        with each class labeled as an integer.
    w0 : int, optional
        Multiplier to the exponential distance loss, default 10 as mentioned
        in UNet paper, by default 10.
    sigma : int, optional
        Standard deviation in the exponential distance term
        e^(-d1 + d2) ** 2 / 2 (sigma ^ 2), default 5 as mentioned in UNet paper,
        by default 5.

    Returns
    -------
    NDArray
        Weight map for borders as specified in U-Net paper.
    """
    # if there is only one label, zero return the array as is
    if np.sum(annotation) == 0:
        return annotation

    # Masks could be saved as .npy bools, if so convert to uint8 and generate
    # labels from binary
    if annotation.dtype == bool:
        annotation = annotation.astype(np.uint8)
    assert annotation.dtype in [
        np.uint8,
        np.uint16,
    ], f"Expected data type uint, it is {annotation.dtype}"

    # cells instances for distance computation
    # 4 connected i.e default (cross-shaped)
    # structuring element to measure connectivy
    # If cells are 8 connected/touching they are labeled as one single object
    # Loss metric on such borders is not useful
    labeled_array, _ = ndimage.measurements.label(annotation)
    # class balance weights w_c(x)
    unique_values = np.unique(labeled_array).tolist()
    weight_map = [0] * len(unique_values)
    for index, unique_value in enumerate(unique_values):
        mask = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
        mask[annotation == unique_value] = 1
        weight_map[index] = 1 / mask.sum()

    # this normalization is important - foreground pixels must have weight 1
    weight_map = [i / max(weight_map) for i in weight_map]

    wc = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    for index, unique_value in enumerate(unique_values):
        wc[annotation == unique_value] = weight_map[index]

    # cells distance map
    border_loss_map = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64
    )
    distance_maps = np.zeros(
        (annotation.shape[0], annotation.shape[1], np.max(labeled_array)),
        dtype=np.float64,
    )

    if np.max(labeled_array) >= 2:
        for index in range(np.max(labeled_array)):
            mask = np.ones_like(labeled_array)
            mask[labeled_array == index + 1] = 0
            distance_maps[:, :, index] = ndimage.distance_transform_edt(mask)
    distance_maps = np.sort(distance_maps, 2)
    d1 = distance_maps[:, :, 0]
    d2 = distance_maps[:, :, 1]
    border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma**2)))

    zero_label = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    zero_label[labeled_array == 0] = 1
    border_loss_map = np.multiply(border_loss_map, zero_label)

    # unet weight map mask
    weight_map_mask = wc + border_loss_map

    return weight_map_mask
