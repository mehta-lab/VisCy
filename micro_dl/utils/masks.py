import numpy as np
import scipy.ndimage as ndimage
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball, binary_opening, binary_erosion, watershed
from micro_dl.utils.image_utils import im_adjust


def create_otsu_mask(input_image,
                     str_elem_size=3,
                     thr=None,
                     kernel_size=3,
                     w_shed=False):
    """Create a binary mask using morphological operations

    Opening removes small objects in the foreground.

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    input_image = im_adjust(cv2.GaussianBlur(input_image, (kernel_size, kernel_size), 0))
    input_image = im_adjust(input_image)
    if thr is None:
        if np.min(input_image) == np.max(input_image):
            thr = np.unique(input_image)
        else:
            thr = 1.1 * threshold_otsu(input_image, nbins=128)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    mask = input_image > thr
    mask = binary_opening(mask, str_elem)
    # mask = binary_fill_holes(mask)
    if not w_shed:
        return mask
    dist = ndimage.distance_transform_edt(mask)
    localMax = peak_local_max(dist, indices=False, min_distance=15,
                              labels=mask)
    # perform a connected component analysis on the local peaks
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=mask, watershed_line=True)
    mask = labels > 0
    # mask = binary_erosion(mask, str_elem)
    return mask

def get_unimodal_threshold(input_image):
    """Determines optimal unimodal threshold

    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding

    :param np.array input_image: generate mask for this image
    :return float best_threshold: optimal lower threshold for the foreground
     hist
    """

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5))
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
    assert best_threshold > -np.inf, 'Error in unimodal thresholding'
    return best_threshold


def create_unimodal_mask(input_image, str_elem_size=3, kernel_size=3):
    """
    Create a mask with unimodal thresholding and morphological operations.
    Unimodal thresholding seems to oversegment, erode it by a fraction

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return mask of input_image, np.array
    """

    input_image = im_adjust(cv2.GaussianBlur(input_image, (kernel_size, kernel_size), 0))
    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    mask = input_image > thr
    mask = binary_opening(mask, str_elem)
    mask = binary_fill_holes(mask)
    return mask


def get_unet_border_weight_map(annotation, w0=10, sigma=5):
    """
    Return weight map for borders as specified in UNet paper
    :param annotation A 2D array of shape (image_height, image_width)
     contains annotation with each class labeled as an integer.
    :param w0 multiplier to the exponential distance loss
     default 10 as mentioned in UNet paper
    :param sigma standard deviation in the exponential distance term
     e^(-d1 + d2) ** 2 / 2 (sigma ^ 2)
     default 5 as mentioned in UNet paper
    :return weight mapt for borders as specified in UNet

    TODO: Calculate boundaries directly and calculate distance
    from boundary of cells to another
    Note: The below method only works for UNet Segmentation only
    """
    # if there is only one label, zero return the array as is
    if np.sum(annotation) == 0:
        return annotation

    # Masks could be saved as .npy bools, if so convert to uint8 and generate
    # labels from binary
    if annotation.dtype == bool:
        annotation = annotation.astype(np.uint8)
    assert annotation.dtype in [np.uint8, np.uint16], (
        "Expected data type uint, it is {}".format(annotation.dtype))

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
        mask = np.zeros(
            (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
        mask[annotation == unique_value] = 1
        weight_map[index] = 1 / mask.sum()

    # this normalization is important - foreground pixels must have weight 1
    weight_map = [i / max(weight_map) for i in weight_map]

    wc = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    for index, unique_value in enumerate(unique_values):
        wc[annotation == unique_value] = weight_map[index]

    # cells distance map
    border_loss_map = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    distance_maps = np.zeros(
        (annotation.shape[0], annotation.shape[1], np.max(labeled_array)),
        dtype=np.float64)

    if np.max(labeled_array) >= 2:
        for index in range(np.max(labeled_array)):
            mask = np.ones_like(labeled_array)
            mask[labeled_array == index + 1] = 0
            distance_maps[:, :, index] = \
                ndimage.distance_transform_edt(mask)
    distance_maps = np.sort(distance_maps, 2)
    d1 = distance_maps[:, :, 0]
    d2 = distance_maps[:, :, 1]
    border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

    zero_label = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    zero_label[labeled_array == 0] = 1
    border_loss_map = np.multiply(border_loss_map, zero_label)

    # unet weight map mask
    weight_map_mask = wc + border_loss_map

    return weight_map_mask
