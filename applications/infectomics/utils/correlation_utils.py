"""Shared image correlation / similarity metrics."""

import numpy as np
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score


def compute_ncc_3d(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Normalized cross-correlation (NCC) between two images (any dimensionality).

    Returns a value in [-1, 1].
    """
    img1 = image1.ravel() - np.mean(image1)
    img2 = image2.ravel() - np.mean(image2)
    denominator = np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))
    if denominator == 0:
        return 0.0
    return float(np.sum(img1 * img2) / denominator)


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Mean SSIM computed slice-by-slice along the first (Z) axis.

    Works on both 2D (adds dummy Z=1) and 3D (Z, Y, X) inputs.
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    if image1.ndim == 2:
        image1 = image1[np.newaxis]
        image2 = image2[np.newaxis]
    data_range = image1.max() - image1.min()
    values = [
        ssim(image1[z], image2[z], win_size=3, data_range=data_range)
        for z in range(image1.shape[0])
    ]
    return float(np.mean(values))


def compute_pcc(image1: np.ndarray, image2: np.ndarray) -> float:
    """Pearson correlation coefficient between two images."""
    return float(pearsonr(image1.ravel(), image2.ravel())[0])


def compute_iou(
    image1: np.ndarray,
    image2: np.ndarray,
    use_otsu: bool = True,
    threshold: float = 300.0,
) -> float:
    """
    Intersection-over-Union of binarized images.

    Parameters
    ----------
    use_otsu : bool
        If True, use Otsu thresholding per image. If False, apply a fixed
        `threshold` value to both images.
    threshold : float
        Fixed threshold used when ``use_otsu=False``.
    """
    if use_otsu:
        bin1 = image1 > threshold_otsu(image1)
        bin2 = image2 > threshold_otsu(image2)
    else:
        bin1 = image1 > threshold
        bin2 = image2 > threshold

    intersection = np.logical_and(bin1, bin2)
    union = np.logical_or(bin1, bin2)
    denom = np.sum(union)
    return float(np.sum(intersection) / denom) if denom > 0 else 0.0


def compute_mutual_information(image1: np.ndarray, image2: np.ndarray) -> float:
    """Mutual information between two images (discretised to integer values)."""
    return float(mutual_info_score(image1.ravel(), image2.ravel()))


def compute_patch_status(
    patch_stack: np.ndarray,
    dsRNA_threshold: int = 870,
    ns3_threshold: int = 670,
) -> tuple[int, int]:
    """
    Count bright pixels as a proxy for dsRNA and NS3 presence in a patch.

    Parameters
    ----------
    patch_stack : np.ndarray
        Shape (C, Z, Y, X) or (C, Y, X). Channel 0 = NS3, Channel 1 = dsRNA.
    dsRNA_threshold, ns3_threshold : int
        Raw intensity cut-offs above which a pixel is considered positive.

    Returns
    -------
    (num_pixels_dsRNA, num_pixels_ns3)
    """
    num_pixels_dsRNA = int(np.sum(patch_stack[1] > dsRNA_threshold))
    num_pixels_ns3 = int(np.sum(patch_stack[0] > ns3_threshold))
    return num_pixels_dsRNA, num_pixels_ns3
