"""Shared organelle and texture feature extraction utilities."""

import numpy as np
from scipy.ndimage import label
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage.util import img_as_float


# ---------------------------------------------------------------------------
# Cell / patch geometry
# ---------------------------------------------------------------------------


def find_centroid(
    nucl_seg_mask: np.ndarray, cell_label: int
) -> tuple[int, int]:
    """
    Return (x, y) pixel centroid of a labeled nucleus.

    Parameters
    ----------
    nucl_seg_mask : np.ndarray
        2D integer label image of nuclei.
    cell_label : int
        The label value to locate.

    Returns
    -------
    (x_centroid, y_centroid)
    """
    props = regionprops((nucl_seg_mask == cell_label).astype(int))
    y_centroid, x_centroid = map(int, props[0].centroid)
    return x_centroid, y_centroid


def find_mem_label(
    nucl_mask: np.ndarray, cyto_mask: np.ndarray, cell_label: int
) -> int:
    """
    Find the membrane/cell label in ``cyto_mask`` that corresponds to the
    nucleus ``cell_label`` in ``nucl_mask`` (uses nuclear centroid lookup).
    """
    props = regionprops((nucl_mask == cell_label).astype(int))
    y_centroid, x_centroid = map(int, props[0].centroid)
    return int(cyto_mask[y_centroid, x_centroid])


# ---------------------------------------------------------------------------
# Texture features (operate on 2D MIP or 2D patches)
# ---------------------------------------------------------------------------


def compute_glcm_features(data: np.ndarray) -> tuple[float, float, float, float]:
    """
    Gray-Level Co-occurrence Matrix texture features.

    Accepts a (Z, Y, X) volume (MIP is taken) or a 2D (Y, X) image.

    Returns
    -------
    (contrast, homogeneity, energy, correlation)
    """
    mip = np.max(data, axis=0) if data.ndim == 3 else data
    img_u8 = ((mip - mip.min()) / (mip.max() - mip.min() + 1e-8) * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_u8,
        distances=[1, 2, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    return contrast, homogeneity, energy, correlation


def compute_edge_density(data: np.ndarray) -> float:
    """
    Fraction of pixels identified as edges by Canny detection.

    Accepts (Z, Y, X) or (Y, X).
    """
    mip = np.max(data, axis=0) if data.ndim == 3 else data
    img_u8 = ((mip - mip.min()) / (mip.max() - mip.min() + 1e-8) * 255).astype(np.uint8)
    edges = canny(img_as_float(img_u8), sigma=1.0)
    return float(np.sum(edges) / edges.size)


def compute_lbp_histogram(data: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Local Binary Pattern histogram (uniform method).

    Accepts (Z, Y, X) or (Y, X).
    """
    mip = np.max(data, axis=0) if data.ndim == 3 else data
    n_points = 8 * radius
    lbp = local_binary_pattern(mip, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


# ---------------------------------------------------------------------------
# Organelle morphology features (require segmentation masks)
# ---------------------------------------------------------------------------


def compute_organelle_volume_fraction(
    org_data: np.ndarray,
    org_mask: np.ndarray,
    cyto_mask: np.ndarray,
    mem_label: int,
) -> tuple[float, float, float]:
    """
    Compute organelle volume, volume fraction and mean intensity within a cell.

    Suitable for continuous / membrane-like organelles (e.g. TOMM20, SEC61B).

    Parameters
    ----------
    org_data : np.ndarray  Shape (Z, Y, X).
    org_mask : np.ndarray  Binary or label mask, shape (Z, Y, X).
    cyto_mask : np.ndarray 2D cell-membrane label image.
    mem_label : int        The label in ``cyto_mask`` for this cell.

    Returns
    -------
    (organelle_volume, organelle_volume_fraction, organelle_intensity)
    """
    cell_mask_2d = (cyto_mask == mem_label).astype(int)
    cell_mask_3d = np.repeat(cell_mask_2d[np.newaxis], org_data.shape[0], axis=0)
    masked_org = org_mask * cell_mask_3d
    organelle_volume = float(np.sum(masked_org))
    organelle_volume_fraction = float(organelle_volume / (np.sum(cell_mask_3d) + 1e-8))
    organelle_intensity = float(np.mean(org_data * masked_org))
    return organelle_volume, organelle_volume_fraction, organelle_intensity


def compute_organelle_count_and_size(
    org_data: np.ndarray,
    org_mask: np.ndarray,
    cyto_mask: np.ndarray,
    mem_label: int,
) -> tuple[int, float]:
    """
    Count organelle objects and compute their mean size within a cell.

    Suitable for punctate organelles (e.g. LAMP1 vesicles, stress granules).

    Parameters
    ----------
    org_data : np.ndarray  Shape (Z, Y, X).
    org_mask : np.ndarray  Binary mask, shape (Z, Y, X).
    cyto_mask : np.ndarray 2D cell-membrane label image.
    mem_label : int        The label in ``cyto_mask`` for this cell.

    Returns
    -------
    (organelle_count, mean_organelle_volume_px)
    """
    cell_mask_2d = (cyto_mask == mem_label).astype(int)
    cell_mask_3d = np.repeat(cell_mask_2d[np.newaxis], org_data.shape[0], axis=0)
    masked_org = org_data * org_mask * cell_mask_3d

    labeled, num_labels = label(masked_org)
    props = regionprops(labeled)
    mean_volume = float(np.mean([p.area for p in props])) if props else 0.0
    return num_labels, mean_volume
