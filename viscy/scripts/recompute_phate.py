import logging
from pathlib import Path
from typing import Any

from xarray import open_zarr

from viscy.representation.evaluation.dimensionality_reduction import (
    _fit_transform_phate,
)

_logger = logging.getLogger(__name__)


def update_phate_embeddings(
    dataset_path: Path,
    phate_kwargs: dict[str, Any],
) -> None:
    """
    Update PHATE embeddings in an existing dataset with new parameters.

    Parameters
    ----------
    dataset_path : Path
        Path to the zarr store containing embeddings
    phate_kwargs : dict
        New PHATE parameters to use for recomputing embeddings.
        Common parameters include:
        - n_components: int, number of dimensions (default: 2)
        - knn: int, number of nearest neighbors (default: 5)
        - decay: int, decay rate for kernel (default: 40)
        - n_jobs: int, number of jobs for parallel processing
        - t: int, number of diffusion steps
        - gamma: float, gamma parameter for kernel
    """
    # Load dataset
    dataset = open_zarr(dataset_path, mode="r+")
    features = dataset["features"].values

    # Compute new PHATE embeddings
    _logger.info(f"Computing PHATE embeddings with parameters: {phate_kwargs}")
    _, phate = _fit_transform_phate(features, **phate_kwargs)

    # Update PHATE coordinates
    dataset["PHATE1"].values = phate[:, 0]
    dataset["PHATE2"].values = phate[:, 1]

    _logger.info(f"Updated PHATE embeddings in {dataset_path}")
