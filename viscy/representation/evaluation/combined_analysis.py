import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from xarray import Dataset

from viscy.representation.embedding_writer import (
    read_embedding_dataset,
    write_embedding_dataset,
)
from viscy.representation.evaluation.data_loading import (
    EmbeddingDataLoader,
    TripletEmbeddingLoader,
)
from viscy.representation.evaluation.dimensionality_reduction import compute_phate

__all__ = ["load_and_combine_features", "compute_phate_for_combined_datasets"]

_logger = logging.getLogger("lightning.pytorch")


def load_and_combine_features(
    feature_paths: list[Path],
    dataset_names: Optional[list[str]] = None,
    loader: Literal[
        EmbeddingDataLoader, TripletEmbeddingLoader
    ] = TripletEmbeddingLoader(),
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load features from multiple datasets and combine them using a pluggable loader.

    Parameters
    ----------
    feature_paths : list[Path]
        Paths to embedding datasets
    dataset_names : list[str], optional
        Names for datasets. If None, uses file stems
    loader : EmbeddingDataLoader | TripletEmbeddingLoader, optional
        Custom data loader. If None, uses TripletEmbeddingLoader

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Combined features array and index DataFrame with dataset_pair column
    """
    if dataset_names is None:
        dataset_names = [path.stem for path in feature_paths]

    if len(dataset_names) != len(feature_paths):
        raise ValueError("Number of dataset names must match number of feature paths")

    all_features = []
    all_indices = []

    for path, dataset_name in zip(feature_paths, dataset_names):
        _logger.info(f"Loading features from {path}")

        dataset = loader.load_dataset(path)
        features = loader.extract_features(dataset)
        index_df = loader.extract_metadata(dataset)

        index_df["dataset_pair"] = dataset_name
        index_df["dataset_path"] = str(path)

        all_features.append(features)
        all_indices.append(index_df)

        _logger.info(f"Loaded {len(features)} samples from {dataset_name}")

    combined_features = np.vstack(all_features)
    combined_indices = pd.concat(all_indices, ignore_index=True)

    _logger.info(
        f"Combined {len(combined_features)} total samples from {len(feature_paths)} datasets"
    )

    return combined_features, combined_indices


def compute_phate_for_combined_datasets(
    feature_paths: list[Path],
    output_path: Path,
    dataset_names: Optional[list[str]] = None,
    phate_kwargs: Optional[dict] = None,
    overwrite: bool = False,
    loader: Literal[
        EmbeddingDataLoader, TripletEmbeddingLoader
    ] = TripletEmbeddingLoader(),
) -> Dataset:
    """
    Compute PHATE embeddings on combined features from multiple datasets.

    Parameters
    ----------
    feature_paths : list[Path]
        List of paths to zarr stores containing embedding datasets
    output_path : Path
        Path to save the combined dataset with PHATE embeddings
    dataset_names : list[str], optional
        Names for each dataset. If None, uses file stems
    phate_kwargs : dict, optional
        Parameters for PHATE computation. Default: {"knn": 5, "decay": 40, "n_components": 2}
    overwrite : bool, optional
        Whether to overwrite existing output file
    loader : EmbeddingDataLoader | TripletEmbeddingLoader, optional
        Custom data loader. If None, uses TripletEmbeddingLoader

    Returns
    -------
    Dataset
        Combined xarray dataset with original features and PHATE coordinates

    Raises
    ------
    FileExistsError
        If output_path exists and overwrite is False
    ImportError
        If PHATE is not installed
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path {output_path} already exists. Use overwrite=True to overwrite."
        )

    if phate_kwargs is None:
        phate_kwargs = {"knn": 5, "decay": 40, "n_components": 2}

    _logger.info(
        f"Computing PHATE for combined datasets with parameters: {phate_kwargs}"
    )

    combined_features, combined_indices = load_and_combine_features(
        feature_paths, dataset_names, loader
    )

    n_samples = len(combined_features)
    if phate_kwargs.get("knn", 5) >= n_samples:
        original_knn = phate_kwargs["knn"]
        phate_kwargs["knn"] = max(2, n_samples // 2)
        _logger.warning(
            f"Reducing knn from {original_knn} to {phate_kwargs['knn']} due to dataset size ({n_samples} samples)"
        )

    _logger.info("Computing PHATE embeddings on combined features")
    try:
        _, phate_embedding = compute_phate(combined_features, **phate_kwargs)
        _logger.info(
            f"PHATE computation successful, embedding shape: {phate_embedding.shape}"
        )
    except Exception as e:
        _logger.error(f"PHATE computation failed: {str(e)}")
        raise

    _logger.info(f"Writing combined dataset with PHATE embeddings to {output_path}")
    write_embedding_dataset(
        output_path=output_path,
        features=combined_features,
        index_df=combined_indices,
        phate_kwargs=phate_kwargs,
        overwrite=overwrite,
    )

    result_dataset = read_embedding_dataset(output_path)
    _logger.info(
        f"Successfully created combined dataset with {len(result_dataset.sample)} samples"
    )

    return result_dataset
