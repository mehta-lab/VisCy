import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from xarray import Dataset

__all__ = ["EmbeddingDataLoader", "TripletEmbeddingLoader"]

_logger = logging.getLogger("lightning.pytorch")


@runtime_checkable
class EmbeddingDataLoader(Protocol):
    """Protocol for embedding dataloaders that can be used with combined analysis."""

    def load_dataset(self, path: Path) -> Dataset:
        """
        Load dataset from path and return xarray Dataset with 'features' data variable.

        Parameters
        ----------
        path : Path
            Path to the dataset file

        Returns
        -------
        Dataset
            Xarray dataset with 'features' data variable containing embeddings
        """
        ...

    def extract_features(self, dataset: Dataset) -> np.ndarray:
        """
        Extract feature embeddings from dataset.

        Parameters
        ----------
        dataset : Dataset
            Xarray dataset containing features

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_features) containing embeddings
        """
        ...

    def extract_metadata(self, dataset: Dataset) -> pd.DataFrame:
        """
        Extract metadata/index information from dataset.

        Parameters
        ----------
        dataset : Dataset
            Xarray dataset containing metadata

        Returns
        -------
        pd.DataFrame
            DataFrame containing metadata for each sample, including any existing
            dimensionality reduction coordinates (PHATE, UMAP, etc.)
        """
        ...


class TripletEmbeddingLoader:
    """Default loader for triplet-based embedding datasets."""

    def load_dataset(self, path: Path) -> Dataset:
        """Load embedding dataset using the standard embedding writer format."""
        from viscy.representation.embedding_writer import read_embedding_dataset

        _logger.debug(f"Loading dataset from {path} using TripletEmbeddingLoader")
        return read_embedding_dataset(path)

    def extract_features(self, dataset: Dataset) -> np.ndarray:
        """Extract features from the 'features' data variable."""
        return dataset["features"].values

    def extract_metadata(self, dataset: Dataset) -> pd.DataFrame:
        """
        Extract metadata from dataset coordinates and data variables.

        This includes sample coordinates and any existing dimensionality reduction
        coordinates like PHATE, UMAP, PCA that were previously computed.
        """
        features_data_array = dataset["features"]

        try:
            coord_df = features_data_array["sample"].to_dataframe()

            if coord_df.index.names != [None]:
                index_df = coord_df.reset_index()
                if "features" in index_df.columns:
                    index_df = index_df.drop(columns=["features"])
            else:
                index_df = coord_df.reset_index(drop=True)

            dim_reduction_cols = [
                col
                for col in index_df.columns
                if any(col.startswith(prefix) for prefix in ["PHATE", "UMAP", "PCA"])
            ]
            if dim_reduction_cols:
                _logger.debug(
                    f"Found dimensionality reduction coordinates: {dim_reduction_cols}"
                )

            _logger.debug(
                f"Extracted metadata with {len(index_df.columns)} columns: {list(index_df.columns)}"
            )
            return index_df

        except Exception as e:
            _logger.error(f"Error extracting metadata: {e}")
            index_df = (
                features_data_array["sample"].to_dataframe().reset_index(drop=True)
            )
            _logger.warning(
                f"Using fallback metadata extraction with {len(index_df.columns)} columns"
            )
            return index_df


# Example of how to implement a custom loader
# TODO: replace with the other dataloaders
class CustomEmbeddingLoader:
    """
    Example implementation of a custom embedding loader.

    This serves as a template for implementing loaders for different data formats.
    Replace the method implementations with your specific loading logic.
    """

    def load_dataset(self, path: Path) -> Dataset:
        """
        Load your custom dataset format.

        This should return an xarray Dataset with at least a 'features' data variable
        containing the embeddings with a 'sample' dimension.
        """
        raise NotImplementedError("Implement your custom dataset loading logic here")

    def extract_features(self, dataset: Dataset) -> np.ndarray:
        """
        Extract features from your dataset format.

        Should return a numpy array of shape (n_samples, n_features).
        """
        return dataset["features"].values  # Modify if your format is different

    def extract_metadata(self, dataset: Dataset) -> pd.DataFrame:
        """
        Extract metadata from your dataset format.

        Should return a DataFrame with one row per sample containing metadata
        like sample IDs, FOV names, track IDs, etc.
        """
        raise NotImplementedError(
            "Implement your custom metadata extraction logic here"
        )
