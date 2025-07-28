import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    features_path: str
    data_path: str
    tracks_path: str
    channels_to_display: List[str]
    z_range: Tuple[int, int]
    yx_patch_size: Tuple[int, int]
    fov_tracks: Dict[str, Union[List[int], str]] = Field(default_factory=dict)

    @field_validator("features_path", "data_path", "tracks_path")
    @classmethod
    def validate_paths(cls, v):
        if not Path(v).exists():
            logging.warning(f"Path does not exist: {v}")
        return v


class VizConfig(BaseModel):
    """Configuration for visualization app."""

    datasets: Dict[str, DatasetConfig] = Field(default_factory=dict)

    num_PC_components: int = Field(default=8, ge=1, le=10)
    phate_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PHATE parameters. If None, PHATE will not be computed.",
    )
    
    # Combined analysis options
    use_cached_combined_phate: bool = Field(
        default=True,
        description="Use cached combined PHATE results if available",
    )
    combined_phate_cache_path: Optional[str] = Field(
        default=None,
        description="Path to cache combined PHATE results. If None, uses cache_path/combined_phate.zarr",
    )

    # File system paths
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save CSV files and other outputs. If None, uses current working directory.",
    )
    cache_path: Optional[str] = Field(
        default=None,
        description="Path to save/load image cache. If None, images will not be cached to disk.",
    )

    @field_validator("output_dir", "cache_path", "combined_phate_cache_path")
    @classmethod
    def validate_optional_paths(cls, v):
        if v is not None:
            # Create parent directory if it doesn't exist
            path = Path(v)
            if not path.parent.exists():
                logging.info(f"Creating parent directory for: {v}")
                path.parent.mkdir(parents=True, exist_ok=True)
        return v

    def get_datasets(self) -> Dict[str, DatasetConfig]:
        """Get the datasets configuration."""
        return self.datasets

    def get_all_fov_tracks(self) -> Dict[str, Union[List[int], str]]:
        """Get all FOV tracks from all datasets combined."""
        all_fov_tracks = {}

        for dataset_name, dataset_config in self.datasets.items():
            all_fov_tracks[dataset_name] = {}
            for fov_name, track_ids in dataset_config.fov_tracks.items():
                # Keep track IDs as original integers
                # Uniqueness will be handled by (dataset, track_id) tuple
                all_fov_tracks[dataset_name][fov_name] = track_ids

        return all_fov_tracks
