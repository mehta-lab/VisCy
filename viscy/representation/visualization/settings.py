import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    num_phate_components: Optional[int] = None
    phate_knn: int = Field(default=5, ge=1)
    phate_decay: int = Field(default=40, ge=1)

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
