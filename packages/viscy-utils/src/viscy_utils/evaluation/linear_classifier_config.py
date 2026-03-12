"""Configuration models for linear classifier training and inference."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Valid classification tasks
VALID_TASKS = Literal["infection_state", "organelle_state", "cell_division_state", "cell_death_state"]

# Valid input channels
VALID_CHANNELS = Literal["phase", "sensor", "organelle"]

WANDB_PROJECT_PREFIX = "linearclassifiers"


class LinearClassifierTrainConfig(BaseModel):
    """Configuration for linear classifier training.

    Parameters
    ----------
    task : str
        Classification task name (one of: infection_state, organelle_state,
        cell_division_state, cell_death_state).
    input_channel : str
        Input channel name (one of: phase, sensor, organelle).
    embedding_model_name : str
        Name of the embedding model (e.g. ``DynaCLR-2D-BagOfChannels-timeaware``).
    embedding_model_version : str
        Version of the embedding model (e.g. ``v3``).
    train_datasets : list[dict]
        List of training datasets with 'embeddings' and 'annotations' paths.
        Each dict may optionally include 'include_wells', a list of well
        prefixes (e.g. ["A/1", "B/2"]) to filter by fov_name.
    use_scaling : bool
        Whether to apply StandardScaler normalization.
    use_pca : bool
        Whether to apply PCA dimensionality reduction.
    n_pca_components : Optional[int]
        Number of PCA components (required if use_pca=True).
    max_iter : int
        Maximum number of iterations for solver.
    class_weight : Optional[str]
        Weighting strategy for classes ('balanced' or None).
    solver : str
        Algorithm to use for optimization.
    split_train_data : float
        Fraction of data to use for training (rest for validation).
    random_seed : int
        Random seed for reproducibility.
    wandb_entity : Optional[str]
        W&B entity (username or team).
    wandb_tags : list[str]
        Tags to add to the run.
    """

    # Task metadata
    task: VALID_TASKS = Field(...)
    input_channel: VALID_CHANNELS = Field(...)
    marker: Optional[str] = Field(
        default=None,
        description="Marker name for marker-specific tasks (e.g. g3bp1, sec61b, tomm20).",
    )
    embedding_model_name: str = Field(..., min_length=1)
    embedding_model_version: str = Field(..., min_length=1)

    # Training datasets
    train_datasets: list[dict] = Field(..., min_length=1)

    # Preprocessing
    use_scaling: bool = Field(default=True)
    use_pca: bool = Field(default=False)
    n_pca_components: Optional[int] = Field(default=None)

    # Classifier parameters
    max_iter: int = Field(default=1000, gt=0)
    class_weight: Optional[Literal["balanced"]] = Field(default="balanced")
    solver: str = Field(default="liblinear")

    # Training parameters
    split_train_data: float = Field(default=0.8, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)

    # W&B configuration
    wandb_entity: Optional[str] = Field(default=None)
    wandb_tags: list[str] = Field(default_factory=list)

    @field_validator("embedding_model_name", "embedding_model_version")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @property
    def wandb_project(self) -> str:
        """Derive W&B project name from embedding model name and version."""
        return f"{WANDB_PROJECT_PREFIX}-{self.embedding_model_name}-{self.embedding_model_version}"

    @model_validator(mode="after")
    def validate_config(self):
        """Validate PCA settings and dataset paths."""
        # PCA validation
        if self.use_pca and self.n_pca_components is None:
            raise ValueError("n_pca_components must be specified when use_pca=True")
        if self.use_pca and self.n_pca_components is not None:
            if self.n_pca_components <= 0:
                raise ValueError("n_pca_components must be positive")

        # Dataset validation
        for i, dataset in enumerate(self.train_datasets):
            if not isinstance(dataset, dict):
                raise ValueError(f"Dataset {i} must be a dict")
            if "embeddings" not in dataset or "annotations" not in dataset:
                raise ValueError(f"Dataset {i} must have 'embeddings' and 'annotations' keys")

            embeddings_path = Path(dataset["embeddings"])
            annotations_path = Path(dataset["annotations"])

            if not embeddings_path.exists():
                raise ValueError(f"Dataset {i}: Embeddings file not found: {dataset['embeddings']}")
            if not annotations_path.exists():
                raise ValueError(f"Dataset {i}: Annotations file not found: {dataset['annotations']}")

        return self


class ClassifierModelSpec(BaseModel):
    """Specification for a single classifier model in batch inference.

    Parameters
    ----------
    model_name : str
        Name of the model artifact in W&B.
    version : str
        Version of the model artifact (e.g., 'latest', 'v0').
    include_wells : Optional[list[str]]
        Well prefixes to restrict prediction to (e.g. ``["A/1", "B/2"]``).
        Cells in other wells will have ``NaN`` for prediction columns.
        When ``None`` (the default), all cells are predicted.
    """

    model_name: str = Field(..., min_length=1)
    version: str = Field(default="latest", min_length=1)
    include_wells: Optional[list[str]] = Field(default=None)


class LinearClassifierInferenceConfig(BaseModel):
    """Configuration for linear classifier inference.

    Parameters
    ----------
    embedding_model_name : str
        Name of the embedding model (e.g. ``DynaCLR-2D-BagOfChannels-timeaware``).
    embedding_model_version : str
        Version of the embedding model (e.g. ``v3``).
    wandb_entity : Optional[str]
        W&B entity (username or team).
    embeddings_path : str
        Path to embeddings zarr file for inference.
    output_path : Optional[str]
        Path to save output zarr file with predictions. When ``None``
        (the default), predictions are written back to ``embeddings_path``.
    overwrite : bool
        Whether to overwrite output if it exists.
    models : list[ClassifierModelSpec]
        List of classifier models to apply. Each model can specify
        its own ``include_wells`` filter.
    """

    embedding_model_name: str = Field(..., min_length=1)
    embedding_model_version: str = Field(..., min_length=1)
    wandb_entity: Optional[str] = Field(default=None)
    embeddings_path: str = Field(..., min_length=1)
    output_path: Optional[str] = Field(default=None)
    overwrite: bool = Field(default=False)
    models: list[ClassifierModelSpec] = Field(..., min_length=1)

    @field_validator("embedding_model_name", "embedding_model_version", "embeddings_path")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @property
    def wandb_project(self) -> str:
        """Derive W&B project name from embedding model name and version."""
        return f"{WANDB_PROJECT_PREFIX}-{self.embedding_model_name}-{self.embedding_model_version}"

    @model_validator(mode="after")
    def validate_paths(self):
        """Validate input exists and output doesn't exist unless overwrite=True."""
        embeddings_path = Path(self.embeddings_path)

        if not embeddings_path.exists():
            raise ValueError(f"Embeddings file not found: {self.embeddings_path}")

        if self.output_path is not None:
            output_path = Path(self.output_path)
            if output_path.exists() and not self.overwrite:
                raise ValueError(f"Output file already exists: {self.output_path}. Set overwrite=true to overwrite.")
        return self
