"""Pydantic configuration models for the DynaCLR evaluation orchestrator."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from dynaclr.evaluation.dimensionality_reduction.config import PCAConfig, PHATEConfig, UMAPConfig


class PredictStepConfig(BaseModel):
    """Configuration for the embedding extraction (predict) step.

    Parameters
    ----------
    batch_size : int
        Batch size for inference. Default: 128.
    num_workers : int
        DataLoader thread workers. Default: 2.
    precision : str
        Mixed-precision setting for Lightning Trainer. Default: "bf16-mixed".
    devices : int
        Number of GPUs. Default: 1.
    """

    batch_size: int = 128
    num_workers: int = 2
    precision: str = "32-true"
    devices: int = 1


class ReduceCombinedStepConfig(BaseModel):
    """Configuration for the joint dimensionality reduction step across experiments.

    Parameters
    ----------
    overwrite_keys : bool
        Whether to overwrite existing obsm keys. Default: True.
    pca : PCAConfig or None
        PCA parameters for joint fit. Results stored as X_pca_combined.
    umap : UMAPConfig or None
        UMAP parameters for joint fit. Results stored as X_umap_combined.
    phate : PHATEConfig or None
        PHATE parameters for joint fit. Results stored as X_phate_combined.
    """

    overwrite_keys: bool = True
    pca: Optional[PCAConfig] = PCAConfig(n_components=32, normalize_features=True)
    umap: Optional[UMAPConfig] = None
    phate: Optional[PHATEConfig] = PHATEConfig(n_components=2, knn=5, decay=40, scale_embeddings=False)


class ReduceStepConfig(BaseModel):
    """Configuration for the dimensionality reduction step.

    Parameters
    ----------
    overwrite_keys : bool
        Whether to overwrite existing obsm keys. Default: True.
    pca : PCAConfig or None
        PCA parameters. None skips PCA.
    umap : UMAPConfig or None
        UMAP parameters. None skips UMAP.
    phate : PHATEConfig or None
        PHATE parameters. None skips PHATE.
    """

    overwrite_keys: bool = True
    pca: Optional[PCAConfig] = PCAConfig(n_components=32, normalize_features=True)
    umap: Optional[UMAPConfig] = None
    phate: Optional[PHATEConfig] = None  # PHATE runs jointly in reduce_combined, not per-experiment


class SmoothnessStepConfig(BaseModel):
    """Configuration for the temporal smoothness evaluation step.

    Parameters
    ----------
    distance_metric : str
        Distance metric. "cosine" or "euclidean". Default: "cosine".
    save_plots : bool
        Save distribution plots. Default: True.
    save_distributions : bool
        Save raw distribution arrays. Default: False.
    verbose : bool
        Print verbose progress. Default: True.
    """

    distance_metric: Literal["cosine", "euclidean"] = "cosine"
    save_plots: bool = True
    save_distributions: bool = False
    verbose: bool = True


class PlotStepConfig(BaseModel):
    """Configuration for the embedding visualization step.

    Parameters
    ----------
    embedding_keys : list[str]
        Per-experiment obsm keys to plot (looped over each split zarr).
        Default: ["X_pca"].
    combined_embedding_keys : list[str]
        Cross-experiment obsm keys to plot once across all zarrs concatenated.
        Default: ["X_pca_combined", "X_phate_combined"].
    color_by : list[str]
        obs columns to color scatter plots by. Default: common metadata columns.
    point_size : float
        Scatter plot point size. Default: 1.0.
    components : tuple[int, int]
        Which components to use as X/Y axes (0-indexed). Default: (0, 1).
    format : str
        Output format. "pdf" or "png". Default: "pdf".
    """

    embedding_keys: list[str] = ["X_pca"]
    combined_embedding_keys: list[str] = ["X_pca_combined", "X_phate_combined"]
    color_by: list[str] = ["perturbation", "hours_post_perturbation", "experiment", "marker"]
    point_size: float = 1.0
    components: tuple[int, int] = (0, 1)
    format: str = "pdf"


class AnnotationSource(BaseModel):
    """Annotation CSV for one experiment.

    Parameters
    ----------
    experiment : str
        Experiment name matching obs["experiment"] in the embeddings zarr.
    path : str
        Absolute path to the annotation CSV. Must have fov_name, id, and
        at least one task column (e.g. infection_state, organelle_state).
    """

    experiment: str
    path: str


class TaskSpec(BaseModel):
    """One classification task to evaluate.

    Parameters
    ----------
    task : str
        Task column name in annotation CSVs (e.g. infection_state, organelle_state).
    marker_filters : list[str] or None
        If set, run one classifier per marker, using only embeddings where
        obs["marker"] == that marker. None (default) runs one classifier using
        all markers combined — useful to compare predictive power across channels.
    """

    task: str
    marker_filters: Optional[list[str]] = None


class LinearClassifiersStepConfig(BaseModel):
    """Configuration for the orchestrated linear classifiers step.

    Parameters
    ----------
    annotations : list[AnnotationSource]
        Per-experiment annotation CSVs. Each entry maps an experiment name
        (matching obs["experiment"] in embeddings.zarr) to a CSV path.
    tasks : list[TaskSpec]
        Tasks to evaluate. Each task can optionally filter by marker.
    use_scaling : bool
        Apply StandardScaler. Default: True.
    use_pca : bool
        Apply PCA before classifier. Default: False.
    n_pca_components : int or None
        Number of PCA components (required if use_pca is True).
    max_iter : int
        Max iterations for solver. Default: 1000.
    class_weight : str or None
        Class weighting. "balanced" or None. Default: "balanced".
    solver : str
        Optimization algorithm. Default: "liblinear".
    split_train_data : float
        Fraction for training. Default: 0.8.
    random_seed : int
        Random seed for reproducibility. Default: 42.
    """

    annotations: list[AnnotationSource]
    tasks: list[TaskSpec]
    use_scaling: bool = True
    use_pca: bool = False
    n_pca_components: Optional[int] = None
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    solver: str = "liblinear"
    split_train_data: float = 0.8
    random_seed: int = 42


class SlurmConfig(BaseModel):
    """SLURM configuration for generated job scripts.

    Parameters
    ----------
    gpu_partition : str
        Partition for GPU jobs. Default: "gpu".
    cpu_partition : str
        Partition for CPU jobs. Default: "cpu".
    gpu_mem : str
        Memory for GPU jobs. Default: "112G".
    cpu_mem : str
        Memory for CPU jobs. Default: "128G".
    gpu_time : str
        Time limit for GPU jobs. Default: "0-04:00:00".
    cpu_time : str
        Time limit for CPU jobs. Default: "0-02:00:00".
    cpus_per_task : int
        CPUs per task for CPU jobs. Default: 16.
    conda_env : str or None
        Conda environment name to activate. None uses uv directly.
    workspace_dir : str
        Path to the viscy repository root.
    """

    gpu_partition: str = "gpu"
    cpu_partition: str = "cpu"
    gpu_mem: str = "112G"
    cpu_mem: str = "128G"
    gpu_time: str = "0-04:00:00"
    cpu_time: str = "0-02:00:00"
    cpus_per_task: int = 16
    conda_env: Optional[str] = None
    workspace_dir: str = "/hpc/mydata/eduardo.hirata/repos/viscy"


class EvaluationConfig(BaseModel):
    """Top-level configuration for the DynaCLR evaluation orchestrator.

    Parameters
    ----------
    training_config : str
        Path to the training YAML config (Lightning CLI format). Model
        architecture, normalizations, and data parameters are auto-extracted.
    ckpt_path : str
        Path to the model checkpoint (.ckpt).
    cell_index_path : str or None
        Override the cell index parquet path from the training config.
        None = use the path from the training config.
    output_dir : str
        Root directory for all evaluation outputs.
    steps : list[str]
        Ordered list of steps to generate configs for.
        Valid values: predict, split, reduce_dimensionality, reduce_combined,
        plot, smoothness, linear_classifiers.
    predict : PredictStepConfig
        Predict step configuration.
    reduce_dimensionality : ReduceStepConfig
        Per-experiment dimensionality reduction step configuration.
    reduce_combined : ReduceCombinedStepConfig
        Joint dimensionality reduction across all experiments.
    smoothness : SmoothnessStepConfig
        Smoothness evaluation configuration.
    plot : PlotStepConfig
        Embedding visualization configuration.
    linear_classifiers : LinearClassifiersStepConfig or None
        Linear classifier configuration. None disables this step.
    slurm : SlurmConfig
        SLURM job configuration for generated scripts.
    """

    training_config: str
    ckpt_path: str
    cell_index_path: Optional[str] = None
    output_dir: str
    steps: list[str] = ["predict", "split", "reduce_dimensionality", "reduce_combined", "plot", "smoothness"]
    predict: PredictStepConfig = PredictStepConfig()
    reduce_dimensionality: ReduceStepConfig = ReduceStepConfig()
    reduce_combined: ReduceCombinedStepConfig = ReduceCombinedStepConfig()
    smoothness: SmoothnessStepConfig = SmoothnessStepConfig()
    plot: PlotStepConfig = PlotStepConfig()
    linear_classifiers: Optional[LinearClassifiersStepConfig] = None
    slurm: SlurmConfig = SlurmConfig()
