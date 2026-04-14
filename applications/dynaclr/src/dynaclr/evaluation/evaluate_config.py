"""Pydantic configuration models for the DynaCLR evaluation orchestrator."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from dynaclr.evaluation.dimensionality_reduction.config import PCAConfig, PHATEConfig, UMAPConfig
from dynaclr.evaluation.mmd.config import ComparisonSpec, MAPSettings, MMDSettings


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
        obs columns for per-experiment plots. Default: perturbation, hours, marker.
    combined_color_by : list[str]
        obs columns for combined (cross-experiment) plots. Adds "experiment" to color_by.
    point_size : float
        Scatter plot point size. Default: 1.0.
    components : tuple[int, int]
        Which components to use as X/Y axes (0-indexed). Default: (0, 1).
    format : str
        Output format. "pdf" or "png". Default: "pdf".
    """

    embedding_keys: list[str] = ["X_pca"]
    combined_embedding_keys: list[str] = ["X_pca_combined", "X_phate_combined"]
    color_by: list[str] = ["perturbation", "hours_post_perturbation", "marker"]
    combined_color_by: list[str] = ["perturbation", "hours_post_perturbation", "experiment", "marker"]
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
        If set, run one classifier per listed marker. None (default) runs one
        classifier per marker discovered in the data (all unique obs["marker"] values).
    """

    task: str
    marker_filters: Optional[list[str]] = None


class MMDStepConfig(BaseModel):
    """Configuration for one MMD evaluation block.

    Comparisons are explicit ``(cond_a, cond_b, label)`` pairs — no auto-discovery.
    Include a null comparison (e.g. uninfected1 vs uninfected2) to establish
    a baseline false-positive rate.

    Parameters
    ----------
    comparisons : list[ComparisonSpec]
        Explicit pairwise comparisons to run.
    group_by : str
        obs column whose values are referenced by ``cond_a``/``cond_b``.
        Default: "perturbation".
    obs_filter : dict[str, str] or None
        Subset adata to rows where obs[key] == value before running MMD.
        Example: ``{perturbation: uninfected}`` to restrict batch-QC
        comparisons to control cells only. None = use all cells.
    embedding_key : str or None
        obsm key to use. None = raw .X. Default: None.
    mmd : MMDSettings
        Kernel MMD algorithm settings (permutations, cell caps, seed, etc.).
    map_settings : MAPSettings
        copairs-based mAP settings. Default: disabled.
    temporal_bin_size : float or None
        Width of each temporal bin in hours. Edges derived from data max.
        None = aggregate MMD.
    combined_temporal_bin_size : float or None
        Override temporal_bin_size for the combined (cross-experiment) run only.
        If not set, falls back to temporal_bin_size. Use None to aggregate across
        all time in the combined run while keeping per-experiment binning.
    save_plots : bool
        Generate kinetics and heatmap plots. Default: True.
    combined_mode : bool
        Also run cross-experiment MMD with per-experiment batch centering.
        Default: False.
    name : str or None
        Short name used in output filenames (e.g. "perturbation", "batch_qc").
        Auto-derived from group_by if None.
    """

    comparisons: list[ComparisonSpec]
    group_by: str = "perturbation"
    obs_filter: Optional[dict[str, str]] = None
    embedding_key: Optional[str] = None
    mmd: MMDSettings = MMDSettings()
    map_settings: MAPSettings = MAPSettings()
    temporal_bin_size: Optional[float] = None
    combined_temporal_bin_size: Optional[float] = None
    save_plots: bool = True
    combined_mode: bool = False
    name: Optional[str] = None


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
        plot, smoothness, mmd, linear_classifiers.
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
    mmd : list[MMDStepConfig]
        MMD evaluation blocks. Each block is an independent run with its own
        group_by, comparisons, and optional obs_filter. Empty list disables MMD.
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
    mmd: list[MMDStepConfig] = []
