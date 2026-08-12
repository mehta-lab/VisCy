"""Pydantic configuration models for the DynaCLR evaluation orchestrator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

from dynaclr.evaluation.dimensionality_reduction.config import PCAConfig, PHATEConfig, UMAPConfig
from dynaclr.evaluation.mmd.config import (
    ComparisonSpec,
    MAPSettings,
    MMDRepresentationConfig,
    MMDSettings,
)


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
    embedding_key : {"features", "projections"}
        Which array the EmbeddingWriter stores as the primary embedding in
        ``adata.X``. ``"features"`` (default) writes the encoder backbone
        output. ``"projections"`` writes the trained projection-head output —
        required when the projection head is the only finetuned component
        (e.g. DINOv3-temporal-MLP, where the DINOv3 backbone is frozen and the
        MLP head carries all the learned task signal). The unselected array
        is still saved to ``obsm["X_projections"]`` / ``obsm["X_backbone"]``
        as a sidecar.
    """

    batch_size: int = 128
    num_workers: int = 2
    precision: str = "32-true"
    devices: int = 1
    embedding_key: Literal["features", "projections"] = "features"


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
    pca: PCAConfig | None = PCAConfig(n_components=32, normalize_features=True)
    umap: UMAPConfig | None = None
    phate: PHATEConfig | None = PHATEConfig(n_components=2, knn=5, decay=40, scale_embeddings=False)


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
    pca: PCAConfig | None = PCAConfig(n_components=32, normalize_features=True)
    umap: UMAPConfig | None = None
    phate: PHATEConfig | None = None  # PHATE runs jointly in reduce_combined, not per-experiment


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
    pairplot_components : int
        Number of leading PCs to render in the pairplot grid (NxN panels).
        Smaller is faster: rendering scales with N^2. Default: 8.
    format : str
        Output format. "pdf" or "png". Default: "pdf".
    """

    embedding_keys: list[str] = ["X_pca"]
    combined_embedding_keys: list[str] = ["X_pca_combined", "X_phate_combined"]
    color_by: list[str] = ["perturbation", "hours_post_perturbation", "marker"]
    combined_color_by: list[str] = ["perturbation", "hours_post_perturbation", "experiment", "marker"]
    point_size: float = 1.0
    components: tuple[int, int] = (0, 1)
    pairplot_components: int = 8
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
    marker_filters: list[str] | None = None


class WitnessLabelSource(BaseModel):
    """Reference spec for one experiment, used to weak-label via the MMD witness.

    The MMD witness function scores each cell by how much it looks like the
    control reference (witness group X) vs the perturbed reference (group Y).
    Those scores are then gated into discrete pseudo-labels that replace
    annotation-CSV labels for the classifier.

    Each side (control / perturbed) is defined in one of two mutually exclusive
    ways:

    - **Well-based** (``control_wells`` / ``perturbed_wells``): the simple case,
      matched against ``obs["fov_name"]`` by path prefix (``"C/1"`` matches
      ``"C/1/000000"`` but not ``"C/10/..."``).
    - **Filter-based** (``control_filter`` / ``perturbed_filter``): an arbitrary
      obs filter (``col -> scalar | list | range-dict``), enabling contrasts
      like early-timepoint vs late-timepoint or control vs perturbed-at-late.
      A range-dict uses ``{lt, le, gt, ge}`` (one bound = half-line, two =
      window); a ``well``/``fov_name`` key routes to well-prefix matching.

    Provide wells XOR a filter for each side (both sides must use the same
    style). Cells matched by neither side are dropped (never silently labeled).

    Parameters
    ----------
    experiment : str
        Experiment name matching obs["experiment"] in the embeddings zarr.
    control_wells : list[str] or None
        Well ids for the control reference (group X). Well-based style.
    perturbed_wells : list[str] or None
        Well ids for the perturbed reference (group Y). Well-based style.
    control_filter : dict or None
        obs filter for the control reference (group X). Filter-based style.
        E.g. ``{"well": "A/2"}`` or
        ``{"perturbation": "infected", "hours_post_perturbation": {"ge": 24}}``.
    perturbed_filter : dict or None
        obs filter for the perturbed reference (group Y). Filter-based style.
    """

    experiment: str
    control_wells: list[str] | None = None
    perturbed_wells: list[str] | None = None
    control_filter: dict | None = None
    perturbed_filter: dict | None = None

    @model_validator(mode="after")
    def _validate_refs(self) -> "WitnessLabelSource":
        control_styles = (self.control_wells is not None) + (self.control_filter is not None)
        perturbed_styles = (self.perturbed_wells is not None) + (self.perturbed_filter is not None)
        if control_styles != 1 or perturbed_styles != 1:
            raise ValueError(
                f"{self.experiment}: each side needs exactly one of wells / filter "
                "(control_wells XOR control_filter, perturbed_wells XOR perturbed_filter)"
            )
        if (self.control_wells is not None) != (self.perturbed_wells is not None):
            raise ValueError(f"{self.experiment}: mix of well-based and filter-based sides is not allowed")
        if self.control_wells is not None:
            if not self.control_wells or not self.perturbed_wells:
                raise ValueError(f"{self.experiment}: control_wells and perturbed_wells must both be non-empty")
            overlap = set(self.control_wells) & set(self.perturbed_wells)
            if overlap:
                raise ValueError(f"{self.experiment}: wells appear in both control and perturbed: {sorted(overlap)}")
        else:
            if not self.control_filter or not self.perturbed_filter:
                raise ValueError(f"{self.experiment}: control_filter and perturbed_filter must both be non-empty")
        return self


class WitnessGmmExperiment(WitnessLabelSource):
    """One experiment for witness-GMM labeling: refs plus its embeddings zarr.

    Extends :class:`WitnessLabelSource` (which carries ``experiment`` and the
    control/perturbed reference specs) with the path to the embeddings zarr the
    witness is scored on.

    Parameters
    ----------
    embeddings_zarr : str
        Path to the embeddings zarr (AnnData) whose ``.obs`` carries
        ``experiment``, ``marker``, ``fov_name``, ``id`` (or ``t``/``track_id``),
        and the ``condition_column``. The witness references and the scored cells
        both come from here.
    """

    embeddings_zarr: str


class WitnessGmmLabelsConfig(BaseModel):
    """Stage-A config: generate an annotation file from the MMD-witness + GMM.

    For each marker, the witness scores every cell against per-experiment
    control/perturbed references, a two-component GMM is fit on the perturbed
    cells' scores per condition, and confident cells are written out as an
    **annotation file** — a named biological-state column (``label_column``) with
    the real class vocabulary (``class_map``), keyed by cell exactly like a hand
    annotation. The Stage-B training path (``run-linear-classifiers`` with
    ``label_source="annotations"``) then consumes it unchanged.

    The label's *meaning* is named by the modality it is computed from: a witness
    over ``viral_sensor`` produces ``infection_state`` (infected/uninfected); over
    an organelle marker it produces ``organelle_remodeling_state``
    (remodel/noremodel). One config = one microscope/marker (no pooling, no LOT).

    Parameters
    ----------
    experiments : list[WitnessGmmExperiment]
        Per-experiment embeddings zarr + control/perturbed reference specs.
    label_column : str
        Name of the biological-state column written to the annotation file (e.g.
        ``"infection_state"``). This is the ``task`` Stage B trains on.
    class_map : dict[str, str]
        Maps the GMM gate outcome to the class vocabulary:
        ``{"positive": <perturbed class>, "negative": <control class>}`` — e.g.
        ``{"positive": "infected", "negative": "uninfected"}``.
    output_dir : str
        Directory for Stage-A outputs. The annotation file is written to
        ``<output_dir>/labels/<marker(s)>_<label_column>.<annotation_format>``
        (the marker prefix disambiguates sibling single-marker configs that share
        a ``label_column``; dropped when ``marker_filters`` is None) and diagnostic
        plots to ``<output_dir>/labels/plots/``. Keeping labels under a
        ``labels/`` subtree lets Stage B write its trained classifiers to a
        sibling ``classifiers/`` subtree under the same checkpoint root.
    annotation_format : str
        Annotation-file format, ``"csv"`` or ``"parquet"``. Default: ``"csv"``.
    marker_filters : list[str] or None
        Markers to label (one annotation column per config; usually one). None =
        all unique ``obs["marker"]``. Default: None.
    condition_column : str
        obs column whose distinct values define per-condition GMM fits and carry
        the biological condition (e.g. ``"perturbation"``). Default: ``"perturbation"``.
    gmm_pos_threshold : float
        GMM remodeled-component posterior at/above which a perturbed cell is a
        confident positive. Symmetrically, ``1 - gmm_pos_threshold`` is the bar for
        a confident **negative**: a perturbed cell at or below it is labeled the
        negative class (the pre-onset and bystander cells), so a perturbed well
        contributes both classes. Cells between the two bars are ambiguous and are
        left unlabeled. Default: 0.8 (so negatives at ≤ 0.2).
    mmd_pvalue_threshold : float
        Target FDR level for the significance gate. Each perturbed condition is
        MMD-permutation-tested against the control reference; the raw p-values
        across the whole run's (marker, condition) family are corrected with
        Benjamini-Yekutieli FDR control, and a condition is skipped when its
        **adjusted** p-value exceeds this level. Set to 1.0 to disable the gate.
        Default: 0.05.
    mmd_n_permutations : int
        Number of permutations for the MMD significance test. Default: 1000.
    bandwidth : float or None
        Gaussian RBF bandwidth for the witness kernel. None = median heuristic on
        the pooled (control, perturbed) reference. Default: None.
    max_reference_cells : int or None
        Subsample each reference group to at most this many cells before fitting
        the witness (bounds kernel cost). None = use all. Default: 5000.
    random_seed : int
        Seed for reference subsampling and the GMM. Default: 42.
    mmd_hpi_bin_hours : float or None
        If set, additionally compute MMD²(control, condition) **per time bin** of
        this width (hours post perturbation) and write an MMD-vs-HPI diagnostic
        plot — the population divergence kinetics, plus a control-vs-control null
        band. Requires an ``hours_post_perturbation`` obs column. None disables
        the diagnostic. Default: None.
    witness_time_bin_hours : float or None
        If set, score the witness with **time-matched references**: each cell is
        scored against control/perturbed reference cells drawn from its own
        ``hours_post_perturbation`` bin of this width, rather than against a single
        pooled all-timepoint reference. The DynaCLR embedding carries a strong
        time/culture axis (uninfected cells drift over a long timelapse), so a
        pooled reference leaks that axis into the witness score and the labels —
        worst for weak channels. Time-matching cancels the shared time component
        so the witness axis reflects perturbation, not culture-time. A single
        global bandwidth (median heuristic on the pooled reference) is kept so
        scores stay comparable across bins. None = pooled reference (original
        behavior). Requires an ``hours_post_perturbation`` obs column. Default: None.
    gate : str
        How confident positives are chosen from the witness scores — pick by
        whether the perturbed population is **bimodal** or merely **shifted**:

        - ``"gmm"``: two-component GMM on the perturbed scores; positive when the
          remodel-mode posterior ≥ ``gmm_pos_threshold``. Right when perturbation
          creates a distinct second state. Abstains (labels nothing) when the fit
          is not bimodal, so it cannot manufacture a split that is not there.
        - ``"percentile"``: cut at the ``control_fp_target`` quantile of the
          **control** scores. Fits nothing and assumes nothing, so it works when
          the perturbation shifts a unimodal population — the case where a GMM has
          no honest split to find. Never abstains. Its parameter is an error rate,
          which means the same thing across markers/plates/timepoints; a posterior
          cut does not.

        Default: ``"gmm"``.
    control_fp_target : float
        For ``gate="percentile"``: the fraction of control cells called positive,
        fixed **by construction** (the gate is a quantile of the control scores).
        Default: 0.05.
    negative_quantile : float
        For ``gate="percentile"``: the control quantile above which a cell is
        called negative — where the control distribution begins. Deliberately NOT
        ``1 - control_fp_target``: perturbed scores usually die out before the
        control's upper tail, so a mirrored cut labels almost nothing. Must exceed
        ``control_fp_target``. Default: 0.10.
    """

    experiments: list[WitnessGmmExperiment]
    label_column: str
    class_map: dict[str, str]
    output_dir: str
    annotation_format: str = "csv"
    marker_filters: list[str] | None = None
    condition_column: str = "perturbation"
    gmm_pos_threshold: float = 0.8
    mmd_pvalue_threshold: float = 0.05
    mmd_n_permutations: int = 1000
    bandwidth: float | None = None
    max_reference_cells: int | None = 5000
    witness_time_bin_hours: float | None = None
    gate: str = "gmm"
    control_fp_target: float = 0.05
    negative_quantile: float = 0.10
    random_seed: int = 42
    mmd_hpi_bin_hours: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> "WitnessGmmLabelsConfig":
        if not self.experiments:
            raise ValueError("witness_gmm_labels requires non-empty experiments")
        missing = {"positive", "negative"} - set(self.class_map)
        if missing:
            raise ValueError(f"class_map must define {sorted(missing)} (got keys {sorted(self.class_map)})")
        if not 0.0 < self.gmm_pos_threshold <= 1.0:
            raise ValueError(f"gmm_pos_threshold must be in (0, 1], got {self.gmm_pos_threshold}")
        if self.gate not in ("gmm", "percentile"):
            raise ValueError(f"gate must be 'gmm' or 'percentile', got {self.gate!r}")
        if not 0.0 < self.control_fp_target < 1.0:
            raise ValueError(f"control_fp_target must be in (0, 1), got {self.control_fp_target}")
        if not self.control_fp_target < self.negative_quantile < 1.0:
            raise ValueError(f"negative_quantile must be in (control_fp_target, 1), got {self.negative_quantile}")
        if not 0.0 < self.mmd_pvalue_threshold <= 1.0:
            raise ValueError(f"mmd_pvalue_threshold must be in (0, 1], got {self.mmd_pvalue_threshold}")
        if self.annotation_format not in ("csv", "parquet"):
            raise ValueError(f"annotation_format must be 'csv' or 'parquet', got {self.annotation_format!r}")
        return self


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
    obs_filter: dict[str, str] | None = None
    embedding_key: str | None = None
    representation: MMDRepresentationConfig = MMDRepresentationConfig()
    mmd: MMDSettings = MMDSettings()
    map_settings: MAPSettings = MAPSettings()
    temporal_bin_size: float | None = None
    combined_temporal_bin_size: float | None = None
    save_plots: bool = True
    combined_mode: bool = False
    name: str | None = None


class LinearClassifiersStepConfig(BaseModel):
    """Configuration for the orchestrated linear classifiers step.

    Parameters
    ----------
    label_source : {"annotations"}
        Where per-cell labels come from. ``"annotations"`` loads labels from
        per-experiment annotation files (``annotations`` + ``tasks``). Witness →
        GMM pseudo-labels are produced upstream by the ``witness-gmm-labels``
        (Stage A) command as an annotation file and consumed here unchanged —
        there is no separate witness label source.
    annotations : list[AnnotationSource]
        Per-experiment annotation files (CSV or parquet). Each entry maps an
        experiment name (matching obs["experiment"] in embeddings.zarr) to a
        path. May be hand annotations or a Stage-A witness-GMM annotation file.
    tasks : list[TaskSpec]
        Tasks to evaluate. Each task can optionally filter by marker.
    publish_dir : str or None
        Central LC registry root for this model (e.g.,
        ``/hpc/projects/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/``).
        When set, pipelines are published as a new versioned bundle
        (``vN/``) with a ``latest`` symlink update. When None, legacy
        behavior: write to ``output_dir/linear_classifiers/pipelines/``.
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
    split_groups_by : list[str] or None
        obs columns whose concatenation defines a "group" that must not
        be split across train and val. When set (e.g. ``["experiment",
        "fov_name", "track_id"]``), the train/val split is a
        ``GroupShuffleSplit`` keyed on the concatenated group id — no
        track lands in both halves. This kills track-level temporal
        leakage for SSL embeddings that pull same-track cells together
        (DynaCLR's positive pairs). When None, behavior is the legacy
        cell-level stratified ``train_test_split``. Default: None.
    control_normalize : bool
        Robust control-reference normalization of the embeddings before
        training/scoring: per experiment, per HPI bin, ``(x - control_median) /
        control_IQR`` referenced against ``uninfected`` cells. Default: False.
    control_normalize_bin_hours : float
        HPI bin width for control-reference normalization. Default: 2.0.
    """

    label_source: Literal["annotations"] = "annotations"
    annotations: list[AnnotationSource] = []
    tasks: list[TaskSpec] = []
    publish_dir: str | None = None
    use_scaling: bool = True
    use_pca: bool = False
    n_pca_components: int | None = None
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    solver: str = "liblinear"
    split_train_data: float = 0.8
    random_seed: int = 42
    split_groups_by: list[str] | None = None
    control_normalize: bool = False
    control_normalize_bin_hours: float = 2.0

    @model_validator(mode="after")
    def _validate_label_source(self) -> "LinearClassifiersStepConfig":
        # tasks are always required. annotations may be empty here when the
        # witness_gmm step auto-fills them at config-generation time; that
        # invariant is enforced by EvaluationConfig, which sees both steps.
        if not self.tasks:
            raise ValueError("label_source='annotations' requires non-empty tasks")
        return self


class AppendPredictionsStepConfig(BaseModel):
    """Configuration for the append-predictions step.

    Parameters
    ----------
    pipelines_dir : str or None
        Directory (or ``latest`` symlink) holding a published LC bundle
        with ``manifest.json`` and ``{task}_{marker}.joblib`` files.
        When None, defaults to ``output_dir/linear_classifiers/pipelines/``
        (legacy layout for runs that both train and apply LCs in the same
        eval). Set this explicitly for Wave-2 evaluations that apply
        pipelines trained by a separate Wave-1 run.
    """

    pipelines_dir: str | None = None


class AppendAnnotationsStepConfig(BaseModel):
    """Configuration for the append-annotations step.

    Used by Wave-2 evaluations that have annotation CSVs but do not train
    linear classifiers (e.g., alfi). Wave-1 evaluations historically
    sourced annotations from ``linear_classifiers.annotations``; this
    field lets datasets carry annotations independently of LC training.
    When both are set, this field takes precedence.

    Parameters
    ----------
    annotations : list[AnnotationSource]
        Per-experiment annotation CSVs to merge into per-experiment zarrs.
    """

    annotations: list[AnnotationSource] = []


class WitnessGmmLabelSource(BaseModel):
    """One witness-GMM label to generate: a marker mapped to a named state column.

    Each marker needs its own ``label_column``/``class_map`` because the label's
    meaning is set by the modality it is computed from (a viral-sensor witness →
    ``infection_state``, an organelle-marker witness → ``organelle_remodeling_state``).

    Parameters
    ----------
    marker : str
        obs["marker"] value to label (one annotation column per source).
    label_column : str
        Name of the biological-state column written to the annotation file
        (e.g. ``"infection_state"``). This is the ``task`` Stage B trains on.
    class_map : dict[str, str]
        Maps the GMM gate outcome to the class vocabulary. Must carry both a
        ``"positive"`` (perturbed class) and ``"negative"`` (control class) key,
        e.g. ``{"positive": "infected", "negative": "uninfected"}``.
    """

    marker: str
    label_column: str
    class_map: dict[str, str]

    @model_validator(mode="after")
    def _validate_class_map(self) -> "WitnessGmmLabelSource":
        missing = {"positive", "negative"} - set(self.class_map)
        if missing:
            raise ValueError(
                f"{self.marker}: class_map must have 'positive' and 'negative' keys; missing {sorted(missing)}"
            )
        return self


class WitnessGmmStepConfig(BaseModel):
    """Config-driven witness-GMM labeling step for the eval orchestrator.

    One block generates one Stage-A YAML per ``label_sources`` entry
    (``{output_dir}/configs/witness_gmm_{marker}.yaml``), each run by
    ``dynaclr witness-gmm-labels`` to write ``{output_dir}/labels/{marker}_{label_column}.csv``.
    The existing ``linear_classifiers`` step then consumes those CSVs as its
    ``annotations`` (the eval-config author points ``linear_classifiers.annotations[].path``
    at the generated label CSVs).

    The control/perturbed reference filters are shared across all label sources;
    each source scores its own marker's embeddings zarr
    (``{output_dir}/embeddings/{marker}.zarr``).

    Parameters
    ----------
    label_sources : list[WitnessGmmLabelSource]
        One entry per marker to label; each carries its ``label_column`` and
        ``class_map``.
    experiments : list[str]
        Experiment names present in ``obs["experiment"]`` of the embeddings
        zarrs. The witness masks references by ``obs["experiment"] == name``
        (see ``compute_marker_scores``), so the reference filters only apply
        within these experiments. One ``WitnessGmmExperiment`` is emitted per
        name, all sharing the control/perturbed filters below.
    control_filter : dict
        obs filter selecting the control reference (e.g.
        ``{"perturbation": "uninfected"}``). A value may be a scalar, a list, or
        a range-dict (``{lt, le, gt, ge}``).
    perturbed_filter : dict
        obs filter selecting the perturbed reference (e.g.
        ``{"perturbation": ["DENV"]}``).
    gate : str
        Gating method for confident positives: ``"gmm"`` (two-component mixture on
        the perturbed scores; abstains when not bimodal) or ``"percentile"`` (cut at
        the ``control_fp_target`` quantile of the control scores; fits nothing, so it
        handles a shifted-but-unimodal perturbed population). Default: ``"gmm"``.
    gmm_pos_threshold : float
        GMM remodeled-component posterior at/above which a perturbed cell is a
        confident positive. Symmetrically, ``1 - gmm_pos_threshold`` is the bar for
        a confident **negative**: a perturbed cell at or below it is labeled the
        negative class (the pre-onset and bystander cells), so a perturbed well
        contributes both classes. Cells between the two bars are ambiguous and are
        left unlabeled. Default: 0.8 (so negatives at ≤ 0.2).
    mmd_pvalue_threshold : float
        Adjusted-p-value ceiling for the MMD significance gate. Default: 0.05.
    mmd_n_permutations : int
        Number of permutations for the MMD significance test. Default: 1000.
    bandwidth : float or None
        Gaussian RBF bandwidth for the witness kernel. None = median heuristic.
        Default: None.
    max_reference_cells : int or None
        Subsample each reference group to at most this many cells. Default: 5000.
    condition_column : str
        obs column whose distinct values define per-condition GMM fits.
        Default: ``"perturbation"``.
    annotation_format : str
        Annotation-file format, ``"csv"`` or ``"parquet"``. Default: ``"csv"``.
    """

    label_sources: list[WitnessGmmLabelSource]
    experiments: list[str]
    control_filter: dict
    perturbed_filter: dict
    gate: str = "gmm"
    gmm_pos_threshold: float = 0.8
    control_fp_target: float = 0.05
    negative_quantile: float = 0.10
    mmd_pvalue_threshold: float = 0.05
    mmd_n_permutations: int = 1000
    bandwidth: float | None = None
    max_reference_cells: int | None = 5000
    condition_column: str = "perturbation"
    annotation_format: str = "csv"
    witness_time_bin_hours: float | None = None
    mmd_hpi_bin_hours: float | None = None
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
        plot, plot_combined, smoothness, mmd, witness_gmm, linear_classifiers,
        append_annotations, append_predictions.
        ``plot`` emits per-experiment scatter plots (fans out one job per
        experiment). ``plot_combined`` emits the joint cross-experiment
        plot only. List both to get both; list neither to skip plotting.
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
    witness_gmm : WitnessGmmStepConfig or None
        Config-driven witness-GMM labeling. Generates annotation CSVs the
        ``linear_classifiers`` step then consumes. None disables this step.
    linear_classifiers : LinearClassifiersStepConfig or None
        Linear classifier configuration. None disables this step.
    append_predictions : AppendPredictionsStepConfig or None
        Append-predictions configuration. Set ``pipelines_dir`` to apply
        pipelines from a separate eval run (e.g., Wave 2 fetching from the
        central LC registry). None keeps legacy behavior.
    mmd : list[MMDStepConfig]
        MMD evaluation blocks. Each block is an independent run with its own
        group_by, comparisons, and optional obs_filter. Empty list disables MMD.
    """

    training_config: str
    # ckpt_path is None for foundation-model baselines (e.g. DINOv3-frozen) where
    # weights are loaded from HuggingFace inside the model __init__ and there is
    # no Lightning checkpoint to restore from.
    ckpt_path: str | None = None
    cell_index_path: str | None = None
    output_dir: str
    steps: list[str] = ["predict", "split", "reduce_dimensionality", "reduce_combined", "plot", "smoothness"]
    predict: PredictStepConfig = PredictStepConfig()
    reduce_dimensionality: ReduceStepConfig = ReduceStepConfig()
    reduce_combined: ReduceCombinedStepConfig = ReduceCombinedStepConfig()
    smoothness: SmoothnessStepConfig = SmoothnessStepConfig()
    plot: PlotStepConfig = PlotStepConfig()
    witness_gmm: WitnessGmmStepConfig | None = None
    linear_classifiers: LinearClassifiersStepConfig | None = None
    append_annotations: AppendAnnotationsStepConfig | None = None
    append_predictions: AppendPredictionsStepConfig | None = None
    mmd: list[MMDStepConfig] = []

    @model_validator(mode="after")
    def _validate_lc_annotation_source(self) -> "EvaluationConfig":
        """LC needs annotations either explicitly or from an active witness step.

        When ``linear_classifiers`` is a step, its labels must come from
        somewhere: explicit ``linear_classifiers.annotations``, or the
        ``witness_gmm`` step (which writes the annotation CSVs the LC generator
        auto-fills). Fail fast if neither is present rather than emitting an LC
        config with an empty annotation list.
        """
        if "linear_classifiers" in self.steps and self.linear_classifiers is not None:
            has_explicit = bool(self.linear_classifiers.annotations)
            has_witness = "witness_gmm" in self.steps and self.witness_gmm is not None
            if not (has_explicit or has_witness):
                raise ValueError(
                    "linear_classifiers step needs annotations: set "
                    "linear_classifiers.annotations, or add 'witness_gmm' to steps "
                    "with a witness_gmm block to generate them."
                )
        return self

    @property
    def model_name(self) -> str:
        """Derive the model identifier from the training config filename stem.

        Example: ``DynaCLR-2D-MIP-BagOfChannels.yml`` → ``"DynaCLR-2D-MIP-BagOfChannels"``.
        Used as the ``feature_space`` tag in LC manifests and as the
        namespace prefix for predicted columns in output zarrs.
        """
        from pathlib import Path as _Path

        return _Path(self.training_config).stem
