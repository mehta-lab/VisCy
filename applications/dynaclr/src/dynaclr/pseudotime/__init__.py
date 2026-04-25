"""DTW-based pseudotime alignment for cellular dynamics from DynaCLR embeddings.

Public API:

- :func:`build_template`             — fit a DBA template from annotated trajectories.
- :func:`dtw_align_tracks`           — warp query tracks onto a template.
- :func:`alignment_results_to_dataframe` — flatten alignment results to a dataframe.
- :func:`extract_dtw_pseudotime`     — pull per-track pseudotime from results.
- :func:`classify_response_groups`   — k-means cluster cells by alignment summary.
- :class:`TemplateResult`            — template + PCA + z-score params bundle.
- :class:`AlignmentResult`           — per-track DTW alignment bundle.
- :data:`DEFAULT_POSITIVE_CLASSES`   — default infection-label mapping.

IO helpers:

- :func:`save_template_zarr`         — write a two-flavor template zarr with provenance.
- :func:`load_template_flavor`       — read one flavor (raw or pca) from a template.
- :func:`read_template_attrs`        — read just the top-level attrs.
- :func:`read_time_calibration`      — read the per-position time calibration array.
- :func:`find_embedding_zarr`        — glob the embedding zarr under a dataset's pred_dir.
- :func:`date_prefix_from_dataset_id` — extract the ``YYYY_MM_DD_`` prefix.
- :func:`get_dynaclr_versions`       — capture viscy/library versions for provenance.

Lower-level helpers and legacy modules (``alignment``, ``signals``,
``metrics``, ``plotting``, ``evaluation``) are still importable but are
not part of the curated public surface.
"""

from dynaclr.pseudotime.dtw_alignment import (
    DEFAULT_POSITIVE_CLASSES,
    AlignmentResult,
    TemplateResult,
    alignment_results_to_dataframe,
    build_template,
    classify_response_groups,
    dtw_align_tracks,
    extract_dtw_pseudotime,
)
from dynaclr.pseudotime.io import (
    date_prefix_from_dataset_id,
    find_embedding_zarr,
    get_dynaclr_versions,
    load_template_flavor,
    read_template_attrs,
    read_time_calibration,
    save_template_zarr,
)

__all__ = [
    "DEFAULT_POSITIVE_CLASSES",
    "AlignmentResult",
    "TemplateResult",
    "alignment_results_to_dataframe",
    "build_template",
    "classify_response_groups",
    "date_prefix_from_dataset_id",
    "dtw_align_tracks",
    "extract_dtw_pseudotime",
    "find_embedding_zarr",
    "get_dynaclr_versions",
    "load_template_flavor",
    "read_template_attrs",
    "read_time_calibration",
    "save_template_zarr",
]
