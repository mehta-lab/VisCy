"""IO helpers for DTW pseudotime: template zarr layout + dataset routing.

Centralizes knowledge of the on-disk schemas so the pipeline scripts do
not have to duplicate ``zarr.open`` plumbing or re-derive dataset paths
from filename conventions. Two responsibilities:

- **Template zarr IO** (``save_template_zarr``, ``load_template_flavor``,
  ``read_template_attrs``, ``read_time_calibration``).
  The template zarr stores both PCA and raw flavors of a DBA template in
  one store, plus shared metadata (z-score params, t_key_event per cell,
  config snapshot, version provenance).

- **Embedding-zarr discovery** (``date_prefix_from_dataset_id``,
  ``find_embedding_zarr``, ``get_dynaclr_versions``).
  Resolves a dataset_id + filename pattern to the single zarr produced
  by the evaluation pipeline.
"""

from __future__ import annotations

import glob
import importlib.metadata as _metadata
import os
import subprocess
from pathlib import Path

import numpy as np
import zarr
from sklearn.decomposition import PCA

from dynaclr.pseudotime.dtw_alignment import TemplateResult


def date_prefix_from_dataset_id(dataset_id: str) -> str:
    """Extract a leading ``YYYY_MM_DD_`` prefix from ``dataset_id``.

    Many embedding zarrs are named with a date prefix derived from the
    experiment id. This helper recovers the prefix used to glob for the
    embedding file under the dataset's ``pred_dir``.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier such as ``2024_07_24_A549_ZIKV_SEC61``.

    Returns
    -------
    str
        Prefix including the trailing underscore, or an empty string if
        the id has fewer than three underscore-separated parts.
    """
    parts = dataset_id.split("_")
    if len(parts) < 3:
        return ""
    return "_".join(parts[:3]) + "_"


def find_embedding_zarr(pred_dir: str | Path, pattern: str) -> str:
    """Find the single embedding zarr matching ``pattern`` in ``pred_dir``.

    Parameters
    ----------
    pred_dir : str or Path
        Directory containing the per-dataset embedding zarrs.
    pattern : str
        Glob pattern (typically ``date_prefix + embedding_pattern``).

    Returns
    -------
    str
        Absolute path to the matching zarr.

    Raises
    ------
    FileNotFoundError
        If zero or more than one zarr matches the pattern.
    """
    matches = glob.glob(str(Path(pred_dir) / pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No zarr matching {pattern} in {pred_dir}")
    if len(matches) > 1:
        names = sorted(Path(m).name for m in matches)
        raise FileNotFoundError(f"Multiple zarrs match {pattern}: {names}")
    return matches[0]


def get_dynaclr_versions() -> dict[str, str]:
    """Return a dict of code/library versions for template provenance.

    Captured fields:

    - ``viscy_git_sha``: short SHA of the current repo HEAD, or
      ``"unknown"`` if the repo is unavailable.
    - ``dtaidistance_version``: installed dtaidistance package version.
    - ``sklearn_version``: installed scikit-learn version.
    - ``numpy_version``: installed numpy version.

    Stamping these into every template zarr is what lets a future
    consumer reproduce or invalidate a published template after the
    embedding model or library stack moves.
    """
    sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    versions = {"viscy_git_sha": sha}
    for pkg in ("dtaidistance", "scikit-learn", "numpy"):
        try:
            versions[f"{pkg.replace('-', '_')}_version"] = _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            versions[f"{pkg.replace('-', '_')}_version"] = "unknown"
    return versions


def compute_tau_event_band(
    template: np.ndarray,
    threshold_fraction: float = 0.5,
) -> tuple[float, float]:
    """Compute the half-rise band of a template's first-derivative magnitude.

    The template's "event" is the region of fastest change. We compute
    the per-position rate of change as the L2 norm of consecutive
    template differences, then return the pseudotime band where the
    rate exceeds ``threshold_fraction`` of its maximum.

    Per discussion §3.4 and the locked execution plan: τ_event is a
    band, not a point, because (1) the template's argmax-derivative has
    a resolution floor of 1/n_frames, and (2) DBA averages flatten
    cell-specific kinks so the template derivative is structurally
    smoother than any individual cell's. The band honestly reflects
    template-derivative resolution.

    Parameters
    ----------
    template : np.ndarray
        DBA template, shape ``(T, D)`` where T is the number of
        template positions and D the embedding dimension.
    threshold_fraction : float
        Fraction of the maximum derivative magnitude that defines the
        band edges. Default 0.5 (half-rise band).

    Returns
    -------
    tuple[float, float]
        ``(τ_lo, τ_hi)`` in pseudotime ∈ [0, 1]. If the template has
        fewer than two positions or the derivative is degenerate,
        returns ``(0.0, 1.0)``.
    """
    if template.ndim != 2 or template.shape[0] < 2:
        return (0.0, 1.0)

    diffs = np.diff(template, axis=0)
    rate = np.linalg.norm(diffs, axis=1)  # shape (T-1,)
    if rate.size == 0 or float(rate.max()) <= 0:
        return (0.0, 1.0)

    threshold = threshold_fraction * float(rate.max())
    above = rate >= threshold
    indices = np.where(above)[0]
    if indices.size == 0:
        return (0.0, 1.0)

    # Map derivative-position indices to pseudotime midpoints.
    # rate[i] reports the change from template[i] to template[i+1], so
    # the rate's natural pseudotime is the midpoint between positions i
    # and i+1: τ = (i + 0.5) / (T - 1).
    n_positions = template.shape[0]
    tau_lo = float(indices.min() + 0.5) / float(n_positions - 1)
    tau_hi = float(indices.max() + 0.5) / float(n_positions - 1)
    return (tau_lo, tau_hi)


def _save_flavor(group, result: TemplateResult, flavor_name: str) -> None:
    """Serialize one ``TemplateResult`` flavor into a zarr group."""
    group.create_array("template", data=result.template)
    if result.time_calibration is not None:
        group.create_array("time_calibration", data=result.time_calibration)
    if result.template_labels is not None:
        labels_grp = group.create_group("template_labels")
        for col, fractions in result.template_labels.items():
            labels_grp.create_array(col, data=fractions)
    if result.pca is not None:
        group.create_array("components", data=result.pca.components_)
        group.create_array("mean", data=result.pca.mean_)
        group.create_array("explained_variance", data=result.pca.explained_variance_)
        group.create_array("explained_variance_ratio", data=result.pca.explained_variance_ratio_)
        group.attrs["n_components"] = int(result.pca.n_components_)
        group.attrs["explained_variance"] = float(result.explained_variance or 0.0)

    # τ_event band: half-rise of the template-derivative magnitude.
    # Stored per-flavor because raw and PCA templates have different
    # geometries and may yield slightly different bands.
    tau_lo, tau_hi = compute_tau_event_band(result.template)
    group.create_array("tau_event_band", data=np.array([tau_lo, tau_hi], dtype=np.float64))

    group.attrs["template_id"] = result.template_id
    group.attrs["n_input_tracks"] = int(result.n_input_tracks)


def save_template_zarr(
    out_path: str | Path,
    raw_result: TemplateResult,
    pca_result: TemplateResult,
    *,
    template_name: str,
    config_snapshot: dict,
    anchor_label: str,
    anchor_positive: str,
    aggregator: str,
    t_key_event_per_cell: np.ndarray,
    build_frame_intervals_minutes: dict[str, float],
    template_duration_minutes: float,
    extra_attrs: dict | None = None,
) -> None:
    """Serialize both template flavors + shared metadata into a single zarr.

    Provenance fields (``viscy_git_sha``, ``dtaidistance_version``, …)
    are stamped automatically via :func:`get_dynaclr_versions`.

    Parameters
    ----------
    out_path : str or Path
        Destination zarr directory. Will be created/overwritten.
    raw_result, pca_result : TemplateResult
        The two template flavors to serialize.
    template_name : str
        Identifier used downstream to name the alignment parquet.
    config_snapshot : dict
        Full config under which this template was built. Stored verbatim.
    anchor_label, anchor_positive : str
        Label column and its positive value (e.g.
        ``("infection_state", "infected")``).
    aggregator : str
        Aggregator name (currently always ``"dba"``).
    t_key_event_per_cell : np.ndarray
        Per-cell event timepoints in the same order as
        ``raw_result.template_cell_ids``.
    build_frame_intervals_minutes : dict[str, float]
        Per-dataset frame intervals so consumers can apply minute-based
        guards without guessing the template's time scale.
    template_duration_minutes : float
        Duration of the calibrated template in real minutes.
    extra_attrs : dict or None
        Additional attrs to merge into the store. Useful for
        method-specific metadata not covered by the shared schema.
    """
    store = zarr.open(str(out_path), mode="w")

    _save_flavor(store.create_group("raw"), raw_result, "raw")
    _save_flavor(store.create_group("pca"), pca_result, "pca")

    if raw_result.zscore_params:
        zgrp = store.create_group("zscore_params")
        for ds_id, (mean, std) in raw_result.zscore_params.items():
            ds_grp = zgrp.create_group(ds_id)
            ds_grp.create_array("mean", data=mean)
            ds_grp.create_array("std", data=std)

    store.create_array("t_key_event", data=np.asarray(t_key_event_per_cell))

    store.attrs["template_name"] = template_name
    store.attrs["template_cell_ids"] = [list(c) for c in raw_result.template_cell_ids]
    store.attrs["anchor_label"] = anchor_label
    store.attrs["anchor_positive"] = anchor_positive
    store.attrs["aggregator"] = aggregator
    store.attrs["template_duration_minutes"] = float(template_duration_minutes)
    store.attrs["build_frame_intervals_minutes"] = {k: float(v) for k, v in build_frame_intervals_minutes.items()}
    store.attrs["config_snapshot"] = config_snapshot

    versions = get_dynaclr_versions()
    for k, v in versions.items():
        store.attrs[k] = v

    if extra_attrs:
        for k, v in extra_attrs.items():
            store.attrs[k] = v


def load_template_flavor(template_path: str | Path, flavor: str) -> tuple[TemplateResult, dict]:
    """Load one flavor from a two-flavor template zarr.

    Parameters
    ----------
    template_path : str or Path
        Path to the template zarr written by :func:`save_template_zarr`.
    flavor : {"raw", "pca"}
        Which flavor to materialize.

    Returns
    -------
    (TemplateResult, dict)
        The selected flavor's :class:`TemplateResult` (template + PCA
        for the ``"pca"`` flavor + shared z-score params), and the raw
        attrs dict for anything else the caller needs.

    Raises
    ------
    ValueError
        If ``flavor`` is not ``"raw"`` or ``"pca"``.
    KeyError
        If the requested flavor is not present in the store.
    """
    store = zarr.open(str(template_path), mode="r")
    attrs = dict(store.attrs)

    if flavor not in ("raw", "pca"):
        raise ValueError(f"flavor must be 'raw' or 'pca', got {flavor!r}")
    if flavor not in store:
        raise KeyError(f"Flavor {flavor!r} not in template zarr {template_path}")
    grp = store[flavor]

    template = np.asarray(grp["template"])
    time_calibration = np.asarray(grp["time_calibration"]) if "time_calibration" in grp else None

    template_labels = None
    if "template_labels" in grp:
        tl_grp = grp["template_labels"]
        template_labels = {col: np.asarray(tl_grp[col]) for col in tl_grp}

    pca = None
    if flavor == "pca" and "components" in grp:
        n_comp = int(grp.attrs["n_components"])
        pca = PCA(n_components=n_comp)
        pca.components_ = np.asarray(grp["components"])
        pca.mean_ = np.asarray(grp["mean"])
        if "explained_variance" in grp:
            pca.explained_variance_ = np.asarray(grp["explained_variance"])
        pca.explained_variance_ratio_ = np.asarray(grp["explained_variance_ratio"])
        pca.n_components_ = n_comp
        pca.n_features_in_ = pca.components_.shape[1]

    zscore_params: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "zscore_params" in store:
        zgrp = store["zscore_params"]
        for ds_id in zgrp:
            zscore_params[ds_id] = (
                np.asarray(zgrp[ds_id]["mean"]),
                np.asarray(zgrp[ds_id]["std"]),
            )

    template_id = str(grp.attrs.get("template_id", ""))
    n_input_tracks = int(grp.attrs.get("n_input_tracks", 0))
    cell_ids = [tuple(c) for c in attrs.get("template_cell_ids", [])]

    result = TemplateResult(
        template=template,
        template_id=template_id,
        pca=pca,
        zscore_params=zscore_params,
        template_cell_ids=cell_ids,
        n_input_tracks=n_input_tracks,
        explained_variance=float(grp.attrs.get("explained_variance", 0.0)) or None,
        template_labels=template_labels,
        time_calibration=time_calibration,
    )
    return result, attrs


def read_template_attrs(template_path: str | Path) -> dict:
    """Read the top-level attrs of a template zarr.

    Convenience wrapper for the common case where a downstream script
    only needs the metadata (config snapshot, anchor label, version
    stamps) and not the template arrays themselves.
    """
    return dict(zarr.open(str(template_path), mode="r").attrs)


def read_time_calibration(template_path: str | Path, flavor: str) -> np.ndarray:
    """Read the per-position ``time_calibration`` array for a flavor.

    Returns the array of mean ``t_relative_minutes`` per template
    position, used for converting DTW pseudotime back to real minutes
    in downstream timing analyses.

    Parameters
    ----------
    template_path : str or Path
        Path to the template zarr.
    flavor : {"raw", "pca"}
        Which flavor's calibration to read.

    Raises
    ------
    KeyError
        If the flavor or its ``time_calibration`` array is missing.
    """
    if flavor not in ("raw", "pca"):
        raise ValueError(f"flavor must be 'raw' or 'pca', got {flavor!r}")
    grp = zarr.open(str(template_path), mode="r")[flavor]
    if "time_calibration" not in grp:
        raise KeyError(f"time_calibration missing for flavor {flavor!r} in {template_path}")
    return np.asarray(grp["time_calibration"])


def read_tau_event_band(template_path: str | Path, flavor: str) -> tuple[float, float]:
    """Read the τ_event band ``[τ_lo, τ_hi]`` from a template flavor.

    The band is computed at template-build time (see
    :func:`compute_tau_event_band`) and stored alongside the template
    arrays. Returns ``(0.0, 1.0)`` if the band array is missing
    (templates built before the band feature was added).

    Parameters
    ----------
    template_path : str or Path
        Path to the template zarr.
    flavor : {"raw", "pca"}
        Which flavor's band to read.
    """
    if flavor not in ("raw", "pca"):
        raise ValueError(f"flavor must be 'raw' or 'pca', got {flavor!r}")
    grp = zarr.open(str(template_path), mode="r")[flavor]
    if "tau_event_band" not in grp:
        return (0.0, 1.0)
    band = np.asarray(grp["tau_event_band"])
    return (float(band[0]), float(band[1]))
